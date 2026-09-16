#!/usr/bin/env python3
"""在 Jetson 上用 TensorRT Python API 从 ONNX 重建 engine（INT8 需自带校准器）。

为什么需要它：仓库里没有 INT8 校准缓存（.cache），而 `trtexec --int8` 没有校准数据
就无法真正启用 INT8。本脚本用 pycuda 实现 IInt8EntropyCalibrator2，从视频抽帧校准，
并把校准表缓存成 .cache 供后续复用（重建一次之后就不用再校准）。

用法：
  # 1) 只看 ONNX 输入信息（不构建）
  python3 build_engine.py --onnx <path> --info

  # 2) 首次构建 INT8（会校准，较慢），并写出 .cache
  python3 build_engine.py --onnx <path> --out /tmp/x.engine --precision int8 \
      --calib-video /home/zhaoqian/prof/test300.mp4 --calib-frames 200 \
      --calib-cache /home/zhaoqian/prof/lm.cache --workspace 2048

  # 3) 之后复用 .cache 快速重建
  python3 build_engine.py --onnx <path> --out /tmp/x.engine --precision int8 \
      --calib-cache /home/zhaoqian/prof/lm.cache --workspace 2048

  # 4) FP16 / FP32
  python3 build_engine.py --onnx <path> --out /tmp/y.engine --precision fp16 --workspace 2048

⚠️ 校准数据用 test300.mp4 只够做「性能」验证；交付必须用业务真实场景数据重新校准，
   否则 INT8 精度没有保证。
"""
import argparse
import os
import sys

import numpy as np


def log(msg):
    print(f'[build_engine] {msg}', flush=True)


def load_calib_frames(video, count, width, height, mean, std, stride=1):
    """从视频抽帧，按 (1,3,H,W) fp32 做校准输入。

    注意：这里是「近似」app 的预处理（直接 resize，不做 letterbox）。
    校准只关心激活分布，近似即可；但若要复现精度，应改成与 app 完全一致。
    """
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f'打不开视频: {video}')
    frames = []
    idx = 0
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - mean) / std
            chw = np.transpose(rgb, (2, 0, 1))
            frames.append(np.ascontiguousarray(chw))
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError('一帧都没抽到')
    log(f'校准帧: {len(frames)} 张 -> {frames[0].shape}')
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--out')
    ap.add_argument('--precision', choices=['int8', 'fp16', 'fp32'], default='int8')
    ap.add_argument('--workspace', type=int, default=2048, help='MB，默认 2048；trtexec 默认只有 16MB')
    ap.add_argument('--calib-video')
    ap.add_argument('--calib-frames', type=int, default=200)
    ap.add_argument('--calib-cache')
    # ⚠️ 默认值必须与 app 一致：depth_model.cpp 走的是 is_normalize=false 分支
    #    （mean=0/std=1），预处理 kernel 是 (val/255 - mean)/std → 输入范围 [0,1]。
    #    这里曾误用 ImageNet 的 0.45/0.225，那会让校准分布整体偏移、激活范围统计错。
    ap.add_argument('--calib-mean', type=float, default=0.0)
    ap.add_argument('--calib-std', type=float, default=1.0)
    ap.add_argument('--info', action='store_true')
    ap.add_argument('--no-fp16', action='store_true', help='INT8 时不额外开启 FP16')
    ap.add_argument('--timing-cache')
    args = ap.parse_args()

    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.INFO)

    if args.info:
        with open(args.onnx, 'rb') as f:
            data = f.read()
        net = trt.Builder(trt_logger).create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(net, trt_logger)
        if not parser.parse(data):
            for i in range(parser.num_errors):
                print('  parse error:', parser.get_error(i))
            sys.exit(1)
        for i in range(net.num_inputs):
            t = net.get_input(i)
            log(f'input[{i}] name={t.name} shape={tuple(t.shape)} dtype={t.dtype}')
        for i in range(net.num_outputs):
            t = net.get_output(i)
            log(f'output[{i}] name={t.name} shape={tuple(t.shape)} dtype={t.dtype}')
        log(f'层数={net.num_layers}')
        return

    if not args.out:
        sys.exit('--out 是必需的（除非只 --info）')

    import pycuda.driver as cuda
    cuda.init()

    # ⚠️ 必须显式创建并 push 一个 CUDA context，否则校准器里的 cuda.mem_alloc /
    #    memcpy_htod 会报 "invalid device context"（pycuda 只 cuda.init() 并不建上下文），
    #    TRT 的 build_serialized_network 也需要 current context 才能跑 tactic profiling。
    #    注意：pycuda 2019.1 **没有** retain_primary_context（Device.__getattr__ 是个
    #    兜底转发到 get_attribute() 的钩子，访问不存在的名字只会抛 AttributeError），
    #    只能 make_context() 建非 primary 上下文 —— 这也是 TRT 官方 python 示例
    #    （pycuda.autoinit）的路径，实测可用。
    _dev = cuda.Device(0)
    _ctx = _dev.make_context()
    _ctx.push()
    log(f'CUDA device: {_dev.name()}')

    # ---------- 校准器 ----------
    class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, frames, cache_file, elem_shape, batch=1):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.frames = frames
            self.batch = batch
            self.cache_file = cache_file
            self.pos = 0
            # ⚠️ 显存按「网络输入形状」分配，不要用 frames[0].shape —— 复用 .cache 时
            #    frames 为空列表（不抽帧），照 frames[0] 取会 IndexError。
            self.dev_mem = cuda.mem_alloc(int(np.prod(elem_shape)) * 4 * batch)

        def get_batch_size(self):
            return self.batch

        def get_batch(self, names):
            if not self.frames:
                return None  # 走 cache 模式：没有图像数据，TRT 直接读缓存
            if self.pos + self.batch > len(self.frames):
                log('校准批次用尽')
                return None
            chunk = np.ascontiguousarray(self.frames[self.pos:self.pos + self.batch])
            cuda.memcpy_htod(self.dev_mem, chunk)
            self.pos += self.batch
            if self.pos % 50 == 0 or self.pos >= len(self.frames):
                log(f'  校准进度 {self.pos}/{len(self.frames)}')
            return [int(self.dev_mem)]

        def read_calibration_cache(self):
            if self.cache_file and os.path.exists(self.cache_file):
                log(f'读取校准缓存 {self.cache_file}')
                with open(self.cache_file, 'rb') as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            if self.cache_file:
                with open(self.cache_file, 'wb') as f:
                    f.write(cache)
                log(f'写出校准缓存 {self.cache_file} ({len(cache)} bytes)')

    log(f'读取 ONNX {args.onnx} ({os.path.getsize(args.onnx)/1e6:.1f} MB)')
    with open(args.onnx, 'rb') as f:
        onnx_data = f.read()

    builder = trt.Builder(trt_logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, trt_logger)
    if not parser.parse(onnx_data):
        for i in range(parser.num_errors):
            print('  parse error:', parser.get_error(i))
        sys.exit(1)

    in_shape = tuple(net.get_input(0).shape)
    in_name = net.get_input(0).name
    log(f'网络输入 {in_name} {in_shape}，层数 {net.num_layers}')

    config = builder.create_builder_config()
    # TRT 8.2 用 max_workspace_size；新 API 是 set_memory_pool_limit
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace << 20)
    else:
        config.max_workspace_size = args.workspace << 20
    log(f'workspace = {args.workspace} MB')

    if args.timing_cache and os.path.exists(args.timing_cache):
        with open(args.timing_cache, 'rb') as f:
            tc = config.create_timing_cache(f.read())
        config.set_timing_cache(tc, False)
        log(f'加载 timing cache {args.timing_cache}')

    if args.precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        if not args.no_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            log('已开启 FP16（INT8 之外的兜底精度）')
        if not args.calib_video and not (args.calib_cache and os.path.exists(args.calib_cache)):
            sys.exit('INT8 需要 --calib-video（抽帧校准）或已存在的 --calib-cache')
        _, _c, _h, _w = in_shape
        elem_shape = (int(_c), int(_h), int(_w))  # CHW，与 load_calib_frames 输出一致
        frames = []
        if not (args.calib_cache and os.path.exists(args.calib_cache)):
            frames = load_calib_frames(args.calib_video, args.calib_frames, int(_w), int(_h),
                                       args.calib_mean, args.calib_std)
        else:
            log(f'复用校准缓存 {args.calib_cache}（不抽帧）')
        config.int8_calibrator = EntropyCalibrator(frames, args.calib_cache, elem_shape)
    elif args.precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)

    log('开始构建（Jetson Nano 上可能要 5-10 分钟，请耐心）…')
    serialized = builder.build_serialized_network(net, config)
    if serialized is None:
        sys.exit('构建失败：build_serialized_network 返回 None（看上面 TRT 日志）')

    with open(args.out, 'wb') as f:
        f.write(serialized)
    log(f'完成 -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)')

    if args.timing_cache:
        with open(args.timing_cache, 'wb') as f:
            f.write(config.get_timing_cache().serialize())
        log(f'写出 timing cache {args.timing_cache}')

    # ⚠️ 必须显式 pop 掉上下文：否则 pycuda 在解释器退出时会报
    #    "PyCUDA ERROR: The context stack was not empty upon module cleanup"
    #    并以 rc=134(SIGABRT) 退出 —— 引擎其实已经写好了，但退出码会让外层脚本误判失败。
    try:
        _ctx.pop()
    except Exception:
        pass
    log('上下文已释放，正常退出')


if __name__ == '__main__':
    main()

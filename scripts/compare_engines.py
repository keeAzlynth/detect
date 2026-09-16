#!/usr/bin/env python3
"""对比多个 engine 在**完全相同输入**下的输出差异 —— INT8 重新校准后的精度体检。

为什么需要：用一段测试视频做校准只保证「性能」，不能保证精度。上线前至少要知道
新引擎的输出相对现役引擎偏移了多少。这个脚本给出可量化的数字（MAE / max|Δ|）。

用法:
  python3 compare_engines.py --video <mp4> --frames 20 \
      --engines 现役=/path/a.engine 新=/path/b.engine [参考fp16=/path/c.engine]
"""
import argparse
import os
import sys

import numpy as np

import pycuda.driver as cuda
import tensorrt as trt


def load_engine(path, logger):
    with open(path, 'rb') as f:
        data = f.read()
    runtime = trt.Runtime(logger)
    eng = runtime.deserialize_cuda_engine(data)
    if eng is None:
        sys.exit(f'反序列化失败: {path}')
    return eng


class Runner:
    """一个 engine 的推理封装：输入输出按 binding 名自动识别。"""

    def __init__(self, name, path, logger, stream):
        self.name = name
        self.path = path
        self.engine = load_engine(path, logger)
        self.ctx = self.engine.create_execution_context()
        self.stream = stream
        self.inputs, self.outputs, self.bindings = [], [], []
        self.host, self.device = {}, {}

        for i in range(self.engine.num_bindings):
            bname = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = int(np.prod(shape))
            dev = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self.device[bname] = dev
            self.bindings.append(int(dev))
            if self.engine.binding_is_input(i):
                self.inputs.append((bname, shape, dtype))
            else:
                h = cuda.pagelocked_empty(size, dtype)
                self.host[bname] = h
                self.outputs.append((bname, shape, dtype))
        self.in_name, self.in_shape, self.in_dtype = self.inputs[0]
        print(f'  [{self.name}] in={self.in_name}{self.in_shape} '
              f'out={[o[0] for o in self.outputs]} file={os.path.getsize(path)/1e6:.1f}MB')

    def infer(self, tensor):
        cuda.memcpy_htod(self.device[self.in_name], tensor)
        self.ctx.execute_v2(self.bindings)
        res = {}
        for bname, shape, _ in self.outputs:
            cuda.memcpy_dtoh(self.host[bname], self.device[bname])
            res[bname] = self.host[bname].copy()
        self.stream.synchronize()
        return res


def prep(frame, w, h, mean=0.0, std=1.0):
    """与 app 的 depthPreprocess 一致：resize + BGR→RGB + (v/255-mean)/std + CHW。"""
    import cv2
    img = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - mean) / std
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--frames', type=int, default=20)
    ap.add_argument('--engines', nargs='+', required=True, help='名字=路径')
    args = ap.parse_args()

    import cv2

    cuda.init()
    dev = cuda.Device(0)
    cuda_ctx = dev.make_context()   # pycuda 2019.1 没有 retain_primary_context
    cuda_ctx.push()
    stream = cuda.Stream()
    logger = trt.Logger(trt.Logger.ERROR)

    runs = []
    for spec in args.engines:
        name, path = spec.split('=', 1)
        runs.append(Runner(name, path, logger, stream))
    base = runs[0]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f'打不开视频 {args.video}')

    w, h = base.in_shape[3], base.in_shape[2]
    acc = {r.name: {o[0]: [] for o in r.outputs} for r in runs}
    n = 0
    while n < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        tensor = prep(frame, w, h)
        for r in runs:
            out = r.infer(tensor)
            for k, v in out.items():
                acc[r.name][k].append(v.astype(np.float64))
        n += 1
    cap.release()
    print(f'\n对比帧数: {n}')

    for out_name in [o[0] for o in base.outputs]:
        print(f'\n=== 输出 {out_name} ===')
        ref = np.stack(acc[base.name][out_name])
        for r in runs:
            cur = np.stack(acc[r.name][out_name])
            print(f'  {r.name:<12} range=[{cur.min():+.4f}, {cur.max():+.4f}] mean={cur.mean():+.4f}')
            if r is base:
                continue
            d = np.abs(cur - ref)
            rng = max(abs(ref.max() - ref.min()), 1e-6)
            print(f'    vs {base.name}: MAE={d.mean():.5f}  max|d|={d.max():.5f}  '
                  f'MAE/range={d.mean()/rng*100:.2f}%  '
                  f'P99|d|={np.percentile(d, 99):.5f}')
    print('\nCHECK_DONE')


if __name__ == '__main__':
    main()

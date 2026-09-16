#!/usr/bin/env python3
"""构建前预检：只验证 build_engine.py 里风险最高的两处，不启动真正的 engine 构建。

检查项：
  1) pycuda retain_primary_context + push 是否可用，cuda.mem_alloc / memcpy_htod 是否正常
  2) load_calib_frames 能否抽帧、shape 是否为 CHW、数值范围是否与 app 一致（[0,1]）
  3) 用真实 ONNX 跑一遍 OnnxParser.parse，确认层数与输入形状
"""
import sys
from importlib.machinery import SourceFileLoader

import numpy as np

sys.path.insert(0, '/home/zhaoqian/prof')
be = SourceFileLoader('build_engine', '/home/zhaoqian/prof/build_engine.py').load_module()

ONNX = '/home/zhaoqian/depth-detect-turbo/model/onnx/lite-mono-tiny/lite-mono-tiny_192x640_op11.onnx'
VIDEO = '/home/zhaoqian/prof/test300.mp4'

import pycuda.driver as cuda
cuda.init()
dev = cuda.Device(0)
ctx = dev.make_context()   # pycuda 2019.1 没有 retain_primary_context
ctx.push()
print('[ok] context push:', dev.name())

# --- 2) 抽帧 ---
frames = be.load_calib_frames(VIDEO, 8, 640, 192, 0.0, 1.0)
print('[ok] frames:', len(frames), frames[0].shape, frames[0].dtype)
print('     value range: min=%.4f max=%.4f mean=%.4f' %
      (frames[0].min(), frames[0].max(), frames[0].mean()))
assert frames[0].shape == (3, 192, 640), 'shape 不符'
assert 0.0 <= frames[0].min() and frames[0].max() <= 1.0001, '归一化范围不对，应落在 [0,1]'

# --- 1) 显存拷贝 ---
buf = cuda.mem_alloc(int(np.prod(frames[0].shape)) * 4)
cuda.memcpy_htod(buf, np.ascontiguousarray(frames[0]))
print('[ok] mem_alloc + memcpy_htod 正常')

# --- 3) ONNX 解析 ---
import tensorrt as trt
lg = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(lg)
net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(net, lg)
with open(ONNX, 'rb') as f:
    ok = parser.parse(f.read())
print('[ok] ONNX parse:', ok, 'num_layers=', net.num_layers)
print('     input:', net.get_input(0).name, tuple(net.get_input(0).shape))
for i in range(net.num_outputs):
    print('     output[%d]:' % i, net.get_output(i).name, tuple(net.get_output(i).shape))

cfg = builder.create_builder_config()
if hasattr(cfg, 'set_memory_pool_limit'):
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2048 << 20)
    print('[ok] set_memory_pool_limit(WORKSPACE, 2048MB)')
else:
    cfg.max_workspace_size = 2048 << 20
    print('[ok] max_workspace_size = 2048MB (旧 API)')
print('ALL_PRECHECK_PASSED')

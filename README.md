# depth-detect-turbo

> Jetson Nano 实时目标检测 + 单目深度估计加速版（重构自 [yzfzzz/depth-detect](https://github.com/yzfzzz/depth-detect)）

YOLOv8n 目标检测 + Lite-Mono-Tiny 单目深度估计的双模型实时管线，TensorRT INT8 引擎加速，
预处理与后处理全部在 GPU 上完成。面向 Jetson Nano（Tegra X1）这类算力受限的边缘设备。

---

## 一、核心特性

- **双模型推理**：YOLOv8n @640×640 + Lite-Mono-Tiny @192×640，TensorRT 8.2 INT8 引擎（FP16 / FP32 备选）
- **全 GPU 管线**：CUDA 预处理（resize / 换布局 / 归一化）+ CUDA 后处理（NMS），`--use_fast_math`
- **目标跟踪**：ByteTrack + 卡尔曼滤波，输出目标运动状态（趋近 / 远离 / 加减速）
- **异步重叠**：读帧与推理重叠；YOLO 与 Depth 各跑独立 CUDA 流（非阻塞）
- **绘图 / 落盘不占关键路径**：`DrawSaveWorker` 独立线程，与下一帧 GPU 推理重叠
- **零分配稳态**：单 `Pipeline` 实例 + 2 槽位帧缓冲池循环复用，稳态无每帧 `cudaMalloc` / `cudaFree`
- **报警**：危险目标（距离变化）生成 AlertMessage，支持 TCP 上报
- **多输入**：视频文件 / USB 摄像头（`/dev/video0`）

---

## 二、环境要求

| 项 | 值 |
|---|---|
| 硬件 | NVIDIA Jetson Nano B01（Tegra X1，Maxwell sm_53，4GB） |
| 系统 | Ubuntu 18.04（JetPack 4.6.x，L4T R32.7.x） |
| CUDA / TensorRT | CUDA 10.2 / TensorRT 8.2（含 `nvonnxparser`） |
| OpenCV (C++) | 4.1.1（`pkg-config` 方式探测） |
| 其他 | yaml-cpp、Eigen3、spdlog（内置子模块）、onnxruntime（内置第三方） |

模型引擎已预置在 `model/engine/`，开箱可用。

---

## 三、编译

默认 Release，无需额外参数：

```bash
cd ~/depth-detect-turbo
mkdir -p build && cd build
cmake ..                # 自动检测平台 / 架构 / nvcc
cmake --build . -- -j4  # 老 cmake 需用 `--` 传 -j
```

可选开关：

```bash
cmake -DENABLE_TIMER=OFF ..          # 关闭逐阶段计时统计
cmake -DPIPELINE_PHASE_TIMER=ON ..   # pipeline 内每阶段 [TIMER] 打印（默认关，零开销）
cmake -DENABLE_PROF_STATS=ON ..      # 逐阶段热点探针，退出时打印 avg/max/占比（默认关，零开销）
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

产物输出到 `bin/`（`main` + 各动态库 `lib*.so`）。

> ⚠️ **4GB 板上编译务必限制并行度与单进程内存**，否则 cc1plus 打满内存会让整机换页卡死：
> `ulimit -v 2621440 && make -j2`（2.5GB 上限；`free -m` 里 available < 3000 时用 `-j1` / `-j2`）

---

## 四、运行

```bash
cd ~/depth-detect-turbo/bin
./perf.sh                            # 性能模式（可选但推荐，自动 sudo）
./main 0 config.yaml                 # 参数是纯数字 = 摄像头设备号
./main ../data/your_video.mp4 config.yaml   # 参数是路径 = 视频文件
```

`perf.sh` 会切到 MAXN、`jetson_clocks`、governor 改 performance、CPU 锁 1479MHz、GPU 锁 921MHz。

处理结果（检测框 + 深度图上下拼接，高度为原图两倍）保存到 `bin/out_dir/`；
日志同时输出控制台与 `bin/latest.log`，每 100 帧打印一次当前 FPS。

> ⚠️ `log_level` 必须保持 `info` 或更低。主循环那句
> `Processing frame N (x fps pipeline, y fps wall)` 是 info 级，设成 `warn` 会把唯一的帧率读数一起吞掉。
>
> ⚠️ 想把 `main` 拷到别的目录跑 A/B，必须 `export LD_LIBRARY_PATH=<那个目录>` 顶到最前面。
> `main` 的 RUNPATH 是指向 `bin/` 的**绝对路径**，否则会静默加载另一份 `libcore` / `libtools`。

---

## 五、配置重点项（bin/config.yaml）

| 配置 | 默认 | 说明 |
|---|---|---|
| `display_manager.is_display` | `false` | true = 实时窗口（需接 HDMI / VNC） |
| `prefer.use_gpu` | `true` | true = TensorRT GPU；false = ONNX CPU（很慢） |
| `prefer.overlap` | `true` | 异步重叠流水线 |
| `depth.depth_interval` | `1` | 深度隔帧推理间隔，调大省算力（见第六节） |
| `io_manager.save_mode` | `images` | images / video / both / none |
| `io_manager.save_interval` | `3` | 图片模式每 N 帧存一张（1 = 每帧都存，省 IO 损耗） |
| `io_manager.save_quality` | `82` | JPEG 压缩质量 |
| `io_manager.send_tcp` | `false` | 是否 TCP 上报报警数据 |
| `logger.log_level` | `debug` | trace / debug / info / warn / err |

---

## 六、性能数据

测试条件：MAXN 锁频（`jetson_clocks`，CPU 1479MHz / GPU 921MHz）+ 固定 300 帧素材 + 生产配置。

### 6.1 端到端吞吐（双口径）

主循环每 100 帧同时打印两个口径，**看优化效果必须看后者**：

- **推理管线 fps** —— 只统计 `processOverlap()`，是历史基准口径，不含读帧 / 绘图 / 落盘
- **墙钟 fps** —— 端到端真实吞吐

| 场景 | 推理管线 fps | 端到端 fps |
|---|---|---|
| 视频文件，INT8 双引擎，`depth_interval: 1` | 7.47（单帧 135ms） | **7.22（单帧 138.5ms）** |
| USB 摄像头 640×360，INT8 双引擎 | 7.60 | 7.1–7.5 |
| 视频文件，INT8 双引擎，`depth_interval: 2` | 10.50 | **10.01（单帧 99.9ms，+38.6%）** |

| 项 | 结果 |
|---|---|
| 冷启动 | ~25s，绝大部分是 CUDA / TensorRT 运行时装载 + 上下文创建的一次性开销；页缓存热时 ~7s |
| 内存 | 4GB 板上稳态无换页抖动 |
| GPU 利用率 | ≈99%，帧内无气泡 |

### 6.2 GPU 时间构成（`trtexec --loadEngine` 直接实测，单次推理）

| 模型 | INT8 | FP16 | FP32 |
|---|---|---|---|
| YOLOv8n @640×640 | **50.1 ms** | 52.2 | — |
| Lite-Mono-Tiny @192×640 | **77.8 ms** | 78.1 ¹ | 98.2 |

两者合计 127.9ms，占单帧墙钟 138.5ms 的 **92.4%**；`1/Throughput ≈ GPU Compute Time`（差 < 1ms），
说明帧内无气泡。账目完全闭合：`138.5 = 50.1(YOLO) + 77.8(Depth) + 6.1(app 侧 CPU) + 4.75(拼接)`。

¹ 仓库预置的 fp16 引擎偏慢（94.3ms），用 `--workspace=2048` 重建后为 78.1ms。
**在 sm_53 上，构建良好的 FP16 与 INT8 延迟几乎相同**（78.1 vs 77.8）——
Maxwell 没有 Tensor Core / DP4A，TensorRT 8.2 仍会发射 INT8 kernel（省的是内存带宽），
所以生产配置选 INT8 主要是**显存占用**考量（权重 / 激活 4× 更小），不是延迟考量。

### 6.3 `depth_interval: 2` 的抖动与代价

> ⚠️ **平均值有欺骗性**：帧时间呈**双峰分布** —— 跑深度的帧 ≈134ms（7.5 fps）、
> 跳过的帧 ≈57ms（17.7 fps），10 fps 只是混合均值，**帧间隔在 57↔134ms 之间跳变**。
> 下游若假定恒定帧间隔（播放器缓冲、编码器码率、告警节流）需按最坏帧时间而不是平均值设计。

除「深度更新率减半」外，还有两处容易漏掉的语义代价：

1. `updateMotionStates()` **每帧都执行**，非深度帧直接用缓存的 depth → 卡尔曼滤波拿到的是
   **当前帧时间戳 + 上一帧深度测量值**的错配组合 → **速度 / 加速度估计有系统性偏差**，不只是「显示慢一帧」
2. `depth_vis` 同样走缓存 → **存出的 jpg 深度半区与 RGB 半区差一帧**

属于业务 / 精度取舍，需产品侧确认后再启用。

---

## 七、性能结论

- 优化后 `infer_pipeline` 占总墙钟 **96.4%**，其中 GPU 执行约 128ms、CPU 发射约 20.7ms
- **瓶颈在 GPU 侧的模型算力**，不在 CPU
- **纯 CPU 侧优化已经到顶**：`yolo_wait` / `depth_wait` 都在 `processOverlap()` 返回前同步各自流，
  本帧全部 GPU 工作必须在同一帧内跑完 → 帧时间下限就是 GPU 忙时，纯 CPU 侧重构理论上限仅 **+3.4%**。
  而且那 20.7ms 的 CPU 发射**本来就是被 GPU 忙时隐藏的**，所以软件流水、CUDA Graphs 都不值得做
  （`--useCudaGraph` 实测仅 +0.2%）
- **可用的两个杠杆**：
  1. `depth_interval: 2` → 实测 **+38.6%**，改一行配置；代价见 6.3，待业务确认
  2. **换更轻的深度模型 / 降低 YOLO 输入分辨率** → 唯一能突破量级的方向，
     但需要重新导出 ONNX + 重新校准 + 业务精度验证
- **已确认走不通**：DLA（Nano 无此硬件）、TF32 / 结构化稀疏（架构不支持）、
  `--tacticSources` 重建（实测更差）、`--directIO`（无增益）、重建 INT8 引擎（实测零增益）

> 完整的测量方法、kernel 级分解、负结论清单与复现步骤见
> [`docs/性能优化报告-2026-09-16.md`](docs/性能优化报告-2026-09-16.md)。

---

## 八、常见问题

| 问题 | 解决 |
|---|---|
| `Failed to open video: 0` | 纯数字设备号走 `/dev/video0`；确认摄像头已插、`ls /dev/video*` 可见 |
| `CMAKE_CUDA_COMPILER could not be found` | 加 `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` |
| 启动要等二十几秒 | 正常，见 6.1 |
| `VIDEOIO ERROR: V4L2: property frame_count is not supported` | 正常：摄像头没有帧总数属性，不影响运行 |
| `Cannot open shared memory file: /dev/shm/trt_engine_*.cache` | 正常：该缓存只写不读，属死代码，不影响运行 |
| 编译到一半整机卡死 | 4GB 板内存打满触发换页，见第三节的 `ulimit -v` |
| pip 装包 ProxyError | 板子走代理时用 `env -u http_proxy -u https_proxy ...` 绕开 |

---

## 九、目录结构

```
depth-detect-turbo/
├── cpp/
│   ├── main.cpp                  # 读帧 → 推理 → 拼接 → 交工作线程绘制/落盘
│   ├── core/                     # 流水线核心（Pipeline / IO / 报警 / 运动状态）
│   ├── inference/                # 推理后端（TensorRT / ONNXRuntime）+ 模型封装
│   ├── bytetrack/                # ByteTrack 跟踪
│   ├── op_kernel/                # CUDA 预处理 / 后处理 kernel
│   ├── tools/                    # 绘制 + DrawSaveWorker（绘制/落盘线程）+ 计时
│   └── utils/                    # 配置 + 日志 + prof_stats 热点探针
├── model/                        # engine/（TRT 引擎）+ onnx/（回退模型）
├── third_party/                  # spdlog / onnxruntime / JsonSender
├── bin/                          # main + config.yaml
├── scripts/                      # 引擎重建工具（重建 / 校准 / 精度体检 / A/B 验证）
├── docs/                         # 性能优化报告
├── perf.sh                       # 性能模式 / 内核调优脚本
└── CMakeLists.txt
```

---

## 十、引擎重建（改模型时才需要）

仓库预置了引擎，正常使用**不需要**重建。改动模型、或想验证引擎构建质量时用 `scripts/`：

```bash
python3 scripts/calib_precheck.py     # 先预检：CUDA 上下文 / 抽帧归一化 / ONNX 解析

python3 scripts/build_engine.py \
    --onnx  model/onnx/lite-mono-tiny/lite-mono-tiny_192x640_op11.onnx \
    --out   /tmp/lm_int8.engine \
    --precision int8 --workspace 2048 \
    --calib-video /path/to/your.mp4 --calib-frames 200 --calib-cache lm.cache

bash  scripts/verify_new_engine.sh /tmp/lm_int8.engine      # 端到端 A/B，自动还原现役引擎
python3 scripts/compare_engines.py --video x.mp4 --frames 20 \
    --engines 现役=model/engine/.../..._int8_trt8.2.engine 新=/tmp/lm_int8.engine
```

`build_engine.py` 自带 INT8 熵校准器 —— 因为 `trtexec` 的 `--calib` 只接受**已有的校准缓存文件**，
不支持从图片 / 视频校准。

两条硬约束：

1. **`--workspace` 必须显式给**。`trtexec` 默认只有 **16MB**，TensorRT 拿不到大 tile 算法的
   scratch 显存就会退到 FP32 sgemm，实测差 **17%**。
2. **校准数据的归一化必须与 app 一致**。本项目深度输入是 `[0,1]`（`mean=0 / std=1`），
   不是 ImageNet 的 0.45/0.225；脚本默认值已对齐，预检会打印实际数值范围供核对。

> ⚠️ 用测试视频校准只够验证**性能**；交付必须用业务真实场景数据重新校准。
> ⚠️ `model/engine/` 已被 `.gitignore` 排除，重建产物请自行留档。

---

## License

MIT（沿用上游 [depth-detect](https://github.com/yzfzzz/depth-detect) License）。

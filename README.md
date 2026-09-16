# depth-detect-turbo

> Jetson Nano 实时目标检测 + 单目深度估计加速版（重构自 [yzfzzz/depth-detect](https://github.com/yzfzzz/depth-detect)）

面向 **Jetson Nano（Tegra X1）** 压榨性能的实时深度检测管线：
YOLOv8n 检测 + Lite-Mono 单目深度估计，TensorRT INT8 引擎加速（FP16/FP32 备选）+ CUDA 预处理/后处理。

---

## 一、核心特性

- **双模型推理**：YOLOv8n 目标检测 + Lite-Mono-Tiny 单目深度估计，TensorRT 8.2 INT8 引擎（FP16/FP32 保留备用）
- **纯 GPU 管线**：CUDA 预处理（letterbox/归一化）+ CUDA 后处理（NMS），`--use_fast_math`
- **目标跟踪**：ByteTrack + 卡尔曼滤波，支持目标运动状态（趋近/远离/加减速）判断
- **异步重叠**：`overlap: true` 时读帧线程与主循环重叠；YOLO 与 Depth 各跑自己的 CUDA 流
- **绘图/落盘不占关键路径**：`DrawSaveWorker` 独立线程负责逐目标绘制与 JPEG 编码写盘，与下一帧的 GPU 推理重叠
- **零分配稳态**：单 `Pipeline` 实例 + 2 槽位帧缓冲池循环复用，稳态无每帧 `cudaMalloc`/`cudaFree`；首帧预热在进入主循环前完成
- **报警机制**：危险目标（距离变化）自动生成 AlertMessage，支持 TCP 上报
- **多输入**：H.264 视频文件 / USB 摄像头（`/dev/video0`）两种输入
- **性能模式脚本**：`perf.sh` 一键内核调优（MAXN + jetson_clocks + performance governor）

### 实测性能（Jetson Nano, MAXN 锁频）

主循环每 100 帧打印两个口径，**看优化效果必须看后者**：

- **推理管线 fps** —— 只统计 `processOverlap()`，是历史基准口径，不含读帧/绘图/落盘
- **墙钟 fps** —— 端到端真实吞吐，含读帧等待、绘图、落盘的重叠情况

| 场景 | 推理管线 fps | 端到端 fps |
|---|---|---|
| 固定 300 帧素材（test300.mp4），INT8 双引擎，`depth_interval: 1` | 7.47（单帧 135ms） | **7.22（单帧 138.5ms）** |
| USB 摄像头 640×360，INT8 双引擎 | 7.60 | 7.1–7.5 |
| 深度隔帧 `depth_interval: 2`（INT8，与上一行同一构建实测） | 10.50 | **10.01（单帧 99.9ms，+38.6%）** |
| FP16 双引擎（量化改造前的对照，管线口径） | 6.53（单帧 153ms） | — |

> ⚠️ `depth_interval: 2` 的**平均值具有欺骗性**：帧时间呈双峰分布 —— 跑深度的帧 ≈134ms（7.5 fps）、
> 跳过的帧 ≈57ms（17.7 fps），平均 10 fps 是这两簇的混合值。帧间隔在 **57↔134ms 之间反复跳变**，
> 下游若假定恒定帧间隔（播放器缓冲、编码器码率、告警节流）要按最坏帧时间而不是平均值设计。
> 还附带两处容易被忽略的语义代价，见第九节。

**GPU 时间去哪了**（`trtexec` 直接实测单次推理，锁频）：

| 模型 | INT8 | FP16 | FP32 |
|---|---|---|---|
| YOLOv8n @640×640 | **50.11 ms** | 52.16 | — |
| Lite-Mono-Tiny @192×640 | **77.76 ms** | 94.30 | 98.19 |

两者合计 127.9ms，占单帧墙钟 138.5ms 的 **92.4%**；且 `1/Throughput ≈ GPU Compute Time`
（相差仅 0.5–0.9ms）说明**帧内没有气泡、GPU 利用率 ≈99%**。
账目完全闭合：`138.5 = 50.1(YOLO) + 77.8(Depth) + 6.1(app 侧 CPU) + 4.75(concat)`。

> ⚠️ 上表 **FP16 那一列是仓库里现役的 fp16 引擎，它是被「建坏」的**（见第九节）：用
> `--workspace=2048` 重建后 fp16 只要 **78.06ms**，与 INT8 的 78.05ms 打平。
> **在 sm_53 上，INT8 相对「构建良好的 FP16」几乎没有延迟优势** —— 它的价值在**显存占用**
> （权重/激活 4× 变小）。生产用的 **INT8 引擎本身是建好的**，重建无增益，
> 深度模型在这张卡上的**延迟底线就是 ~78ms**。

> 同一个 INT8 版本，管线口径 7.40 而端到端只有 6.58 —— 那 17ms 差在主循环里画框和压 JPEG，
> 期间 GPU 是空转的。这是 2026-09-16 那轮优化解决的事，见第九节。

| 项 | 结果 |
|---|---|
| 启动 | 冷启动 **~25s**（清页缓存实测：日志起 → 两个引擎就绪 24.9s。其中**第一个引擎独占 24.1s**，第二个只 0.8s —— 主要是 CUDA/TensorRT 运行时装载 + 上下文创建的一次性开销，不是"反序列化两个引擎"）；页缓存热时 ~7s |
| 内存 | 4GB 板上无换页抖动 |
| 退出 | 收尾偏慢（`cuModuleUnload` ×594 ≈ 2.6s），不影响稳态帧率 |

> 历史问题（commit 469f09b 已修）：更早版本 `main.cpp` 与 `AsyncPipeline` 各自隐式加载一份
> Pipeline（4 次引擎反序列化、显存翻倍），在 4GB 板上触发换页抖动导致阶段耗时 7~85ms 剧烈波动。

---

## 二、环境要求

| 项 | 值 |
|---|---|
| 硬件 | NVIDIA Jetson Nano B01（Tegra X1, 4G） |
| 系统 | Ubuntu 18.04（JetPack 4.6.x, L4T R32.7.x） |
| CUDA | 10.2 |
| TensorRT | 8.2（`libnvinfer8`，含 `nvonnxparser`） |
| OpenCV (C++) | 4.1.1（`pkg-config` 方式探测） |
| 其他 | yaml-cpp, Eigen3, spdlog(内置子模块), onnxruntime(内置第3方) |

> 模型引擎已预导出至 `model/engine/`，无需重新导出。

---

## 三、目录结构

```
depth-detect-turbo/
├── cpp/
│   ├── main.cpp                  # 主程序（读帧→推理→拼接→交给工作线程绘制/落盘）
│   ├── core/                     # 流水线核心（Pipeline、IO、报警、运动状态）
│   ├── inference/                # 推理后端(TensorRT/ONNXRuntime) + 模型封装
│   ├── bytetrack/                # ByteTrack 目标跟踪
│   ├── op_kernel/                # CUDA 预处理/后处理 kernel
│   ├── tools/                    # 绘制(FPS/检测框) + DrawSaveWorker(绘制/落盘线程) + 计时
│   └── utils/                    # 配置加载 + 日志 + prof_stats(热点探针)
├── docs/                         # 优化报告
├── model/
│   ├── engine/                   # 已导出的 INT8/FP16/FP32 TRT 引擎
│   └── onnx/                     # ONNX 回退模型
├── third_party/                  # spdlog / onnxruntime / JsonSender
├── bin/
│   ├── main                      # 可执行文件（cmake 生成）
│   └── config.yaml               # 运行配置
├── perf.sh                       # 性能模式 / 内核调优脚本
├── scripts/                      # 引擎重建工具（build_engine.py / calib_precheck.py），见第十节
└── CMakeLists.txt                # 精简版构建脚本
```

---

## 四、编译

默认 **Release**，无需额外参数：

```bash
cd ~/depth-detect-turbo
mkdir -p build && cd build
cmake ..                # 自动检测平台/架构/nvcc，默认 Release
cmake --build . -- -j4  # 并行编译（老 cmake 需用 `--` 传 -j，或直接 make -j4）
```

可选开关：

```bash
cmake -DENABLE_TIMER=OFF ..                 # 关闭逐阶段计时统计
cmake -DPIPELINE_PHASE_TIMER=ON ..          # 开启 pipeline 内每阶段 [TIMER] 打印（默认关，零开销）
cmake -DENABLE_PROF_STATS=ON ..             # 开启逐阶段热点探针，退出时打印 avg/max/占比（默认关，零开销）
cmake -DENABLE_JESTON_MEM_MANAGED=ON ..     # Jetson 统一内存（实验性）
cmake -DTARGET_CUDA_ARCHS=53 ..             # 手动指定 CUDA 架构
cmake -DCMAKE_BUILD_TYPE=Debug ..           # 调试构建
```

产物输出到 `bin/`（`main` + 各动态库 `lib*.so`）。

> ⚠️ **4GB 板上编译务必限制并行度与单进程内存**，否则 cc1plus 打满内存会让整机换页卡死：
> ```bash
> ulimit -v 2621440 && make -j2      # 2.5GB 上限；free -m 里 available < 3000 时用 -j1/-j2
> ```

> ⚠️ **`main` 的 RUNPATH 是绝对路径 `bin/`**。想把二进制拷到别处跑 A/B，必须
> `export LD_LIBRARY_PATH=<那个目录>` 顶到最前面，否则会静默加载 `bin/` 里的另一份 `libcore/libtools`。

---

## 五、运行

### 5.1 跑摄像头（USB /dev/video0）

```bash
cd ~/depth-detect-turbo/bin
./perf.sh                 # 开启性能模式（可选但推荐，自动 sudo）
./main 0 config.yaml      # 第一个参数 0 表示摄像头设备号
```

### 5.2 跑视频文件

```bash
cd ~/depth-detect-turbo/bin
./main ../data/your_video.mp4 config.yaml    # 需要先把视频放到 data/ 目录
```

### 5.3 参数说明

| 参数 | 说明 |
|---|---|
| `0` | 摄像头设备号（`./main 0`） |
| `任意路径.mp4` | 视频文件路径 |
| `config.yaml` | 配置文件路径 |

处理结果（检测框 + 深度图逐帧拼接图，高度是原图两倍）保存到 `bin/out_dir/`。

日志默认同时输出到控制台与 `bin/latest.log`，每 100 帧打印一次当前 FPS。

---

## 六、性能模式：perf.sh

```bash
./perf.sh            # 应用优化（需要 sudo，自动提升）
./perf.sh --dry-run  # 只预览，不改动
./perf.sh --reset    # 恢复默认
```

应用的内容：

| 项 | 调优前 | 调优后 |
|---|---|---|
| 电源模式 | MAXN | MAXN（锁定） |
| CPU 频率 | 动态 1224MHz | 全核锁 1479MHz |
| GPU 频率 | 待机 76.8MHz | 锁 921MHz |
| CPU 调度器 | schedutil | performance（4核） |
| vm.swappiness | 60 | 10 |

建议运行：

```bash
cd ~/depth-detect-turbo/bin
sudo nice -n -10 ./main 0 config.yaml
```

---

## 七、配置文件（bin/config.yaml）重点项

| 配置 | 默认 | 说明 |
|---|---|---|
| `display_manager.is_display` | `false` | true = 实时窗口（需接 HDMI/VNC） |
| `prefer.use_gpu` | `true` | true = TensorRT GPU，false = ONNX CPU（很慢） |
| `prefer.overlap` | `true` | true = 异步重叠流水线 |
| `depth.depth_interval` | `1` | 隔帧做深度推理的间隔（调大省算力，见第九节） |
| `io_manager.save_mode` | `images` | images / video / both / none |
| `io_manager.send_tcp` | `false` | 是否 TCP 上报报警数据 |
| `logger.log_level` | `debug` | trace / debug / info / warn / err |

> ⚠️ `log_level` 一定要保留 `info` 或更低。主循环那句
> `Processing frame N (x fps pipeline, y fps wall)` 是 info 级别，
> 设成 `warn` 会把唯一的帧率读数一起吞掉。

---

## 八、常见问题

| 问题 | 解决 |
|---|---|
| `Failed to open video: 0` | 程序内置支持纯数字设备号走 `/dev/video0`；确认 USB 摄像头已插、`ls /dev/video*` 可见 |
| `cmake: CMAKE_CUDA_COMPILER could not be found` | 本新版 CMakeLists 会自动找 nvcc；若仍失败，`sudo apt install nvidia-cuda-toolkit` 或用 `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` |
| 启动要等二十几秒 | 正常：冷启动约 25s，绝大部分是 CUDA/TensorRT 运行时装载 + 上下文创建的一次性开销（第一个引擎就占 24s，第二个只 0.8s），页缓存热了之后约 7s。首帧自动调优已由预热吸收，主循环首帧即稳态 |
| `VIDEOIO ERROR: V4L2: property frame_count is not supported` | 正常：摄像头没有帧总数属性，不影响运行 |
| `Cannot open shared memory file: /dev/shm/trt_engine_*.cache` | 正常：`saveEngineToShm()` 写缓存失败，而 `loadEngineFromShm()` 根本没被调用，属死代码，不影响运行 |
| 编译到一半整机卡死 | 4GB 板内存打满触发换页。见第四节：`ulimit -v 2621440` + 低 `-j` |
| pip 装包 ProxyError | 板子走代理时用 `env -u http_proxy -u https_proxy ...` 绕开 |
| `trtexec --dumpProfile` 的单层耗时加起来远大于 `GPU Compute Time` | 正常：该表被 CUPTI 仪器开销**放大约 1.5×**，且 `{ForeignNode[...]}` / `Conv` 这类孤立大条目是**归因伪影**（真 kernel 的耗时会随精度变化，伪影几乎不变）。**千万不要照着「表里最慢的那一层」去重构网络** |
| 自己重建的引擎比现役的还快 | 先查 `--workspace`：`trtexec` 默认**只有 16MB**，TRT 拿不到大 tile 的 scratch 显存就会退到 FP32 sgemm。生产构建请显式 `--workspace=2048`（本次实测差 17%） |
| `trtexec --int8` 建出来的引擎并没有更快 | 正常：**没有校准数据时 INT8 无法真正启用**，缺 dynamic range 的层会整体退回 FP16。INT8 必须配校准器，见第十节 |
| python 建引擎报 `invalid device context` | pycuda 只 `cuda.init()` **不会**创建 CUDA 上下文，必须先 `cuda.Device(0).make_context()` 再 push。注意 pycuda 2019.1 **没有** `retain_primary_context`（`Device.__getattr__` 是个兜底转发到 `get_attribute()` 的钩子，访问不存在的名字只抛 `AttributeError`） |
| python 里没有 `config.set_memory_pool_limit` | Jetson 版 TRT 8.2 的 python 绑定**没有暴露 `MemoryPoolType`**，只能用 `config.max_workspace_size = 2048 << 20`（8.2 里虽标 deprecated 但生效；8.5+ 才移除） |
| `sudo: nvprof: command not found` / `Error: Encountered invalid option: --loadEngine=` | `sudo` 会重置 PATH → 用绝对路径 `/usr/local/cuda/bin/nvprof`；且**先写被测可执行文件**再写它的参数：`nvprof /usr/src/tensorrt/bin/trtexec --loadEngine=...` |

---

## 九、性能优化记录与方向

### 当前状态（2026-09-16）

锁频（MAXN + jetson_clocks，CPU 1479MHz）+ 固定 300 帧素材，生产配置：

| 版本 | 单帧墙钟 | 端到端 fps | 推理管线 fps |
|---|---|---|---|
| 基线 `77659e3` | 152.05 ms | 6.577 | 7.40 |
| **本轮** | **138.53 ms** | **7.219（+9.8%）** | 7.47 |

完整数据、kernel 级分解、负结论与复现步骤见
[`docs/性能优化报告-2026-09-16.md`](docs/性能优化报告-2026-09-16.md)。

**热点在哪**：主循环里「逐目标绘制 7.16ms + JPEG 编码落盘 10.05ms」是纯 CPU 串行，
这 17.2ms 里 GPU 完全空转 —— 就是「单帧 152ms 而推理只占 135ms」的缺口。

**怎么改的**：新增 `DrawSaveWorker`（有界队列容量 2 + 独立线程）把这两件事挪出主循环，
与下一帧的 GPU 推理重叠；主线程只留拼接（4.7ms）与入队（0.06ms）。
显示走 `promise<void>` 信号 —— 工作线程画完立刻唤醒主线程 `imshow`，JPEG 编码继续在后台跑。
验证方式是开/关落盘两个配置的耗时完全重合（改前差 10.25ms，改后差 0.1%）。

**顺带拆掉的一个隐患**：`io_manager` 的读帧 H2D 原先不传 stream，走 legacy default stream，
而模型流是 `cudaStreamCreate` 出来的 blocking stream —— 按 CUDA 语义两者会互相插入隐式依赖。
已全部改为 `cudaStreamNonBlocking`。**但实测只快了 0.8%**：原先怀疑的
「`depth_launch` 12.8ms 由隐式同步造成」被数据推翻，那 12.8ms 其实是 TRT `enqueueV2`
逐层发射数百个 kernel 的 CPU 固有开销。

**瓶颈现状**：优化后 `infer_pipeline` 占墙钟 **96.4%**，其中 CPU 发射约 20.7ms、GPU 执行约 128ms
（`trtexec` 直接实测：YOLO 50.1 + Depth 77.8）。GPU 利用率 ≈99%、帧内无气泡，
**所以真正的出路只能在 GPU 侧**（模型 / 分辨率 / 精度 / 帧率策略）。

> **一个必须记住的分析范式**：别凭直觉估「把 CPU 工作藏进 GPU 等待能省多少」，先做归零判断。
> `p3.yolo_wait` 与 `p5.depth_wait` 都在 `processOverlap()` 返回前同步了各自的流
> → **本帧全部 GPU 工作都必须落在这 134ms 内**，而帧时间 ≥ 每帧 GPU 忙时。
> 所以纯 CPU 侧重构的绝对上限是 134ms → 7.46 fps，相对当前 7.219 **最多 +3.4%**。
> 顺带纠正一个直觉错误：那 20.7ms 的 CPU 发射**早就是被隐藏的**（深度预处理 kernel 一发射
> GPU 就开工，后面的 `enqueueV2` 都是在 GPU 忙着时做的），并不存在「把发射藏进等待」的收益。

### ⭐ 引擎构建质量：一个被自己推翻的假设

**背景**：在得出「INT8 在这张卡上已经到头」之前，我们做了一件本该更早做的事 ——
**拿原始 ONNX 用最朴素的参数裸重建一遍，跟现役引擎对着测**。结果先推翻了「到头」，后来又推翻了「重建就能提速」。

**第一次对照（发现 fp16 引擎有问题）**：

| 引擎 | GPU Compute Time |
|---|---|
| 现役 `..._fp16_trt8.2.engine` | **94.32 ms** |
| 现役 `..._int8_trt8.2.engine` | **77.70 ms** |
| 裸重建 `--fp16 --workspace=2048` | **78.09 ms** |

裸 fp16 重建比现役 fp16 快 16.2ms（−17%）。原因大概率是 **`--workspace` 太小**：
`trtexec` 默认只有 **16MB**，TRT 拿不到大 tile / winograd 需要的 scratch 显存，只能退到 FP32 sgemm ——
与 `nvprof` 看到的「29 次/帧 FP32 sgemm」吻合。

**第二次对照（推翻「重建 INT8 能提速」）** —— 用 `--workspace=2048` + 200 帧校准
重建现役那个 INT8 引擎，同会话对照：

| 引擎（同一次会话内测量） | Throughput | GPU Compute Time |
|---|---|---|
| 现役 INT8 | 12.6756 qps | **77.75 ms** |
| **重建 INT8**（`--workspace=2048` + 200 帧校准） | 12.6522 qps | **78.05 ms** |
| 裸重建 FP16（`--workspace=2048`） | 12.6582 qps | **78.06 ms** |

**三个引擎落在 0.4% 之内 —— 所以：**

1. **重建 INT8 没有任何增益**（78.05 vs 77.75，差异在噪声内）。
   → 现役生产引擎（INT8）本来就是**构建良好**的，**不存在「重建一下白得 11%」这条路**。
   被建坏的只有**不用于生产**的 fp16 引擎。
2. **在 sm_53 上，构建良好的 FP16 ≈ INT8**（78.06 vs 77.75，差 0.4%）。
   → 本文档先前写的「INT8 比 FP16 快 17.5%」是**拿建坏的 FP16 当对照**得出的，**该结论不成立**。
   INT8 的价值在**显存占用**（权重/激活 4× 变小，4GB 板上不无小补），**不在延迟**。
3. 深度模型在 Nano 上的**延迟底线就是 ~78ms**，无论 INT8 还是 FP16、无论怎么重建。
   这也从另一侧加固了「出路只剩 `depth_interval` / 换更轻的模型」这个结论。

> 📌 **方法论纪律（比结论更重要）：对照实验的「对照组」本身必须先被验证。**
> 两次对照各自都对，但第一次得出的「INT8 比 FP16 快 17.5%」把一个**被建坏的 fp16 引擎**
> 当成了 FP16 的正常水平。**任何 A/B 结论都要问一句：对照组是健康的吗？**
>
> 📌 第二条纪律：**拿到一个「已优化」的引擎，先自己从 ONNX 重建一遍做对照。**
> 构建参数（workspace / timing cache / tacticSources / 构建机）任何一个不同都可能差 10–20%。
>
> ⚠️ 构建很慢要有耐心：Lite-Mono-Tiny（2.2M 参数）在 Nano 上单次构建 **约 9–10 分钟**；
> INT8 还要先跑 200 批校准（1.76s/批 ≈ 6 分钟），
> 全程「校准 + 构建」实测 **1374s（≈23 分钟）**。多变体扫描请写脚本丢后台跑。

**构建期其它旋钮实测（全是负结论，别重复试）**：

| 变体 | GPU Compute Time | 结论 |
|---|---|---|
| 裸重建 `--fp16 --workspace=2048` | 78.09 ms | ✅ 对 fp16 有效（对生产无意义，生产用 INT8） |
| `--tacticSources=-CUDNN` | 103.65 ms | ❌ 更差 |
| `--tacticSources=-CUDNN,-CUBLAS,+CUBLAS_LT` | 105.21 ms | ❌ 更差 |
| `--directIO` | 78.09 ms | ➖ 与裸重建打平，无增益 |
| `--int8`（**不给校准数据**） | ≈ fp16 | ❌ 无意义：缺 dynamic range 的层会整体退回 FP16 |
| `--workspace=2048` + 校准重建 INT8 | 78.05 ms | ➖ **无增益**（现役 77.75 已到底） |

### kernel 级真实成本（`nvprof`，基线构建，单帧）

| 类别 | ms/帧 | 占比 | 次/帧 |
|---|---|---|---|
| **FP32 GEMM (sgemm)** | 23.81 | 17.1% | 29 |
| winograd / cuDNN 卷积 | 22.41 | 16.1% | 32 |
| TRT pointwise（生成内核） | 20.26 | 14.5% | **98** |
| 其它 | 18.30 | 13.1% | 63 |
| `__myl_*`（融合激活 / Norm / gating） | 12.03 | 8.6% | 34 |
| INT8 layout 转换 | 6.79 | 4.9% | 42 |
| Reformat / Copy | 6.59 | 4.7% | 21 |
| grouped / direct 卷积 | 6.44 | 4.6% | 17 |
| Slice / Concat / Shuffle | 4.81 | 3.5% | 7 |
| Resize | 3.97 | 2.8% | 7 |
| 自研 CUDA kernel（letterbox / resize / transpose） | 3.52 | 2.5% | 3 |
| FP16 GEMM (hgemm) | 3.29 | 2.4% | 10 |
| Softmax | 2.11 | 1.5% | 1 |

三个可直接读出的结论：

1. **卷积只占约 35%**，别默认「卷积是瓶颈」。
2. **非数学的图层开销**（pointwise + INT8 layout + reformat + slice + transpose + memcpy）
   ≈ **43.6 ms/帧 = 31%** —— 这是 ONNX 图形态造成的，不是模型算力需要的。
3. **29 次/帧的 FP32 GEMM = 23.8ms 全部来自深度模型**（单测深度引擎可复现 29 次 / 23.5ms）。
   FP16 hgemm 只有 3.0ms → **GEMM 占深度模型的 34%，其中 88% 跑在 FP32 上**。
   这些 GEMM 来自 Lite-Mono 的注意力 MatMul；`__myl_*` 的 kernel 名字直接写着融合了什么
   （`...AddDivErfAddMulMul` = Erf 版 GELU、`...MeaSubMulMeaAddSqrDiv...` = LayerNorm），
   是免费的模型结构情报。

> ⚠️ `nvprof` 的**绝对值**也带仪器开销（本次合计 139.45ms vs 真实 GPU 忙时 127.9ms，约 9% 膨胀），
> 但**相对占比可用**。要绝对值请以 `trtexec` 的 `GPU Compute Time` 为准。

### `depth_interval: 2` 的完整代价

实测 **7.220 → 10.009 fps（+38.6%，单帧 138.5 → 99.9ms）**。除了「深度更新率减半」，
还有两处容易被漏掉的语义代价（都在 `pipeline.cpp::processOverlap`）：

1. `updateMotionStates()` **每帧都执行**，非深度帧直接用 `cached_depth_` → 卡尔曼滤波拿到的是
   **当前帧 `timestamp` + 上一帧 depth 测量值**的错配组合 → 速度/加速度估计有**系统性偏差**，
   不是单纯「显示慢一帧」。业务若对「趋近 / 远离」判定敏感需单独评估。
2. `depth_vis` 走的也是 `cached_depth_vis_` → **存出的 jpg 深度半区与 RGB 半区差一帧**。


### 更早的轮次

- **2026-09-05（`ebf5077`，管线口径 6.53 → 7.40 fps）**：INT8 双引擎落地（熵校准 370 帧，
  与 FP16 视觉无可感知差异）；读帧 H2D 改 pinned 暂存，消掉 pageable 拷贝触发的驱动隐式设备同步
  （Depth 发射 49ms → 12ms）；修掉落盘缺深度半区 bug（之前用 swap 把 `depth_vis` 从 context 换走，
  存出的图只有上半原图，1280×720 而不是设计的 1280×1440）。
- **2026-08-28（`469f09b`，6.54 fps）**：单 `Pipeline` 实例消除双份引擎加载；2 槽位帧缓冲池；
  YOLO/Depth 双 CUDA 流；首帧预热；`BaseModel` 输出指针表 static 改实例成员；letterbox 常量复用。

### 已经确认走不通的方向

| 想法 | 结论 |
|---|---|
| 听信「Nano 没有 Tensor Core，所以 INT8 没用」改用 FP16 | ➖ **两句都别信**：官方那句的**理由**不成立（Maxwell 确实没有 Tensor Core/DP4A，但 TRT 8.2 照发 INT8 kernel，省的是内存带宽）；而「INT8 快 17.5%」也是跟**建坏的** fp16 引擎比出来的 —— 两者都建好之后 **INT8 ≈ FP16（78.05 vs 78.06ms，差 0.4%）**。见第九节 |
| **DLA 卸载**（官方文档的常规建议） | ❌ Jetson Nano / Tegra X1 **没有 DLA**（Xavier 起才有） |
| TF32 / 结构化稀疏 `--sparsity` / `--builderOptimizationLevel` | ❌ Maxwell 不支持 TF32；稀疏需 Ampere+；`builderOptimizationLevel` 要 TRT 8.6+（板上是 8.2） |
| **CUDA Graphs** 消掉 20.7ms 发射 | ❌ **已实测否决**：`--useCudaGraph` 19.771 vs 19.735 qps = **+0.2%**，独立验证了 +3.4% 上界 |
| 软件流水：拆 `launch(N)` / `collect(N-1)` | ❌ 上界只有 +3.4%，且前提就错了（发射早被隐藏，见上文的归零判断） |
| legacy default stream 耦合导致 `depth_launch` 12.8ms | ❌ 已实测：改非阻塞流后仍是 12.8ms |
| 融合 letterbox + CHW 预处理 kernel | ❌ 自研 kernel **全量**才 7.4ms/帧（letterbox 2.8 + depth resize 1.5 + transpose 1.3 + decode 0.7 + process 0.5 + normalize 0.2），天花板 <2% |
| 把拼接搬进工作线程 | ❌ `depth_vis` 直接包在深度模型的 pinned 主机缓冲上，下一帧深度的 D2H 会覆盖它 —— 数据竞争，必须留在主线程 |
| `04.concat` 用 3 槽环形预分配 buffer | ⚠️ 槽数不够（需 ≥4）、必须按引用计数回收，收益仅 ~3ms，且保存图会撕裂，不划算 |
| `--tacticSources` 调 cuDNN / cuBLAS 选型 | ❌ 实测**更差**：103.7 / 105.2ms（对照 78.1ms） |
| `--directIO` | ➖ 78.09ms，与裸重建打平，无增益 |
| `--useSpinWait` | ⚠️ 未试。理论只省 CPU 侧事件同步延迟，量级 <1ms |
| 「`/model.22/dfl/conv/Conv` 占了 YOLO 一半算力」去重构检测头 | ❌ 那是逐层表的**归因伪影**，真实不存在，见「常见问题」 |

> **「Nano 上 INT8 没用」这句话结论对、理由不成立。** 官方论坛常答「INT8 需要 Tensor Core（sm>7.x），
> Nano 请用 FP16/FP32」。Maxwell（sm_53）确实**没有 Tensor Core 也没有 DP4A**，但 TRT 8.2 仍会
> 发射 INT8 kernel —— 省的是**内存带宽**（权重/激活 4× 变小），带宽型网络直接受益。
> **但反过来也别以为 INT8 能提速**：和一份**构建良好**的 FP16 引擎比，两者延迟几乎一样
> （78.05 vs 78.06ms）。**先把引擎构建参数对齐，再按实测选精度**，别按说法选。

### 待做 / 候选方向

- [x] ~~重建引擎~~ **已实测否决（2026-09-16）**：用 `--workspace=2048` + 200 帧校准重建的 INT8 是
      **78.05ms**，与现役 77.75ms 无差异 → **不存在「重建一下就提速」这条路**；
      被建坏的只有不用于生产的 fp16 引擎。工具保留在 `scripts/`（将来换模型后仍要用它重建/校准）
- [ ] **`depth_interval: 2`**：已实测 **+38.6%**（7.219 → 10.009 fps），与粗估只差 1%。
      代码与配置都已支持，**卡在业务确认**（三处语义代价见上文）
- [ ] **换更轻的深度模型**（唯一能突破量级的方向）：Lite-Mono-Tiny 实测 77.8ms；
      论文里 RT-MonoDepth-S 在同板同分辨率可跑 30.5 FPS（≈32.8ms），约 **2.4×**。
      但需从 `.pt` 重新导出 ONNX + 重新校准 + 业务精度验证 —— 板上与 PC 上都没有
      torch / ultralytics，**当前无法重导 ONNX**（得先解决这个前置条件）
- [ ] **减小 YOLO 输入分辨率**（640→512/416）：社区实测 Nano 上 320² ≈42 FPS vs 640² ≈18 FPS；
      本项目 YOLO 占 50.1ms，预期能压到 12–21ms。同样需要重导 ONNX
      （板上只有固定 640 的 ONNX，直接改图会破坏 neck 里 Resize 的常数尺度），且小目标会退化
- [ ] 深度模型的 3 个 `{ForeignNode[...]}` 打包节点 + 29 次 FP32 GEMM（23.8ms，占深度 34%）——
      来自 Lite-Mono 的注意力 MatMul，**要动只能改模型结构**
- [ ] 深度模型 `naiveSlice` kernel ~4.8ms/帧（TRT 内部切片 op，需改模型导出才能消除）
- [ ] 引擎反序列化 / 启动加速：冷启动 24.9s 里第一个引擎独占 24.1s，说明大头是
      CUDA/TensorRT 运行时的一次性装载，而非反序列化本身。顺带清理死代码：
      `/dev/shm` 引擎缓存**只写不读**（`loadEngineFromShm()` 从未被调用，写入本身也失败）
- [ ] `main` 的 RUNPATH 由绝对路径改为 `$ORIGIN` —— 否则任何 A/B 都活在「加载错库」的阴影里
- [ ] 报警上报 JSON 轻量化（剥离 JsonSender 或改共享内存）

---

## 十、重建引擎（含 INT8 校准）

仓库里预置了引擎，但**只要你想改构建参数、或想验证「现役引擎是不是建好的」，就得自己重建**。
`scripts/build_engine.py` 用纯 TensorRT Python API 在板上直接把 ONNX 转成 engine
（不需要 torch / ultralytics，Jetson 上原生可跑）。

**为什么不能用 `trtexec` 建 INT8**：`trtexec` 的 `--calib` 只接受**已有的校准缓存文件**，
不支持从图片/视频校准，而本仓库没有校准缓存 —— 所以 INT8 重建必须自带校准器。

| 脚本 | 用途 |
|---|---|
| `scripts/calib_precheck.py` | 建之前先跑（≈40s）：验证 CUDA 上下文 / 抽帧归一化 / ONNX 解析 / workspace API |
| `scripts/build_engine.py` | 从 ONNX 建 engine（INT8 自带熵校准器 + 校准缓存复用） |
| `scripts/verify_new_engine.sh` | 把新引擎接进 app 跑端到端 A/B，**跑完自动还原**现役引擎 |
| `scripts/compare_engines.py` | 精度体检：同一批输入喂两个 engine，输出 MAE / max\|Δ\| |

```bash
# 0) 预检（约 40s）：验证 CUDA 上下文 / 抽帧归一化 / ONNX 解析 / workspace API
python3 scripts/calib_precheck.py

# 1) 首次构建 INT8：抽 200 帧校准（≈6 分钟）+ 构建（≈9 分钟），并写出校准缓存
python3 scripts/build_engine.py \
    --onnx model/onnx/lite-mono-tiny/lite-mono-tiny_192x640_op11.onnx \
    --out  /tmp/lm_int8.engine \
    --precision int8 --workspace 2048 \
    --calib-video /path/to/your.mp4 --calib-frames 200 \
    --calib-cache lm.cache

# 2) 之后复用缓存重建（跳过校准，只花构建时间）
python3 scripts/build_engine.py \
    --onnx model/onnx/lite-mono-tiny/lite-mono-tiny_192x640_op11.onnx \
    --out  /tmp/lm_int8.engine \
    --precision int8 --workspace 2048 --calib-cache lm.cache

# 3) 实测（务必与现役引擎在**同一会话**里对照，跨会话的时钟漂移会骗人）
/usr/src/tensorrt/bin/trtexec --loadEngine=/tmp/lm_int8.engine \
    --iterations=50 --warmUp=500 --avgRuns=20 | grep -E 'Throughput|GPU Compute Time'

# 4) 接进 app 跑端到端 A/B（自动备份/还原现役引擎）+ 精度体检
bash scripts/verify_new_engine.sh /tmp/lm_int8.engine
python3 scripts/compare_engines.py --video /path/to/x.mp4 --frames 20 \
    --engines 现役=model/engine/lite-mono-tiny/lite-mono-tiny_192x640_op11_int8_trt8.2.engine \
               新=/tmp/lm_int8.engine
```

三条硬约束（都是踩过的坑）：

1. **`--workspace` 必须显式给**。`trtexec` 默认只有 **16MB**，TRT 拿不到大 tile 的 scratch 显存
   就会退到 FP32 sgemm —— 实测同一份 ONNX 差 **17%**（94.32 → 78.09ms）。
2. **校准归一化必须与 app 一致**。深度模型走 `depth_model.cpp` 的 `is_normalize=false` 分支，
   预处理 kernel 是 `(val/255 - mean)/std` 且 mean=0、std=1 → 校准输入应落在 **[0,1]**，
   **不是** ImageNet 的 0.45/0.225。脚本默认值已是 0/1，预检会打印实际数值范围供核对。
3. **交付前必须用业务真实场景数据重新校准**。拿测试视频校准只够验证**性能**，INT8 精度没有保证。

> 📌 **先看结果再决定要不要建（能省 23 分钟）**：2026-09-16 实测，用本办法重建的 INT8 是
> **78.05ms**，与仓库里现役的 INT8（77.75ms）**无差异** —— 现役生产引擎本来就是建好的，
> **重建不带来提速**。这套工具的真正用途是 ① 换模型后重建/校准；② 怀疑引擎构建质量时做对照。
> 顺带一条纪律：**任何 A/B 都要先确认「对照组」是健康的** —— 我们曾拿一份建坏的 fp16 引擎
> 当作 FP16 的正常水平，得出「INT8 快 17.5%」这个错误结论（见第九节）。

> ⚠️ `model/engine/` 已被 `.gitignore` 排除，重建产物请自行留档，不要指望它进仓库。

---

## License

MIT（沿用上游 [depth-detect](https://github.com/yzfzzz/depth-detect) License）。

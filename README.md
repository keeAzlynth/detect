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
| 视频文件 (1shu_east_0514.mp4, 30fps)，INT8 双引擎 | 7.47（单帧 135ms） | **7.22（单帧 138.5ms）** |
| USB 摄像头 640×360，INT8 双引擎 | 7.60 | 7.1–7.5 |
| 深度隔帧 `depth_interval: 2`（INT8） | 10.42（旧构建） | 待复测，见第九节 |
| FP16 双引擎（量化改造前的对照，管线口径） | 6.53（单帧 153ms） | — |

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

**瓶颈现状与优化上界**：优化后 `infer_pipeline` 占墙钟 **96.4%**，其中 CPU 发射约 20.7ms、
GPU 执行约 112–115ms。用 `depth_interval: 2` 反推，深度模型占约 85ms GPU、YOLO 约 30–50ms。

而且可以**只用已有数据证明纯 CPU 侧优化已经到顶**：`p3.yolo_wait` 与 `p5.depth_wait` 都在
`processOverlap()` 返回前同步了各自的流 → **本帧全部 GPU 工作必须在这 134ms 内跑完**，
而帧时间 ≥ 每帧 GPU 忙时。所以任何纯 CPU 侧重构的绝对上限是 134ms → 7.46 fps，
相对当前的 7.219 **最多 +3.4%**。

> 顺带纠正一个直觉错误：那 20.7ms 的 CPU 发射**早就是被隐藏的**（深度预处理 kernel 一发射
> GPU 就开工，后面的 `enqueueV2` 都是在 GPU 忙着时做的），并不存在「把发射藏进等待」的收益。

**所以出路只有两条**：① `depth_interval: 2`（改一行配置，业务取舍）；② 转向模型层面
（更小的 YOLO 输入、更轻的深度模型、降输入帧率）—— 后者需要重新导出 ONNX + 重新校准，属大工程。

### 更早的轮次

- **2026-09-05（`ebf5077`，管线口径 6.53 → 7.40 fps）**：INT8 双引擎落地（熵校准 370 帧，
  与 FP16 视觉无可感知差异）；读帧 H2D 改 pinned 暂存，消掉 pageable 拷贝触发的驱动隐式设备同步
  （Depth 发射 49ms → 12ms）；修掉落盘缺深度半区 bug（之前用 swap 把 `depth_vis` 从 context 换走，
  存出的图只有上半原图，1280×720 而不是设计的 1280×1440）。
- **2026-08-28（`469f09b`，6.54 fps）**：单 `Pipeline` 实例消除双份引擎加载；2 槽位帧缓冲池；
  YOLO/Depth 双 CUDA 流；首帧预热；`BaseModel` 输出指针表 static 改实例成员；letterbox 常量复用。

### 已经确认走不通的方向

- **继续在量化层面抠**：sm_53（Maxwell）在 TRT 8.2 下**没有真正的 INT8 卷积 kernel**，
  nvprof 显示卷积全部回退 FP16 winograd / FP32 sgemm（FP32 sgemm 合计 ~25–30ms/帧）。
  这不是配置错误，是架构限制 —— INT8 在这张卡上已经到头。
- **融合 letterbox + CHW 两个预处理 kernel**：全量 nvprof 统计后，自研 CUDA kernel
  （letterbox 2.8 + depth resize 1.5 + transpose 1.3 + decode 0.7 + process 0.5 + normlize 0.2）
  **一共才 ~7.4ms/帧**，融合省不下 2%，不值得动。
- **把拼接也搬进工作线程**：`depth_vis` 直接包在深度模型的 pinned 主机缓冲上，
  下一帧深度的 D2H 会覆盖它 —— 搬过去就是数据竞争，必须留在主线程。

### 待做 / 候选方向

- [ ] `depth_interval: 2`：**当前唯一有意义的杠杆**。旧数字 10.42 fps 是「管线口径 + Round 1 之前的
      旧构建」，不能直接和现在的「端到端 7.219」比，**必须先在当前构建上实测端到端 fps**。
      代价是深度更新率减半（需业务侧确认精度，注意 `motion_state_engine` 用深度算速度/加速度）
- [ ] 模型层面（唯一能突破 +3.4% 上界的方向）：减小 YOLO 输入分辨率（640→512/416）、换更轻的深度模型、
      降输入帧率。都需从 .pt 重新导出 ONNX 并重新校准，属跨轮次大工程
- [ ] CUDA Graphs：静态形状已满足（`1×3×640×640` / `1×3×192×640`），
      把整条 enqueue 链捕获成一张图回放。**但同受 +3.4% 上界约束**，只有当实测 GPU 忙时
      明显低于 134ms（帧内有可观气泡）时才值得做
- [ ] 减小 YOLO 输入分辨率（640→512/416）：需从 .pt 重新导出 ONNX（板上只有固定 640 的 ONNX，
      直接改图会破坏 neck 里 Resize 的常数尺度）
- [ ] Depth 模型 `naiveSlice` kernel ~4.8ms/帧（TRT 内部切片 op，需改模型导出才能消除）
- [ ] 引擎反序列化 / 启动加速：冷启动 24.9s 里第一个引擎独占 24.1s，说明大头是
      CUDA/TensorRT 运行时的一次性装载，而非反序列化本身。顺带清理死代码：
      `/dev/shm` 引擎缓存**只写不读**（`loadEngineFromShm()` 从未被调用，写入本身也失败）
- [ ] `main` 的 RUNPATH 由绝对路径改为 `$ORIGIN`
- [ ] 报警上报 JSON 轻量化（剥离 JsonSender 或改共享内存）

---

## License

MIT（沿用上游 [depth-detect](https://github.com/yzfzzz/depth-detect) License）。

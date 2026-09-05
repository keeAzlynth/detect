# depth-detect-turbo

> Jetson Nano 实时目标检测 + 单目深度估计加速版（重构自 [yzfzzz/depth-detect](https://github.com/yzfzzz/depth-detect)）

面向 **Jetson Nano（Tegra X1）** 压榨性能的实时深度检测管线：
YOLOv8n 检测 + Lite-Mono 单目深度估计，TensorRT INT8 引擎加速（FP16/FP32 备选）+ CUDA 预处理/后处理。

---

## 一、核心特性

- **双模型推理**：YOLOv8n 目标检测 + Lite-Mono-Tiny 单目深度估计，TensorRT 8.2 INT8 引擎（FP16/FP32 保留备用）
- **纯 GPU 管线**：CUDA 预处理（letterbox/归一化）+ CUDA 后处理（NMS），`--use_fast_math`
- **目标跟踪**：ByteTrack + 卡尔曼滤波，支持目标运动状态（趋近/远离/加减速）判断
- **异步重叠**：`overlap: true` 时读帧线程与主循环重叠；YOLO 与 Depth 通过各自 CUDA 流真并行（Depth 先发车，YOLO 等待/跟踪期间 Depth 持续计算）
- **零分配稳态**：单 `Pipeline` 实例 + 2 槽位帧缓冲池循环复用，稳态无每帧 `cudaMalloc`/`cudaFree`；首帧预热在进入主循环前完成
- **报警机制**：危险目标（距离变化）自动生成 AlertMessage，支持 TCP 上报
- **多输入**：H.264 视频文件 / USB 摄像头（`/dev/video0`）两种输入
- **性能模式脚本**：`perf.sh` 一键内核调优（MAXN + jetson_clocks + performance governor）

### 实测性能（Jetson Nano, MAXN 锁频）

| 场景 | 结果 |
|---|---|
| 视频文件 (1shu_east_0514.mp4, 30fps)，**INT8 双引擎** | **7.40 fps**（单帧推理 ~135ms），1000 帧稳态波动 ≤0.01（2026-09-05） |
| 深度隔帧 `depth_interval: 2`（INT8） | **10.42 fps**，深度更新率减半，待业务确认后可启用 |
| FP16 双引擎（对照） | 6.53 fps（单帧推理 ~153ms） |
| USB 摄像头 | 未单独基准（GPU 工作量与视频模式相同，预期同量级） |
| 启动 | ~24s（反序列化 2 个引擎）；首帧 TRT/cuDNN 自动调优已由预热吸收，主循环第一帧即稳态 |
| 内存 | 4GB 板上无换页抖动 |

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
│   ├── main.cpp                  # 主程序（读帧→推理→绘图→保存）
│   ├── core/                     # 流水线核心（Pipeline、IO、报警、运动状态）
│   ├── inference/                # 推理后端(TensorRT/ONNXRuntime) + 模型封装
│   ├── bytetrack/                # ByteTrack 目标跟踪
│   ├── op_kernel/                # CUDA 预处理/后处理 kernel
│   ├── tools/                    # 绘制(FPS/检测框) + 计时
│   └── utils/                    # 配置加载 + 日志
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
cmake -DENABLE_JESTON_MEM_MANAGED=ON ..     # Jetson 统一内存（实验性）
cmake -DTARGET_CUDA_ARCHS=53 ..             # 手动指定 CUDA 架构
cmake -DCMAKE_BUILD_TYPE=Debug ..           # 调试构建
```

产物输出到 `bin/`（`main` + 各动态库 `lib*.so`），已放在同一目录，运行时无需额外设置 `LD_LIBRARY_PATH`。

> 提示：`cmake --build .` 串行可能较慢，建议追加 `-- -j4`。

---

## 五、运行

### 5.1 跑摄像头（USB /dev/video0）

```bash
cd ~/depth-detect-turbo/bin
# 先开启性能模式（可选但推荐，自动 sudo）
./perf.sh
# 运行：第一个参数 0 表示摄像头设备号
./main 0 config.yaml
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

处理结果（检测框 + 深度图逐帧拼接图）保存到 `bin/out_dir/`。

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
| `depth.depth_interval` | `1` | 隔帧做深度推理的间隔（调大省算力） |
| `io_manager.save_mode` | `images` | images / video / both / none |
| `io_manager.send_tcp` | `false` | 是否 TCP 上报报警数据 |
| `logger.log_level` | `debug` | trace / debug / info / warn / err |

---

## 八、常见问题

| 问题 | 解决 |
|---|---|
| `Failed to open video: 0` | 程序内置支持纯数字设备号走 `/dev/video0`；确认 USB 摄像头已插、`ls /dev/video*` 可见 |
| `cmake: CMAKE_CUDA_COMPILER could not be found` | 本新版 CMakeLists 会自动找 nvcc；若仍失败，`sudo apt install nvidia-cuda-toolkit` 或用 `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` |
| 启动慢（约20s+） | 正常：需反序列化 2 个 TRT 引擎（~24s）；首帧自动调优已由预热吸收，主循环首帧即稳态 |
| `VIDEOIO ERROR: V4L2: property frame_count is not supported` | 正常：摄像头无帧总数属性，不影响运行 |
| pip 装包 ProxyError | 板子走代理时用 `env -u http_proxy -u https_proxy ...` 绕开 |

---

## 九、性能优化记录与方向

已完成（2026-08-28，commit 469f09b）：

- [x] 单 `Pipeline` 实例：`AsyncPipeline` 改持外部引用，消除双份引擎加载（启动/显存减半）
- [x] 2 槽位帧缓冲池：稳态零 `cudaMalloc`/`cudaFree`（后者隐式同步全设备，是隐性停顿点）
- [x] YOLO/Depth 双 CUDA 流真重叠：Depth 先异步发车，与 YOLO 等待/跟踪期并行
- [x] 首帧预热：TRT/cuDNN 首帧 ~35s 自动调优移出主循环
- [x] 深度缓存 swap 顺序修复：运动状态判定不再拿到空深度图
- [x] `BaseModel` 输出指针表 static → 实例成员（消除多实例互踩隐患）
- [x] letterbox 常量真正复用 + 乘倒数替代除法；`cv::Mat` 热路径传参改 `const &`
- [x] `PhaseTimer` 编译期开关（`-DPIPELINE_PHASE_TIMER=ON` 可开，默认零开销）

已完成（2026-09-05，本轮，6.53 → 7.40 fps / +13%）：

- [x] **INT8 引擎落地**：YOLOv8n + Lite-Mono-Tiny 双模型 INT8（熵校准 370 帧），应用内 6.53 → 7.40 fps
      （1000 帧稳态）。检测框与深度伪彩视觉验证与 FP16 无可感知差异（同帧对比）。
      ⚠️ 实测结论：Maxwell（sm_53）在 TRT 8.2 下**没有真正的 INT8 卷积 kernel**——nvprof 显示卷积
      全部回退 FP16 winograd / FP32 sgemm（`cuInt8::nchwToNchhw2` 只有量化搬运），收益来自
      逐层策略重选：YOLO +4%、Depth +17%（trtexec 单测），应用内合计 +13%。INT8 在本卡上已到头，
      进一步提速只能减少 GPU 工作量（见待做）。
- [x] **读帧 H2D 改 pinned 暂存**（`frame.h` / `io_manager.cpp`）：pageable 内存直接
      `cudaMemcpyAsync` 会触发驱动隐式设备同步，与主循环 TRT enqueueV2 相撞，把 kernel
      发射拖大 4 倍（Depth 发射实测 49ms → 12ms，trtexec 单测 11.4ms）。
      注：该 CPU 停顿此前被 GPU 饱和掩盖，对帧率无直接影响，但消除了 p90 抖动源。
- [x] **修复落盘缺深度半区 bug**（`pipeline.cpp`）：`process/processOverlap` 末尾用 swap 把
      `depth_vis` 换入缓存，导致主循环绘图/落盘拿到空 Mat——`depth_interval=1` 时每帧存出的
      图都只有上半原图（1280x720 而非设计的 1280x1440 拼接图）。改为 Mat 浅拷贝缓存（仍零像素拷贝）。
- [x] INT8 校准工具链：`~/build_int8.py`（pycuda + IInt8EntropyCalibrator2），
      校准缓存保留在 `~/trt_int8/`（重建引擎免重校准），引擎已装入
      `model/engine/*/`（gitignore 不入库）。校准图已清理（370 帧，49MB），
      需重新校准时从任意测试视频一键重抽：
      `ffmpeg -i <视频> -vf select='not(mod(n,7))' -vsync vfr -q:v 2 ~/calib_int8/img_%04d.jpg`
- [x] 性能分析方法论：trtexec 单引擎基线 + nvprof 逐 kernel 分解 + tegrastats。
      帧时间 ≈ 两模型 GPU 时间之和（双流在饱和 GPU 上无真并行），该结论已实测闭环。

本轮问题记录（2026-09-05，均与代码/性能直接相关）：

- **pageable 内存 H2D 隐式设备同步**（已修）：读帧线程直接 `cudaMemcpyAsync(pageable→device)`
  会与主循环 TRT enqueueV2 相撞，kernel 发射拖大 4 倍（Depth 发射实测 49ms vs trtexec 11.4ms，
  p90 46ms 双峰）。已改 pinned 暂存；该 CPU 停顿此前被 GPU 饱和掩盖，对帧率无直接影响。
- **swap 缓存清空 depth_vis**（已修）：上一轮 swap 零拷贝缓存把 `depth_vis` 从 context 换走，
  落盘图长期缺深度半区（1280x720 ≠ 设计的 1280x1440）。教训：对"零拷贝优化"要用输出物验证，
  对比落盘图片尺寸即可发现。
- **sm_53 的 INT8 架构限制**（已实测闭环）：INT8 引擎能建能跑，但无 INT8 卷积 kernel，卷积全部
  回退 FP16/FP32。校准工具链保留（`~/build_int8.py` + `~/trt_int8/*cache`），不要再期待量化层面
  的进一步收益。
- **遗留问题**：应用收尾退出偏慢（teardown 阶段 cuModuleUnload ×594 ≈ 2.6s + cudaFree），
  不影响稳态帧率；启动 ~24s 引擎反序列化仍是待做项。

待做 / 候选方向：

- [ ] `depth_interval: 2`：深度隔帧，INT8 后**实测 10.42 fps**，代价是深度更新率减半（需业务侧确认精度）
- [ ] 减小 YOLO 输入分辨率（640→512/416）：需从 .pt 重新导出 ONNX（板上只有固定 640 的 ONNX，
      直接改图会破坏 neck 里 Resize 的常数尺度）
- [ ] YOLO letterbox 双 kernel 融合（letterbox+CHW 各写一遍 640x640，nvprof 实测 letterbox 2.8ms/帧）
- [ ] Depth 模型 `naiveSlice` kernel ~4.8ms/帧（模型结构决定的切片 op，需改模型导出才能消除）
- [ ] 引擎反序列化加速（当前 `/dev/shm` 缓存只写不读，反序列化本身仍是启动 ~24s 的主体）
- [ ] 报警上报 JSON 轻量化（剥离 JsonSender 或改共享内存）

---

## License

MIT（沿用上游 [depth-detect](https://github.com/yzfzzz/depth-detect) License）。
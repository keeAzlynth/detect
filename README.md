# depth-detect-turbo

> Jetson Nano 实时目标检测 + 单目深度估计加速版（重构自 [yzfzzz/depth-detect](https://github.com/yzfzzz/depth-detect)）

面向 **Jetson Nano（Tegra X1）** 压榨性能的实时深度检测管线：
YOLOv8n 检测 + Lite-Mono 单目深度估计，TensorRT FP16 引擎加速 + CUDA 预处理/后处理。

---

## 一、核心特性

- **双模型推理**：YOLOv8n 目标检测 + Lite-Mono-Tiny 单目深度估计，TensorRT 8.2 FP16 引擎
- **纯 GPU 管线**：CUDA 预处理（letterbox/归一化）+ CUDA 后处理（NMS），`--use_fast_math`
- **目标跟踪**：ByteTrack + 卡尔曼滤波，支持目标运动状态（趋近/远离/加减速）判断
- **异步重叠**：`overlap: true` 时推理与主循环重叠执行，降低帧间等待
- **报警机制**：危险目标（距离变化）自动生成 AlertMessage，支持 TCP 上报
- **多输入**：H.264 视频文件 / USB 摄像头（`/dev/video0`）两种输入
- **性能模式脚本**：`perf.sh` 一键内核调优（MAXN + jetson_clocks + performance governor）

### 实测性能（Jetson Nano, FP16, MAXN）

| 场景 | 结果 |
|---|---|
| 视频文件 (1shu_east_0514.mp4, 30fps) | 约 6.3 fps（单帧管线 ~159ms） |
| USB 摄像头 (性能模式) | 稳态约 6 fps |

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
│   ├── engine/                   # 已导出的 FP16/FP32 TRT 引擎
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
| 启动慢（约20s） | 正常：启动需反序列化 2 个 TRT 引擎，之后进入稳态 FPS |
| `VIDEOIO ERROR: V4L2: property frame_count is not supported` | 正常：摄像头无帧总数属性，不影响运行 |
| pip 装包 ProxyError | 板子走代理时用 `env -u http_proxy -u https_proxy ...` 绕开 |

---

## 九、性能优化方向（TODO / 重构点）

- [ ] 引擎反序列化缓存到共享内存，跳过启动 20s 冷加载
- [ ] 深度推理隔帧（`depth_interval`）与延迟渲染分离
- [ ] 生产者-消费者线程池，读帧/推理/绘图流水线重排
- [ ] 裁剪 onnxruntime 依赖（仅 TRT 后端运行时可不链接）
- [ ] 报警上报 JSON 轻量化（剥离 JsonSender 或改共享内存）

---

## License

MIT（沿用上游 [depth-detect](https://github.com/yzfzzz/depth-detect) License）。
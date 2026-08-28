# JsonSender

一个轻量的 TCP JSON 发送封装库，基于 C++11，包含可选的 CLI 测试工具和 UDS 代理进程。

## 文件说明

| 文件 | 说明 |
|---|---|
| `JsonSender.hpp` | **核心头文件**，TCP 连接 + JSON 发送封装，其他工程直接 include 此文件即可 |
| `ansi_color.hpp` | ANSI 终端颜色辅助头文件，供上层使用 |
| `JsonSenderCli.cpp` | 可选：交互式 CLI 测试工具，支持 Tab 补全、历史记录、JSON 格式化打印 |
| `JsonSenderUds.cpp` | 可选：UDS 代理进程，将本地 UDS 客户端的 JSON 数据透传到 TCP 服务器 |
| `server_test.cpp` | 简单 TCP echo 服务器，用于本地测试（打印收到的数据并回复 JSON） |

## 依赖（子模块）

```bash
git submodule update --init --recursive
```

| 子模块 | 路径 | 说明 |
|---|---|---|
| [nlohmann/json](https://github.com/nlohmann/json) | `lib/json` | JSON 解析库（header-only） |
| [linenoise](https://github.com/Sheep118/linenoise) | `lib/linenoise` | 行编辑库，供 CLI 工具使用 |

---

## 方式一：作为头文件直接使用

只需将本仓库加入工程，include `JsonSender.hpp` 即可使用核心功能。

### 在外部工程的 CMakeLists.txt 中集成

```cmake
# 将本仓库作为子目录添加（例如放在 third_party/JsonSender）
add_subdirectory(third_party/JsonSender)

# 链接到你的目标
target_link_libraries(your_target PRIVATE JsonSender)
```

之后在代码中：

```cpp
#include <JsonSender.hpp>

// 连接到 TCP 服务器
JsonSender sender("127.0.0.1", 9000);

// 发送 JSON 对象
nlohmann::json j = {{"cmd", "hello"}, {"value", 42}};
sender.send(j);

// 发送 JSON 字符串（会自动校验格式）
sender.send(R"({"cmd": "ping"})");

// 获取底层 fd，自行用 epoll 监听服务器返回数据
int fd = sender.get_fd();
```

---

## 方式二：通过 UDS 代理进程通信

如果你的进程不方便直接引入本库，可以先启动 `JsonSenderUds` 代理进程，
然后通过 Unix Domain Socket 与之通信，代理会负责 JSON 校验和 TCP 转发。

### 启动代理进程

```bash
# 构建
./build.sh --uds

# 运行（单向转发：UDS → TCP）
./build/JsonSenderUds --host 127.0.0.1 --port 9000

# 双向转发（TCP 服务器回包也会转发回 UDS 客户端）
./build/JsonSenderUds --host 127.0.0.1 --port 9000 --sync

# 自定义 UDS 路径（默认 /tmp/jsonsender.sock）
./build/JsonSenderUds --host 127.0.0.1 --port 9000 --uds /tmp/my.sock
```

### 外部工程通过 UDS 发送数据

只需连接 UDS socket 并发送 JSON 字符串，**无需引入本库**：

```cpp
#include <sys/un.h>
#include <sys/socket.h>
#include <unistd.h>
#include <string>

int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
sockaddr_un addr{};
addr.sun_family = AF_UNIX;
strncpy(addr.sun_path, "/tmp/jsonsender.sock", sizeof(addr.sun_path) - 1);
connect(fd, (sockaddr*)&addr, sizeof(addr));

std::string msg = R"({"cmd": "test", "value": 1})";
send(fd, msg.c_str(), msg.size(), 0);
```

> 发送非 JSON 格式数据时，代理会丢弃并打印错误，不会转发到 TCP 服务器。

---

## 方式三：CLI 交互测试

```bash
./build.sh --cli
./build/JsonSenderCli [--sync]
```

| 命令 | 说明 |
|---|---|
| `socket open <ip> <port>` | 连接到 TCP 服务器 |
| `socket send '<json>'` | 发送 JSON（严格校验，不合法则拒绝） |
| `socket close` | 断开连接 |
| `exit` / `quit` | 退出程序 |

支持 Tab 补全、右侧输入提示、历史记录持久化（保存在 `~/.jsonsender_cli_history`）。

---

## 构建脚本

```bash
./build.sh [选项]

  --cli      构建 JsonSenderCli
  --uds      构建 JsonSenderUds
  --clean    清理后重新构建
  --release  Release 模式（默认 Debug）
  -j N       并行线程数（默认 nproc）
```

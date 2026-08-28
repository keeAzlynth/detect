#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <linenoise.h>
#include "ansi_color.hpp"

// ── 全局配置 ──────────────────────────────────────────────────────
static bool        g_sync     = false;  // --sync  : 接收服务器数据并转发/打印
static bool        g_clitest  = false;  // --clitest: 启用 CLI 测试模式
static std::string g_host;              // --host  : UDS 模式下目标服务器 IP
static int         g_port     = 0;      // --port  : UDS 模式下目标服务器端口
static std::string g_uds_path = "/tmp/project2.sock"; // --uds

// ── 工具 ──────────────────────────────────────────────────────────
static void set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static int do_tcp_connect(const std::string& host, int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return -1; }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
        std::cerr << ansi::colored("[error] 无效地址: " + host + "\n", ansi::fg::red);
        close(fd); return -1;
    }
    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("connect"); close(fd); return -1;
    }
    return fd;
}

static bool parse_args(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--sync")    { g_sync    = true; }
        else if (a == "--clitest") { g_clitest = true; }
        else if ((a == "--host" || a == "-H") && i + 1 < argc) { g_host     = argv[++i]; }
        else if ((a == "--port" || a == "-p") && i + 1 < argc) { g_port     = std::stoi(argv[++i]); }
        else if ((a == "--uds"  || a == "-u") && i + 1 < argc) { g_uds_path = argv[++i]; }
        else {
            std::cerr << "未知参数: " << a << "\n"
                      << "用法: " << argv[0]
                      << " [--sync] [--clitest] [--host <ip>] [--port <n>] [--uds <path>]\n";
            return false;
        }
    }
    return true;
}

// ══════════════════════════════════════════════════════════════════
// CLI 测试模式
//   --clitest        : 启动 linenoise 终端，支持以下命令
//   socket open <ip> <port>   连接到 TCP 服务器
//   socket send "data"        发送字符串
//   socket close              断开连接
//   --sync           : 同时把服务器返回数据打印到 stdout
// ══════════════════════════════════════════════════════════════════
static int g_cli_tcp  = -1;
static int g_cli_epfd = -1;

static void cli_epoll_add(int fd) {
    if (g_cli_epfd < 0 || fd < 0) return;
    epoll_event ev{};
    ev.events  = EPOLLIN;
    ev.data.fd = fd;
    epoll_ctl(g_cli_epfd, EPOLL_CTL_ADD, fd, &ev);
}

static void cli_epoll_del(int fd) {
    if (g_cli_epfd < 0 || fd < 0) return;
    epoll_ctl(g_cli_epfd, EPOLL_CTL_DEL, fd, nullptr);
}

static void cli_open(const std::string& host, int port) {
    if (g_cli_tcp >= 0) {
        std::cout << ansi::colored("[warn] 已有连接，请先 socket close\n", ansi::fg::yellow);
        return;
    }
    g_cli_tcp = do_tcp_connect(host, port);
    if (g_cli_tcp < 0) return;
    set_nonblocking(g_cli_tcp);
    if (g_sync) cli_epoll_add(g_cli_tcp); // --sync 才需要监听服务器数据
    std::cout << ansi::colored("[ok] 已连接 " + host + ":" + std::to_string(port) + "\n",
                               ansi::fg::green);
}

static void cli_send(const std::string& data) {
    if (g_cli_tcp < 0) {
        std::cout << ansi::colored("[error] 未连接，请先 socket open <ip> <port>\n",
                                   ansi::fg::red);
        return;
    }
    if (::send(g_cli_tcp, data.c_str(), data.size(), MSG_NOSIGNAL) < 0) {
        perror("send");
    } else {
        std::cout << ansi::colored("[sent] ", ansi::fg::green) << data << "\n";
    }
}

static void cli_close() {
    if (g_cli_tcp < 0) {
        std::cout << ansi::colored("[warn] 当前没有连接\n", ansi::fg::yellow);
        return;
    }
    cli_epoll_del(g_cli_tcp);
    close(g_cli_tcp);
    g_cli_tcp = -1;
    std::cout << ansi::colored("[ok] 连接已关闭\n", ansi::fg::green);
}

// 解析并执行 CLI 命令，返回 false 表示应退出
static bool dispatch_cli(const char* raw) {
    std::string line(raw);
    if (line.empty()) return true;
    linenoiseHistoryAdd(raw);

    std::istringstream ss(line);
    std::string tok;
    ss >> tok;

    if (tok == "exit" || tok == "quit") {
        cli_close();
        return false;
    }
    if (tok == "help") {
        std::cout << "  socket open <ip> <port>   连接到 TCP 服务器\n"
                  << "  socket send \"data\"         发送字符串\n"
                  << "  socket close              关闭当前连接\n"
                  << "  exit / quit               退出程序\n";
        return true;
    }
    if (tok != "socket") {
        std::cout << ansi::colored("[error] 未知命令: " + tok + "（输入 help 查看帮助）\n",
                                   ansi::fg::red);
        return true;
    }

    ss >> tok;
    if (tok == "open") {
        std::string host; int port;
        if (!(ss >> host >> port))
            std::cout << ansi::colored("[error] 用法: socket open <ip> <port>\n", ansi::fg::red);
        else
            cli_open(host, port);
    } else if (tok == "send") {
        std::string rest;
        std::getline(ss >> std::ws, rest);
        // 去掉首尾英文引号
        if (rest.size() >= 2 && rest.front() == '"' && rest.back() == '"')
            rest = rest.substr(1, rest.size() - 2);
        cli_send(rest);
    } else if (tok == "close") {
        cli_close();
    } else {
        std::cout << ansi::colored("[error] 未知子命令: " + tok + "\n", ansi::fg::red);
    }
    return true;
}

// 处理从 TCP 服务器读到的数据（--sync 时在 epoll 事件中调用）
static void cli_handle_recv(linenoiseState* ls) {
    char buf[4096];
    ssize_t r = recv(g_cli_tcp, buf, sizeof(buf) - 1, 0);
    if (r > 0) {
        buf[r] = '\0';
        linenoiseHide(ls);
        std::cout << ansi::colored("[recv] ", ansi::fg::cyan) << buf << std::flush;
        linenoiseShow(ls);
    } else {
        // 服务器主动断开
        linenoiseHide(ls);
        std::cout << ansi::colored("[warn] 服务器已关闭连接\n", ansi::fg::yellow);
        linenoiseShow(ls);
        cli_epoll_del(g_cli_tcp);
        close(g_cli_tcp);
        g_cli_tcp = -1;
    }
}

// ── linenoise Tab 补全回调 ────────────────────────────────────────
static void cli_completion(const char* buf, linenoiseCompletions* lc) {
    static const char* candidates[] = {
        "socket open ",
        "socket send ",
        "socket close",
        "help",
        "exit",
        "quit",
        nullptr
    };
    std::string input(buf);
    for (int i = 0; candidates[i]; ++i) {
        std::string c(candidates[i]);
        if (c.size() >= input.size() && c.substr(0, input.size()) == input)
            linenoiseAddCompletion(lc, candidates[i]);
    }
}

// ── linenoise 右侧提示回调 ────────────────────────────────────────
static char* cli_hints(const char* buf, int* color, int* bold) {
    struct Entry { const char* cmd; const char* hint; int clr; };
    static const Entry table[] = {
        {"socket",       " open|send|close",  33},
        {"socket open",  " <ip> <port>",      33},
        {"socket send",  " \"data\"",         32},
        {"socket close", "",                   0},
        {"help",         "",                   0},
        {"exit",         "",                   0},
        {"quit",         "",                   0},
        {nullptr, nullptr, 0}
    };
    std::string input(buf);
    for (int i = 0; table[i].cmd; ++i) {
        if (input == table[i].cmd) {
            *color = table[i].clr;
            *bold  = 0;
            return const_cast<char*>(table[i].hint);
        }
    }
    return nullptr;
}

static void run_clitest() {
    std::cout << ansi::styled("CLI 测试模式", ansi::fg::bright_green, ansi::style::bold)
              << "  --sync=" << (g_sync ? "ON（打印服务器返回数据）" : "OFF") << "\n"
              << "  命令: socket open <ip> <port> | socket send \"data\""
                 " | socket close | exit\n\n";

    linenoiseHistorySetMaxLen(100);
    linenoiseSetCompletionCallback(cli_completion);
    linenoiseSetHintsCallback(cli_hints);
    const std::string prompt = std::string(ansi::fg::cyan) + "cli> " + ansi::reset;

    // epoll：始终监听 stdin；连接成功 + --sync 后加入 tcp_fd
    g_cli_epfd = epoll_create1(0);
    {
        epoll_event ev{};
        ev.events  = EPOLLIN;
        ev.data.fd = STDIN_FILENO;
        epoll_ctl(g_cli_epfd, EPOLL_CTL_ADD, STDIN_FILENO, &ev);
    }

    linenoiseState ls{};
    char buf[4096];
    epoll_event events[4];
    bool running = true;

    while (running) {
        linenoiseEditStart(&ls, STDIN_FILENO, STDOUT_FILENO,
                           buf, sizeof(buf), prompt.c_str());

        // 内层：等待当前行输入完成，期间响应服务器数据
        bool got_line = false;
        while (!got_line) {
            int n = epoll_wait(g_cli_epfd, events, 4, -1);
            if (n < 0) { perror("epoll_wait"); break; }

            for (int i = 0; i < n; ++i) {
                int fd = events[i].data.fd;

                if (fd == STDIN_FILENO) {
                    char* line = linenoiseEditFeed(&ls);
                    if (line == linenoiseEditMore) continue; // 用户还在输入
                    linenoiseEditStop(&ls);
                    got_line = true;
                    if (line == nullptr) {          // Ctrl+D / EOF
                        std::cout << "\n";
                        cli_close();
                        running = false;
                    } else {
                        running = dispatch_cli(line);
                        linenoiseFree(line);
                    }
                    break; // 跳出 for，回到外层重新 EditStart

                } else if (g_sync && fd == g_cli_tcp && g_cli_tcp >= 0) {
                    cli_handle_recv(&ls);
                }
            }
        }
    }

    close(g_cli_epfd);
    g_cli_epfd = -1;
}

// ══════════════════════════════════════════════════════════════════
// UDS 服务器模式（无 --clitest）
//   监听 UDS socket，客户端连入时建立 TCP 连接
//   UDS 客户端 → TCP 服务器（单向）
//   TCP 服务器 → UDS 客户端（仅 --sync 时开启，双向）
//   客户端断开时同步断开 TCP 连接
// ══════════════════════════════════════════════════════════════════
static void run_uds_server() {
    if (g_host.empty() || g_port == 0) {
        std::cerr << ansi::colored(
            "[fatal] UDS 模式需要指定目标服务器: --host <ip> --port <n>\n",
            ansi::fg::red);
        return;
    }

    int uds_srv = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (uds_srv < 0) { perror("socket(AF_UNIX)"); return; }

    unlink(g_uds_path.c_str());
    sockaddr_un uaddr{};
    uaddr.sun_family = AF_UNIX;
    strncpy(uaddr.sun_path, g_uds_path.c_str(), sizeof(uaddr.sun_path) - 1);

    if (bind(uds_srv, reinterpret_cast<sockaddr*>(&uaddr), sizeof(uaddr)) < 0) {
        perror("bind(UDS)"); close(uds_srv); return;
    }
    if (listen(uds_srv, 5) < 0) {
        perror("listen(UDS)"); close(uds_srv); return;
    }
    set_nonblocking(uds_srv);

    std::cout << ansi::colored("[uds] 监听 " + g_uds_path + "\n", ansi::fg::cyan)
              << ansi::colored("[uds] 目标 TCP " + g_host + ":" + std::to_string(g_port) + "\n",
                               ansi::fg::cyan)
              << ansi::colored(std::string("[uds] --sync=") +
                               (g_sync ? "ON（双向转发）" : "OFF（单向 UDS→TCP）") + "\n",
                               ansi::fg::cyan);

    int epfd = epoll_create1(0);
    epoll_event ev{}, events[8];
    ev.events  = EPOLLIN;
    ev.data.fd = uds_srv;
    epoll_ctl(epfd, EPOLL_CTL_ADD, uds_srv, &ev);

    int uds_cli = -1;
    int tcp_fd  = -1;

    while (true) {
        int n = epoll_wait(epfd, events, 8, -1);
        if (n < 0) { perror("epoll_wait"); break; }

        for (int i = 0; i < n; ++i) {
            int fd = events[i].data.fd;

            // ── 新 UDS 客户端连入 ────────────────────────────────
            if (fd == uds_srv) {
                int cli = accept(uds_srv, nullptr, nullptr);
                if (cli < 0) continue;

                // 已有会话：先清理旧的
                if (uds_cli >= 0) {
                    std::cout << ansi::colored("[uds] 替换旧连接\n", ansi::fg::yellow);
                    epoll_ctl(epfd, EPOLL_CTL_DEL, uds_cli, nullptr);
                    close(uds_cli);
                    if (tcp_fd >= 0) {
                        epoll_ctl(epfd, EPOLL_CTL_DEL, tcp_fd, nullptr);
                        close(tcp_fd);
                    }
                    uds_cli = tcp_fd = -1;
                }

                int tfd = do_tcp_connect(g_host, g_port);
                if (tfd < 0) {
                    std::cerr << ansi::colored("[uds] 无法建立 TCP 连接，拒绝客户端\n",
                                               ansi::fg::red);
                    close(cli);
                    continue;
                }

                uds_cli = cli;
                tcp_fd  = tfd;
                set_nonblocking(uds_cli);
                set_nonblocking(tcp_fd);

                ev.events = EPOLLIN;
                ev.data.fd = uds_cli;
                epoll_ctl(epfd, EPOLL_CTL_ADD, uds_cli, &ev);

                if (g_sync) {
                    ev.data.fd = tcp_fd;
                    epoll_ctl(epfd, EPOLL_CTL_ADD, tcp_fd, &ev);
                }

                std::cout << ansi::colored(
                    "[uds] 客户端连接，TCP 已建立 → " +
                    g_host + ":" + std::to_string(g_port) + "\n",
                    ansi::fg::green);

            // ── UDS 客户端 → TCP 服务器 ──────────────────────────
            } else if (fd == uds_cli) {
                char buf[4096];
                ssize_t r = recv(uds_cli, buf, sizeof(buf), 0);
                if (r <= 0) {
                    std::cout << ansi::colored("[uds] 客户端断开，关闭 TCP\n",
                                               ansi::fg::yellow);
                    epoll_ctl(epfd, EPOLL_CTL_DEL, uds_cli, nullptr);
                    close(uds_cli); uds_cli = -1;
                    if (tcp_fd >= 0) {
                        epoll_ctl(epfd, EPOLL_CTL_DEL, tcp_fd, nullptr);
                        close(tcp_fd); tcp_fd = -1;
                    }
                } else if (tcp_fd >= 0) {
                    ::send(tcp_fd, buf, static_cast<size_t>(r), MSG_NOSIGNAL);
                }

            // ── TCP 服务器 → UDS 客户端（--sync）────────────────
            } else if (fd == tcp_fd) {
                char buf[4096];
                ssize_t r = recv(tcp_fd, buf, sizeof(buf), 0);
                if (r <= 0) {
                    std::cout << ansi::colored("[uds] TCP 服务器断开\n", ansi::fg::yellow);
                    epoll_ctl(epfd, EPOLL_CTL_DEL, tcp_fd, nullptr);
                    close(tcp_fd); tcp_fd = -1;
                    if (uds_cli >= 0) {
                        epoll_ctl(epfd, EPOLL_CTL_DEL, uds_cli, nullptr);
                        close(uds_cli); uds_cli = -1;
                    }
                } else if (uds_cli >= 0) {
                    ::send(uds_cli, buf, static_cast<size_t>(r), MSG_NOSIGNAL);
                }
            }
        }
    }

    if (uds_cli >= 0) close(uds_cli);
    if (tcp_fd  >= 0) close(tcp_fd);
    close(uds_srv);
    close(epfd);
    unlink(g_uds_path.c_str());
}

// ── 入口 ──────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (!parse_args(argc, argv)) return 1;

    if (g_clitest) {
        run_clitest();
    } else {
        run_uds_server();
    }
    return 0;
}
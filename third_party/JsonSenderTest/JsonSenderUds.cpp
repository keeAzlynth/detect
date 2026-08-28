// JsonSenderUds.cpp
// UDS 代理服务器：通过 JsonSender 将本地 UDS 客户端的 JSON 数据转发到 TCP 服务器
//
// 用法:
//   JsonSenderUds --host <ip> --port <n> [--sync] [--uds <path>]
//   --host / -H   目标 TCP 服务器 IP
//   --port / -p   目标 TCP 服务器端口
//   --sync        开启后将 TCP 服务器返回 JSON 转发回 UDS 客户端（双向）
//   --uds  / -u   UDS socket 路径（默认 /tmp/jsonsender.sock）
//
// 数据流:
//   UDS 客户端 ──[JSON]──▶ JsonSenderUds ──[JSON]──▶ TCP 服务器
//   UDS 客户端 ◀──[JSON]── JsonSenderUds ◀──[JSON]── TCP 服务器   (仅 --sync)

#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <nlohmann/json.hpp>
#include "JsonSender.hpp"

static bool        g_sync    = false;
static std::string g_host;
static int         g_port    = 0;
static std::string g_uds_path = "/tmp/jsonsender.sock";

static void set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

// 从 src_fd 读取数据，校验 JSON 后通过 sender 发送到 TCP
// 返回 false 表示连接已断开
static bool relay_uds_to_tcp(int src_fd, JsonSender* sender) {
    char buf[65536];
    ssize_t r = recv(src_fd, buf, sizeof(buf) - 1, 0);
    if (r <= 0) return false;
    buf[r] = '\0';
    try {
        auto j = nlohmann::json::parse(buf, buf + r);
        sender->send(j);
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << ansi::colored("[uds-warn] 收到非法 JSON，已丢弃:\n", ansi::fg::yellow)
                  << "  原始内容: " << std::string(buf, r) << "\n"
                  << ansi::colored("  解析错误: ", ansi::fg::red) << e.what() << "\n";
    }
    return true;
}

// 从 TCP 服务器读取数据，校验 JSON 后转发到 dst_fd（UDS 客户端）
// 返回 false 表示连接已断开
static bool relay_tcp_to_uds(int tcp_fd, int dst_fd) {
    char buf[65536];
    ssize_t r = recv(tcp_fd, buf, sizeof(buf) - 1, 0);
    if (r <= 0) return false;
    buf[r] = '\0';
    try {
        auto j = nlohmann::json::parse(buf, buf + r);
        std::string out = j.dump();
        ::send(dst_fd, out.c_str(), out.size(), MSG_NOSIGNAL);
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << ansi::colored("[tcp-warn] 服务器返回非法 JSON，已丢弃:\n", ansi::fg::yellow)
                  << "  原始内容: " << std::string(buf, r) << "\n"
                  << ansi::colored("  解析错误: ", ansi::fg::red) << e.what() << "\n";
    }
    return true;
}

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--sync")                       { g_sync = true; }
        else if ((a == "--host" || a == "-H") && i + 1 < argc) { g_host     = argv[++i]; }
        else if ((a == "--port" || a == "-p") && i + 1 < argc) { g_port     = std::atoi(argv[++i]); }
        else if ((a == "--uds"  || a == "-u") && i + 1 < argc) { g_uds_path = argv[++i]; }
        else {
            std::cerr << "用法: " << argv[0]
                      << " --host <ip> --port <n> [--sync] [--uds <path>]\n";
            return 1;
        }
    }

    if (g_host.empty() || g_port == 0) {
        std::cerr << ansi::colored("[fatal] 必须指定 --host <ip> --port <n>\n", ansi::fg::red);
        return 1;
    }

    // ── 创建 UDS 监听 socket ──────────────────────────────────────
    int uds_srv = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (uds_srv < 0) { perror("socket(AF_UNIX)"); return 1; }

    unlink(g_uds_path.c_str());
    sockaddr_un uaddr{};
    uaddr.sun_family = AF_UNIX;
    strncpy(uaddr.sun_path, g_uds_path.c_str(), sizeof(uaddr.sun_path) - 1);

    if (bind(uds_srv, reinterpret_cast<sockaddr*>(&uaddr), sizeof(uaddr)) < 0) {
        perror("bind(UDS)"); close(uds_srv); return 1;
    }
    if (listen(uds_srv, 5) < 0) {
        perror("listen(UDS)"); close(uds_srv); return 1;
    }
    set_nonblocking(uds_srv);

    std::cout << ansi::colored("[uds] 监听 " + g_uds_path + "\n", ansi::fg::cyan)
              << ansi::colored("[uds] 目标 TCP " + g_host + ":" + std::to_string(g_port) + "\n",
                               ansi::fg::cyan)
              << ansi::colored(std::string("[uds] --sync=") +
                               (g_sync ? "ON（双向 JSON 转发）" : "OFF（单向 UDS→TCP）") + "\n",
                               ansi::fg::cyan);

    // ── epoll 初始化 ──────────────────────────────────────────────
    int epfd = epoll_create1(0);
    epoll_event ev{}, events[8];
    ev.events  = EPOLLIN;
    ev.data.fd = uds_srv;
    epoll_ctl(epfd, EPOLL_CTL_ADD, uds_srv, &ev);

    int         uds_cli = -1;
    JsonSender* sender  = nullptr;

    // 清理当前会话（析构 JsonSender 自动关闭 TCP）
    auto cleanup = [&]() {
        if (uds_cli >= 0) {
            epoll_ctl(epfd, EPOLL_CTL_DEL, uds_cli, nullptr);
            close(uds_cli);
            uds_cli = -1;
        }
        if (sender) {
            if (g_sync)
                epoll_ctl(epfd, EPOLL_CTL_DEL, sender->get_fd(), nullptr);
            delete sender;  // 析构函数关闭 TCP
            sender = nullptr;
        }
    };

    // ── 主循环 ────────────────────────────────────────────────────
    while (true) {
        int n = epoll_wait(epfd, events, 8, -1);
        if (n < 0) { perror("epoll_wait"); break; }

        for (int i = 0; i < n; ++i) {
            int fd = events[i].data.fd;

            // ── 新 UDS 客户端连入 ─────────────────────────────────
            if (fd == uds_srv) {
                int cli = accept(uds_srv, nullptr, nullptr);
                if (cli < 0) continue;

                if (uds_cli >= 0) {
                    std::cout << ansi::colored("[uds] 替换旧连接\n", ansi::fg::yellow);
                    cleanup();
                }

                // 建立 JsonSender（TCP 连接）
                auto* s = new JsonSender(g_host, g_port);
                if (s->get_fd() < 0) {
                    std::cerr << ansi::colored("[uds] 无法建立 TCP 连接，拒绝客户端\n",
                                               ansi::fg::red);
                    delete s;
                    close(cli);
                    continue;
                }

                uds_cli = cli;
                sender  = s;
                sender->set_nonblocking();
                set_nonblocking(uds_cli);

                ev.events  = EPOLLIN;
                ev.data.fd = uds_cli;
                epoll_ctl(epfd, EPOLL_CTL_ADD, uds_cli, &ev);

                if (g_sync) {
                    ev.data.fd = sender->get_fd();
                    epoll_ctl(epfd, EPOLL_CTL_ADD, sender->get_fd(), &ev);
                }

                std::cout << ansi::colored(
                    "[uds] 客户端已连接，TCP 已建立 → " +
                    g_host + ":" + std::to_string(g_port) + "\n",
                    ansi::fg::green);

            // ── UDS 客户端 → TCP（JSON 校验转发）─────────────────
            } else if (fd == uds_cli) {
                if (!relay_uds_to_tcp(uds_cli, sender)) {
                    std::cout << ansi::colored("[uds] 客户端断开，关闭 TCP\n",
                                               ansi::fg::yellow);
                    cleanup();
                }

            // ── TCP 服务器 → UDS 客户端（--sync，JSON 校验转发）──
            } else if (sender && fd == sender->get_fd()) {
                if (!relay_tcp_to_uds(sender->get_fd(), uds_cli)) {
                    std::cout << ansi::colored("[uds] TCP 服务器断开\n", ansi::fg::yellow);
                    cleanup();
                }
            }
        }
    }

    cleanup();
    close(uds_srv);
    close(epfd);
    unlink(g_uds_path.c_str());
    return 0;
}

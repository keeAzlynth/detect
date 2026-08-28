#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include "ansi_color.hpp"
#include "nlohmann/json.hpp"

static void set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static std::string peer_addr(int fd) {
    sockaddr_in addr{};
    socklen_t len = sizeof(addr);
    if (getpeername(fd, reinterpret_cast<sockaddr*>(&addr), &len) < 0)
        return "unknown";
    char ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &addr.sin_addr, ip, sizeof(ip));
    return std::string(ip) + ":" + std::to_string(ntohs(addr.sin_port));
}

int main(int argc, char* argv[]) {
    int port = 9000;
    if (argc == 2) port = std::atoi(argv[1]);

    int srv = ::socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return 1; }

    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(static_cast<uint16_t>(port));

    if (bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("bind"); return 1;
    }
    if (listen(srv, 16) < 0) {
        perror("listen"); return 1;
    }
    set_nonblocking(srv);

    std::cout << ansi::colored("[server] 监听端口 " + std::to_string(port) + "\n",
                               ansi::fg::bright_cyan);

    int epfd = epoll_create1(0);
    epoll_event ev{}, events[32];
    ev.events  = EPOLLIN;
    ev.data.fd = srv;
    epoll_ctl(epfd, EPOLL_CTL_ADD, srv, &ev);

    unsigned long counter = 0; // 递增数据戳

    while (true) {
        int n = epoll_wait(epfd, events, 32, -1);
        if (n < 0) { perror("epoll_wait"); break; }

        for (int i = 0; i < n; ++i) {
            int fd = events[i].data.fd;

            if (fd == srv) {
                // 新客户端连入
                int cli = accept(srv, nullptr, nullptr);
                if (cli < 0) continue;
                set_nonblocking(cli);
                ev.events  = EPOLLIN | EPOLLET;
                ev.data.fd = cli;
                epoll_ctl(epfd, EPOLL_CTL_ADD, cli, &ev);
                std::cout << ansi::colored(
                    "[+] 客户端连接  " + peer_addr(cli) + "\n",
                    ansi::fg::green);

            } else {
                // 客户端数据
                char buf[4096];
                ssize_t r = recv(fd, buf, sizeof(buf) - 1, 0);
                if (r <= 0) {
                    std::cout << ansi::colored(
                        "[-] 客户端断开  " + peer_addr(fd) + "\n",
                        ansi::fg::yellow);
                    epoll_ctl(epfd, EPOLL_CTL_DEL, fd, nullptr);
                    close(fd);
                } else {
                    buf[r] = '\0';
                    std::cout << ansi::colored("[recv] ", ansi::fg::cyan)
                              << peer_addr(fd) << "  "
                              << std::string(buf, static_cast<size_t>(r));
                    if (buf[r - 1] != '\n') std::cout << "\n";

                    // 回复 ok + 递增数据戳
                    // 封装成json格式返回
                    nlohmann::json reply_json;
                    reply_json["status"] = "ok";
                    reply_json["counter"] = ++counter;
                    std::string reply = reply_json.dump() + "\n";
                    ::send(fd, reply.c_str(), reply.size(), MSG_NOSIGNAL);
                }
            }
        }
    }

    close(srv);
    close(epfd);
    return 0;
}

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
#include <nlohmann/json.hpp>
#include "ansi_color.hpp"


class JsonSender {
private:
    std::string host_;
    int port_;
    int sock_fd_;
    int do_tcp_connect() {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) { perror("socket"); return -1; }

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port   = htons(static_cast<uint16_t>(port_));
        if (inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) <= 0) {
            std::cerr << ansi::colored("[error] 无效地址: " + host_ + "\n", ansi::fg::red);
            close(fd); return -1;
        }
        if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            perror("connect"); close(fd); return -1;
        }
        return fd;
    }
    void close_socket() {
        if (sock_fd_ >= 0) {
            close(sock_fd_);
            sock_fd_ = -1;
        }
    }
public:
    JsonSender(const std::string& host, int port) : 
        host_(host), port_(port), sock_fd_(-1) {
        sock_fd_ = do_tcp_connect();
        if (sock_fd_ < 0) {
            std::cerr << ansi::colored("[error] 无法连接到 " + host_ + ":" + std::to_string(port_) + "\n", ansi::fg::red);
        }
    }
    ~JsonSender() { close_socket(); }
    int get_fd() const { return sock_fd_; }
    void set_nonblocking() {
        int flags = fcntl(sock_fd_, F_GETFL, 0);
        if (flags >= 0) fcntl(sock_fd_, F_SETFL, flags | O_NONBLOCK);
    }
    bool send(const nlohmann::json& j) {
        if (sock_fd_ < 0) return false;
        std::string data = j.dump() + "\n";
        size_t total = 0;
        // 保证发完，否则最后的 \n 漏发导致接收区堆积
        while (total < data.size()) {
            ssize_t sent = ::send(sock_fd_, data.c_str() + total,
                                data.size() - total, MSG_NOSIGNAL);
            if (sent <= 0) {
                perror("send failed");
                return false;
            }
            total += sent;
        }
        return true;
    }
    bool send(const std::string& data) {
        if (sock_fd_ < 0) return false;
        try {
            auto parsed = nlohmann::json::parse(data);
            ssize_t sent = ::send(sock_fd_, parsed.dump().c_str(), parsed.dump().size(), MSG_NOSIGNAL);
            if (sent < 0) {
                perror("send failed");
                return false;
            }
            return true;
        } catch (const nlohmann::json::parse_error& e) {
            std::cerr << ansi::colored(std::string("[error] 非法JSON字符串: ") + e.what() + "\n", ansi::fg::red);
            return false;
        }
    }
    // 只负责提供发送，接受数据方面由外界自己获取fd epoll监听socket数据
};
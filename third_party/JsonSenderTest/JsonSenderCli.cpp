// JsonSenderCli.cpp
// CLI 测试工具：通过 JsonSender 发送 / 接收 JSON 数据
//
// 用法:
//   JsonSenderCli [--sync]
//   --sync  开启后将服务器返回数据格式化打印到 stdout
//
// 命令:
//   socket open <ip> <port>   连接到 TCP 服务器
//   socket send '<json>'      发送 JSON（严格校验）
//   socket close              断开连接
//   exit / quit               退出

#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <sys/epoll.h>
#include <unistd.h>
#include <linenoise.h>
#include <nlohmann/json.hpp>
#include "JsonSender.hpp"

static bool        g_sync    = false;
static JsonSender* g_sender  = nullptr;
static int         g_cli_epfd = -1;

// ── linenoise Tab 补全 ────────────────────────────────────────────
static void cli_completion(const char* buf, linenoiseCompletions* lc) {
    static const char* candidates[] = {
        "socket open ",
        "socket send ",
        "socket close",
        "help", "exit", "quit",
        nullptr
    };
    std::string input(buf);
    for (int i = 0; candidates[i]; ++i) {
        std::string c(candidates[i]);
        if (c.size() >= input.size() && c.substr(0, input.size()) == input)
            linenoiseAddCompletion(lc, candidates[i]);
    }
}

// ── linenoise 右侧提示 ────────────────────────────────────────────
static char* cli_hints(const char* buf, int* color, int* bold) {
    struct Entry { const char* cmd; const char* hint; int clr; };
    static const Entry table[] = {
        {"socket",       " open|send|close",    33},
        {"socket open",  " <ip> <port>",        33},
        {"socket send",  " '<json>'",           32},
        {"socket close", "",                     0},
        {"help",         "",                     0},
        {"exit",         "",                     0},
        {"quit",         "",                     0},
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

// ── 打印 JSON 数据（格式化或报错）────────────────────────────────
static void print_json_recv(const char* data, ssize_t len) {
    try {
        auto j = nlohmann::json::parse(data, data + len);
        std::cout << ansi::colored("[recv]\n", ansi::fg::cyan)
                  << j.dump(4) << "\n";
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << ansi::colored("[recv-warn] 服务器返回非 JSON 数据:\n", ansi::fg::yellow)
                  << "  原始内容: " << std::string(data, len) << "\n"
                  << ansi::colored("  解析错误: ", ansi::fg::red) << e.what() << "\n";
    }
}

// ── 处理服务器返回数据 ────────────────────────────────────────────
static void cli_handle_recv(linenoiseState* ls) {
    char buf[65536];
    ssize_t r = recv(g_sender->get_fd(), buf, sizeof(buf) - 1, 0);
    linenoiseHide(ls);
    if (r > 0) {
        buf[r] = '\0';
        print_json_recv(buf, r);
    } else {
        std::cout << ansi::colored("[warn] 服务器已关闭连接\n", ansi::fg::yellow);
        epoll_ctl(g_cli_epfd, EPOLL_CTL_DEL, g_sender->get_fd(), nullptr);
        delete g_sender;
        g_sender = nullptr;
    }
    linenoiseShow(ls);
}

// ── socket 命令实现 ───────────────────────────────────────────────
static void cmd_open(const std::string& host, int port) {
    if (g_sender) {
        std::cout << ansi::colored("[warn] 已有连接，请先 socket close\n", ansi::fg::yellow);
        return;
    }
    g_sender = new JsonSender(host, port);
    if (g_sender->get_fd() < 0) {
        delete g_sender;
        g_sender = nullptr;
        return;
    }
    g_sender->set_nonblocking();
    if (g_sync) {
        epoll_event ev{};
        ev.events  = EPOLLIN;
        ev.data.fd = g_sender->get_fd();
        epoll_ctl(g_cli_epfd, EPOLL_CTL_ADD, g_sender->get_fd(), &ev);
    }
    std::cout << ansi::colored("[ok] 已连接 " + host + ":" + std::to_string(port) + "\n",
                               ansi::fg::green);
}

static void cmd_send(const std::string& raw) {
    if (!g_sender) {
        std::cout << ansi::colored("[error] 未连接，请先 socket open <ip> <port>\n",
                                   ansi::fg::red);
        return;
    }
    try {
        auto j = nlohmann::json::parse(raw);
        if (g_sender->send(j)) {
            std::cout << ansi::colored("[sent] ", ansi::fg::green)
                      << j.dump(4) << "\n";
        }
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << ansi::colored("[error] 非法 JSON，发送取消:\n", ansi::fg::red)
                  << "  输入内容: " << raw << "\n"
                  << "  解析错误: " << e.what() << "\n";
    }
}

static void cmd_close() {
    if (!g_sender) {
        std::cout << ansi::colored("[warn] 当前没有连接\n", ansi::fg::yellow);
        return;
    }
    if (g_sync)
        epoll_ctl(g_cli_epfd, EPOLL_CTL_DEL, g_sender->get_fd(), nullptr);
    delete g_sender;
    g_sender = nullptr;
    std::cout << ansi::colored("[ok] 连接已关闭\n", ansi::fg::green);
}

// ── 命令分发 ──────────────────────────────────────────────────────
static bool dispatch_cli(const char* raw) {
    std::string line(raw);
    if (line.empty()) return true;
    linenoiseHistoryAdd(raw);

    std::istringstream ss(line);
    std::string tok;
    ss >> tok;

    if (tok == "exit" || tok == "quit") {
        cmd_close();
        return false;
    }
    if (tok == "help") {
        std::cout << "  socket open <ip> <port>   连接到 TCP 服务器\n"
                  << "  socket send '<json>'      发送 JSON 字符串（严格校验）\n"
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
            cmd_open(host, port);
    } else if (tok == "send") {
        std::string rest;
        std::getline(ss >> std::ws, rest);
        // 去掉首尾引号（单引号或双引号均支持）
        if (rest.size() >= 2 &&
            ((rest.front() == '"'  && rest.back() == '"') ||
             (rest.front() == '\'' && rest.back() == '\'')))
            rest = rest.substr(1, rest.size() - 2);
        cmd_send(rest);
    } else if (tok == "close") {
        cmd_close();
    } else {
        std::cout << ansi::colored("[error] 未知子命令: " + tok + "\n", ansi::fg::red);
    }
    return true;
}

// ── 入口 ──────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--sync") {
            g_sync = true;
        } else {
            std::cerr << "用法: " << argv[0] << " [--sync]\n";
            return 1;
        }
    }

    std::cout << ansi::styled("JsonSender CLI", ansi::fg::bright_green, ansi::style::bold)
              << "  --sync=" << (g_sync ? "ON（接收并打印服务器 JSON）" : "OFF") << "\n"
              << "  命令: socket open <ip> <port>"
                 " | socket send '<json>' | socket close | exit\n\n";

    const std::string history_file = std::string(getenv("HOME") ? getenv("HOME") : ".") + "/.jsonsender_cli_history";
    linenoiseHistorySetMaxLen(200);
    linenoiseHistoryLoad(history_file.c_str());
    linenoiseSetCompletionCallback(cli_completion);
    linenoiseSetHintsCallback(cli_hints);

    const std::string prompt = std::string(ansi::fg::cyan) + "json-cli> " + ansi::reset;

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

        bool got_line = false;
        while (!got_line) {
            int n = epoll_wait(g_cli_epfd, events, 4, -1);
            if (n < 0) { perror("epoll_wait"); break; }

            for (int i = 0; i < n; ++i) {
                int fd = events[i].data.fd;

                if (fd == STDIN_FILENO) {
                    char* line = linenoiseEditFeed(&ls);
                    if (line == linenoiseEditMore) continue;
                    linenoiseEditStop(&ls);
                    got_line = true;
                    if (line == nullptr) {
                        std::cout << "\n";
                        cmd_close();
                        running = false;
                    } else {
                        running = dispatch_cli(line);
                        linenoiseFree(line);
                    }
                    break;
                } else if (g_sync && g_sender && fd == g_sender->get_fd()) {
                    cli_handle_recv(&ls);
                }
            }
        }
    }

    linenoiseHistorySave(history_file.c_str());
    close(g_cli_epfd);
    return 0;
}

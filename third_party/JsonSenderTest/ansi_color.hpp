#pragma once
#include <string>

// ANSI 转义序列格式：\x1b[<code>m
// 兼容 C++11：使用 constexpr const char* 代替 inline 变量
//
// 使用示例：
//   std::cout << ansi::fg::red << "错误信息" << ansi::reset << std::endl;
//   std::string prompt = ansi::colored("输入>", ansi::fg::cyan);

namespace ansi {

// ── 重置 ──────────────────────────────────────────────────
constexpr const char* reset      = "\x1b[0m";

// ── 文字样式 ──────────────────────────────────────────────
namespace style {
    constexpr const char* bold      = "\x1b[1m";
    constexpr const char* dim       = "\x1b[2m";
    constexpr const char* underline = "\x1b[4m";
    constexpr const char* blink     = "\x1b[5m";
    constexpr const char* reverse   = "\x1b[7m";
} // namespace style

// ── 前景色（文字颜色）────────────────────────────────────
namespace fg {
    constexpr const char* black          = "\x1b[30m";
    constexpr const char* red            = "\x1b[31m";
    constexpr const char* green          = "\x1b[32m";
    constexpr const char* yellow         = "\x1b[33m";
    constexpr const char* blue           = "\x1b[34m";
    constexpr const char* magenta        = "\x1b[35m";
    constexpr const char* cyan           = "\x1b[36m";
    constexpr const char* white          = "\x1b[37m";

    constexpr const char* bright_black   = "\x1b[90m";
    constexpr const char* bright_red     = "\x1b[91m";
    constexpr const char* bright_green   = "\x1b[92m";
    constexpr const char* bright_yellow  = "\x1b[93m";
    constexpr const char* bright_blue    = "\x1b[94m";
    constexpr const char* bright_magenta = "\x1b[95m";
    constexpr const char* bright_cyan    = "\x1b[96m";
    constexpr const char* bright_white   = "\x1b[97m";
} // namespace fg

// ── 背景色 ────────────────────────────────────────────────
namespace bg {
    constexpr const char* black          = "\x1b[40m";
    constexpr const char* red            = "\x1b[41m";
    constexpr const char* green          = "\x1b[42m";
    constexpr const char* yellow         = "\x1b[43m";
    constexpr const char* blue           = "\x1b[44m";
    constexpr const char* magenta        = "\x1b[45m";
    constexpr const char* cyan           = "\x1b[46m";
    constexpr const char* white          = "\x1b[47m";

    constexpr const char* bright_black   = "\x1b[100m";
    constexpr const char* bright_red     = "\x1b[101m";
    constexpr const char* bright_green   = "\x1b[102m";
    constexpr const char* bright_yellow  = "\x1b[103m";
    constexpr const char* bright_blue    = "\x1b[104m";
    constexpr const char* bright_magenta = "\x1b[105m";
    constexpr const char* bright_cyan    = "\x1b[106m";
    constexpr const char* bright_white   = "\x1b[107m";
} // namespace bg

// ── 便捷函数：给文字着色后自动追加 reset ─────────────────
inline std::string colored(const std::string& text, const char* color) {
    return std::string(color) + text + reset;
}

// ── 便捷函数：同时设置前景 + 样式 ────────────────────────
inline std::string styled(const std::string& text,
                          const char* color,
                          const char* st) {
    return std::string(st) + color + text + reset;
}

} // namespace ansi

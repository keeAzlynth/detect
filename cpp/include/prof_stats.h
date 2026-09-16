#pragma once
// 轻量热点统计探针（编译期开关，默认零开销）
//
// 目的：把每个流水线阶段的耗时按帧累加，程序正常退出时打印
//       avg / max / n / 占单帧墙钟比例，用来定位 hot path。
//
// 启用：cmake -DENABLE_PROF_STATS=ON ..
// 关闭时 PROF_* 宏全部展开为空，调用点不产生任何代码与运行时开销
// （存储实现仍会编进 libutils，但无人调用）。
//
// ⚠️ 存储与打印的实现必须放在 .cpp（libutils/prof_stats.cpp）里，不能用
//    header 内的函数局部 static：main 与 libcore.so 会各自拿到一份独立的表，
//    libcore 里记录的阶段数据永远打不出来。

#include <chrono>
#include <map>
#include <string>

namespace prof {

struct Accum {
    double sum_ms = 0.0;
    double max_ms = 0.0;
    long   n      = 0;

    void add(double ms) {
        sum_ms += ms;
        if (ms > max_ms) {
            max_ms = ms;
        }
        ++n;
    }
};

// 全局阶段表（定义在 libutils，跨 SO 唯一）
std::map<std::string, Accum> & table();

// 打印汇总（stderr）
void dump();

#ifdef ENABLE_PROF_STATS

// RAII：构造时打点，析构时把耗时累加进对应阶段
struct Scope {
    const char *                          name;
    std::chrono::steady_clock::time_point t0;

    explicit Scope(const char * n) :
        name(n),
        t0(std::chrono::steady_clock::now()) {}

    ~Scope() {
        table()[name].add(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count());
    }
};

// 帧计数（值为 0，只看 n）
inline void markFrame() { table()["*frames"].add(0.0); }

#define PROF_CAT2(a, b) a##b
#define PROF_CAT(a, b) PROF_CAT2(a, b)
#define PROF_SCOPE(name) ::prof::Scope PROF_CAT(prof_scope_, __LINE__)(name)
#define PROF_FRAME() ::prof::markFrame()
#define PROF_DUMP() ::prof::dump()

#else

#define PROF_SCOPE(name) ((void) 0)
#define PROF_FRAME() ((void) 0)
#define PROF_DUMP() ((void) 0)

#endif  // ENABLE_PROF_STATS

}  // namespace prof

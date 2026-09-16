#include "prof_stats.h"

#include <cstdio>

namespace prof {

std::map<std::string, Accum> & table() {
    // 单一定义点：链接进 libutils，main / libcore 共享同一份表
    static std::map<std::string, Accum> t;
    return t;
}

void dump() {
    auto & t      = table();
    long   frames = 0;
    if (auto it = t.find("*frames"); it != t.end()) {
        frames = it->second.n;
    }

    // 单帧墙钟由主循环用 PROF_SCOPE("00.wall/frame") 包住整轮迭代累加
    double wall_sum = 0.0;
    if (auto it = t.find("00.wall/frame"); it != t.end()) {
        wall_sum = it->second.sum_ms;
    }

    fprintf(stderr, "\n========== PROF STATS (frames=%ld) ==========\n", frames);
    fprintf(stderr, "%-28s %10s %11s %8s %10s\n", "phase", "avg(ms)", "max(ms)", "n", "wall %");
    for (const auto & kv : t) {
        if (kv.first == "*frames") {
            continue;
        }
        const double avg = kv.second.n ? kv.second.sum_ms / kv.second.n : 0.0;
        const double pct = wall_sum > 0.0 ? kv.second.sum_ms / wall_sum * 100.0 : 0.0;
        fprintf(stderr, "%-28s %10.3f %11.3f %8ld %9.2f%%\n", kv.first.c_str(), avg,
                kv.second.max_ms, kv.second.n, pct);
    }
    if (frames > 0 && wall_sum > 0.0) {
        fprintf(stderr, "%-28s %10.3f  -> %.3f fps end-to-end (wall clock)\n", "== 单帧墙钟合计",
                wall_sum / frames, frames * 1000.0 / wall_sum);
    }
    fprintf(stderr, "==============================================\n");
    fflush(stderr);
}

}  // namespace prof

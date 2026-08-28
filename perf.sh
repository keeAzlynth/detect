#!/bin/bash
# ============================================================
#  perf.sh — Jetson Nano 性能模式 / 内核调优
#
#  用法：
#    ./perf.sh            # 应用所有性能优化（需要 sudo，会提示输入密码）
#    sudo ./perf.sh       # 已 root 直接跑
#    ./perf.sh --dry-run  # 只打印当前状态，不改任何东西
#    ./perf.sh --reset    # 恢复默认（schedutil 调度器 + 默认电源模式）
# ============================================================
set -u

DRY=0
RESET=0
case "${1:-}" in
    --dry-run) DRY=1 ;;
    --reset)   RESET=1 ;;
esac

# 若没有 root，自动以 root 重跑
if [ "$(id -u)" -ne 0 ] && [ "$DRY" -eq 0 ]; then
    echo "[*] 需要 root 权限，自动提升 sudo ..."
    exec sudo bash "$0" "$@"
fi

say()  { echo -e "\033[1;36m[*]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[+]\033[0m $*"; }
warn() { echo -e "\033[1;33m[!]\033[0m $*"; }

# 安全执行：非 dry-run 才真正执行
run() {
    if [ "$DRY" -eq 1 ]; then
        echo -e "\033[90m[dry-run] $*\033[0m"
        return 0
    fi
    "$@"
}

echo "============================================"
echo "  Jetson Nano 性能模式 / 内核调优"
echo "============================================"

# ---------- 1. 电源模式 -> MAXN（Tegra X1 最高功耗档 10W） ----------
if command -v nvpmodel > /dev/null; then
    MODE="0"
    if [ "$RESET" -eq 1 ]; then
        MODE="1"
    fi
    say "nvpmodel -> 模式 ${MODE} ($([ "$RESET" -eq 1 ] && echo '5W 省电' || echo 'MAXN 满血'))"
    run nvpmodel -m "$MODE" > /dev/null 2>&1 && ok "电源模式已设置" || warn "nvpmodel 设置失败(可能已是最新模式)"
    # nvpmodel -m 会重置 EMC/GPU 频率上限，必须先于 jetson_clocks
    sleep 1
else
    warn "未找到 nvpmodel，跳过电源模式"
fi

# ---------- 2. 锁定 CPU/GPU/EMC 最高频率 ----------
if command -v jetson_clocks > /dev/null; then
    say "jetson_clocks 锁频（CPU/GPU/EMC @ max）"
    if [ "$RESET" -eq 1 ]; then
        run jetson_clocks --restore > /dev/null 2>&1 && ok "已恢复默认频率" || warn "恢复失败"
    else
        run jetson_clocks > /dev/null 2>&1 && ok "频率已锁定最高" || warn "jetson_clocks 执行失败"
    fi
else
    warn "未找到 jetson_clocks，跳过锁频"
fi

# ---------- 3. CPU 调度器 -> performance ----------
GOV="performance"
if [ "$RESET" -eq 1 ]; then
    GOV="schedutil"
fi
say "CPU governor -> ${GOV}（写入后回读校验）"
GOV_OK=0
for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
    [ -e "$f" ] || continue
    if [ "$DRY" -eq 1 ]; then
        echo -e "\033[90m[dry-run] echo $GOV > $f\033[0m"
        GOV_OK=1
    elif [ -w "$f" ]; then
        if sh -c "echo $GOV > $f" 2>/dev/null && grep -q "$GOV" "$f"; then
            ok "  ${f} -> ${GOV}"
            GOV_OK=1
        else
            warn "  ${f}: 无法写入 ${GOV}，保持当前 $(cat "$f" 2>/dev/null)"
        fi
    else
        warn "  ${f}: 不可写"
    fi
done
[ "$GOV_OK" -eq 0 ] && warn "governor ${GOV} 不可用（内核未编译该 governor）"

if [ "$RESET" -eq 1 ]; then
    exit 0   # 恢复默认到这就够了
fi

# ---------- 4. VM 调优：减少 swap 依赖，提升实时性 ----------
say "VM 参数调优"
run sh -c 'echo 10 > /proc/sys/vm/swappiness'            && ok "  vm.swappiness = 10"  || warn "swappiness 设置失败"
# 有 1.9GB swap 兜底，不强制关闭 swap

# ---------- 5. （可选）禁用无关服务，释放 CPU/内存 ----------
# 默认不执行，按需取消注释：
# say "禁用无关服务（桌面包/蓝牙/avahi）"
# systemctl --quiet stop gdm lightdm bluetooth avahi-daemon 2>/dev/null
# systemctl --quiet disable bluetooth avahi-daemon 2>/dev/null

# ---------- 6. 状态快照 ----------
echo ""
echo "================= 优化后状态 ================="
[ "$DRY" -eq 1 ] && echo "(dry-run: 预览) "

if command -v nvpmodel > /dev/null; then
    echo "* nvpmodel: $(nvpmodel -q 2>/dev/null | head -1)"
else
    warn "nvpmodel 不可用"
fi
echo "* governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
echo "* cpu 频率:"
for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq; do
    hz=$(cat "$c" 2>/dev/null)
    echo "    $c: $((hz / 1000)) MHz"
done
echo "* GPU/VIC 频率(devfreq):"
for d in /sys/class/devfreq/*; do
    n=$(basename "$d")
    r=$(cat "$d/cur_freq" 2>/dev/null)
    [ -n "$r" ] && echo "    $n: $((r / 1000000)) MHz"
done
echo "* swappiness: $(cat /proc/sys/vm/swappiness 2>/dev/null)"
echo "* 内存: $(free -h | awk '/Mem:/{print $3"/"$2}')  可用"
echo ""
echo "================ 跑项目建议 ================"
echo "  cd ~/depth-detect-turbo/bin && sudo nice -n -10 ./main 0 config.yaml"
echo "============================================"
exit 0
#!/usr/bin/env bash
# ============================================================
#  build.sh — 内存感知构建（防 4GB Jetson OOM thrash 卡死）
#  用法: ./build.sh [--jobs N] [--clean]
# ============================================================
set -uo pipefail

cd "$(dirname "$0")"

JOBS="auto"
CLEAN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        *)
            echo "Unknown argument: $1 (--jobs N / --clean)"
            exit 1
            ;;
    esac
done

# ---- 检测 main 是否正在运行（会占用 ~2.5GB 内存，需先停掉）----
if pgrep -f "bin/main" >/dev/null 2>&1; then
    echo "[!] 检测到 bin/main 正在运行（约占用 2.5GB 内存），继续构建极易 OOM 卡死。"
    read -r -p "    是否强制终止 main 并继续构建？[y/N] " ans
    if [[ "$ans" == "y" || "$ans" == "Y" ]]; then
        pkill -f "bin/main"
        sleep 2
    else
        echo "    已取消构建。请先停止 main 再执行 ./build.sh"
        exit 1
    fi
fi

# ---- 内存感知并行度 ----
if [[ "$JOBS" == "auto" ]]; then
    avail_mb=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
    if (( avail_mb < 1800 )); then
        JOBS=1
    elif (( avail_mb < 3000 )); then
        JOBS=2
    else
        JOBS=4
    fi
fi
echo "[*] 可用内存: ${avail_mb}MB, 并行度: -j${JOBS}"

# ---- 限制 cc1plus 虚拟内存，防止单进程 OOM（阈值 2.5GB）----
ulimit -v 2621440 2>/dev/null || true

mkdir -p build
cd build

if [[ "$CLEAN" == true ]]; then
    echo "[*] 清理构建..."
    make clean >/dev/null 2>&1 || true
fi

cmake .. && make -j"$JOBS"
echo ""
echo "============================================"
echo "  ✓ 构建完成: ./bin/main 0 config.yaml"
echo "============================================"

#!/bin/bash
# 把新构建的深度引擎接进 app，跑端到端 A/B（现役引擎 vs 新引擎），跑完自动还原。
#
# 用法:
#   bash scripts/verify_new_engine.sh <新引擎文件路径> [视频文件] [二进制目录]
#
# 说明:
#   - 会临时替换 <repo>/model/engine/lite-mono-tiny/lite-mono-tiny_192x640_op11_int8_trt8.2.engine，
#     结束后从备份自动还原（trap EXIT），备份默认放 <repo>/../engine_backup/
#   - 第 3 个参数是「A/B 隔离用的独立二进制目录」，里面要有 main + lib*.so + config.yaml。
#     不传则用 <repo>/bin（**注意这会直接跑构建目录，不算隔离**）。
#   - 环境变量可覆盖：VIDEO / LOCKCLK
#
# ⚠️ 两个必须记住的坑:
#   1) 换目录跑 main 必须 export LD_LIBRARY_PATH —— main 的 RUNPATH 是指向 bin/ 的**绝对路径**，
#      不顶置库目录会静默加载另一份 libcore/libtools（或报 undefined symbol）。
#   2) 跑测前先锁频（LOCKCLK 指向 lockclk.sh），否则 ARM DVFS 会让 A/B 结果漂移 ±15%。
set -u

REPO=$(cd "$(dirname "$0")/.." && pwd)
NEW=${1:?用法: bash scripts/verify_new_engine.sh <新引擎> [视频] [二进制目录]}
VIDEO=${2:-${VIDEO:-$REPO/data/test300.mp4}}
BIN=${3:-${BIN:-$REPO/bin}}
LOCKCLK=${LOCKCLK:-$REPO/../prof/lockclk.sh}

ENG=$REPO/model/engine/lite-mono-tiny
CUR=$ENG/lite-mono-tiny_192x640_op11_int8_trt8.2.engine
BK=$(cd "$REPO/.." && pwd)/engine_backup
OUT=$(cd "$REPO/.." && pwd)/prof_out
TRT=${TRT:-/usr/src/tensorrt/bin/trtexec}

[ -f "$NEW" ] || { echo "!! 新引擎不存在: $NEW"; exit 1; }
[ -f "$CUR" ] || { echo "!! 现役引擎不存在: $CUR"; exit 1; }
[ -f "$VIDEO" ] && echo "[env] 素材: $VIDEO" || echo "!! 素材不存在: $VIDEO"
mkdir -p "$BK" "$OUT"

[ -f "$LOCKCLK" ] && { echo "=== 锁频 ==="; bash "$LOCKCLK"; }

echo "=== 备份现役引擎 ==="
if [ ! -f "$BK/orig_int8.engine" ]; then
    cp -a "$CUR" "$BK/orig_int8.engine" && echo "  已备份 -> $BK/orig_int8.engine"
else
    echo "  备份已存在，跳过（如需重取请先删除 $BK/orig_int8.engine）"
fi
ls -la "$BK/"

restore() {
    echo
    echo "=== 还原现役引擎 ==="
    cp -a "$BK/orig_int8.engine" "$CUR" && echo "  已还原"
}
trap restore EXIT

# ---- trtexec 单测对比（同会话，规避跨会话时钟漂移）----
echo
echo "=== trtexec 单测（同会话）==="
echo "--- 现役 INT8 ---"
"$TRT" --loadEngine="$BK/orig_int8.engine" --iterations=50 --warmUp=500 --avgRuns=20 2>&1 \
    | grep -E 'Throughput|GPU Compute Time' | sed 's/^/  /'
echo "--- 新引擎 ---"
"$TRT" --loadEngine="$NEW" --iterations=50 --warmUp=500 --avgRuns=20 2>&1 \
    | grep -E 'Throughput|GPU Compute Time' | sed 's/^/  /'

# ---- 接进 app 跑端到端 ----
echo
echo "=== 端到端 A/B（300 帧，生产配置；注意 log_level 必须是 info 才有 fps 读数）==="
cd "$REPO/bin" || exit 1
export LD_LIBRARY_PATH="$BIN"
echo "[env] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

run_app() {
    tag=$1
    rm -rf out_dir && mkdir -p out_dir
    "$BIN/main" "$VIDEO" "$REPO/bin/config.yaml" > "$OUT/ve_$tag.log" 2>&1
    echo "  --- $tag ---"
    grep -E 'Processing frame' "$OUT/ve_$tag.log" | sed 's/^/    /'
    grep -E '\[Infer Pipeline\]|wall/frame|infer_pipeline' "$OUT/ve_$tag.log" | sed 's/^/    /'
}

echo "### 现役引擎 ###"
cp -a "$BK/orig_int8.engine" "$CUR"
run_app orig

echo "### 新引擎 ###"
cp -a "$NEW" "$CUR"
run_app new

echo
echo "=== 产物校验 ==="
ls out_dir | wc -l
python3 - "$REPO/bin/out_dir" <<'PY'
import glob, os, sys, cv2
files = sorted(glob.glob(os.path.join(sys.argv[1], '*.jpg')))
print('jpg count:', len(files))
for f in files[:2] + files[-2:]:
    im = cv2.imread(f)
    print(' ', os.path.basename(f), None if im is None else im.shape)
PY
echo "提示: 精度体检请另跑 scripts/compare_engines.py（对比两引擎在相同输入下的输出 MAE）"

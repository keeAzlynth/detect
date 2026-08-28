#!/usr/bin/env bash
set -e

BUILD_DIR="build"
CMAKE_ARGS=""

usage() {
    echo "用法: $0 [选项]"
    echo "  --cli      启用 JsonSenderCli"
    echo "  --uds      启用 JsonSenderUds"
    echo "  --clean    清理后重新构建"
    echo "  --release  使用 Release 模式（默认 Debug）"
    echo "  -j N       并行编译线程数（默认 nproc）"
    echo "  --help     显示此帮助"
    exit 0
}

BUILD_TYPE="Debug"
CLEAN=0
JOBS=$(nproc)
OPT_CLI=OFF
OPT_UDS=OFF

while [[ $# -gt 0 ]]; do
    case $1 in
        --cli)     OPT_CLI=ON ;;
        --uds)     OPT_UDS=ON ;;
        --clean)   CLEAN=1 ;;
        --release) BUILD_TYPE="Release" ;;
        -j)        JOBS="$2"; shift ;;
        --help)    usage ;;
        *)         echo "未知参数: $1"; usage ;;
    esac
    shift
done

CMAKE_ARGS="-DBUILD_JSONSENDER_CLI=$OPT_CLI -DBUILD_JSONSENDER_UDS=$OPT_UDS"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $CLEAN -eq 1 && -d "$BUILD_DIR" ]]; then
    echo "[build] 清理 $BUILD_DIR ..."
    rm -rf "$BUILD_DIR"
fi

echo "[build] cmake 配置  BUILD_TYPE=$BUILD_TYPE $CMAKE_ARGS"
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" $CMAKE_ARGS

echo "[build] 开始编译  -j$JOBS"
cmake --build "$BUILD_DIR" -- -j"$JOBS"

echo "[build] 完成，产物在 $BUILD_DIR/"

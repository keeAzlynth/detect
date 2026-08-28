cmake_minimum_required(VERSION 3.10)

# ============================================================
#  depth_detect_turbo — Jetson Nano 深度检测（YOLOv8n + Lite-Mono）
#  重构版：默认 Release，cmake .. && cmake --build . 即可编译
# ============================================================

# ---- 自动定位 nvcc（让裸 `cmake ..` 即可用，无需手动指定编译器）----
find_program(_TURBO_NVCC nvcc HINTS /usr/local/cuda/bin /usr/bin)
if(_TURBO_NVCC AND NOT DEFINED CUDACXX)
    set(ENV{CUDACXX} "${_TURBO_NVCC}")
endif()

project(depth_detect_turbo VERSION 1.0.0 LANGUAGES CXX CUDA)

# ==================== 默认构建类型 ====================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# Debug 下关闭掉优化，其余保持用户指定
message(STATUS "Project: depth_detect_turbo (${CMAKE_BUILD_TYPE})")

# ==================== 语言标准 ====================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ==================== 平台/架构检测 ====================
set(TARGET_PLATFORM "AUTO" CACHE STRING "Build platform (AUTO/NANO/TX2/X86_64)")
set(TARGET_CUDA_ARCHS "" CACHE STRING "CUDA arch e.g. 53 / 62 / 75")

if(TARGET_PLATFORM STREQUAL "AUTO")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        file(READ "/proc/device-tree/model" _MODEL_RAW)
        if(_MODEL_RAW MATCHES "TX2|T186|Quill")
            set(TARGET_PLATFORM "TX2")
        else()
            set(TARGET_PLATFORM "NANO")
        endif()
    else()
        set(TARGET_PLATFORM "X86_64")
    endif()
endif()
message(STATUS "TARGET_PLATFORM = ${TARGET_PLATFORM}")

# ---- CUDA 架构（用 CMake 原生属性，避免 gencode/-maxrregcount 的坑）----
if(NOT TARGET_CUDA_ARCHS)
    if(TARGET_PLATFORM STREQUAL "TX2")
        set(TARGET_CUDA_ARCHS 62)
    elseif(TARGET_PLATFORM STREQUAL "NANO")
        set(TARGET_CUDA_ARCHS 53)
    else()
        set(TARGET_CUDA_ARCHS 75)
    endif()
endif()
set(CMAKE_CUDA_ARCHITECTURES ${TARGET_CUDA_ARCHS})
message(STATUS "CUDA arch = ${TARGET_CUDA_ARCHS}")

set(IS_JETSON OFF)
if(TARGET_PLATFORM STREQUAL "TX2" OR TARGET_PLATFORM STREQUAL "NANO")
    set(IS_JETSON ON)
endif()

# ==================== 编译优化 ====================
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fomit-frame-pointer -ftree-vectorize")
    if(IS_JETSON)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=cortex-a57")
    endif()
    # 函数/数据节分离 + 链接时收割未用段，减小体积
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -DNDEBUG --expt-relaxed-constexpr")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections -s")
    # LTO: 链接时优化，允许跨编译单元内联
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -flto")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
add_definitions(-DAPI_EXPORTS)

# ==================== 可选特性开关 ====================
option(ENABLE_TIMER "Collect per-stage timing (main.cpp summary)" ON)
if(ENABLE_TIMER)
    add_definitions(-DENABLE_TIMER)
endif()
option(ENABLE_JESTON_MEM_MANAGED "Use unified memory on Jetson (experimental)" OFF)
if(ENABLE_JESTON_MEM_MANAGED)
    add_definitions(-DENABLE_JESTON_MEM_MANAGED)
endif()

# ==================== 依赖 ====================
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# TensorRT
set(TENSORRT_HINT_DIRS
    /usr/local/TensorRT
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib
)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_HINT_DIRS} PATH_SUFFIXES include)
find_library(TENSORRT_NVINFER_LIBRARY NAMES nvinfer HINTS ${TENSORRT_HINT_DIRS} PATH_SUFFIXES lib lib64)
find_library(TENSORRT_NVONNXPARSER_LIBRARY NAMES nvonnxparser HINTS ${TENSORRT_HINT_DIRS} PATH_SUFFIXES lib lib64)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIBRARY OR NOT TENSORRT_NVONNXPARSER_LIBRARY)
    message(FATAL_ERROR "TensorRT not found. Check TENSORRT_HINT_DIRS or install TensorRT.")
endif()

# CUDA Runtime
find_library(CUDA_CUDART_LIBRARY NAMES cudart HINTS /usr/local/cuda PATH_SUFFIXES lib lib64)
if(NOT CUDA_CUDART_LIBRARY)
    message(FATAL_ERROR "libcudart not found. Ensure CUDA toolkit is installed.")
endif()

# Eigen3
find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core PATHS /usr/include/eigen3)
if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen3 not found, please install: sudo apt install libeigen3-dev")
endif()

# spdlog + onnxruntime + JsonSender（子模块/内置第三方库）
set(ONNXRUNTIME_ROOT ${PROJECT_SOURCE_DIR}/third_party/onnxruntime)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/spdlog)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# yaml-cpp target 兼容
if(TARGET yaml-cpp::yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
else()
    set(YAML_CPP_TARGET yaml-cpp)
endif()
list(APPEND YAML_CPP_TARGET dl)

# ==================== 公共包含 ====================
set(PROJECT_CPP_DIR ${PROJECT_SOURCE_DIR}/cpp)
function(configure_common_includes target)
    target_include_directories(${target} PRIVATE
        ${PROJECT_CPP_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${ONNXRUNTIME_ROOT}/include
    )
endfunction()

# ==================== 模块 <-- 依赖方向 ====================
# utils -> spdlog / yaml-cpp
file(GLOB_RECURSE UTILS_SRCS ${PROJECT_CPP_DIR}/utils/src/*.cpp)
add_library(utils SHARED ${UTILS_SRCS})
configure_common_includes(utils)
target_include_directories(utils PUBLIC ${PROJECT_CPP_DIR}/utils/include)
target_link_libraries(utils PUBLIC spdlog::spdlog ${YAML_CPP_TARGET})

# bytetrack -> utils / opencv
file(GLOB_RECURSE BYTETRACK_SRCS ${PROJECT_CPP_DIR}/bytetrack/src/*.cpp)
add_library(bytetrack SHARED ${BYTETRACK_SRCS})
configure_common_includes(bytetrack)
target_include_directories(bytetrack PUBLIC ${PROJECT_CPP_DIR}/bytetrack/include)
target_link_libraries(bytetrack PRIVATE utils ${OpenCV_LIBS})

# op_kernel (CUDA kernels) -> cudart / tensorrt / opencv
file(GLOB_RECURSE OP_KERNEL_SRCS ${PROJECT_CPP_DIR}/op_kernel/src/*.cu)
add_library(op_kernel SHARED ${OP_KERNEL_SRCS})
configure_common_includes(op_kernel)
target_include_directories(op_kernel PUBLIC ${PROJECT_CPP_DIR}/op_kernel/include)
if(IS_JETSON)
    target_compile_definitions(op_kernel PRIVATE ENABLE_FP16=1)
endif()
target_link_libraries(op_kernel PRIVATE
    ${CUDA_CUDART_LIBRARY}
    ${TENSORRT_NVINFER_LIBRARY}
    ${TENSORRT_NVONNXPARSER_LIBRARY}
    ${OpenCV_LIBS}
)

# inference -> op_kernel / utils / onnxruntime / tensorrt
file(GLOB_RECURSE INFERENCE_SRCS
    ${PROJECT_CPP_DIR}/inference/infer_models/src/*.cpp
    ${PROJECT_CPP_DIR}/inference/infer_backend/src/*.cpp
)
add_library(inference SHARED ${INFERENCE_SRCS})
configure_common_includes(inference)
target_include_directories(inference PUBLIC
    ${PROJECT_CPP_DIR}/inference/infer_models/include
    ${PROJECT_CPP_DIR}/inference/infer_backend/include
)
target_link_libraries(inference PRIVATE
    op_kernel
    utils
    ${CUDA_CUDART_LIBRARY}
    ${TENSORRT_NVINFER_LIBRARY}
    ${OpenCV_LIBS}
    onnxruntime
)

# JsonSender（TCP 报警上报）
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/JsonSenderTest)

# core -> inference / bytetrack / JsonSender  （主流水线）
file(GLOB_RECURSE CORE_SRCS ${PROJECT_CPP_DIR}/core/src/*.cpp)
add_library(core SHARED ${CORE_SRCS})
configure_common_includes(core)
target_include_directories(core PUBLIC
    ${PROJECT_CPP_DIR}/core/include
    ${PROJECT_CPP_DIR}/bytetrack/include
)
target_link_libraries(core PUBLIC
    inference
    bytetrack
    utils
    ${OpenCV_LIBS}
    JsonSender
)

# tools -> utils / core / bytetrack  （绘制/FPS 等）
file(GLOB_RECURSE TOOL_SRCS ${PROJECT_CPP_DIR}/tools/src/*.cpp)
add_library(tools SHARED ${TOOL_SRCS})
configure_common_includes(tools)
target_include_directories(tools PUBLIC
    ${PROJECT_CPP_DIR}/tools/include
    ${PROJECT_CPP_DIR}/core/include
    ${PROJECT_CPP_DIR}/bytetrack/include
)
target_link_libraries(tools PRIVATE utils ${OpenCV_LIBS} core)

# ==================== 主程序 ====================
add_executable(main ${PROJECT_CPP_DIR}/main.cpp)
configure_common_includes(main)
target_link_libraries(main PRIVATE core bytetrack tools utils ${CUDA_CUDART_LIBRARY})

message(STATUS "=== depth_detect_turbo build ready: ./bin/main <video|[camera_id]> <config.yaml> ===")
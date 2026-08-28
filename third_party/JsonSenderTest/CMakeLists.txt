cmake_minimum_required(VERSION 3.10)
project(JsonSenderTest)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── 基础库 ────────────────────────────────────────────────────────

# linenoise（纯 C 库）
add_library(linenoise STATIC
    lib/linenoise/linenoise.c
)
target_include_directories(linenoise PUBLIC lib/linenoise)

# nlohmann/json（header-only）
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE lib/json/include)

# JsonSender（header-only，作为子仓库被引用时的核心目标）
add_library(JsonSender INTERFACE)
target_include_directories(JsonSender INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(JsonSender INTERFACE nlohmann_json linenoise)

# ── 开发 / 测试用可执行文件（本仓库内部使用）────────────────────

# 原 main.cpp 测试程序
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        linenoise
        nlohmann_json
)

# 简单 TCP echo 服务器（用于手动测试）
add_executable(server_test server_test.cpp)
target_include_directories(server_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(server_test PRIVATE nlohmann_json linenoise)

# ── 可选目标（作为子仓库时按需开启）─────────────────────────────

# JsonSender CLI 测试工具
# 启用方式: cmake -DBUILD_JSONSENDER_CLI=ON ..
option(BUILD_JSONSENDER_CLI "构建 JsonSender CLI 测试工具 (JsonSenderCli)" OFF)
if(BUILD_JSONSENDER_CLI)
    add_executable(JsonSenderCli JsonSenderCli.cpp)
    target_link_libraries(JsonSenderCli PRIVATE JsonSender linenoise)
    message(STATUS "[JsonSender] 已启用 JsonSenderCli")
endif()

# JsonSender UDS 代理服务器
# 启用方式: cmake -DBUILD_JSONSENDER_UDS=ON ..
option(BUILD_JSONSENDER_UDS "构建 JsonSender UDS 代理服务器 (JsonSenderUds)" OFF)
if(BUILD_JSONSENDER_UDS)
    add_executable(JsonSenderUds JsonSenderUds.cpp)
    target_link_libraries(JsonSenderUds PRIVATE JsonSender)
    message(STATUS "[JsonSender] 已启用 JsonSenderUds")
endif()

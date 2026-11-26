# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(TensorApiHeaderChecker)

if(NOT ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
endif()
set(INCLUDE_PATHS ${ASCENDC_DIR}/include ${ASCENDC_DIR})
if(ASCEND_CANN_PACKAGE_PATH)
    list(APPEND INCLUDE_PATHS "${ASCEND_CANN_PACKAGE_PATH}/asc/include"
        "${ASCEND_CANN_PACKAGE_PATH}/asc" "${ASCEND_CANN_PACKAGE_PATH}/include")
endif()

if (TARGET tikcfw_headers)
    message(STATUS "tikcfw_headers target exists, gathering its include dirs")
    get_target_property(TIKCFW_HEADERS_INCLUDE_DIRS tikcfw_headers INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "tikcfw_headers include dirs: ${TIKCFW_HEADERS_INCLUDE_DIRS}")
    if (TIKCFW_HEADERS_INCLUDE_DIRS)
        foreach(DIR ${TIKCFW_HEADERS_INCLUDE_DIRS})
            if(DIR MATCHES "^\\$<INSTALL_INTERFACE:.*>$")
                continue()
            endif()
            if(DIR MATCHES "^\\$<BUILD_INTERFACE:(.*)>$")
                list(APPEND INCLUDE_PATHS "${CMAKE_MATCH_1}")
            else()
                list(APPEND INCLUDE_PATHS "${DIR}")
            endif()
        endforeach()
    endif()
else()
    message(WARNING "tikcfw_headers target does not exist")
endif()

find_package(tikicpulib REQUIRED)
if(tikicpulib_FOUND)
    message(STATUS "tikicpulib found, adding include directories")
    message(STATUS "tikicpulib cmake path: ${tikicpulib_DIR}")
    get_target_property(TIKICPULIB_INCLUDE_DIRS tikicpulib_ascend910 INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "tikicpulib include dirs: ${TIKICPULIB_INCLUDE_DIRS}")
    if(TIKICPULIB_INCLUDE_DIRS)
        list(APPEND INCLUDE_PATHS ${TIKICPULIB_INCLUDE_DIRS})
    endif()
endif()

set(ARCH_LIST 3510)
# Use the first version header the compiler would see. Without a known version, keep the 3510 checks.
foreach(INC IN LISTS INCLUDE_PATHS)
    set(VERSION_HEADER "${INC}/version/asc_devkit_version.h")
    if(NOT EXISTS "${VERSION_HEADER}" OR IS_DIRECTORY "${VERSION_HEADER}")
        continue()
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${VERSION_HEADER}")
    file(STRINGS "${VERSION_HEADER}" VERSION_DEFINE LIMIT_COUNT 1
        REGEX "^[ \t]*#[ \t]*define[ \t]+ASC_DEVKIT_VERSION_NUM[ \t]+[0-9]+[ \t]*($|//|/\\*)")
    string(REGEX MATCH "[0-9]+" DEVKIT_VERSION_NUM "${VERSION_DEFINE}")
    if(DEVKIT_VERSION_NUM AND DEVKIT_VERSION_NUM GREATER_EQUAL 90300000)
        list(APPEND ARCH_LIST 9201)
    endif()
    break()
endforeach()

# ==========================================
# 测试文件
# ==========================================

set(ALL_TEST_FILES "")

file(GLOB_RECURSE ALL_TEST_FILES
    LIST_DIRECTORIES false
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

if(NOT ALL_TEST_FILES)
    message(FATAL_ERROR "No test files found to test.")
endif()

list(REMOVE_DUPLICATES ALL_TEST_FILES)
list(LENGTH ALL_TEST_FILES LEN)
message(STATUS "Total test files to test: ${LEN}")

# ==========================================
# 构建逻辑
# ==========================================

# 生成 Include 参数
set(INC_FLAGS "")
list(REMOVE_DUPLICATES INCLUDE_PATHS)
list(FILTER INCLUDE_PATHS EXCLUDE REGEX "^$")
foreach(INC ${INCLUDE_PATHS})
    list(APPEND INC_FLAGS "-I${INC}")
endforeach()
message(STATUS "INC_FLAGS: ${INC_FLAGS}")

# 函数: 为指定模式生成检查 targets
# 参数:  MODE_SUFFIX - 模式后缀 (如 "" 或 "_single")
#       MODE_DEFINES - 额外的宏定义列表
function(create_check_targets MODE_SUFFIX MODE_DEFINES)
    set(TARGET_VAR "CHECK_TARGETS_${NPU_MODE}${MODE_SUFFIX}")
    set(${TARGET_VAR})

    # 根据 NPU 模式设置编译器和参数
    set(COMPILER "bisheng")
    set(COMPILER_FLAGS -x cce -std=c++17 -w)
    set(COMPILER_TYPE "NPU")

    message(STATUS "Creating ${COMPILER_TYPE}${MODE_SUFFIX} targets")

    # 遍历所有文件和所有配置
    foreach(TEST_FILE ${ALL_TEST_FILES})
        set(CHECK_FLAGS ${COMPILER_FLAGS})
        if(TEST_FILE MATCHES "\\.verify\\.cpp$")
            # Diagnostic tests include the module directly and must fail for the expected reason.
            if(MODE_SUFFIX STREQUAL "")
                continue()
            endif()
            list(APPEND CHECK_FLAGS -Xclang -verify -Xclang -verify-ignore-unexpected=note)
        endif()
        # 获取相对路径用于生成唯一的 Target 名字
        file(RELATIVE_PATH REL_TEST_FILE ${ASCENDC_DIR} ${TEST_FILE})

        # 清理文件名中的特殊字符
        string(REPLACE "/" "_" TEST_FILE_ID ${REL_TEST_FILE})
        string(REPLACE "." "_" TEST_FILE_ID ${TEST_FILE_ID})
        string(REPLACE "-" "_" TEST_FILE_ID ${TEST_FILE_ID})
        foreach(ARCH IN LISTS ARCH_LIST)
            set(ARCH_FLAG "--npu-arch=dav-${ARCH}")

            # 创建子 Target
            set(TARGET_NAME "tensor_api_check_npu${MODE_SUFFIX}_${ARCH}_${TEST_FILE_ID}")
            add_custom_target(${TARGET_NAME}
                COMMAND ${COMPILER} ${CHECK_FLAGS} ${INC_FLAGS} ${MODE_DEFINES}
                    -fsyntax-only ${TEST_FILE} ${ARCH_FLAG}
                COMMENT "Checking host (${COMPILER_TYPE})${MODE_SUFFIX} ${ARCH} ${REL_TEST_FILE}"
                VERBATIM
                USES_TERMINAL
            )

            list(APPEND ${TARGET_VAR} ${TARGET_NAME})
        endforeach()
    endforeach()

    # 创建汇总 Target
    set(RUN_ALL_TARGET "ascendc_run_all_tensor_api_header_checks_npu${MODE_SUFFIX}")

    if(MODE_SUFFIX STREQUAL "")
        set(MODE_DESC "normal")
    else()
        string(REPLACE "_" " " MODE_DESC "${MODE_SUFFIX}")
        string(STRIP "${MODE_DESC}" MODE_DESC)
    endif()
    add_custom_target(${RUN_ALL_TARGET}
        DEPENDS ${${TARGET_VAR}}
        COMMENT "Running all tensor api header checks (${COMPILER_TYPE} ${MODE_DESC} mode)..."
    )
endfunction()

# 模式 1: NPU 普通模式
set(NORMAL_DEFINES)
create_check_targets("" "${NORMAL_DEFINES}")

# 模式 2: NPU 单一头文件模式
set(SINGLE_DEFINES -DVERIFY_SINGLE_HEADER)
create_check_targets("_single" "${SINGLE_DEFINES}")

add_custom_target(ascendc_run_all_tensor_api_header_checks ALL
    DEPENDS
        ascendc_run_all_tensor_api_header_checks_npu
        ascendc_run_all_tensor_api_header_checks_npu_single
    COMMENT "Running all tensor api header checks (all modes, all architectures)..."
)

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 后续如果更新asc-comm代码需要同步更新至devkit run包需要更新change id
set(ASC_COMM_TAG_ID cc2a082a7cb4f8d09f55bb22a676ec6841de4b32)

# asc-comm 与 asc-devkit 按同级目录放置。路径以本文件所在目录(<devkit>/cmake/third_party)为锚，
# 对于CI环境已经存在asc-comm代码, 不去拉取代码否则使用submodule方式拉取asc-comm仓代码
message(STATUS "[ThirdPartyLib][asc-comm] project source dir: ${PROJECT_SOURCE_DIR}")
get_filename_component(_ASC_COMM_DEVKIT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." REALPATH)
if(NOT ASC_COMM_SOURCE_DIR)
    foreach(_asc_comm_candidate
        "${_ASC_COMM_DEVKIT_ROOT}/asc-comm"
        "${_ASC_COMM_DEVKIT_ROOT}/../asc-comm"
        "${_ASC_COMM_DEVKIT_ROOT}/../asc/asc-comm"
        "${_ASC_COMM_DEVKIT_ROOT}/../../asc-comm"
        "${_ASC_COMM_DEVKIT_ROOT}/../../asc/asc-comm"
        "${_ASC_COMM_DEVKIT_ROOT}/../../../asc-comm"
        "${_ASC_COMM_DEVKIT_ROOT}/../../../asc/asc-comm")
        if(EXISTS "${_asc_comm_candidate}/CMakeLists.txt")
            set(ASC_COMM_SOURCE_DIR "${_asc_comm_candidate}")
            break()
        endif()
    endforeach()
endif()

if(ASC_COMM_SOURCE_DIR AND EXISTS "${ASC_COMM_SOURCE_DIR}")
    get_filename_component(ASC_COMM_SOURCE_PATH ${ASC_COMM_SOURCE_DIR} REALPATH)
    message(STATUS "[ThirdPartyLib][asc-comm] Find source dir: ${ASC_COMM_SOURCE_PATH}")
else()
    set(ASC_COMM_SOURCE_DIR "${_ASC_COMM_DEVKIT_ROOT}/../asc-comm")
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/asc-comm-subbuild")
        file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/_deps/asc-comm-subbuild)
    endif()
    include(FetchContent)

    FetchContent_Declare(
        asc-comm
        GIT_REPOSITORY https://gitcode.com/cann/asc-comm.git
        GIT_TAG ${ASC_COMM_TAG_ID}
        GIT_PROGRESS TRUE
        SOURCE_DIR ${ASC_COMM_SOURCE_DIR}
    )
    FetchContent_Populate(asc-comm)

    get_filename_component(ASC_COMM_SOURCE_PATH ${ASC_COMM_SOURCE_DIR} REALPATH)
endif()

# asc-comm 迁出后，src 与 include 的公共头统一打包到 comm_api（ain/jetty/asccomm）；
# hcomm 已下沉到 src/aicore/legacy/hcomm（仅保留兼容旧接口所需内容）：
#   - legacy 公开头打包到 asc/include/adv_api/hcomm/，安装路径与软链接保持不变
#   - 引擎实现（src/aicore/asccomm）随 src 扫描打包到 asc/impl/comm_api/aicore/asccomm/
#   - legacy 适配层打包到 asccomm/impl/，新旧公开头以相对路径包含 comm_api 下实现源码
#   - asccomm 公开头实体在 asc/include/comm_api/aicore/asccomm/（无软链接）
set(ASC_COMM_SRC_DIR ${ASC_COMM_SOURCE_PATH}/src)
set(ASC_COMM_INCLUDE_DIR ${ASC_COMM_SOURCE_PATH}/include)
if(NOT EXISTS ${ASC_COMM_SRC_DIR})
    message(FATAL_ERROR "[ThirdPartyLib][asc-comm] Missing src dir: ${ASC_COMM_SRC_DIR}")
endif()
if(NOT EXISTS ${ASC_COMM_INCLUDE_DIR})
    message(FATAL_ERROR "[ThirdPartyLib][asc-comm] Missing include dir: ${ASC_COMM_INCLUDE_DIR}")
endif()

# 将 src 下代码（legacy 除外）统一打包到 asc/impl/comm_api/
# （aicore/asccomm 引擎实现随扫描进入 asc/impl/comm_api/aicore/asccomm/）
install(DIRECTORY ${ASC_COMM_SRC_DIR}/
    DESTINATION ${INSTALL_LIBRARY_DIR}/asc/impl/comm_api
    COMPONENT asc-devkit
    FILES_MATCHING PATTERN "*.h"
    # 这些目录只有源码（.cc/CMakeLists/README），FILES_MATCHING 过滤后只剩空目录，直接排除
    PATTERN "cc" EXCLUDE
    PATTERN "cpu" EXCLUDE
    PATTERN "ccu/ccu_api/primitives" EXCLUDE
    PATTERN "aicpu/asccomm/experimental" EXCLUDE
    REGEX "legacy" EXCLUDE
)

# 将 legacy jetty 公开头实体打包到 asc/include/comm_api/aicore/jetty/（原路径不变）
install(DIRECTORY ${ASC_COMM_SRC_DIR}/aicore/legacy/jetty/
    DESTINATION ${INSTALL_LIBRARY_DIR}/asc/include/comm_api/aicore/jetty
    COMPONENT asc-devkit
    FILES_MATCHING PATTERN "*.h"
)

# 将 include 下代码（direct_drive 除外）统一打包到 asc/include/comm_api/
# （ain/jetty/asccomm 公开头实体随扫描进入 asc/include/comm_api/aicore/）
install(DIRECTORY ${ASC_COMM_INCLUDE_DIR}/
    DESTINATION ${INSTALL_LIBRARY_DIR}/asc/include/comm_api
    COMPONENT asc-devkit
    FILES_MATCHING PATTERN "*.h"
    PATTERN "cc" EXCLUDE
    REGEX "direct_drive" EXCLUDE
)


# （安装路径与原版本一致，asc/include/comm_api/aicore/hcomm 为兼容软链接）
install(DIRECTORY ${ASC_COMM_SRC_DIR}/aicore/legacy/hcomm/
    DESTINATION ${INSTALL_LIBRARY_DIR}/asc/include/adv_api/hcomm
    COMPONENT asc-devkit
    FILES_MATCHING PATTERN "*.h"
    REGEX "impl" EXCLUDE
)

# 将 legacy hcomm 适配层打包到 asc/impl/comm_api/aicore/asccomm/impl/
# （引擎实现已在 asc/impl/comm_api/aicore/asccomm/；新旧公开头均以相对路径
#   指向 comm_api 下的实现源码，适配层与引擎同树解析）
install(DIRECTORY ${ASC_COMM_SRC_DIR}/aicore/legacy/hcomm/impl/
    DESTINATION ${INSTALL_LIBRARY_DIR}/asc/impl/comm_api/aicore/asccomm/impl
    COMPONENT asc-devkit
    FILES_MATCHING PATTERN "*.h"
)

# 为 legacy hcomm 建立 adv_api 兼容软链接（实体在 comm_api/aicore/hcomm）；
# 引擎与 asccomm 公开头均为 comm_api 下的实际文件，无需软链接
install(CODE "file(MAKE_DIRECTORY \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${INSTALL_LIBRARY_DIR}/asc/include/comm_api/aicore)" COMPONENT asc-devkit)
install(CODE "file(CREATE_LINK ../../adv_api/hcomm \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${INSTALL_LIBRARY_DIR}/asc/include/comm_api/aicore/hcomm SYMBOLIC)" COMPONENT asc-devkit)

# AICPU 公共 API 头由 asc-devkit 直接交付；运行期库仍由 runtime 包提供。
# asc-comm include 布局沿革：aicpu/hcomm（旧）→ aicpu/asccomm → aicpu_ts/asccomm（现行，cann 9.3.0）。
set(ASC_COMM_AICPU_HEADER_DIR "")
foreach(_asc_comm_hdr_layout "aicpu_ts/asccomm" "aicpu/asccomm" "aicpu/hcomm")
    if(EXISTS "${ASC_COMM_SOURCE_PATH}/include/${_asc_comm_hdr_layout}")
        set(ASC_COMM_AICPU_HEADER_DIR "${ASC_COMM_SOURCE_PATH}/include/${_asc_comm_hdr_layout}")
        break()
    endif()
endforeach()
if(ASC_COMM_AICPU_HEADER_DIR)
    install(DIRECTORY ${ASC_COMM_AICPU_HEADER_DIR}/
        DESTINATION ${INSTALL_LIBRARY_DIR}/asc/include/comm_api/aicpu/hcomm
        COMPONENT asc-devkit
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
endif()

# mc2_client 硬依赖 libasccomm_aicpu.so，须随消费者交付到顶层 lib64；-DASC_COMM_AICPU_SO 可指定预编译件。
set(ASC_COMM_AICPU_SO "" CACHE PATH "Prebuilt libasccomm_aicpu.so to deliver under top-level lib64")
option(ENABLE_MOD_EXT "Enable modular-extension build components" OFF)
if(NOT ASC_COMM_AICPU_SO)
    set(_asc_comm_aicpu_out "${CMAKE_BINARY_DIR}/package-aicpu-device/asc-comm-build/libasccomm_aicpu.so")
    # The device sub-project resolves cann-cmake itself; prefer the devkit-resolved
    # CANN_CMAKE_DIR when present so both stay on one checkout.
    if(NOT DEFINED CANN_CMAKE_DIR OR CANN_CMAKE_DIR STREQUAL "")
        set(_asc_comm_toolchain_file "")
    else()
        set(_asc_comm_toolchain_file "-DCMAKE_TOOLCHAIN_FILE=${CANN_CMAKE_DIR}/toolchain/aarch64-hcc-toolchain.cmake")
    endif()
    if(NOT ENABLE_MOD_EXT AND NOT EXISTS "${_asc_comm_aicpu_out}" AND EXISTS "${ASC_COMM_SOURCE_PATH}/cmake/device/CMakeLists.txt"
        AND ASCEND_CANN_PACKAGE_PATH)
        include(ExternalProject)
        # Mirror the superbuild's device-pass -I set (CANN arch/toplevel, runtime
        # pkg_inc, libc_sec); component-isolated builds point ASCEND_CANN_PACKAGE_PATH
        # at a build-area tree whose layout differs from an installed package.
        # Keep only real directories — -I<file> warns.
        set(_asc_comm_aicpu_extra_inc "")
        set(_asc_comm_aicpu_inc_candidates "")
        foreach(_asc_arch aarch64-linux x86_64-linux)
            file(GLOB _asc_comm_cann_glob_dirs
                    "${ASCEND_CANN_PACKAGE_PATH}/${_asc_arch}/pkg_inc"
                    "${ASCEND_CANN_PACKAGE_PATH}/${_asc_arch}/pkg_inc/*"
                    "${ASCEND_CANN_PACKAGE_PATH}/${_asc_arch}/pkg_inc/*/*"
                    "${ASCEND_CANN_PACKAGE_PATH}/${_asc_arch}/include"
                    "${ASCEND_CANN_PACKAGE_PATH}/${_asc_arch}/include/*")
            list(APPEND _asc_comm_aicpu_inc_candidates ${_asc_comm_cann_glob_dirs})
        endforeach()
        file(GLOB _asc_comm_cann_glob_dirs
                "${ASCEND_CANN_PACKAGE_PATH}/include"
                "${ASCEND_CANN_PACKAGE_PATH}/include/*"
                "${ASCEND_CANN_PACKAGE_PATH}/include/*/*"
                "${ASCEND_CANN_PACKAGE_PATH}/pkg_inc"
                "${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/*")
        list(APPEND _asc_comm_aicpu_inc_candidates ${_asc_comm_cann_glob_dirs})
        foreach(_rt_anchor ".." "../..")
            file(GLOB _asc_comm_rt_pkg_inc_dirs
                    "${PROJECT_SOURCE_DIR}/${_rt_anchor}/runtime/pkg_inc"
                    "${PROJECT_SOURCE_DIR}/${_rt_anchor}/runtime/pkg_inc/*")
            list(APPEND _asc_comm_aicpu_inc_candidates ${_asc_comm_rt_pkg_inc_dirs})
            foreach(_rt_sub
                    "src/dfx/log/inc"
                    "src/dfx/log/inc/toolchain"
                    "include/dfx"
                    "include/dfx/base"
                    "include/external"
                    "include/external/acl")
                list(APPEND _asc_comm_aicpu_inc_candidates
                    "${PROJECT_SOURCE_DIR}/${_rt_anchor}/runtime/${_rt_sub}")
            endforeach()
        endforeach()
        list(APPEND _asc_comm_aicpu_inc_candidates
                "${CANN_3RD_LIB_PATH}/libc_sec/include"
                "${PROJECT_SOURCE_DIR}/../open_source/libc_sec/include"
                "${PROJECT_SOURCE_DIR}/../../open_source/libc_sec/include")
        foreach(_inc ${_asc_comm_aicpu_inc_candidates})
            if(IS_DIRECTORY "${_inc}")
                string(APPEND _asc_comm_aicpu_extra_inc " -I${_inc}")
            endif()
        endforeach()
        ExternalProject_Add(asccomm_aicpu_provider
            SOURCE_DIR "${ASC_COMM_SOURCE_PATH}/cmake/device"
            BINARY_DIR "${CMAKE_BINARY_DIR}/package-aicpu-device"
            CMAKE_ARGS
                "-DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}"
                "-DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"
                "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                "-DTOOLCHAIN_DIR=${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc"
                "-DCMAKE_C_FLAGS=${_asc_comm_aicpu_extra_inc}"
                "-DCMAKE_CXX_FLAGS=${_asc_comm_aicpu_extra_inc}"
                ${_asc_comm_toolchain_file}
            BUILD_COMMAND "${CMAKE_COMMAND}" --build . --target pack_asccomm_aicpu
            INSTALL_COMMAND ""
            BUILD_BYPRODUCTS "${_asc_comm_aicpu_out}")
        # The custom ALL target drives build order that install() cannot express.
    endif()
    if(EXISTS "${_asc_comm_aicpu_out}")
        set(ASC_COMM_AICPU_SO "${_asc_comm_aicpu_out}")
    endif()
endif()

# ASC_COMM_AICPU_SO 始终为 aarch64 设备 ELF：aarch64 host 可直接占位顶层 lib64；
# x86 host 上 loader 拒绝异构 ELF（561000），host 槽位须用 host 编译的同名 stub
# （cmake/stub/asccomm_aicpu_host_stub.cc）替代，devlib 槽位两种 host 均留真身。
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(NOT TARGET asccomm_aicpu_host_stub)
        add_library(asccomm_aicpu_host_stub SHARED
            ${CMAKE_CURRENT_LIST_DIR}/../stub/asccomm_aicpu_host_stub.cc)
        set_target_properties(asccomm_aicpu_host_stub PROPERTIES
            OUTPUT_NAME asccomm_aicpu
            POSITION_INDEPENDENT_CODE ON)
        # Header surface: asccomm primitives + hcomm entity defs + libc_sec.
        target_include_directories(asccomm_aicpu_host_stub PRIVATE
            ${ASC_COMM_AICPU_HEADER_DIR}
            ${ASC_COMM_SOURCE_PATH}/pkg_inc/hcomm
            ${CANN_3RD_LIB_PATH}/libc_sec/include
            ${PROJECT_SOURCE_DIR}/../open_source/libc_sec/include
            ${PROJECT_SOURCE_DIR}/../../open_source/libc_sec/include)
        target_compile_options(asccomm_aicpu_host_stub PRIVATE -fvisibility=hidden)
    endif()
endif()

if(ASC_COMM_AICPU_SO AND EXISTS "${ASC_COMM_AICPU_SO}")
    message(STATUS "[ThirdPartyLib][asc-comm] Deliver AICPU data plane: ${ASC_COMM_AICPU_SO}")
    # devlib 槽位走既有 devlib 打包；此处只处理顶层 host 槽位。
    if(TARGET asccomm_aicpu_host_stub)
        # x86_64 host: host-facing slots get the host-compiled stub.
        install(TARGETS asccomm_aicpu_host_stub
            LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/lib64
            COMPONENT asc-devkit)
        install(TARGETS asccomm_aicpu_host_stub
            LIBRARY DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
            COMPONENT asc-devkit)
    else()
        # aarch64 host: the original doubles as the host copy (same ELF class).
        install(FILES "${ASC_COMM_AICPU_SO}"
            DESTINATION ${INSTALL_LIBRARY_DIR}/lib64
            COMPONENT asc-devkit)
        install(FILES "${ASC_COMM_AICPU_SO}"
            DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
            COMPONENT asc-devkit)
    endif()
elseif(TARGET asccomm_aicpu_provider)
    # Defer the artifact pick to install time (ExternalProject builds after configure);
    # fail loud when the build did not produce it.
    # NOTE: file(INSTALL) prepends $ENV{DESTDIR} itself — DESTINATION must not repeat it.
    message(STATUS "[ThirdPartyLib][asc-comm] AICPU data plane delivery deferred to install: ${_asc_comm_aicpu_out}")
    if(TARGET asccomm_aicpu_host_stub)
        # x86_64 host: device slot gets the deferred artifact; host slots use the stub.
        install(CODE "
            set(_out \"${_asc_comm_aicpu_out}\")
            if(NOT EXISTS \"\${_out}\")
                message(FATAL_ERROR \"asccomm_aicpu_provider did not produce \${_out}; cannot deliver libasccomm_aicpu.so\")
            endif()
        " COMPONENT asc-devkit)
        install(TARGETS asccomm_aicpu_host_stub
            LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/lib64
            COMPONENT asc-devkit)
        install(TARGETS asccomm_aicpu_host_stub
            LIBRARY DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
            COMPONENT asc-devkit)
    else()
        install(CODE "
            set(_out \"${_asc_comm_aicpu_out}\")
            if(NOT EXISTS \"\${_out}\")
                message(FATAL_ERROR \"asccomm_aicpu_provider did not produce \${_out}; cannot deliver libasccomm_aicpu.so\")
            endif()
            file(INSTALL \"\${_out}\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${INSTALL_LIBRARY_DIR}/lib64\")
        " COMPONENT asc-devkit)
    endif()
else()
    message(WARNING "[ThirdPartyLib][asc-comm] libasccomm_aicpu.so unavailable: no prebuilt (-DASC_COMM_AICPU_SO), "
        "no source device build (asc-comm cmake/device), no ASCEND_CANN_PACKAGE_PATH — top-level lib64 delivery "
        "skipped; the assembled toolkit will miss the mc2_client DT_NEEDED provider")
endif()

# 构建 asc-comm CCU 动态库（数据面实现 + host-launch 兼容 wrapper），
# 并将库与公开 DSL 头文件交付到 asc-devkit run 包。
# CCU 源码归属 asc-comm，跨 SO 依赖通过已安装 CANN pkg_inc 和 POD ABI 解决。
set(ASC_COMM_CCU_CMAKE_DIR ${ASC_COMM_SOURCE_PATH}/src/ccu)
if(EXISTS "${ASC_COMM_CCU_CMAKE_DIR}/CMakeLists.txt")
    # The compiler-facing runtime is delivered as a static archive; the archive below
    # compiles the same sources with the real HComm backend.
    set(ASCCOMM_BUILD_CCU_RUNTIME OFF CACHE BOOL "" FORCE)
    if(NOT TARGET asccomm_ccu)
        add_subdirectory(${ASC_COMM_CCU_CMAKE_DIR} ${CMAKE_BINARY_DIR}/asccomm_ccu)
    endif()

    if(TARGET asccomm_ccu)
        if(NOT TARGET asccomm_ccu_build)
            add_custom_target(asccomm_ccu_build ALL DEPENDS asccomm_ccu)
        endif()
        install(TARGETS asccomm_ccu
            LIBRARY DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64 ${INSTALL_OPTIONAL}
            COMPONENT asc-devkit)
        install(DIRECTORY ${ASC_COMM_SOURCE_PATH}/include/ccu/ccu_api/
            DESTINATION ${INSTALL_LIBRARY_DIR}/asc/include/comm_api/ccu/ccu_api
            COMPONENT asc-devkit
            FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
        install(FILES ${ASC_COMM_SOURCE_PATH}/cmake/asccomm-config.cmake
            DESTINATION ${INSTALL_LIBRARY_DIR}/lib64/cmake
            COMPONENT asc-devkit)
    endif()

    # ccu_runtime 由 asc-devkit 统一交付（子目录构建时 asc-comm/experimental 的独立 install 不生效）。
    # Static archive so generated launch stubs resolve into the final host image.
    get_filename_component(_ASCCOMM_RUNTIME_SOURCE_PATH
        "${ASC_COMM_SOURCE_PATH}/../../runtime" REALPATH BASE_DIR "${ASC_COMM_SOURCE_PATH}")

    if(EXISTS "${ASC_COMM_CCU_CMAKE_DIR}/experimental/ccu_runtime.cpp")
        if(NOT TARGET ccu_runtime_static)
            add_library(ccu_runtime_static STATIC)
            target_sources(ccu_runtime_static PRIVATE
                ${ASC_COMM_CCU_CMAKE_DIR}/experimental/ccu_runtime.cpp
                ${ASC_COMM_CCU_CMAKE_DIR}/experimental/ccu_runtime_backend.cpp
                ${ASC_COMM_CCU_CMAKE_DIR}/experimental/ccu_aot_adapter.cpp
                ${ASC_COMM_CCU_CMAKE_DIR}/experimental/ccu_runtime_asccomm_backend.cpp)
            target_compile_features(ccu_runtime_static PRIVATE cxx_std_17)
            target_compile_options(ccu_runtime_static PRIVATE -Wall -Wextra -Werror)
            target_compile_definitions(ccu_runtime_static PRIVATE
                CCU_TRANSLATOR_HAS_HCOMM_BACKEND
                ASCCOMM_CCU_LOCAL_LOGGING)
            target_include_directories(ccu_runtime_static PRIVATE
                ${ASC_COMM_SOURCE_PATH}/include
                ${ASC_COMM_SOURCE_PATH}/pkg_inc
                ${ASC_COMM_CCU_CMAKE_DIR}/experimental)
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
                set(_ASCCOMM_CANN_ARCH "aarch64-linux")
            else()
                set(_ASCCOMM_CANN_ARCH "x86_64-linux")
            endif()
            set(_ASCCOMM_CANN_ARCH_DIR
                "${ASCEND_CANN_PACKAGE_PATH}/${_ASCCOMM_CANN_ARCH}")
            target_include_directories(ccu_runtime_static SYSTEM PRIVATE
                ${ASCEND_CANN_PACKAGE_PATH}/include
                ${ASCEND_CANN_PACKAGE_PATH}/runtime/include
                ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
                ${_ASCCOMM_RUNTIME_SOURCE_PATH}/pkg_inc
                ${_ASCCOMM_CANN_ARCH_DIR}/include
                ${_ASCCOMM_CANN_ARCH_DIR}/asc/include/comm_api
                ${_ASCCOMM_CANN_ARCH_DIR}/pkg_inc)
            # Static archives do not link dependencies into the archive: keep the
            # dependency surface PUBLIC and deliver ccu_runtime.pc for external link lines.
            target_link_libraries(ccu_runtime_static PUBLIC
                acl_rt runtime c_sec pthread)
            set_target_properties(ccu_runtime_static PROPERTIES
                OUTPUT_NAME ccu_runtime
                POSITION_INDEPENDENT_CODE ON)
            configure_file(
                "${CMAKE_CURRENT_LIST_DIR}/ccu_runtime.pc.in"
                "${CMAKE_BINARY_DIR}/ccu_runtime.pc"
                @ONLY)
        endif()
        if(NOT TARGET ccu_runtime_build)
            add_custom_target(ccu_runtime_build ALL DEPENDS ccu_runtime_static)
        endif()
        install(TARGETS ccu_runtime_static
            ARCHIVE DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
            COMPONENT asc-devkit)
        install(FILES "${CMAKE_BINARY_DIR}/ccu_runtime.pc"
            DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64/pkgconfig
            COMPONENT asc-devkit)
    endif()
else()
    message(STATUS "[ThirdPartyLib][asc-comm] Missing CCU cmake dir: ${ASC_COMM_CCU_CMAKE_DIR}, skip CCU so build")
endif()

# 构建 asc-comm CPU TS 同步库，并交付到 asc-devkit run 包。
# 该库仅依赖运行时已导出的 notify API，不依赖 hcomm 源码或符号。
# 同步将 include/cpu 改名为 include/cpu_ts（旧树回退探测）。
set(ASC_COMM_CPU_TS_CMAKE_DIR ${ASC_COMM_SOURCE_PATH}/src/cpu)
if(EXISTS "${ASC_COMM_CPU_TS_CMAKE_DIR}/CMakeLists.txt")
    if(NOT TARGET asccomm_cpu_ts)
        add_subdirectory(${ASC_COMM_CPU_TS_CMAKE_DIR} ${CMAKE_BINARY_DIR}/asccomm_cpu_ts)
    endif()

    if(TARGET asccomm_cpu_ts)
        if(NOT TARGET asccomm_cpu_ts_build)
            add_custom_target(asccomm_cpu_ts_build ALL DEPENDS asccomm_cpu_ts)
        endif()
        install(TARGETS asccomm_cpu_ts
            LIBRARY DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64 ${INSTALL_OPTIONAL}
            COMPONENT asc-devkit)
        set(ASC_COMM_CPU_TS_HDR_DIR "")
        foreach(_asc_comm_cpu_layout "cpu_ts" "cpu")
            if(EXISTS "${ASC_COMM_SOURCE_PATH}/include/${_asc_comm_cpu_layout}")
                set(ASC_COMM_CPU_TS_HDR_DIR "${ASC_COMM_SOURCE_PATH}/include/${_asc_comm_cpu_layout}")
                break()
            endif()
        endforeach()
        if(ASC_COMM_CPU_TS_HDR_DIR)
            install(DIRECTORY ${ASC_COMM_CPU_TS_HDR_DIR}/
                DESTINATION ${INSTALL_LIBRARY_DIR}/asc/include/comm_api/cpu
                COMPONENT asc-devkit
                FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
        endif()
    endif()
else()
    message(STATUS "[ThirdPartyLib][asc-comm] Missing CPU TS cmake dir: ${ASC_COMM_CPU_TS_CMAKE_DIR}, skip CPU TS so build")
endif()

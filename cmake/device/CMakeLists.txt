# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
include(../fetch_cann_cmake.cmake)
project(ascendc_tiling_api)

init_cann_project(PRODUCT_SIDE "device" PREPEND_MODULE_PATH)
add_cann_target_options()

set(HI_PYTHON "python3" CACHE STRING "python executor")
set(ASCENDC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(ASCENDC_INCLUDE_DIR ${ASCENDC_DIR}/include)
set(ASCENDC_IMPL_DIR ${ASCENDC_DIR}/impl)
set(INSTALL_LIBRARY_DIR ${CMAKE_HOST_SYSTEM_PROCESSOR}-linux/lib64/device)
get_filename_component(ASC_DEVKIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

find_cann_package(aicpu_sharder REQUIRED)
find_cann_package(tilingdata_base REQUIRED OBJECTS)
find_cann_package(metadef REQUIRED)
find_cann_package(unified_dlog REQUIRED)
find_cann_package(ascend_hal REQUIRED)
find_cann_package(securec REQUIRED)
find_cann_package(runtime REQUIRED)
find_cann_package(mmpa REQUIRED)
find_cann_package(acl_rt REQUIRED)
find_cann_package(atrace REQUIRED)
find_cann_package(error_manager REQUIRED)

include(${ASCENDC_DIR}/impl/adv_api/gen_kernel_tiling.cmake)
include(${ASCENDC_DIR}/version.cmake)

set(DEVICE_MODE TRUE)
set(KERNEL_MODE TRUE)

# asccomm public header dir: asc-comm include layout history is aicpu/hcomm (oldest)
# -> aicpu/asccomm -> aicpu_ts/asccomm (current, cann 9.3.0); probe newest first.
function(_asccomm_hdr_dir _root _out_var)
    if(EXISTS "${_root}/include/aicpu_ts/asccomm/asccomm_primitives.h")
        set(${_out_var} "${_root}/include/aicpu_ts/asccomm" PARENT_SCOPE)
    elseif(EXISTS "${_root}/include/aicpu/asccomm/asccomm_primitives.h")
        set(${_out_var} "${_root}/include/aicpu/asccomm" PARENT_SCOPE)
    elseif(EXISTS "${_root}/include/aicpu/hcomm/asccomm_primitives.h")
        set(${_out_var} "${_root}/include/aicpu/hcomm" PARENT_SCOPE)
    else()
        set(${_out_var} "" PARENT_SCOPE)
    endif()
endfunction()

_asccomm_hdr_dir("${ASC_COMM_SOURCE_DIR}" _asc_hdr)
if(NOT _asc_hdr)
    get_filename_component(_ASC_COMM_DEVKIT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." REALPATH)
    foreach(_asc_comm_candidate
        "${_ASC_COMM_DEVKIT_ROOT}/asc-comm"
        "${_ASC_COMM_DEVKIT_ROOT}/../asc-comm"
        "${_ASC_COMM_DEVKIT_ROOT}/../asc/asc-comm"
        "${_ASC_COMM_DEVKIT_ROOT}/../../asc-comm"
        "${_ASC_COMM_DEVKIT_ROOT}/../../asc/asc-comm"
        "${_ASC_COMM_DEVKIT_ROOT}/../../../asc-comm"
        "${_ASC_COMM_DEVKIT_ROOT}/../../../asc/asc-comm")
        _asccomm_hdr_dir("${_asc_comm_candidate}" _asc_hdr)
        if(_asc_hdr)
            set(ASC_COMM_SOURCE_DIR "${_asc_comm_candidate}")
            break()
        endif()
    endforeach()
endif()
if(NOT _asc_hdr)
    message(FATAL_ERROR
        "[asc-comm] no usable asc-comm tree found (needs the cann 9.3.0 asccomm public-header "
        "layout; current layouts: include/aicpu_ts/asccomm or include/aicpu/asccomm): place an "
        "asc-comm checkout next to the devkit tree in the joint workspace, or pass "
        "-DASC_COMM_SOURCE_DIR=<asc-comm tree> for single-repo builds.")
endif()
if(_asc_hdr)
    set(ASCCOMM_PRIMITIVES_HDR "${_asc_hdr}")
    # The entity contract header includes "hcomm_res_defs.h" by bare name and needs the
    # ThreadType definition from the current hcomm public header (the old vendored copy lacks
    # the enum and breaks the build). Redundant with the derivation in impl/.../cc/src/CMakeLists.txt:
    # whichever probe entry (device or top-level shape) runs first, the variable is set.
    get_filename_component(_ASC_COMM_ROOT_DEVKIT "${_asc_hdr}/../../.." ABSOLUTE)
    if(EXISTS "${_ASC_COMM_ROOT_DEVKIT}/pkg_inc/hcomm/hcomm_res_defs.h")
        set(ASCOMM_RES_DEFS_HDR "${_ASC_COMM_ROOT_DEVKIT}/pkg_inc/hcomm")
    endif()
endif()

add_subdirectory(../../impl/aicpu_api impl/aicpu_api)
add_subdirectory(../../impl/adv_api/tiling impl/adv_api/tiling)
add_subdirectory(../../impl/adv_api/detail/hccl/cc impl/adv_api/detail/hccl/cc)

set_cann_subprj_package()

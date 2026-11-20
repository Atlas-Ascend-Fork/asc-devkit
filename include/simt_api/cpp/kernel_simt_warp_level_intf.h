/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_MODULE_SIMT_WARP_LEVEL_INTERFACE_H
#define ASCENDC_MODULE_SIMT_WARP_LEVEL_INTERFACE_H

#include "simt_api/device_types.h"
#include "simt_api/cpp/kernel_simt_utils.h"

namespace AscendC {
namespace Simt {
ASC_SIMT_CPP_DEPRECATED(9.3.0, "2027/11/20", asc_all)
__SIMT_DEVICE_FUNCTIONS_DECL__ inline int32_t WarpAllSync(int32_t predicate);

ASC_SIMT_CPP_DEPRECATED(9.3.0, "2027/11/20", asc_any)
__SIMT_DEVICE_FUNCTIONS_DECL__ inline int32_t WarpAnySync(int32_t predicate);

ASC_SIMT_CPP_DEPRECATED(9.3.0, "2027/11/20", asc_uni)
__SIMT_DEVICE_FUNCTIONS_DECL__ inline int32_t Uni(int32_t predicate);

ASC_SIMT_CPP_DEPRECATED(9.3.0, "2027/11/20", asc_ballot)
__SIMT_DEVICE_FUNCTIONS_DECL__ inline uint32_t WarpBallotSync(int32_t predicate);

ASC_SIMT_CPP_DEPRECATED(9.3.0, "2027/11/20", asc_activemask)
__SIMT_DEVICE_FUNCTIONS_DECL__ inline uint32_t WarpActiveMask();

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T WarpShflSync(T var, int32_t srcLane, int32_t width = warpSize);

template <typename T>
ASC_SIMT_CPP_DEPRECATED(9.3.0, "2027/11/20", asc_shfl_up)
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T WarpShflUpSync(T var, uint32_t delta, int32_t width = warpSize);

template <typename T>
ASC_SIMT_CPP_DEPRECATED(9.3.0, "2027/11/20", asc_shfl_down)
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T WarpShflDownSync(T var, uint32_t delta, int32_t width = warpSize);

template <typename T>
ASC_SIMT_CPP_DEPRECATED(9.3.0, "2027/11/20", asc_shfl_xor)
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T WarpShflXorSync(T var, int32_t laneMask, int32_t width = warpSize);

template <typename T>
ASC_SIMT_CPP_DEPRECATED(9.3.0, "2027/11/20", asc_reduce_add)
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T WarpReduceAddSync(T val);

template <typename T>
ASC_SIMT_CPP_DEPRECATED(9.3.0, "2027/11/20", asc_reduce_max)
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T WarpReduceMaxSync(T val);

template <typename T>
ASC_SIMT_CPP_DEPRECATED(9.3.0, "2027/11/20", asc_reduce_min)
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T WarpReduceMinSync(T val);

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void ThreadBarrier();

ASC_SIMT_CPP_DEPRECATED(9.3.0, "2027/11/20", asc_threadfence)
__SIMT_DEVICE_FUNCTIONS_DECL__ inline void ThreadFence();
} // namespace Simt
} // namespace AscendC

#include "impl/simt_api/cpp/kernel_simt_warp_level_intf_impl.h"
#endif // ASCENDC_MODULE_SIMT_WARP_LEVEL_INTERFACE_H

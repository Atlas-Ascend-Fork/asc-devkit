/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef VERIFY_SINGLE_HEADER
#include "tensor_api/experimental/vector_ops.h"
#else
#include "tensor_api/experimental/arch/vector/compute/register_logic.h"
#endif

using namespace asc::te::experimental;

template <typename T, typename RegisterCount, typename ScalarCount>
__simd_callee__ inline void check_shifts()
{
    reg_tensor<T> src;
    reg_tensor<RegisterCount> shift;
    ScalarCount count = 1;
    auto result = src << shift;
    result = src >> shift;
    result = src << count;
    result = src >> count;
}

__simd_vf__ inline void check_reg_logical_compute()
{
#if __NPU_ARCH__ == 3510
    check_shifts<int8_t, int8_t, int16_t>();
    check_shifts<uint8_t, int8_t, int16_t>();
    check_shifts<int16_t, int16_t, int16_t>();
    check_shifts<uint16_t, int16_t, int16_t>();
    check_shifts<int32_t, int32_t, int16_t>();
    check_shifts<uint32_t, int32_t, int>();
#elif __NPU_ARCH__ == 9201
    check_shifts<int8_t, uint8_t, uint16_t>();
    check_shifts<uint8_t, uint8_t, uint16_t>();
    check_shifts<int16_t, uint16_t, uint16_t>();
    check_shifts<uint16_t, uint16_t, uint16_t>();
    check_shifts<int32_t, uint32_t, uint32_t>();
    check_shifts<uint32_t, uint32_t, uint32_t>();
#endif
}

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
#include "tensor_api/experimental/arch/vector/compute/register_permute_select.h"
#include "tensor_api/experimental/arch/vector/compute/register_mask.h"
#endif

using namespace asc::te::experimental;

template <typename DataType>
__simd_callee__ inline void check_data_interleave()
{
    reg_tensor<DataType> source0;
    reg_tensor<DataType> source1;
    auto full = interleave(source0, source1);
    auto reordered = deinterleave(source0, source1);
    static_assert(AscendC::Std::is_same_v<decltype(full), reg_pair<DataType>>);
    static_assert(AscendC::Std::is_same_v<decltype(reordered), reg_pair<DataType>>);
}

template <typename DataType>
__simd_callee__ inline void check_mask_interleave()
{
    reg_tensor<bool> source0;
    reg_tensor<bool> source1;
    auto full = interleave<DataType>(source0, source1);
    auto reordered = deinterleave<DataType>(source0, source1);
    static_assert(AscendC::Std::is_same_v<decltype(full), reg_pair<bool>>);
    static_assert(AscendC::Std::is_same_v<decltype(reordered), reg_pair<bool>>);
}

__simd_vf__ inline void check_reg_data_reorder()
{
    check_data_interleave<int8_t>();
    check_data_interleave<uint8_t>();
    check_data_interleave<int16_t>();
    check_data_interleave<uint16_t>();
    check_data_interleave<int32_t>();
    check_data_interleave<uint32_t>();
    check_data_interleave<half>();
    check_data_interleave<bfloat16_t>();
    check_data_interleave<float>();
    check_data_interleave<fp8_e4m3fn_t>();
    check_data_interleave<fp8_e5m2_t>();
    check_data_interleave<fp8_e8m0_t>();
    check_data_interleave<hifloat8_t>();
    check_mask_interleave<uint8_t>();
    check_mask_interleave<uint16_t>();
    check_mask_interleave<uint32_t>();
}

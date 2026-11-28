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
#include "tensor_api/tensor.h"
#endif
#include "tensor_api/experimental/arch/vector/defs/register_tensor.h"

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201)) && \
    !defined(IMPL_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_TENSOR_IMPL_H)
#error register tensor implementation is missing for the target architecture
#endif

using namespace asc::te::experimental;

template <typename DataType>
__simd_callee__ inline void check_reg_tensor_type()
{
    reg_tensor<bool> active;
    reg_tensor<DataType> source(active.reg);
    auto& masked = source.with_mask(active);
    static_assert(AscendC::Std::is_same_v<decltype(masked), reg_tensor<DataType>&>);
    static_assert(AscendC::Std::is_same_v<typename reg_tensor<DataType>::elem_type, DataType>);
    static_assert(is_reg_tensor<reg_tensor<DataType>>::value);
    static_assert(is_reg_tensor_v<const reg_tensor<DataType>&>);
    static_assert(!is_reg_tensor_v<DataType>);
    reg_pair<DataType> pair;
    reg_pair<DataType, bool> carry_pair;
    static_assert(AscendC::Std::is_same_v<decltype(pair.first), reg_tensor<DataType>>);
    static_assert(AscendC::Std::is_same_v<decltype(carry_pair.second), reg_tensor<bool>>);
}

__simd_vf__ inline void check_reg_tensor()
{
    check_reg_tensor_type<bool>();
    check_reg_tensor_type<int8_t>();
    check_reg_tensor_type<uint8_t>();
    check_reg_tensor_type<int16_t>();
    check_reg_tensor_type<uint16_t>();
    check_reg_tensor_type<int32_t>();
    check_reg_tensor_type<uint32_t>();
    check_reg_tensor_type<half>();
    check_reg_tensor_type<bfloat16_t>();
    check_reg_tensor_type<float>();
}

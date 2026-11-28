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
#include "tensor_api/experimental/arch/vector/compute/register_copy.h"

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201)) && \
    !defined(IMPL_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_COPY_IMPL_H)
#error register data copy implementation is missing for the target architecture
#endif

using namespace asc::te::experimental;

template <typename DataType>
__simd_callee__ inline void check_data_move()
{
    reg_tensor<DataType> source;
    reg_tensor<DataType> destination;
    auto result = move(destination, source);
    static_assert(AscendC::Std::is_same_v<decltype(result), reg_tensor<DataType>>);
}

template <typename DataType, int16_t Offset>
__simd_callee__ inline void check_mask_move()
{
    reg_tensor<DataType> source;
    auto result = move_mask<Offset>(source);
    static_assert(AscendC::Std::is_same_v<decltype(result), reg_tensor<bool>>);
    if constexpr (Offset > 0) {
        check_mask_move<DataType, Offset - 1>();
    }
}

__simd_vf__ inline void check_reg_data_copy()
{
    check_data_move<uint8_t>();
    check_data_move<int8_t>();
    check_data_move<uint16_t>();
    check_data_move<int16_t>();
    check_data_move<half>();
    check_data_move<bfloat16_t>();
    check_data_move<uint32_t>();
    check_data_move<int32_t>();
    check_data_move<float>();
    reg_tensor<bool> source;
    auto result = move(source);
    static_assert(AscendC::Std::is_same_v<decltype(result), reg_tensor<bool>>);
    constexpr int16_t last_halfword_block = 15;
    constexpr int16_t last_word_block = 31;
    check_mask_move<uint16_t, last_halfword_block>();
    check_mask_move<uint32_t, last_word_block>();
}

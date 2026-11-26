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
#include "tensor_api/experimental/arch/vector/basic_arithmetic.h"
#include "tensor_api/experimental/arch/vector/basic_arithmetic.h"

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201)) && \
    !defined(IMPL_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_BASIC_ARITHMETIC_IMPL_H)
#error basic arithmetic implementation is missing for the target architecture
#endif

using namespace asc::te::experimental;

template <typename DataType>
__simd_callee__ inline void check_arithmetic()
{
    reg_tensor<DataType> source0;
    reg_tensor<DataType> source1;
    DataType scalar{};
    auto sum = source0 + source1;
    sum = source0 + scalar;
    sum = scalar + source0;
    auto difference = source0 - source1;
    difference = source0 - scalar;
    difference = scalar - source0;
    auto maximum = max(source0, source1);
    maximum = max(source0, scalar);
    maximum = max(scalar, source0);
    auto minimum = min(source0, source1);
    minimum = min(source0, scalar);
    minimum = min(scalar, source0);
    if constexpr (detail::supports_neg_v<DataType>) {
        auto negative = neg(source0);
        auto magnitude = abs(source0);
    }
    if constexpr (detail::supports_mul_v<DataType>) {
        auto product = source0 * source1;
    }
    if constexpr (detail::supports_mul_scalar_v<DataType>) {
        auto product = source0 * scalar;
        product = scalar * source0;
    }
    if constexpr (detail::supports_div_v<DataType>) {
        auto quotient = source0 / source1;
        quotient = div(source0, source1);
    }
    if constexpr (detail::supports_relu_v<DataType>) {
        auto rectified = relu(source0);
    }
    static_assert(AscendC::Std::is_same_v<decltype(sum), reg_tensor<DataType>>);
}

template <typename DataType>
__simd_callee__ inline void check_float_arithmetic()
{
    reg_tensor<DataType> source0;
    reg_tensor<DataType> source1;
    auto natural_log = log(source0);
    natural_log = ln(source0);
    auto binary_log = log2(source0);
    auto decimal_log = log10(source0);
    auto exponential = exp(source0);
    auto root = sqrt(source0);
    auto rectified = prelu(source0, source1);
    DataType slope{};
    rectified = leaky_relu(source0, slope);
    auto quotient = div<false, div_algo::PRECISION_1ULP_FTZ_TRUE>(source0, source1);
    quotient = div<false, div_algo::PRECISION_1ULP_FTZ_FALSE>(source0, source1);
}

template <typename DataType>
__simd_callee__ inline void check_carry_arithmetic()
{
    reg_tensor<DataType> source0;
    reg_tensor<DataType> source1;
    reg_tensor<bool> carry;
    auto sum = add_carry(source0, source1);
    sum = add_carry(source0, source1, carry);
    auto difference = sub_carry(source0, source1);
    difference = sub_carry(source0, source1, carry);
    auto product = mull(source0, source1);
    static_assert(AscendC::Std::is_same_v<decltype(sum), reg_pair<DataType, bool>>);
    static_assert(AscendC::Std::is_same_v<decltype(product), reg_pair<DataType>>);
}

__simd_vf__ inline void check_reg_basic_arithmetic()
{
    check_arithmetic<int8_t>();
    check_arithmetic<uint8_t>();
    check_arithmetic<int16_t>();
    check_arithmetic<uint16_t>();
    check_arithmetic<int32_t>();
    check_arithmetic<uint32_t>();
    check_arithmetic<half>();
    check_arithmetic<bfloat16_t>();
    check_arithmetic<float>();
    check_float_arithmetic<half>();
    check_float_arithmetic<float>();
    check_carry_arithmetic<int32_t>();
    check_carry_arithmetic<uint32_t>();
    reg_tensor<float> source0;
    reg_tensor<float> source1;
    auto quotient = div<false, div_algo::DIFF_COMPENSATION>(source0, source1);
    quotient = div<false, div_algo::PRECISION_0ULP_FTZ_TRUE>(source0, source1);
    quotient = div<false, div_algo::PRECISION_0ULP_FTZ_FALSE>(source0, source1);
    quotient = div<true, div_algo::INTRINSIC>(source0, source1);
    quotient = div<true, div_algo::PRECISION_0ULP_FTZ_TRUE>(source0, source1);
    quotient = div<true, div_algo::PRECISION_0ULP_FTZ_FALSE>(source0, source1);
}

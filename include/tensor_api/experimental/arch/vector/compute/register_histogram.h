/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if !defined(ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_HISTOGRAM_H
#endif

#ifndef INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_HISTOGRAM_H
#define INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_HISTOGRAM_H

#include "tensor_api/experimental/arch/vector/defs/reg_enum.h"
#include "tensor_api/experimental/arch/vector/defs/register_tensor.h"

namespace asc {
namespace te {
namespace experimental {

struct histogram_options {
    histogram_half half = histogram_half::low;
    histogram_mode mode = histogram_mode::frequency;
};

inline constexpr histogram_options default_histogram_options{};

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_histogram
 * @brief Accumulates histogram or cumulative histogram counts into dst and returns the updated tensor.
 *
 * @override_id te_experimental_histograms__reg_tensor_T_R_const_reg_tensor_U_R__template_options_T_U
 * @archs 3510, 9201
 * @return The updated destination tensor with its original mask unchanged.
 * @param [in,out] dst Destination tensor containing the initial and accumulated counts.
 * @param [in] src Source operand.
 */
template <const histogram_options& options = default_histogram_options, typename T, typename U>
__simd_callee__ inline reg_tensor<T> histograms(reg_tensor<T>& dst, const reg_tensor<U>& src);

} // namespace experimental
} // namespace te
} // namespace asc

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201))
#include "impl/tensor_api/experimental/arch/vector/compute/register_histogram_impl.h"
#endif

#endif // INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_HISTOGRAM_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_HISTOGRAM_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_HISTOGRAM_H
#endif

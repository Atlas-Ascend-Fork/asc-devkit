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
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_MASK_H
#endif

/*!
 * \file register_mask.h
 * \brief Register mask construction and update interfaces.
 */
#ifndef INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_MASK_H
#define INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_MASK_H

#include "tensor_api/experimental/arch/vector/defs/register_tensor.h"
#include "tensor_api/experimental/arch/vector/defs/reg_enum.h"

namespace asc {
namespace te {
namespace experimental {

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_mask
 * @brief Creates a register mask with the specified pattern for an element type.
 *
 * @override_id te_experimental_make_mask__void__template_Pattern_DataType
 * @archs 3510, 9201
 * @return A register mask containing the selected pattern.
 * @note Supported element widths are b8, b16 and b32.
 */
template <mask_pattern Pattern = mask_pattern::all, typename DataType>
__simd_callee__ inline reg_tensor<bool> make_mask();

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_mask
 * @brief Creates a register mask with all lanes enabled for an element type.
 *
 * @override_id te_experimental_all_mask__void__template_DataType
 * @archs 3510, 9201
 * @return A register mask with all lanes enabled.
 * @note Supported element widths are b8, b16 and b32.
 */
template <typename DataType>
__simd_callee__ inline reg_tensor<bool> all_mask();

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_mask
 * @brief Creates a register mask with all lanes disabled for an element type.
 *
 * @override_id te_experimental_none_mask__void__template_DataType
 * @archs 3510, 9201
 * @return A register mask with all lanes disabled.
 * @note Supported element widths are b8, b16 and b32.
 */
template <typename DataType>
__simd_callee__ inline reg_tensor<bool> none_mask();

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_mask
 * @brief Creates a mask for a vector iteration and updates the remaining element count.
 *
 * @override_id te_experimental_update_mask__uint32_t_R__template_DataType
 * @archs 3510, 9201
 * @return A register mask enabling the elements processed by the current iteration.
 * @note Supported element widths are b8, b16 and b32.
 * @param [in,out] remain Element count used to create the mask and updated for the subsequent call.
 */
template <typename DataType>
__simd_callee__ inline reg_tensor<bool> update_mask(uint32_t& remain);

} // namespace experimental
} // namespace te
} // namespace asc

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201))
#include "impl/tensor_api/experimental/arch/vector/compute/register_mask_impl.h"
#endif

#endif // INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_MASK_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_MASK_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_MASK_H
#endif

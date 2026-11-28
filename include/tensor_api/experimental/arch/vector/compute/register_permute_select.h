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
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_PERMUTE_SELECT_H
#endif

#ifndef INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_PERMUTE_SELECT_H
#define INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_PERMUTE_SELECT_H

#include "version/asc_devkit_version.h"
#include "tensor_api/experimental/arch/vector/defs/register_tensor.h"

#include "tensor_api/experimental/arch/vector/defs/reg_enum.h"

namespace asc {
namespace te {
namespace experimental {
/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_permute_select
 * @brief Interleaves elements from two register tensors.
 *
 * @override_id te_experimental_interleave__reg_tensor_T_reg_tensor_T__template_T
 * @archs 3510, 9201
 * @tparam T Element type of the register tensors.
 * @return A pair of register tensors containing the interleaved elements. Both result tensors use a full mask.
 * @param [in] src0 First source register tensor.
 * @param [in] src1 Second source register tensor.
 */
template <typename T>
__simd_callee__ inline reg_pair<T> interleave(reg_tensor<T> src0, reg_tensor<T> src1);

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_permute_select
 * @brief Deinterleaves elements from two register tensors.
 *
 * @override_id te_experimental_deinterleave__reg_tensor_T_reg_tensor_T__template_T
 * @archs 3510, 9201
 * @tparam T Element type of the register tensors.
 * @return A pair of register tensors containing the deinterleaved elements. Both result tensors use a full mask.
 * @param [in] src0 First source register tensor.
 * @param [in] src1 Second source register tensor.
 */
template <typename T>
__simd_callee__ inline reg_pair<T> deinterleave(reg_tensor<T> src0, reg_tensor<T> src1);

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_permute_select
 * @brief Interleaves the lanes of two register masks for the specified element type.
 *
 * @override_id te_experimental_interleave__reg_tensor_bool_reg_tensor_bool__template_DataType
 * @archs 3510, 9201
 * @tparam DataType Element type whose width determines the mask lane granularity.
 * @return A register-mask pair containing the interleaved sequence. The first and second halves of the sequence are
 * stored in the first and second members, respectively.
 * @note Supported element widths are b8, b16 and b32.
 * @param [in] src0 First source register mask.
 * @param [in] src1 Second source register mask.
 */
template <typename DataType>
__simd_callee__ inline reg_pair<bool> interleave(reg_tensor<bool> src0, reg_tensor<bool> src1);

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_permute_select
 * @brief Deinterleaves the lanes of two register masks for the specified element type.
 *
 * @override_id te_experimental_deinterleave__reg_tensor_bool_reg_tensor_bool__template_DataType
 * @archs 3510, 9201
 * @tparam DataType Element type whose width determines the mask lane granularity.
 * @return A register-mask pair containing the deinterleaved sequence. The first member contains the even lanes, and the
 * second member contains the odd lanes.
 * @note Supported element widths are b8, b16 and b32.
 * @param [in] src0 First half of the interleaved register-mask sequence.
 * @param [in] src1 Second half of the interleaved register-mask sequence.
 */
template <typename DataType>
__simd_callee__ inline reg_pair<bool> deinterleave(reg_tensor<bool> src0, reg_tensor<bool> src1);

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_permute_select
 * @brief Selects elements from two register tensors according to a condition tensor.
 *
 * @override_id te_experimental_select__const_reg_tensor_bool_R_const_reg_tensor_T_R_const_reg_tensor_T_R__template_T
 * @archs 3510, 9201
 * @return A register tensor containing the selected elements.
 * @param [in] condition Per-element selection condition.
 * @param [in] src0 Source tensor selected when the corresponding condition is true.
 * @param [in] src1 Source tensor selected when the corresponding condition is false.
 * @note A condition value of 1 selects the corresponding element from src0; a value of 0 selects it from src1. This
 * condition does not zero unselected elements. The result preserves src0.mask.
 * @note Selection uses condition.reg; condition.mask and src1.mask do not select participating lanes.
 */
template <typename T>
__simd_callee__ inline reg_tensor<T> select(
    const reg_tensor<bool>& condition, const reg_tensor<T>& src0, const reg_tensor<T>& src1);

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_permute_select
 * @brief Packs a register tensor into the selected low or high half-width representation.
 *
 * @override_id te_experimental_pack__reg_tensor_T__template_part_T
 * @archs 3510, 9201
 * @tparam part Destination half selected for the pack operation.
 * @tparam T Source element type: bool, uint16_t, int16_t, uint32_t, or int32_t.
 * @return A register tensor containing the packed data with an all-active result mask.
 * @note part selects the destination half, not which bits to extract from each source element.
 * @param [in] src Source register tensor.
 */
template <high_low_part part = high_low_part::low, typename T>
__simd_callee__ inline decltype(auto) pack(reg_tensor<T> src);

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_permute_select
 * @brief Unpacks a register tensor from the selected low or high half-width representation.
 *
 * @override_id te_experimental_unpack__reg_tensor_T__template_part_T
 * @archs 3510, 9201
 * @tparam part Source half selected for the unpack operation.
 * @tparam T Element type of the source register tensor.
 * @return A register tensor containing the unpacked data with an all-active result mask.
 * @param [in] src Source register tensor.
 */
template <high_low_part part = high_low_part::low, typename T>
__simd_callee__ inline decltype(auto) unpack(reg_tensor<T> src);

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_permute_select
 * @brief Compresses the lanes selected by the source tensor mask into contiguous low lanes.
 *
 * @override_id te_experimental_compress__reg_tensor_T__template_T
 * @archs 3510, 9201
 * @tparam T Element type of the source register tensor.
 * @return A register tensor containing selected elements packed into low lanes.
 * @note The result mask is copied from src.mask, not adjusted to the compacted lane positions.
 * @param [in] src Source register tensor whose mask selects the lanes to compress.
 */
template <typename T>
__simd_callee__ inline reg_tensor<T> compress(reg_tensor<T> src);

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_permute_select
 * @brief Generates exclusive prefix-count indexes from a mask.
 *
 * @override_id te_experimental_expand__reg_tensor_bool__template_T
 * @archs 3510, 9201
 * @tparam T Element type of the generated index register tensor.
 * @return A register tensor containing prefix-count indexes with an all-active result mask.
 * @param [in] mask Lane-selection mask.
 */
template <typename T>
__simd_callee__ inline reg_tensor<T> expand(reg_tensor<bool> mask);

} // namespace experimental
} // namespace te
} // namespace asc

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201))
#include "impl/tensor_api/experimental/arch/vector/compute/register_permute_select_impl.h"
#endif

#endif // INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_PERMUTE_SELECT_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_PERMUTE_SELECT_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_PERMUTE_SELECT_H
#endif

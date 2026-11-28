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
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_GATHER_H
#endif

/*!
 * \file register_load_gather.h
 * \brief Register data load and broadcast interfaces.
 */
#ifndef INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_GATHER_H
#define INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_GATHER_H

#include "version/asc_devkit_version.h"
#include "tensor_api/experimental/arch/vector/defs/register_tensor.h"
#include "tensor_api/experimental/arch/vector/defs/reg_enum.h"

namespace asc {
namespace te {
namespace experimental {

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_load
 * @brief Gathers elements or data blocks from a UB tensor at the specified coordinate.
 *
 * @override_id
 *
 *
 * te_experimental_load_gather__const_Tensor_R_const_Coord_R_const_reg_tensor_Index_const_reg_tensor_bo
 * ol_R__template_mode_Tensor_Coord_Index
 * @archs 3510, 9201
 * @return A register tensor containing the gathered data.
 * @param [in] tensor Source tensor located in UB.
 * @param [in] coord Coordinate used as the base source position.
 * @param [in] index Register tensor containing the source offsets relative to the base position.
 * @param [in] gather_mask Register mask that selects the values to gather.
 * @note The active-lane mask is taken from gather_mask.reg. In element mode, a value of 1 selects the corresponding
 * element; a value of 0 excludes it. Unselected elements are set to zero in element mode. In datablock mode, mask bits
 * 0, 4, 8, 12, 16, 20, 24, and 28 select the eight data blocks. The result preserves gather_mask.reg as its active-lane
 * mask.
 */
template <gather_mode mode = gather_mode::element, typename Tensor, typename Coord, typename Index>
__simd_callee__ inline auto load_gather(
    const Tensor& tensor, const Coord& coord, const reg_tensor<Index> index, const reg_tensor<bool>& gather_mask);

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_load
 * @brief Gathers elements or data blocks from the beginning of a UB tensor.
 *
 * @override_id
 *
 *
 * te_experimental_load_gather__const_Tensor_R_const_reg_tensor_Index_const_reg_tensor_bool_R__template
 * _mode_Tensor_Index
 * @archs 3510, 9201
 * @return A register tensor containing the gathered data.
 * @param [in] tensor Source tensor located in UB.
 * @param [in] index Register tensor containing the source offsets.
 * @param [in] gather_mask Register mask that selects the values to gather.
 * @note The active-lane mask is taken from gather_mask.reg. In element mode, a value of 1 selects the corresponding
 * element; a value of 0 excludes it. Unselected elements are set to zero in element mode. In datablock mode, mask bits
 * 0, 4, 8, 12, 16, 20, 24, and 28 select the eight data blocks. The result preserves gather_mask.reg as its active-lane
 * mask.
 */
template <gather_mode mode = gather_mode::element, typename Tensor, typename Index>
__simd_callee__ inline auto load_gather(
    const Tensor& tensor, const reg_tensor<Index> index, const reg_tensor<bool>& gather_mask);

} // namespace experimental
} // namespace te
} // namespace asc

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201 && ASC_DEVKIT_VERSION_NUM >= 90300000))
#include "impl/tensor_api/experimental/arch/vector/memory/register_load_gather_impl.h"
#endif

#endif // INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_GATHER_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_GATHER_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_GATHER_H
#endif

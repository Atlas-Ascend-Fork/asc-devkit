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
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_STORE_SCATTER_H
#endif

/*!
 * \file register_store_scatter.h
 * \brief Register data store interfaces.
 */
#ifndef INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_STORE_SCATTER_H
#define INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_STORE_SCATTER_H

#include "version/asc_devkit_version.h"
#include "tensor_api/experimental/arch/vector/defs/register_tensor.h"
#include "tensor_api/experimental/arch/vector/defs/reg_enum.h"

namespace asc {
namespace te {
namespace experimental {

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_store
 * @brief Scatters register data from active source lanes to a UB tensor.
 *
 * @override_id
 * te_experimental_store_scatter__Tensor_R_const_reg_tensor_T_R_const_reg_tensor_Index_R__template_Tensor_T_Index
 * @archs 3510, 9201
 * @note The tensor and source register element types must match.
 * @param [out] tensor Destination tensor located in UB.
 * @param [in] src Source register tensor. Its active-lane mask selects the elements to store.
 * @param [in] index Register tensor containing the destination offsets.
 * @note The active-lane mask is taken from src.mask. A value of 1 selects the corresponding source element for storage;
 * a value of 0 excludes it. Unselected elements are not written; the corresponding destination memory is unchanged.
 */
template <typename Tensor, typename T, typename Index>
__simd_callee__ inline void store_scatter(Tensor& tensor, const reg_tensor<T>& src, const reg_tensor<Index>& index);

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_store
 * @brief Scatters register data from active source lanes to a UB tensor at the specified coordinate.
 *
 * @override_id
 *
 *
 * te_experimental_store_scatter__Tensor_R_const_Coord_R_const_reg_tensor_T_R_const_reg_tensor_Index_R_
 * _template_Tensor_Coord_T_Index
 * @archs 3510, 9201
 * @note The tensor and source register element types must match.
 * @param [out] tensor Destination tensor located in UB.
 * @param [in] coord Coordinate used as the base destination position.
 * @param [in] src Source register tensor. Its active-lane mask selects the elements to store.
 * @param [in] index Register tensor containing the destination offsets relative to the base position.
 * @note The active-lane mask is taken from src.mask. A value of 1 selects the corresponding source element for storage;
 * a value of 0 excludes it. Unselected elements are not written; the corresponding destination memory is unchanged.
 */
template <typename Tensor, typename Coord, typename T, typename Index>
__simd_callee__ inline void store_scatter(
    Tensor& tensor, const Coord& coord, const reg_tensor<T>& src, const reg_tensor<Index>& index);

} // namespace experimental
} // namespace te
} // namespace asc

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201 && ASC_DEVKIT_VERSION_NUM >= 90300000))
#include "impl/tensor_api/experimental/arch/vector/memory/register_store_scatter_impl.h"
#endif

#endif // INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_STORE_SCATTER_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_STORE_SCATTER_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_STORE_SCATTER_H
#endif

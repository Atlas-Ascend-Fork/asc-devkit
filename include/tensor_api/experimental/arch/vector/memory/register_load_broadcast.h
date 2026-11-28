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
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_BROADCAST_H
#endif

/*!
 * \file register_load_broadcast.h
 * \brief Register data load and broadcast interfaces.
 */
#ifndef INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_BROADCAST_H
#define INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_BROADCAST_H

#include "version/asc_devkit_version.h"
#include "tensor_api/experimental/arch/vector/defs/register_tensor.h"
#include "tensor_api/experimental/arch/vector/defs/reg_enum.h"

namespace asc {
namespace te {
namespace experimental {

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_load
 * @brief Loads and broadcasts data from a UB tensor at the specified coordinate.
 *
 * @override_id te_experimental_load_broadcast__const_Tensor_R_const_Coord_R__template_brc_mode_Tensor_Coord
 * @archs 3510, 9201
 * @return A register tensor populated according to the selected broadcast mode.
 * @param [in] tensor Source tensor located in UB.
 * @param [in] coord Coordinate of the first source element.
 * @note The source tensor element type must be supported by brc_mode. An int4b_t source
 * returns reg_tensor<int4x2_t>; other supported source types return a register tensor
 * with the corresponding element type. Only the returned reg member is initialized;
 * call with_mask before using the result in an operation that reads its mask.
 */
template <broadcast_mode brc_mode = broadcast_mode::elem, typename Tensor, typename Coord>
__simd_callee__ inline auto load_broadcast(const Tensor& tensor, const Coord& coord);

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_load
 * @brief Loads and broadcasts data from the beginning of a UB tensor.
 *
 * @override_id te_experimental_load_broadcast__const_Tensor_R__template_brc_mode_Tensor
 * @archs 3510, 9201
 * @return A register tensor populated according to the selected broadcast mode.
 * @param [in] tensor Source tensor located in UB.
 * @note The source tensor element type must be supported by brc_mode. An int4b_t source
 * returns reg_tensor<int4x2_t>; other supported source types return a register tensor
 * with the corresponding element type. Only the returned reg member is initialized;
 * call with_mask before using the result in an operation that reads its mask.
 */
template <broadcast_mode brc_mode = broadcast_mode::elem, typename Tensor>
__simd_callee__ inline auto load_broadcast(const Tensor& tensor);

} // namespace experimental
} // namespace te
} // namespace asc

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201 && ASC_DEVKIT_VERSION_NUM >= 90300000))
#include "impl/tensor_api/experimental/arch/vector/memory/register_load_broadcast_impl.h"
#endif

#endif // INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_BROADCAST_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_BROADCAST_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_BROADCAST_H
#endif

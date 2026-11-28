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
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_ALIGN_H
#endif

/*!
 * \file register_load_align.h
 * \brief Register data load and broadcast interfaces.
 */
#ifndef INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_ALIGN_H
#define INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_ALIGN_H

#include "version/asc_devkit_version.h"
#include "tensor_api/experimental/arch/vector/defs/register_tensor.h"
#include "tensor_api/experimental/arch/vector/defs/reg_enum.h"

namespace asc {
namespace te {
namespace experimental {

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_load
 * @brief Loads data from a UB tensor at the specified coordinate into one or two register tensors.
 *
 * @override_id te_experimental_load__const_Tensor_R_const_Coord_R__template_sideband_mode_Tensor_Coord
 * @archs 3510, 9201
 * @return A register pair for deintlv mode; otherwise, a register tensor.
 * @param [in] tensor Source tensor located in UB.
 * @param [in] coord Coordinate of the first source element.
 */
template <load_sideband_mode sideband_mode = load_sideband_mode::direct, typename Tensor, typename Coord>
__simd_callee__ inline auto load(const Tensor& tensor, const Coord& coord);

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_load
 * @brief Loads data from the beginning of a UB tensor into one or two register tensors.
 *
 * @override_id te_experimental_load__const_Tensor_R__template_sideband_mode_Tensor
 * @archs 3510, 9201
 * @return A register pair for deintlv mode; otherwise, a register tensor.
 * @param [in] tensor Source tensor located in UB.
 */
template <load_sideband_mode sideband_mode = load_sideband_mode::direct, typename Tensor>
__simd_callee__ inline auto load(const Tensor& tensor);

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_load
 * @brief Loads packed mask bits from a UB tensor at the specified coordinate.
 *
 * @override_id te_experimental_load__const_Tensor_R_const_Coord_R__template_RegDataType_sideband_mode_Tensor_Coord
 * @archs 3510, 9201
 * @return A Boolean register tensor containing the loaded mask bits in its reg member.
 * @param [in] tensor Source uint32_t tensor located in UB.
 * @param [in] coord Coordinate of the first source word.
 * @note RegDataType must be bool. Supported modes are direct, upsample, and downsample.
 * Direct mode loads packed bits unchanged; upsample repeats each bit twice; downsample retains even-indexed bits.
 * The effective source address must be 16-byte aligned for upsample and 32-byte aligned otherwise.
 * For a 256-byte vector data register, these modes read 32, 16, and 64 bytes, respectively.
 * The full load must remain within the source buffer.
 */
template <
    typename RegDataType, load_sideband_mode sideband_mode = load_sideband_mode::direct, typename Tensor,
    typename Coord>
__simd_callee__ inline auto load(const Tensor& tensor, const Coord& coord);

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_load
 * @brief Loads packed mask bits from the beginning of a UB tensor.
 *
 * @override_id te_experimental_load__const_Tensor_R__template_RegDataType_sideband_mode_Tensor
 * @archs 3510, 9201
 * @return A Boolean register tensor containing the loaded mask bits in its reg member.
 * @param [in] tensor Source uint32_t tensor located in UB.
 * @note RegDataType must be bool. Supported modes are direct, upsample, and downsample.
 * Direct mode loads packed bits unchanged; upsample repeats each bit twice; downsample retains even-indexed bits.
 * The source address must be 16-byte aligned for upsample and 32-byte aligned otherwise.
 * For a 256-byte vector data register, these modes read 32, 16, and 64 bytes, respectively.
 * The full load must remain within the source buffer.
 */
template <typename RegDataType, load_sideband_mode sideband_mode = load_sideband_mode::direct, typename Tensor>
__simd_callee__ inline auto load(const Tensor& tensor);

} // namespace experimental
} // namespace te
} // namespace asc

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201 && ASC_DEVKIT_VERSION_NUM >= 90300000))
#include "impl/tensor_api/experimental/arch/vector/memory/register_load_align_impl.h"
#endif

#endif // INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_ALIGN_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_ALIGN_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_ALIGN_H
#endif

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
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_STORE_ALIGN_H
#endif

/*!
 * \file register_store_align.h
 * \brief Register data store interfaces.
 */
#ifndef INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_STORE_ALIGN_H
#define INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_STORE_ALIGN_H

#include "version/asc_devkit_version.h"
#include "tensor_api/experimental/arch/vector/defs/register_tensor.h"
#include "tensor_api/experimental/arch/vector/defs/reg_enum.h"

namespace asc {
namespace te {
namespace experimental {

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_store
 * @brief Stores register data to a UB tensor at the specified coordinate.
 *
 * @override_id
 *
 *
 * te_experimental_store__Tensor_R_const_Coord_R_const_reg_tensor_RegDataType_R__template_sideband_mode
 * _Tensor_Coord_RegDataType
 * @archs 3510, 9201
 * @note For vector data registers, the tensor and register element types must match, except that an int4b_t
 * tensor requires an int4x2_t register. Mask registers (reg_tensor<bool>) support only store_sideband_mode::direct
 * and store_sideband_mode::pack, and only to uint8_t, uint16_t, or uint32_t tensors. Both modes ignore src.mask.
 * @param [out] tensor Destination tensor located in UB.
 * @param [in] coord Coordinate of the first destination element.
 * @param [in] src Source register tensor.
 */
template <
    store_sideband_mode sideband_mode = store_sideband_mode::direct, typename Tensor, typename Coord,
    typename RegDataType>
__simd_callee__ inline void store(Tensor& tensor, const Coord& coord, const reg_tensor<RegDataType>& src);

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_store
 * @brief Stores register data at the beginning of a UB tensor.
 *
 * @override_id
 * te_experimental_store__Tensor_R_const_reg_tensor_RegDataType_R__template_sideband_mode_Tensor_RegDataType
 * @archs 3510, 9201
 * @note For vector data registers, the tensor and register element types must match, except that an int4b_t
 * tensor requires an int4x2_t register. Mask registers (reg_tensor<bool>) support only store_sideband_mode::direct
 * and store_sideband_mode::pack, and only to uint8_t, uint16_t, or uint32_t tensors. Both modes ignore src.mask.
 * @param [out] tensor Destination tensor located in UB.
 * @param [in] src Source register tensor.
 */
template <store_sideband_mode sideband_mode = store_sideband_mode::direct, typename Tensor, typename RegDataType>
__simd_callee__ inline void store(Tensor& tensor, const reg_tensor<RegDataType>& src);

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_store
 * @brief Interleaves two register tensors and stores the result at the specified coordinate of a UB tensor.
 *
 * @override_id
 *
 *
 * te_experimental_store__Tensor_R_const_Coord_R_const_reg_tensor_RegDataType_R_const_reg_tensor_RegDat
 * aType_R__template_sideband_mode_Tensor_Coord_RegDataType
 * @archs 3510, 9201
 * @note This overload supports only store_sideband_mode::intlv. The tensor and register element types must match.
 * @param [out] tensor Destination tensor located in UB.
 * @param [in] coord Coordinate of the first destination element.
 * @param [in] src0 First source register tensor.
 * @param [in] src1 Second source register tensor.
 */
template <
    store_sideband_mode sideband_mode = store_sideband_mode::intlv, typename Tensor, typename Coord,
    typename RegDataType>
__simd_callee__ inline void store(
    Tensor& tensor, const Coord& coord, const reg_tensor<RegDataType>& src0, const reg_tensor<RegDataType>& src1);

/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_store
 * @brief Interleaves two register tensors and stores the result at the beginning of a UB tensor.
 *
 * @override_id
 *
 *
 * te_experimental_store__Tensor_R_const_reg_tensor_RegDataType_R_const_reg_tensor_RegDataType_R__templ
 * ate_sideband_mode_Tensor_RegDataType
 * @archs 3510, 9201
 * @note This overload supports only store_sideband_mode::intlv. The tensor and register element types must match.
 * @param [out] tensor Destination tensor located in UB.
 * @param [in] src0 First source register tensor.
 * @param [in] src1 Second source register tensor.
 */
template <store_sideband_mode sideband_mode = store_sideband_mode::intlv, typename Tensor, typename RegDataType>
__simd_callee__ inline void store(
    Tensor& tensor, const reg_tensor<RegDataType>& src0, const reg_tensor<RegDataType>& src1);

} // namespace experimental
} // namespace te
} // namespace asc

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201 && ASC_DEVKIT_VERSION_NUM >= 90300000))
#include "impl/tensor_api/experimental/arch/vector/memory/register_store_align_impl.h"
#endif

#endif // INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_STORE_ALIGN_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_STORE_ALIGN_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_STORE_ALIGN_H
#endif

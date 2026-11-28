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
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_GATHER_H
#endif

/*!
 * \file register_gather.h
 * \brief Register gather interface.
 */
#ifndef INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_GATHER_H
#define INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_GATHER_H

#include "tensor_api/experimental/arch/vector/defs/register_tensor.h"

namespace asc {
namespace te {
namespace experimental {
/**
 * @ingroup simd_api.tensor_api.reg_vector_compute.register_gather
 * @brief Gathers elements from a source register tensor according to per-lane indexes.
 *
 * @override_id
 * te_experimental_gather__const_reg_tensor_DataType_R_const_reg_tensor_IndexType_R__template_DataType_IndexType
 * @archs 3510, 9201
 * @tparam DataType Element type of the source and result register tensors. Supported types are int8_t, uint8_t,
 * fp8_e4m3fn_t, hifloat8_t, fp8_e5m2_t, fp8_e8m0_t, int16_t, uint16_t, half, bfloat16_t, int32_t, uint32_t, and float.
 * @tparam IndexType Unsigned integer type with the same element width as DataType: uint8_t, uint16_t, or uint32_t.
 * @return A register tensor containing the gathered data with an all-active result mask.
 * @note Indexes are element positions, not byte offsets. Each result lane is src.reg[index.reg[i] % VL_T], where
 * VL_T is the number of DataType elements in one vector register.
 * @note Neither src.mask nor index.mask selects participating lanes. All index lanes and all referenced source
 * elements must be initialized. Bind a new result mask before consuming only part of the result.
 * @note This operation is effective only on AIV.
 * @param [in] src Source register tensor.
 * @param [in] index Register tensor containing per-lane source indexes.
 */
template <typename DataType, typename IndexType>
__simd_callee__ inline reg_tensor<DataType> gather(const reg_tensor<DataType>& src, const reg_tensor<IndexType>& index);

} // namespace experimental
} // namespace te
} // namespace asc

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201))
#include "impl/tensor_api/experimental/arch/vector/compute/register_gather_impl.h"
#endif

#endif // INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_GATHER_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_GATHER_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_GATHER_H
#endif

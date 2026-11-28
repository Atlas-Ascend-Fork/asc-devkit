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
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_UNALIGN_H
#endif

/*!
 * \file register_load_unalign.h
 * \brief Register data load and broadcast interfaces.
 */
#ifndef INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_UNALIGN_H
#define INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_UNALIGN_H

#include "version/asc_devkit_version.h"
#include "tensor_api/experimental/arch/vector/defs/register_tensor.h"
#include "tensor_api/experimental/arch/vector/defs/reg_enum.h"

namespace asc {
namespace te {
namespace experimental {

/**
 * @brief Maintains the hardware state for an unaligned UB-to-register load.
 * @note load() does not advance the prepared state. Call prepare() for each new
 * source address, using a tensor whose beginning matches the load address.
 */
struct unalign_load {
    __simd_callee__ unalign_load() = default;

    /**
     * @brief Prepares an unaligned load stream at the beginning of a UB tensor.
     * @param src_base Tensor whose beginning is the first address of the stream.
     */
    template <typename Tensor>
    __simd_callee__ inline void prepare(const Tensor& src_base);

    /**
     * @brief Loads one full vector register from the prepared source address.
     * @param src_tile Tensor whose beginning is the source address for this load.
     * @return One register tensor containing the loaded data. An int4b_t source
     * returns reg_tensor<int4x2_t>.
     */
    template <typename Tensor>
    __simd_callee__ inline auto load(const Tensor& src_tile);

    /**
     * @brief Loads one full vector register from a UB tensor at the specified coordinate.
     * @param src_tile Source tensor in the prepared unaligned stream.
     * @param coord Coordinate of the first source element.
     * @return One register tensor containing the loaded data. An int4b_t source
     * returns reg_tensor<int4x2_t>.
     */
    template <typename Tensor, typename Coord>
    __simd_callee__ inline auto load(const Tensor& src_tile, const Coord& coord);

private:
    vector_load_unalign ureg;
};

} // namespace experimental
} // namespace te
} // namespace asc

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201 && ASC_DEVKIT_VERSION_NUM >= 90300000))
#include "impl/tensor_api/experimental/arch/vector/memory/register_load_unalign_impl.h"
#endif

#endif // INCLUDE_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_UNALIGN_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_UNALIGN_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_UNALIGN_H
#endif

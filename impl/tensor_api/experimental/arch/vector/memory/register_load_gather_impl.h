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
#warning "impl/tensor_api/experimental/arch/vector/memory/register_load_gather_impl.h is an internal header file" \
    "and must not be used directly. Functions or variables defined in this file may be removed in the future." \
    "Please use \"#include \"tensor_api/experimental/arch/vector/memory/register_load_gather.h\"\" and use" \
    "public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_GATHER_IMPL_H
#endif

#ifndef IMPL_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_GATHER_IMPL_H
#define IMPL_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_GATHER_IMPL_H

#include "impl/tensor_api/experimental/arch/utils/reg_utils.h"
#include "tensor_api/tensor/tensor_interface.h"

namespace asc {
namespace te {
namespace experimental {

template <gather_mode mode, typename Tensor, typename Coord, typename Index>
__simd_callee__ inline auto load_gather(
    const Tensor& tensor, const Coord& coord, const reg_tensor<Index> index, const reg_tensor<bool>& gather_mask)
{
    using engine_type = typename Tensor::engine_type;
    using data_type = typename Tensor::data_type;
    using reg_data_type = detail::get_reg_data_type<data_type>;
    static_assert(
        Std::is_same_v<get_mem_location<engine_type>, location::ub>, "load_gather only supports tensors located in UB");
    TENSOR_API_DEBUG_CHECK(debug_check_coord, tensor.layout(), coord, "load_gather");

    auto src_engine = tensor.engine() + tensor.layout()(coord);
    reg_tensor<reg_data_type> dst;
    if constexpr (mode == gather_mode::element) {
        asc_gather(dst.reg, src_engine.begin().get(), index.reg, gather_mask.reg);
    } else {
        asc_gather_datablock(dst.reg, src_engine.begin().get(), index.reg, gather_mask.reg);
    }
    dst.mask = gather_mask.reg;
    return dst;
}

template <gather_mode mode, typename Tensor, typename Index>
__simd_callee__ inline auto load_gather(
    const Tensor& tensor, const reg_tensor<Index> index, const reg_tensor<bool>& gather_mask)
{
    using engine_type = typename Tensor::engine_type;
    using data_type = typename Tensor::data_type;
    using reg_data_type = detail::get_reg_data_type<data_type>;
    static_assert(
        Std::is_same_v<get_mem_location<engine_type>, location::ub>, "load_gather only supports tensors located in UB");
    reg_tensor<reg_data_type> dst;
    if constexpr (mode == gather_mode::element) {
        asc_gather(dst.reg, tensor.engine().begin().get(), index.reg, gather_mask.reg);
    } else {
        asc_gather_datablock(dst.reg, tensor.engine().begin().get(), index.reg, gather_mask.reg);
    }
    dst.mask = gather_mask.reg;
    return dst;
}

} // namespace experimental
} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_GATHER_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_GATHER_IMPL_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_GATHER_IMPL_H
#endif

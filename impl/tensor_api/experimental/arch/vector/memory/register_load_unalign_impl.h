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
#warning "impl/tensor_api/experimental/arch/vector/memory/register_load_unalign_impl.h is an internal header file" \
    "and must not be used directly. Functions or variables defined in this file may be removed in the future." \
    "Please use \"#include \"tensor_api/experimental/arch/vector/memory/register_load_unalign.h\"\" and use" \
    "public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_UNALIGN_IMPL_H
#endif

#ifndef IMPL_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_UNALIGN_IMPL_H
#define IMPL_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_UNALIGN_IMPL_H

#include "impl/tensor_api/experimental/arch/utils/reg_utils.h"
#include "tensor_api/tensor/tensor_interface.h"

namespace asc {
namespace te {
namespace experimental {

template <typename Tensor>
__simd_callee__ inline void unalign_load::prepare(const Tensor& src_base)
{
    using engine_type = typename Tensor::engine_type;
    using data_type = typename Tensor::data_type;
    using reg_data_type = detail::get_reg_data_type<data_type>;
    static_assert(
        Std::is_same_v<get_mem_location<engine_type>, location::ub>,
        "unalign_load only supports tensors located in UB");
    static_assert(
        detail::is_b4_type_v<reg_data_type> || detail::is_b8_type_v<reg_data_type> ||
            detail::is_b16_type_v<reg_data_type> || detail::is_b32_type_v<reg_data_type> ||
            detail::is_b64_type_v<reg_data_type>,
        "unalign_load supports b4, b8, b16, b32, and b64 element types only");

    asc_loadunalign_pre(ureg, src_base.engine().begin().get());
}

template <typename Tensor>
__simd_callee__ inline auto unalign_load::load(const Tensor& src_tile)
{
    using engine_type = typename Tensor::engine_type;
    using data_type = typename Tensor::data_type;
    using reg_data_type = detail::get_reg_data_type<data_type>;
    static_assert(
        Std::is_same_v<get_mem_location<engine_type>, location::ub>,
        "unalign_load only supports tensors located in UB");
    static_assert(
        detail::is_b4_type_v<reg_data_type> || detail::is_b8_type_v<reg_data_type> ||
            detail::is_b16_type_v<reg_data_type> || detail::is_b32_type_v<reg_data_type> ||
            detail::is_b64_type_v<reg_data_type>,
        "unalign_load supports b4, b8, b16, b32, and b64 element types only");

    reg_tensor<reg_data_type> dst;
    auto src_addr = src_tile.engine().begin().get();
    asc_loadunalign(dst.reg, ureg, src_addr);
    return dst;
}

template <typename Tensor, typename Coord>
__simd_callee__ inline auto unalign_load::load(const Tensor& src_tile, const Coord& coord)
{
    using engine_type = typename Tensor::engine_type;
    using data_type = typename Tensor::data_type;
    using reg_data_type = detail::get_reg_data_type<data_type>;
    static_assert(
        Std::is_same_v<get_mem_location<engine_type>, location::ub>,
        "unalign_load only supports tensors located in UB");
    static_assert(
        detail::is_b4_type_v<reg_data_type> || detail::is_b8_type_v<reg_data_type> ||
            detail::is_b16_type_v<reg_data_type> || detail::is_b32_type_v<reg_data_type> ||
            detail::is_b64_type_v<reg_data_type>,
        "unalign_load supports b4, b8, b16, b32, and b64 element types only");
    TENSOR_API_DEBUG_CHECK(debug_check_coord, src_tile.layout(), coord, "unalign_load::load");

    reg_tensor<reg_data_type> dst;
    auto src_addr = (src_tile.data() + src_tile.layout()(coord)).get();
    asc_loadunalign(dst.reg, ureg, src_addr);
    return dst;
}

} // namespace experimental
} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_UNALIGN_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_UNALIGN_IMPL_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_UNALIGN_IMPL_H
#endif

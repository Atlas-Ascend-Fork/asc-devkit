/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef VERIFY_SINGLE_HEADER
#include "tensor_api/tensor.h"
#endif
#include "tensor_api/experimental/arch/vector/reg_data_load.h"
#include "tensor_api/experimental/arch/vector/reg_data_load.h"

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201)) && \
    !defined(IMPL_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REG_DATA_LOAD_IMPL_H)
#error register data load implementation is missing for the target architecture
#endif

using namespace asc::te;
using namespace asc::te::experimental;
namespace reg_detail = asc::te::experimental::detail;

template <load_sideband_mode Mode, typename Tensor>
__simd_callee__ inline void check_load_mode(const Tensor& tensor)
{
    auto first = load<Mode>(tensor);
    auto positioned = load<Mode>(tensor, make_coord(0));
    using data_type = reg_detail::get_reg_data_type<typename Tensor::data_type>;
    using result_type =
        Std::conditional_t<Mode == load_sideband_mode::deintlv, reg_pair<data_type>, reg_tensor<data_type>>;
    static_assert(Std::is_same_v<decltype(first), result_type>);
    static_assert(Std::is_same_v<decltype(positioned), result_type>);
}

template <broadcast_mode Mode, typename Tensor>
__simd_callee__ inline void check_broadcast_mode(const Tensor& tensor)
{
    auto first = load_broadcast<Mode>(tensor);
    auto positioned = load_broadcast<Mode>(tensor, make_coord(0));
    using result_type = reg_tensor<reg_detail::get_reg_data_type<typename Tensor::data_type>>;
    static_assert(Std::is_same_v<decltype(first), result_type>);
    static_assert(Std::is_same_v<decltype(positioned), result_type>);
}

template <gather_mode Mode, typename Index, typename Tensor>
__simd_callee__ inline void check_load_gather(const Tensor& tensor)
{
    reg_tensor<Index> index;
    reg_tensor<bool> mask;
    auto first = load_gather<Mode>(tensor, index, mask);
    auto positioned = load_gather<Mode>(tensor, make_coord(0), index, mask);
    using result_type = reg_tensor<reg_detail::get_reg_data_type<typename Tensor::data_type>>;
    static_assert(Std::is_same_v<decltype(first), result_type>);
    static_assert(Std::is_same_v<decltype(positioned), result_type>);
}

template <typename DataType>
__aicore__ inline void check_reg_data_load(uint32_t address)
{
    constexpr uint32_t tensor_elements = 1024;
    auto tensor = make_tensor(
        make_mem_ptr<location::ub, DataType>(address), make_layout(make_shape(tensor_elements), make_stride(1)));
    using reg_data_type = reg_detail::get_reg_data_type<DataType>;
    check_load_mode<load_sideband_mode::direct>(tensor);
    if constexpr (!Std::is_same_v<DataType, int64_t>) {
        check_load_mode<load_sideband_mode::deintlv>(tensor);
        check_load_mode<load_sideband_mode::unpack>(tensor);
        check_broadcast_mode<broadcast_mode::elem>(tensor);
        check_broadcast_mode<broadcast_mode::datablock>(tensor);
    }
    if constexpr (
        reg_detail::is_b4_type_v<reg_data_type> || reg_detail::is_b8_type_v<reg_data_type> ||
        reg_detail::is_b16_type_v<reg_data_type>) {
        check_load_mode<load_sideband_mode::downsample>(tensor);
        check_load_mode<load_sideband_mode::upsample>(tensor);
    }
    if constexpr (reg_detail::is_b4_type_v<reg_data_type> || reg_detail::is_b8_type_v<reg_data_type>) {
        check_load_mode<load_sideband_mode::unpack4>(tensor);
    }
    if constexpr (reg_detail::is_b16_type_v<reg_data_type> || reg_detail::is_b32_type_v<reg_data_type>) {
        check_broadcast_mode<broadcast_mode::elem2datablock>(tensor);
        check_load_gather<gather_mode::element, uint32_t>(tensor);
    }
    if constexpr (reg_detail::is_b8_type_v<reg_data_type> || reg_detail::is_b16_type_v<reg_data_type>) {
        check_load_gather<gather_mode::element, uint16_t>(tensor);
    }
    check_load_gather<gather_mode::data_block, uint32_t>(tensor);
    unalign_load stream;
    stream.prepare(tensor);
    auto first = stream.load(tensor);
    auto positioned = stream.load(tensor, make_coord(0));
    static_assert(Std::is_same_v<decltype(first), reg_tensor<reg_data_type>>);
    static_assert(Std::is_same_v<decltype(positioned), reg_tensor<reg_data_type>>);
}

template void check_reg_data_load<int8_t>(uint32_t);
template void check_reg_data_load<uint8_t>(uint32_t);
template void check_reg_data_load<int16_t>(uint32_t);
template void check_reg_data_load<uint16_t>(uint32_t);
template void check_reg_data_load<half>(uint32_t);
template void check_reg_data_load<bfloat16_t>(uint32_t);
template void check_reg_data_load<int32_t>(uint32_t);
template void check_reg_data_load<uint32_t>(uint32_t);
template void check_reg_data_load<float>(uint32_t);
template void check_reg_data_load<int64_t>(uint32_t);
template void check_reg_data_load<hifloat8_t>(uint32_t);
template void check_reg_data_load<fp8_e4m3fn_t>(uint32_t);
template void check_reg_data_load<fp8_e5m2_t>(uint32_t);
template void check_reg_data_load<fp8_e8m0_t>(uint32_t);
template void check_reg_data_load<fp4x2_e1m2_t>(uint32_t);
template void check_reg_data_load<fp4x2_e2m1_t>(uint32_t);
template void check_reg_data_load<int4b_t>(uint32_t);

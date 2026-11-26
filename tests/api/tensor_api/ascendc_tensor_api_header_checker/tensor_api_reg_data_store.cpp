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
#include "tensor_api/experimental/arch/vector/reg_data_store.h"
#include "tensor_api/experimental/arch/vector/reg_data_store.h"

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201)) && \
    !defined(IMPL_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REG_DATA_STORE_IMPL_H)
#error register data store implementation is missing for the target architecture
#endif

using namespace asc::te;
using namespace asc::te::experimental;
namespace reg_detail = asc::te::experimental::detail;

template <store_sideband_mode Mode, typename Tensor>
__simd_callee__ inline void check_store_mode(Tensor& tensor)
{
    using reg_data_type = reg_detail::get_reg_data_type<typename Tensor::data_type>;
    reg_tensor<reg_data_type> source;
    store<Mode>(tensor, source);
    store<Mode>(tensor, make_coord(0), source);
}

template <typename Index, typename Tensor>
__simd_callee__ inline void check_store_scatter(Tensor& tensor)
{
    reg_tensor<typename Tensor::data_type> source;
    reg_tensor<Index> index;
    store_scatter(tensor, source, index);
    store_scatter(tensor, make_coord(0), source, index);
}

template <typename DataType>
__aicore__ inline void check_reg_data_store(uint32_t address)
{
    constexpr uint32_t tensor_elements = 1024;
    auto tensor = make_tensor(
        make_mem_ptr<location::ub, DataType>(address), make_layout(make_shape(tensor_elements), make_stride(1)));
    using reg_data_type = reg_detail::get_reg_data_type<DataType>;
    reg_tensor<reg_data_type> source0;
    reg_tensor<reg_data_type> source1;
    if constexpr (!Std::is_same_v<DataType, int64_t>) {
        check_store_mode<store_sideband_mode::direct>(tensor);
        check_store_mode<store_sideband_mode::store_1st>(tensor);
    }
    if constexpr (!Std::is_one_of_v<DataType, float, int64_t>) {
        store<store_sideband_mode::intlv>(tensor, source0, source1);
        store<store_sideband_mode::intlv>(tensor, make_coord(0), source0, source1);
    }
    if constexpr (
        reg_detail::is_b16_type_v<reg_data_type> || reg_detail::is_b32_type_v<reg_data_type> ||
        reg_detail::is_b64_type_v<reg_data_type>) {
        check_store_mode<store_sideband_mode::pack>(tensor);
    }
    if constexpr (reg_detail::is_b32_type_v<reg_data_type>) {
        check_store_mode<store_sideband_mode::pack_quarter>(tensor);
        check_store_scatter<uint32_t>(tensor);
    }
    if constexpr (Std::is_one_of_v<DataType, int8_t, uint8_t, int16_t, uint16_t, half, bfloat16_t>) {
        check_store_scatter<uint16_t>(tensor);
    }
    unalign_store stream;
    constexpr uint32_t stored_elements = 1;
    stream.store(tensor, source0, stored_elements);
    stream.store(tensor, make_coord(0), source1, stored_elements);
    stream.flush(tensor, 0);
}

template void check_reg_data_store<int8_t>(uint32_t);
template void check_reg_data_store<uint8_t>(uint32_t);
template void check_reg_data_store<int16_t>(uint32_t);
template void check_reg_data_store<uint16_t>(uint32_t);
template void check_reg_data_store<half>(uint32_t);
template void check_reg_data_store<bfloat16_t>(uint32_t);
template void check_reg_data_store<int32_t>(uint32_t);
template void check_reg_data_store<uint32_t>(uint32_t);
template void check_reg_data_store<float>(uint32_t);
template void check_reg_data_store<int64_t>(uint32_t);
template void check_reg_data_store<hifloat8_t>(uint32_t);
template void check_reg_data_store<fp8_e4m3fn_t>(uint32_t);
template void check_reg_data_store<fp8_e5m2_t>(uint32_t);
template void check_reg_data_store<fp8_e8m0_t>(uint32_t);
template void check_reg_data_store<fp4x2_e1m2_t>(uint32_t);
template void check_reg_data_store<fp4x2_e2m1_t>(uint32_t);
template void check_reg_data_store<int4b_t>(uint32_t);

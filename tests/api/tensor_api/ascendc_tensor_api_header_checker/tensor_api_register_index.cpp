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
#include "tensor_api/experimental/arch/vector/compute/register_index.h"

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201)) && \
    !defined(IMPL_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_INDEX_IMPL_H)
#error register index implementation is missing for the target architecture
#endif

using namespace asc::te::experimental;

template <typename DataType>
__simd_callee__ inline void check_index_compute()
{
    const DataType start{};
    auto ascending = iota(start);
    auto explicit_ascending = iota<iota_order::increase>(start);
    auto descending = iota<iota_order::decrease>(start);
    static_assert(AscendC::Std::is_same_v<decltype(ascending), reg_tensor<DataType>>);
    static_assert(AscendC::Std::is_same_v<decltype(explicit_ascending), reg_tensor<DataType>>);
    static_assert(AscendC::Std::is_same_v<decltype(descending), reg_tensor<DataType>>);
}

__simd_vf__ inline void check_reg_index_compute()
{
    check_index_compute<int8_t>();
    check_index_compute<int16_t>();
    check_index_compute<int32_t>();
    check_index_compute<half>();
    check_index_compute<float>();
}

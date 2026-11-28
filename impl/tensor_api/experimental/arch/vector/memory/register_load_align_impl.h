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
#warning "impl/tensor_api/experimental/arch/vector/memory/register_load_align_impl.h is an internal header file and" \
    "must not be used directly. Functions or variables defined in this file may be removed in the future." \
    "Please use \"#include \"tensor_api/experimental/arch/vector/memory/register_load_align.h\"\" and use" \
    "public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_ALIGN_IMPL_H
#endif

#ifndef IMPL_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_ALIGN_IMPL_H
#define IMPL_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_ALIGN_IMPL_H

#include "impl/tensor_api/experimental/arch/utils/reg_utils.h"
#include "tensor_api/tensor/tensor_interface.h"

namespace asc {
namespace te {
namespace experimental {

namespace detail {

template <load_sideband_mode sideband_mode, typename RegDataType = void>
struct load_from_ptr_impl {
    template <typename DataType, typename Pointer>
    __simd_callee__ static inline reg_tensor<bool> run(Pointer src)
    {
        static_assert(
            Std::is_same_v<RegDataType, void> || Std::is_same_v<RegDataType, bool>,
            "load with an explicit register type only supports bool");
        static_assert(
            sideband_mode == load_sideband_mode::direct || sideband_mode == load_sideband_mode::upsample ||
                sideband_mode == load_sideband_mode::downsample,
            "load mask only supports direct, upsample, and downsample modes");
        return {};
    }
};

template <>
struct load_from_ptr_impl<load_sideband_mode::direct, void> {
    template <typename DataType, typename Pointer>
    __simd_callee__ static inline reg_tensor<DataType> run(Pointer src)
    {
        static_assert(
            is_b4_type_v<DataType> || is_b8_type_v<DataType> || is_b16_type_v<DataType> || is_b32_type_v<DataType> ||
                is_b64_type_v<DataType>,
            "load direct supports b4, b8, b16, b32, and b64 element types only");
        reg_tensor<DataType> dst;
        asc_loadalign(dst.reg, src);
        return dst;
    }
};

template <>
struct load_from_ptr_impl<load_sideband_mode::downsample, void> {
    template <typename DataType, typename Pointer>
    __simd_callee__ static inline reg_tensor<DataType> run(Pointer src)
    {
        static_assert(
            is_b4_type_v<DataType> || is_b8_type_v<DataType> || is_b16_type_v<DataType>,
            "load downsample supports b4, b8, and b16 element types only");
        reg_tensor<DataType> dst;
        asc_loadalign_downsample(dst.reg, src);
        return dst;
    }
};

template <>
struct load_from_ptr_impl<load_sideband_mode::unpack, void> {
    template <typename DataType, typename Pointer>
    __simd_callee__ static inline reg_tensor<DataType> run(Pointer src)
    {
        static_assert(
            is_b4_type_v<DataType> || is_b8_type_v<DataType> || is_b16_type_v<DataType> || is_b32_type_v<DataType>,
            "load unpack supports b4, b8, b16, and b32 element types only");
        reg_tensor<DataType> dst;
        asc_loadalign_unpack(dst.reg, src);
        return dst;
    }
};

template <>
struct load_from_ptr_impl<load_sideband_mode::unpack4, void> {
    template <typename DataType, typename Pointer>
    __simd_callee__ static inline reg_tensor<DataType> run(Pointer src)
    {
        static_assert(
            is_b4_type_v<DataType> || is_b8_type_v<DataType>, "load unpack4 supports b4 and b8 element types only");
        reg_tensor<DataType> dst;
        asc_loadalign_unpack4(dst.reg, src);
        return dst;
    }
};

template <>
struct load_from_ptr_impl<load_sideband_mode::upsample, void> {
    template <typename DataType, typename Pointer>
    __simd_callee__ static inline reg_tensor<DataType> run(Pointer src)
    {
        static_assert(
            is_b4_type_v<DataType> || is_b8_type_v<DataType> || is_b16_type_v<DataType>,
            "load upsample supports b4, b8, and b16 element types only");
        reg_tensor<DataType> dst;
        asc_loadalign_upsample(dst.reg, src);
        return dst;
    }
};

template <>
struct load_from_ptr_impl<load_sideband_mode::deintlv, void> {
    template <typename DataType, typename Pointer>
    __simd_callee__ static inline reg_pair<DataType> run(Pointer src)
    {
        static_assert(
            is_b4_type_v<DataType> || is_b8_type_v<DataType> || is_b16_type_v<DataType> || is_b32_type_v<DataType>,
            "load deintlv supports b4, b8, b16, and b32 element types only");
        reg_pair<DataType> dst;
        asc_loadalign_deintlv(dst.first.reg, dst.second.reg, src);
        return dst;
    }
};

template <>
struct load_from_ptr_impl<load_sideband_mode::direct, bool> {
    template <typename DataType, typename Pointer>
    __simd_callee__ static inline reg_tensor<bool> run(Pointer src)
    {
        static_assert(Std::is_same_v<DataType, uint32_t>, "load mask only supports uint32_t source tensors");
        reg_tensor<bool> dst;
        dst.reg = asc_loadalign_mask(src);
        return dst;
    }
};

template <>
struct load_from_ptr_impl<load_sideband_mode::upsample, bool> {
    template <typename DataType, typename Pointer>
    __simd_callee__ static inline reg_tensor<bool> run(Pointer src)
    {
        static_assert(Std::is_same_v<DataType, uint32_t>, "load mask only supports uint32_t source tensors");
        reg_tensor<bool> dst;
        dst.reg = asc_loadalign_mask_upsample(src);
        return dst;
    }
};

template <>
struct load_from_ptr_impl<load_sideband_mode::downsample, bool> {
    template <typename DataType, typename Pointer>
    __simd_callee__ static inline reg_tensor<bool> run(Pointer src)
    {
        static_assert(Std::is_same_v<DataType, uint32_t>, "load mask only supports uint32_t source tensors");
        reg_tensor<bool> dst;
        dst.reg = asc_loadalign_mask_downsample(src);
        return dst;
    }
};

template <load_sideband_mode sideband_mode, typename DataType, typename RegDataType = void, typename Pointer>
__simd_callee__ inline auto load_from_ptr(Pointer src)
{
    return load_from_ptr_impl<sideband_mode, RegDataType>::template run<get_reg_data_type<DataType>>(src);
}

template <typename RegDataType, load_sideband_mode sideband_mode, typename Tensor, typename Coord>
__simd_callee__ inline auto load_impl(const Tensor& tensor, const Coord& coord)
{
    using engine_type = typename Tensor::engine_type;
    using data_type = typename Tensor::data_type;
    static_assert(
        Std::is_same_v<get_mem_location<engine_type>, location::ub>, "load only supports tensors located in UB");
    TENSOR_API_DEBUG_CHECK(debug_check_coord, tensor.layout(), coord, "load");

    auto src_engine = tensor.engine() + tensor.layout()(coord);
    return load_from_ptr<sideband_mode, data_type, RegDataType>(src_engine.begin().get());
}

template <typename RegDataType, load_sideband_mode sideband_mode, typename Tensor>
__simd_callee__ inline auto load_impl(const Tensor& tensor)
{
    using engine_type = typename Tensor::engine_type;
    using data_type = typename Tensor::data_type;
    static_assert(
        Std::is_same_v<get_mem_location<engine_type>, location::ub>, "load only supports tensors located in UB");

    return load_from_ptr<sideband_mode, data_type, RegDataType>(tensor.engine().begin().get());
}

} // namespace detail

template <load_sideband_mode sideband_mode, typename Tensor, typename Coord>
__simd_callee__ inline auto load(const Tensor& tensor, const Coord& coord)
{
    return detail::load_impl<void, sideband_mode>(tensor, coord);
}

template <load_sideband_mode sideband_mode, typename Tensor>
__simd_callee__ inline auto load(const Tensor& tensor)
{
    return detail::load_impl<void, sideband_mode>(tensor);
}

template <typename RegDataType, load_sideband_mode sideband_mode, typename Tensor, typename Coord>
__simd_callee__ inline auto load(const Tensor& tensor, const Coord& coord)
{
    static_assert(Std::is_same_v<RegDataType, bool>, "load with an explicit register type only supports bool");
    return detail::load_impl<RegDataType, sideband_mode>(tensor, coord);
}

template <typename RegDataType, load_sideband_mode sideband_mode, typename Tensor>
__simd_callee__ inline auto load(const Tensor& tensor)
{
    static_assert(Std::is_same_v<RegDataType, bool>, "load with an explicit register type only supports bool");
    return detail::load_impl<RegDataType, sideband_mode>(tensor);
}

} // namespace experimental
} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_REGISTER_LOAD_ALIGN_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_ALIGN_IMPL_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REGISTER_LOAD_ALIGN_IMPL_H
#endif

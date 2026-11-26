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
#warning \
    "impl/tensor_api/experimental/arch/vector/logical_compute_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"tensor_api/experimental/arch/vector/logical_compute.h\"\" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_LOGICAL_COMPUTE_IMPL_H
#endif

#ifndef IMPL_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_LOGICAL_COMPUTE_IMPL_H
#define IMPL_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_LOGICAL_COMPUTE_IMPL_H

#include "impl/tensor_api/experimental/arch/utils/reg_utils.h"

namespace asc {
namespace te {
namespace experimental {
namespace detail {

template <uint32_t version, typename DataType, typename ShiftType>
struct shift_type_check;

template <typename DataType, typename ShiftType>
struct shift_type_check<arch_version::v3510, DataType, ShiftType> {
    __simd_callee__ static constexpr void check()
    {
        static_assert(
            (AscendC::Std::is_one_of_v<DataType, int8_t, uint8_t> && AscendC::Std::is_same_v<ShiftType, int8_t>) ||
                (AscendC::Std::is_one_of_v<DataType, int16_t, uint16_t> &&
                 AscendC::Std::is_same_v<ShiftType, int16_t>) ||
                (AscendC::Std::is_one_of_v<DataType, int32_t, uint32_t> && AscendC::Std::is_same_v<ShiftType, int32_t>),
            "3510 register shift supports these (DataType, ShiftType) pairs: "
            "(int8_t/uint8_t, int8_t), (int16_t/uint16_t, int16_t), (int32_t/uint32_t, int32_t)");
    }
};

template <typename DataType, typename ShiftType>
struct shift_type_check<arch_version::v9201, DataType, ShiftType> {
    __simd_callee__ static constexpr void check()
    {
        static_assert(
            (AscendC::Std::is_one_of_v<DataType, int8_t, uint8_t> && AscendC::Std::is_same_v<ShiftType, uint8_t>) ||
                (AscendC::Std::is_one_of_v<DataType, int16_t, uint16_t> &&
                 AscendC::Std::is_same_v<ShiftType, uint16_t>) ||
                (AscendC::Std::is_one_of_v<DataType, int32_t, uint32_t> &&
                 AscendC::Std::is_same_v<ShiftType, uint32_t>),
            "9201 register shift supports these (DataType, ShiftType) pairs: "
            "(int8_t/uint8_t, uint8_t), (int16_t/uint16_t, uint16_t), (int32_t/uint32_t, uint32_t)");
    }
};

template <uint32_t version, typename DataType, typename ShiftType>
struct shift_scalar_type_check;

template <typename DataType, typename ShiftType>
struct shift_scalar_type_check<arch_version::v3510, DataType, ShiftType> {
    using scalar_type = int16_t;

    __simd_callee__ static constexpr void check()
    {
        static_assert(
            AscendC::Std::is_one_of_v<DataType, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t>,
            "3510 scalar shift requires an 8-, 16-, or 32-bit integer register element");
    }
};

template <typename DataType, typename ShiftType>
struct shift_scalar_type_check<arch_version::v9201, DataType, ShiftType> {
    using scalar_type = ShiftType;

    __simd_callee__ static constexpr void check()
    {
        static_assert(
            (AscendC::Std::is_one_of_v<DataType, int8_t, uint8_t, int16_t, uint16_t> &&
             AscendC::Std::is_same_v<ShiftType, uint16_t>) ||
                (AscendC::Std::is_one_of_v<DataType, int32_t, uint32_t> &&
                 AscendC::Std::is_same_v<ShiftType, uint32_t>),
            "9201 scalar shift supports these (DataType, ShiftType) pairs: "
            "(int8_t/uint8_t/int16_t/uint16_t, uint16_t), (int32_t/uint32_t, uint32_t)");
    }
};

} // namespace detail

template <typename T>
__simd_callee__ inline reg_tensor<T> operator!(const reg_tensor<T>& src)
{
    static_assert(detail::supports_not_v<T>, "operator! does not support this element type");
    reg_tensor<T> dst;
    asc_not(dst.reg, src.reg, src.mask);
    dst.mask = src.mask;
    return dst;
}

template <typename T>
__simd_callee__ inline reg_tensor<T> operator&(const reg_tensor<T>& src0, const reg_tensor<T>& src1)
{
    static_assert(detail::supports_and_v<T>, "operator& does not support this element type");
    reg_tensor<T> dst;
    asc_and(dst.reg, src0.reg, src1.reg, src0.mask);
    dst.mask = src0.mask;
    return dst;
}

template <typename T>
__simd_callee__ inline reg_tensor<T> operator|(const reg_tensor<T>& src0, const reg_tensor<T>& src1)
{
    static_assert(detail::supports_or_v<T>, "operator| does not support this element type");
    reg_tensor<T> dst;
    asc_or(dst.reg, src0.reg, src1.reg, src0.mask);
    dst.mask = src0.mask;
    return dst;
}

template <typename T>
__simd_callee__ inline reg_tensor<T> operator^(const reg_tensor<T>& src0, const reg_tensor<T>& src1)
{
    static_assert(detail::supports_xor_v<T>, "operator^ does not support this element type");
    reg_tensor<T> dst;
    asc_xor(dst.reg, src0.reg, src1.reg, src0.mask);
    dst.mask = src0.mask;
    return dst;
}

template <typename T, typename ShiftType>
__simd_callee__ inline reg_tensor<T> operator<<(const reg_tensor<T>& src, const reg_tensor<ShiftType>& shift)
{
    detail::shift_type_check<current_arch_version, T, ShiftType>::check();
    reg_tensor<T> dst;
    asc_shiftleft(dst.reg, src.reg, shift.reg, src.mask);
    dst.mask = src.mask;
    return dst;
}

template <typename T, typename ShiftType>
__simd_callee__ inline reg_tensor<T> operator<<(const reg_tensor<T>& src, const ShiftType& shift)
{
    using shift_traits = detail::shift_scalar_type_check<current_arch_version, T, ShiftType>;
    shift_traits::check();
    const typename shift_traits::scalar_type& shiftCount = shift;
    reg_tensor<T> dst;
    asc_shiftleft_scalar(dst.reg, src.reg, shiftCount, src.mask);
    dst.mask = src.mask;
    return dst;
}

template <typename T, typename ShiftType>
__simd_callee__ inline reg_tensor<T> operator>>(const reg_tensor<T>& src, const reg_tensor<ShiftType>& shift)
{
    detail::shift_type_check<current_arch_version, T, ShiftType>::check();
    reg_tensor<T> dst;
    asc_shiftright(dst.reg, src.reg, shift.reg, src.mask);
    dst.mask = src.mask;
    return dst;
}

template <typename T, typename ShiftType>
__simd_callee__ inline reg_tensor<T> operator>>(const reg_tensor<T>& src, const ShiftType& shift)
{
    using shift_traits = detail::shift_scalar_type_check<current_arch_version, T, ShiftType>;
    shift_traits::check();
    const typename shift_traits::scalar_type& shiftCount = shift;
    reg_tensor<T> dst;
    asc_shiftright_scalar(dst.reg, src.reg, shiftCount, src.mask);
    dst.mask = src.mask;
    return dst;
}

} // namespace experimental
} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_EXPERIMENTAL_ARCH_VECTOR_LOGICAL_COMPUTE_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_LOGICAL_COMPUTE_IMPL_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_LOGICAL_COMPUTE_IMPL_H
#endif

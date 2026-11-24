/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "tests/api/c_api/stub/cce_stub.h"
#include "include/c_api/reg_compute/compute/reg_fused.h"

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

// These cases verify that every overload forwards the requested even/odd position to vmulscvt.
// The multiply-and-convert rounding is defined by the vmulscvt instruction itself, so it is not
// re-verified here.
#define TEST_VECTOR_COMPUTE_MUL_SCALAR_FLOAT2HALF_INSTR(                                              \
    class_name, c_api_name, cce_name, dst_type, data_type, scalar_type)                               \
                                                                                                      \
    class TestVectorCompute##class_name##_##data_type##_##scalar_type##_CApi : public testing::Test { \
    protected:                                                                                        \
        void SetUp()                                                                                  \
        {                                                                                             \
            originalCoreType = g_coreType;                                                            \
            g_coreType = AscendC::AIV;                                                                \
        }                                                                                             \
        void TearDown()                                                                               \
        {                                                                                             \
            g_coreType = originalCoreType;                                                            \
            GlobalMockObject::verify();                                                               \
        }                                                                                             \
        int32_t originalCoreType = AscendC::MIX;                                                      \
    };                                                                                                \
                                                                                                      \
    namespace {                                                                                       \
    int c_api_name##_call_index = 0;                                                                  \
    const int c_api_name##_expected_modes[] = {PART_EVEN, PART_ODD, PART_EVEN, PART_ODD};             \
    void cce_name##c_api_name##_##data_type##_##scalar_type##_Stub(                                   \
        dst_type& dst, data_type src0, scalar_type src1, vector_bool mask, int mode)                  \
    {                                                                                                 \
        ASSERT_LT(c_api_name##_call_index, 4);                                                        \
        EXPECT_EQ(c_api_name##_expected_modes[c_api_name##_call_index], mode);                        \
        ++c_api_name##_call_index;                                                                    \
        dst[0] = half(1.0f);                                                                          \
    }                                                                                                 \
    }                                                                                                 \
                                                                                                      \
    TEST_F(                                                                                           \
        TestVectorCompute##class_name##_##data_type##_##scalar_type##_CApi,                           \
        c_api_name##_##data_type##_##scalar_type##_Succ)                                              \
    {                                                                                                 \
        dst_type dst{};                                                                               \
        data_type src0{};                                                                             \
        scalar_type src1 = 1.0f;                                                                      \
        vector_bool mask{};                                                                           \
        c_api_name##_call_index = 0;                                                                  \
                                                                                                      \
        MOCKER_CPP(cce_name, void(dst_type&, data_type, scalar_type, vector_bool, int))               \
            .times(4)                                                                                 \
            .will(invoke(cce_name##c_api_name##_##data_type##_##scalar_type##_Stub));                 \
                                                                                                      \
        c_api_name(dst, src0, src1, mask, ASC_POSITION_EVEN);                                         \
        EXPECT_EQ(static_cast<float>(dst[0]), 1.0f);                                                  \
        c_api_name(dst, src0, src1, mask, ASC_POSITION_ODD);                                          \
        EXPECT_EQ(static_cast<float>(dst[0]), 1.0f);                                                  \
        dst = c_api_name(src0, src1, mask, ASC_POSITION_EVEN);                                        \
        EXPECT_EQ(static_cast<float>(dst[0]), 1.0f);                                                  \
        dst = c_api_name(src0, src1, mask, ASC_POSITION_ODD);                                         \
        EXPECT_EQ(static_cast<float>(dst[0]), 1.0f);                                                  \
        EXPECT_EQ(c_api_name##_call_index, 4);                                                        \
    }

TEST_VECTOR_COMPUTE_MUL_SCALAR_FLOAT2HALF_INSTR(
    MUL_SCALAR_FLOAT2HALF_RN, asc_mul_scalar_float2half_rn, vmulscvt, vector_half, vector_float, float);

TEST_VECTOR_COMPUTE_MUL_SCALAR_FLOAT2HALF_INSTR(
    MUL_SCALAR_FLOAT2HALF_RNA, asc_mul_scalar_float2half_rna, vmulscvt, vector_half, vector_float, float);

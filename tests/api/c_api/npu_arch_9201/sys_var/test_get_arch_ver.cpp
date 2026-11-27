/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <gtest/gtest.h>
#include "tests/api/c_api/stub/cce_stub.h"
#include "c_api/utils/sys_var.h"

namespace {
constexpr uint32_t EXPECTED_ARCH_VER = 9201U;
}

class TestSysVarGetArchVer9201CApi : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestSysVarGetArchVer9201CApi, get_arch_ver_Succ)
{
    uint32_t core_version = 0;
    asc_get_arch_ver(core_version);

    EXPECT_EQ(core_version, EXPECTED_ARCH_VER);
}

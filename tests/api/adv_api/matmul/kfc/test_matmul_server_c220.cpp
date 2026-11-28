/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "kernel_operator.h"
#include "impl/adv_api/detail/matmul/kfc/matmul_server_impl.h"
#include "impl/adv_api/detail/matmul/kfc/matmul_server_impl_c220.h"
#include "kfc_fake_modules.h"
#include "../copy_cube_in/base_tiling_struct.h"

// UT not support multi thread, only aic or aiv thread, func that contain aic and aiv can not simulated
using namespace std;

namespace AscendC {
constexpr uint32_t NUM_FIFTYSIX = 56;
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using Scheduler = CustomMatmulScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG>;
    using C1Buffer = Impl::Detail::C1Buffer<IMPL, A_TYPE, BIAS_TYPE, MM_CFG>;
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class BatchCustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using BatchScheduler = CustomBatchScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG>;
};

__aicore__ constexpr MatmulConfig GetNBatchConfig()
{
    MatmulBatchParams bmmParams = {true, BatchMode::BATCH_LESS_THAN_L1, true};
    auto bmmCfg = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(bmmParams);
    bmmCfg.enableSetBias = false;
    bmmCfg.enableSetTail = false;
    bmmCfg.enableSetOrgShape = false;
    bmmCfg.enableSetDefineData = false;
    bmmCfg.enableQuantVector = false;
    bmmCfg.iterateMode = IterateMode::ITERATE_MODE_DEFAULT;
    return bmmCfg;
}

static constexpr MatmulConfig N_BATCH_CFG = GetNBatchConfig();

class TestMatmulServerC220 : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using A_TYPE_BSNGD = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::BSNGD>;

    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true>;
    using B_TYPE_BSNGD = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true, LayoutMode::BSNGD>;

    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using C_TYPE_BSNGD = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, false, LayoutMode::BSNGD>;

    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    using A_TYPE_IBSHARE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::NONE, true>;

    using B_TYPE_IBSHARE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::NONE, true>;

    template <typename T>
    void InitTestServer(T& server, TCubeTiling& tiling)
    {
        server.instID = 1;
        server.devEvtID = 1;
        server.tiling_.SetTiling(&tiling);
        server.mul.SetSubBlockIdx(0);
    }
};

TEST_F(TestMatmulServerC220, ProcessEnd)
{
    MatmulService<
        A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy>
        mmServer;
    KfcMsg kfcMsg{};
    uint8_t cGM[2048] = {0};
    kfcMsg.body.cAddr = (uint64_t)(cGM);
    auto funID = KFC_Enum::MMFUN_END;
    mmServer.Process(&kfcMsg, funID);
}

TEST_F(TestMatmulServerC220, StartIterateNotify)
{
    TilingParams tilingParams = {1, 384, 2048, 192, 384, 2048, 192, 128, 256, 64, 2, 8, 3, 2, 3, 3, 0, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    MatmulService<
        A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy>
        mmServer;
    InitTestServer(mmServer, tiling);

    KfcMsg kfcMsg{};
    uint8_t cGM[2048] = {0};
    kfcMsg.body.cAddr = reinterpret_cast<uint64_t>(cGM);
    uint32_t cntIterator = 0;
    ASSERT_TRUE(mmServer.StartIterate(&kfcMsg.body, KFC_Enum::MMFUN_ITERATE, 0, cntIterator));
}

TEST_F(TestMatmulServerC220, GetTensorCNotify)
{
    TilingParams tilingParams = {1, 384, 2048, 192, 384, 2048, 192, 128, 256, 64, 2, 8, 3, 2, 3, 3, 0, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    MatmulService<
        A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy>
        mmServer;
    InitTestServer(mmServer, tiling);

    KfcMsg kfcMsg{};
    uint8_t cGM[2048] = {0};
    kfcMsg.body.cAddr = reinterpret_cast<uint64_t>(cGM);
    kfcMsg.body.sync = 1;
    ASSERT_FALSE(mmServer.GetTensorC(&kfcMsg));
}

TEST_F(TestMatmulServerC220, StartIterateNBatchNotify)
{
    TilingParamsBatch tilingParams = {1, 128, 64, 256, 128, 64, 256, 128, 64, 128, 2, 1, 1,   1, 2,   2, 0, 0,
                                      2, 2,   2,  2,   128, 2,  1,   256, 2,  64,  2, 1, 256, 2, 128, 2, 1, 64};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    MatmulService<
        A_TYPE_BSNGD, B_TYPE_BSNGD, C_TYPE_BSNGD, BIAS_TYPE, N_BATCH_CFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>,
        BatchCustomMatmulPolicy>
        mmServer;
    InitTestServer(mmServer, tiling);

    KfcMsg kfcMsg{};
    uint8_t cGM[2048] = {0};
    kfcMsg.body.cAddr = reinterpret_cast<uint64_t>(cGM);
    kfcMsg.body.sizeAmatrix = 256;
    kfcMsg.body.sizeBmatrix = 256;
    kfcMsg.body.batchA = 2;
    kfcMsg.body.batchB = 2;
    kfcMsg.body.batchLoop = 2;
    uint32_t cntIterator = 0;
    mmServer.StartIterateNBatch(&kfcMsg.body, cntIterator);
}

TEST_F(TestMatmulServerC220, IterateNBatchNotify)
{
    TilingParamsBatch tilingParams = {1, 128, 64, 256, 128, 64, 256, 128, 64, 128, 2, 1, 1,   1, 2,   2, 0, 0,
                                      2, 2,   2,  2,   128, 2,  1,   256, 2,  64,  2, 1, 256, 2, 128, 2, 1, 64};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    MatmulService<
        A_TYPE_BSNGD, B_TYPE_BSNGD, C_TYPE_BSNGD, BIAS_TYPE, N_BATCH_CFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>,
        BatchCustomMatmulPolicy>
        mmServer;
    InitTestServer(mmServer, tiling);

    KfcMsg kfcMsg{};
    uint8_t cGM[2048] = {0};
    kfcMsg.body.cAddr = reinterpret_cast<uint64_t>(cGM);
    kfcMsg.body.sizeAmatrix = 256;
    kfcMsg.body.sizeBmatrix = 256;
    kfcMsg.body.batchA = 2;
    kfcMsg.body.batchB = 2;
    kfcMsg.body.batchLoop = 16;
    ASSERT_TRUE(mmServer.IterateNBatch(&kfcMsg));
}

TEST_F(TestMatmulServerC220, IterateNBatchSyncNotify)
{
    TilingParamsBatch tilingParams = {1, 128, 64, 256, 128, 64, 256, 128, 64, 128, 2, 1, 1,   1, 2,   2, 0, 0,
                                      2, 2,   2,  2,   128, 2,  1,   256, 2,  64,  2, 1, 256, 2, 128, 2, 1, 64};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    MatmulService<
        A_TYPE_BSNGD, B_TYPE_BSNGD, C_TYPE_BSNGD, BIAS_TYPE, N_BATCH_CFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>,
        BatchCustomMatmulPolicy>
        mmServer;
    InitTestServer(mmServer, tiling);

    KfcMsg kfcMsg{};
    uint8_t cGM[2048] = {0};
    kfcMsg.body.cAddr = reinterpret_cast<uint64_t>(cGM);
    kfcMsg.body.sizeAmatrix = 256;
    kfcMsg.body.sizeBmatrix = 256;
    kfcMsg.body.batchA = 2;
    kfcMsg.body.batchB = 2;
    kfcMsg.body.batchLoop = 2;
    kfcMsg.body.sync = 1;
    ASSERT_TRUE(mmServer.IterateNBatch(&kfcMsg));
}

TEST_F(TestMatmulServerC220, IterateFakeMsgNotify)
{
    TilingParams tilingParams = {1, 384, 2048, 192, 384, 2048, 192, 128, 256, 64, 2, 8, 3, 2, 3, 3, 0, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    MatmulService<
        A_TYPE_IBSHARE, B_TYPE_IBSHARE, C_TYPE, BIAS_TYPE, CFG_IBSHARE_NORM,
        MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy>
        mmServer;
    InitTestServer(mmServer, tiling);
    KFC_COMM_SERVER kfcServer{};
    kfcServer.subBlockID = 0;
    mmServer.kfcCommSrv = &kfcServer;

    KfcMsg kfcMsg{};
    kfcMsg.body.iterateFakeMsg = 1;
    ASSERT_TRUE(mmServer.Iterate(&kfcMsg, KFC_Enum::MMFUN_ITERATE_ALL));
}

TEST_F(TestMatmulServerC220, IterateAllNotify)
{
    TilingParams tilingParams = {1, 384, 2048, 192, 384, 2048, 192, 128, 256, 64, 2, 8, 3, 2, 3, 3, 0, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    MatmulService<
        A_TYPE_IBSHARE, B_TYPE_IBSHARE, C_TYPE, BIAS_TYPE, CFG_IBSHARE_NORM,
        MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy>
        mmServer;
    InitTestServer(mmServer, tiling);

    KfcMsg kfcMsg{};
    uint8_t cGM[2048] = {0};
    kfcMsg.body.cAddr = reinterpret_cast<uint64_t>(cGM);
    kfcMsg.body.sync = 1;
    ASSERT_TRUE(mmServer.Iterate(&kfcMsg, KFC_Enum::MMFUN_ITERATE_ALL));
}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
TEST_F(TestMatmulServerC220, IterateAsyncNotify)
{
    TilingParams tilingParams = {1, 384, 2048, 192, 384, 2048, 192, 128, 256, 64, 2, 8, 3, 2, 3, 3, 0, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    MatmulService<
        A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy>
        mmServer;
    InitTestServer(mmServer, tiling);

    KfcMsg kfcMsg{};
    uint8_t cGM[2048] = {0};
    kfcMsg.body.cAddr = reinterpret_cast<uint64_t>(cGM);
    ASSERT_TRUE(mmServer.Iterate(&kfcMsg, KFC_Enum::MMFUN_ITERATE));
}
#endif
} // namespace AscendC

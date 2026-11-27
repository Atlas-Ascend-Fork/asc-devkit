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
#include <cstring>
#include <memory>
#include <vector>
#include <iterator>
#include <set>
#include <unordered_map>
#include <cstdlib>

#include "dtype_common.h"
#include "base.h"
#include "include/adv_api/hccl/hccl_mc2.h"
#include "include/adv_api/hccl/internal/hccl_msg.h"
#include "hccl_alloc_ctx_res.h"
#include "external_alg_resolver.h"
#include "external_alg_rules.h"
#include "topo_pattern.h"
#include "kfc_server_protocol.h"
#include "sim_communicator.h"
#include "sim_stream.h"
#include "sim_world.h"
#include "topo_model.h"
#include "sim_common.h"
#include "stub/cann_host_bridge_stub.h"

namespace mc2_ops_hccl {
DevType g_stubDeviceType = DevType::DEV_TYPE_950;
// 桩态复位声明（hcomm_ccu_stub.cc 内强定义，直接 include 会二次包含致链接冲突）。
void ResetCcuDynResStub();
extern bool g_stubCcuAlgorithmRegistered;
extern bool g_stubCcuAlgExecNull;
extern std::string g_stubCcuAlgExecNullName;
extern std::string g_stubSelectorAlgName;
extern bool g_stubCcuAlgResUnavailable;
extern std::string g_stubCcuAlgResUnavailableName;
extern bool g_stubAclrtMemcpyFail;
extern uint32_t g_stubAclrtMemcpyCallCount;
extern uint32_t g_stubAclrtMemcpyFailOnCall;
extern HcclResult g_stubHcomCheckDataTypeResult;
extern HcclResult g_stubHcomCheckReductionOpResult;
extern HcclResult g_stubCalcTopoInfoResult;
extern bool g_stubTopoPcieMix;
extern bool g_stubTopoBigClosRange;
extern bool g_stubHcommCcuKernelLaunchFail;
extern uint32_t g_stubHcommCcuKernelLaunchCallCount;
extern std::vector<std::vector<uint64_t>> g_stubHcommCcuKernelLaunchArgs;
extern bool g_stubHcommIsSupportCcuV2;
} // namespace mc2_ops_hccl

// 夹具：本目标不编 executor 文件（-DAICPU_COMPILE 裁掉产品 CCU sidecar 注册行），
// 按产品现场同款注册供 EXTERNAL 用例使用。
// 共享测试模式取单条全 Any：测试注册只验证注册机制（sidecar 落库/定名/候选表），不验证拓扑
// 语义——本目标 topo 恒为 make_unique 预分配的默认构造值，全 Any 模式在 MatchOne 中逐子句
// 短路命中，等价于不设拓扑过滤。
namespace mc2_ops_hccl {
static const TopoPattern kTestPatterns[] = {{}};
REGISTER_ALG_META(
    HcclCMDType::HCCL_CMD_ALLGATHER, CcuSchedAllGatherSoleMesh, AlgEngine::CCU, "sole[mesh]", COND_NONE, kTestPatterns,
    0);
REGISTER_ALG_META(
    HcclCMDType::HCCL_CMD_ALLGATHER, InsAllGatherMesh1D, AlgEngine::AICPU, "sole[mesh]", COND_NONE, kTestPatterns, 0);
REGISTER_ALG_META(
    HcclCMDType::HCCL_CMD_ALLREDUCE, CcuSchedAllReduceSoleMesh, AlgEngine::CCU, "sole[mesh]", COND_NONE, kTestPatterns,
    0);
REGISTER_ALG_META(
    HcclCMDType::HCCL_CMD_ALLREDUCE, CcuSchedAllGatherSoleMesh, AlgEngine::CCU, "sole[mesh]", COND_NONE, kTestPatterns,
    0);
REGISTER_ALG_META(
    HcclCMDType::HCCL_CMD_ALLREDUCE, InsAllReduceMesh1D, AlgEngine::AICPU, "sole[mesh]", COND_NONE, kTestPatterns, 0);
REGISTER_ALG_META(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CcuSchedReduceScatterSoleMesh, AlgEngine::CCU, "sole[mesh]", COND_NONE,
    kTestPatterns, 0);
} // namespace mc2_ops_hccl

// hccl_mc2.cc 的 CheckCcuAlgorithmsRegistered（非 static 自由函数，须在全局作用域声明）
bool CheckCcuAlgorithmsRegistered(const void* ccTilingList[], uint32_t tilingNum);

// Mc2GetSupportAlgList 候选可见性：查询链以 CollAlgExecRegistryV2 的 tag 在册性为候选入口
// ：本目标不编 executor 文件，为查询用例涉足的注册名镜像落 tag（ALLGATHER/REDUCE_SCATTER）。查询只消费 tag 在册性，
// creator 不被调用；IsRegistered/GetAlgExec 走桩与此表无关。CcuSchedAgExecutorOnly 只落
// tag 不落元数据，验证候选 ∩ 元数据语义；LenGated 条目带 PER_RANK LE k1MB 长度子句，
// 供查询接口 dataLen 区间语义（哨兵 (0,0) ⊇ 非空窗）用例使用；LongExternal 见其注册处注释。
namespace mc2_ops_hccl {
static const TopoPattern kLenGatedPatterns[] = {[] {
    TopoPattern pattern{};
    pattern.lenClauses.push_back(LenClause{LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k1MB});
    return pattern;
}()};
REGISTER_ALG_META(
    HcclCMDType::HCCL_CMD_ALLGATHER, CcuSchedAllGatherLenGated, AlgEngine::CCU, "len[rank]", COND_NONE,
    kLenGatedPatterns, 0);

// D18 降级链可观测性：长外部名（52 字符）把 ALLGATHER+CCU 名单 JSON 撑到 ~120B；error JSON
// 含 need/len 实值约 110B——中等 len 窗口 [110,119] 由此存在，CAbiJsonContract 才能验证"名单
// JSON 装不下、error JSON 装得下"的中间降级态。全 Any 模式与 kTestPatterns 同款，无长度子句，
// 既有名单语义断言（去重/集合包含/基线相等）对其均成立。
REGISTER_ALG_META(
    HcclCMDType::HCCL_CMD_ALLGATHER, CcuSchedAllGatherLongExternal, AlgEngine::CCU,
    "pipeline[mesh.multi_channel,nhr.two_shot,mesh.chunk]", COND_NONE, kTestPatterns, 0);

static InsCollAlgBase* QueryTagOnlyCreator() { return nullptr; }
static const HcclResult g_queryTagAllGatherSoleMesh = CollAlgExecRegistryV2::Instance().Register(
    HcclCMDType::HCCL_CMD_ALLGATHER, "CcuSchedAllGatherSoleMesh", QueryTagOnlyCreator);
static const HcclResult g_queryTagAllGatherLongExternal = CollAlgExecRegistryV2::Instance().Register(
    HcclCMDType::HCCL_CMD_ALLGATHER, "CcuSchedAllGatherLongExternal", QueryTagOnlyCreator);
static const HcclResult g_queryTagAllGatherLenGated = CollAlgExecRegistryV2::Instance().Register(
    HcclCMDType::HCCL_CMD_ALLGATHER, "CcuSchedAllGatherLenGated", QueryTagOnlyCreator);
static const HcclResult g_queryTagAllGatherAicpu = CollAlgExecRegistryV2::Instance().Register(
    HcclCMDType::HCCL_CMD_ALLGATHER, "InsAllGatherMesh1D", QueryTagOnlyCreator);
static const HcclResult g_queryTagAllGatherNoMeta = CollAlgExecRegistryV2::Instance().Register(
    HcclCMDType::HCCL_CMD_ALLGATHER, "CcuSchedAgExecutorOnly", QueryTagOnlyCreator);
static const HcclResult g_queryTagReduceScatterSoleMesh = CollAlgExecRegistryV2::Instance().Register(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "CcuSchedReduceScatterSoleMesh", QueryTagOnlyCreator);

// 静态注册自检（fixture 前置校验）：tag 落库失败（如重名）会静默产生空名单，
// 后续断言随之失去意义，消费名单的用例统一先经此 helper 校验一次
static void VerifyQueryTagRegistrations()
{
    ASSERT_EQ(g_queryTagAllGatherSoleMesh, HCCL_SUCCESS);
    ASSERT_EQ(g_queryTagAllGatherLongExternal, HCCL_SUCCESS);
    ASSERT_EQ(g_queryTagAllGatherLenGated, HCCL_SUCCESS);
    ASSERT_EQ(g_queryTagAllGatherAicpu, HCCL_SUCCESS);
    ASSERT_EQ(g_queryTagAllGatherNoMeta, HCCL_SUCCESS);
    ASSERT_EQ(g_queryTagReduceScatterSoleMesh, HCCL_SUCCESS);
}
} // namespace mc2_ops_hccl

HcclResult CheckMc2CcDeviceType(const char* apiName);

namespace {

static u32 g_stubRankSize = 8;
static u32 g_stubRankId = 0;
static std::vector<void*> g_allocatedPtrs;
static std::unordered_map<uint32_t, std::string> g_opTypeToAlgName = {
    {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER), "CcuSchedAllGatherSoleMesh"},
    {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE), "CcuAllReduceMesh1D"},
    {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER), "CcuSchedReduceScatterSoleMesh"},
};

static void StubCleanup()
{
    for (void* p : g_allocatedPtrs) {
        free(p);
    }
    g_allocatedPtrs.clear();
    g_stubRankSize = 8;
    g_stubRankId = 0;
    mc2_ops_hccl::g_stubDeviceType = DevType::DEV_TYPE_950;
    mc2_ops_hccl::g_stubCcuAlgExecNull = false;
    mc2_ops_hccl::g_stubCcuAlgExecNullName.clear();
    mc2_ops_hccl::g_stubSelectorAlgName.clear();
    mc2_ops_hccl::g_stubCcuAlgResUnavailable = false;
    mc2_ops_hccl::g_stubCcuAlgResUnavailableName.clear();
    mc2_ops_hccl::g_cannBridgeTestState = {};
    mc2_ops_hccl::g_stubAclrtMemcpyFail = false;
    mc2_ops_hccl::g_stubAclrtMemcpyCallCount = 0U;
    mc2_ops_hccl::g_stubAclrtMemcpyFailOnCall = 0U;
    mc2_ops_hccl::g_stubHcomCheckDataTypeResult = HCCL_SUCCESS;
    mc2_ops_hccl::g_stubHcomCheckReductionOpResult = HCCL_SUCCESS;
    mc2_ops_hccl::g_stubCalcTopoInfoResult = HCCL_SUCCESS;
    mc2_ops_hccl::g_stubTopoPcieMix = false;
    mc2_ops_hccl::g_stubTopoBigClosRange = false;
    mc2_ops_hccl::g_stubHcommCcuKernelLaunchFail = false;
    mc2_ops_hccl::g_stubHcommCcuKernelLaunchCallCount = 0U;
    mc2_ops_hccl::g_stubHcommCcuKernelLaunchArgs.clear();
    mc2_ops_hccl::g_stubHcommIsSupportCcuV2 = true;
    // hcomm_ccu_stub.cc 内 ccu_dynamic_res_stub.h 的桩态复位（desc 句柄/assignedInsByComm），
    // 与 ST fixture 的 ResetCcuDynResStub 同源：comm 指针跨用例复用会命中陈旧 assigned 条目。
    mc2_ops_hccl::ResetCcuDynResStub();
    unsetenv("HCCL_OP_EXPANSION_MODE");
}

static void SetCommEngineEnv(uint8_t commEngine)
{
    if (commEngine == static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED)) {
        setenv("HCCL_OP_EXPANSION_MODE", "CCU_SCHED", 1);
    } else if (commEngine == COMM_ENGINE_CCU) {
        setenv("HCCL_OP_EXPANSION_MODE", "CCU_MS", 1);
    } else if (commEngine == COMM_ENGINE_AICPU) {
        setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    } else {
        unsetenv("HCCL_OP_EXPANSION_MODE");
    }
}

static TopoMeta BuildTopoMeta(uint32_t rankSize)
{
    TopoMeta topoMeta;
    SuperPodMeta superPod;
    ServerMeta server;
    for (uint32_t i = 0; i < rankSize; ++i) {
        server.push_back(i);
    }
    superPod.push_back(server);
    topoMeta.push_back(superPod);
    return topoMeta;
}

struct Mc2TilingTestData {
    uint32_t version;
    uint32_t cnt;
    uint32_t offset[Hccl::MC2_MAX_OP_NUM];
    Mc2ServerCfg serverCfg;
    Mc2CcTilingInner ccTiling[Hccl::MC2_MAX_OP_NUM];
};

static Mc2TilingTestData BuildMc2Tiling(
    uint32_t tilingNum, uint8_t commEngine, const uint32_t opTypes[], const uint8_t srcDataTypes[])
{
    SetCommEngineEnv(commEngine);
    Mc2TilingTestData tiling{};
    tiling.version = commEngine == static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED) ? INIT_TILING_CCU_NEW_VERSION :
                                                                                      INIT_TILING_VERSION;
    tiling.cnt = tilingNum;
    uint32_t baseOffset = offsetof(Mc2TilingTestData, ccTiling);
    for (uint32_t i = 0; i < Hccl::MC2_MAX_OP_NUM; ++i) {
        tiling.offset[i] = baseOffset + i * sizeof(Mc2CcTilingInner);
    }
    for (uint32_t i = 0; i < tilingNum; ++i) {
        tiling.ccTiling[i].commEngine = commEngine;
        tiling.ccTiling[i].opType = opTypes[i];
        tiling.ccTiling[i].srcDataType = srcDataTypes ? srcDataTypes[i] : HCCL_DATA_TYPE_FP16;
        tiling.ccTiling[i].dstDataType = HCCL_DATA_TYPE_FP16;
        tiling.ccTiling[i].reduceType = HCCL_REDUCE_SUM;
        strcpy(tiling.ccTiling[i].groupName, "test_group");
        if (commEngine == static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED) &&
            opTypes[i] == static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER)) {
            strcpy(tiling.ccTiling[i].algConfig, "CcuSchedAllGatherSoleMesh");
        } else if (
            commEngine == static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED) &&
            opTypes[i] == static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER)) {
            strcpy(tiling.ccTiling[i].algConfig, "CcuSchedReduceScatterSoleMesh");
        } else {
            strcpy(tiling.ccTiling[i].algConfig, "default");
        }
    }
    return tiling;
}

static HcclResult RunCcuSelectAlg(
    HcclComm comm, void* stream, const std::string topoTag[], const void* ccTilingList[], uint32_t tilingNum,
    OpResCtx& resCtx)
{
    Mc2InitTilingInner initTiling{};
    return CcuSelectAlg(comm, stream, topoTag, ccTilingList, tilingNum, &initTiling, resCtx);
}

static void* CreateMc2CcArgs(uint8_t commEngine, const char* algConfig)
{
    void* ccArgs = nullptr;
    if (Mc2GetCcArgs(&ccArgs) != HCCL_SUCCESS || ccArgs == nullptr) {
        return nullptr;
    }
    if (Mc2SetCcCommEngine(ccArgs, commEngine) != HCCL_SUCCESS ||
        (algConfig != nullptr && Mc2SetCcAlgConfig(ccArgs, algConfig) != HCCL_SUCCESS)) {
        (void)Mc2FreeCcArgs(ccArgs);
        return nullptr;
    }
    return ccArgs;
}

static OpResCtx BuildLaunchOpResCtx(OpParam& opParam, HcclComm comm, CommEngine engine)
{
    opParam.hcclComm = comm;
    opParam.engine = engine;
    opParam.resCtx = reinterpret_cast<void*>(0x1);
    opParam.ctxSize = 1U;

    OpResCtx opResCtx{};
    opResCtx.commEngine = static_cast<uint64_t>(engine);
    opResCtx.workSpace = 1U;
    opResCtx.workSpaceSize = 1U;
    opResCtx.algInfo[0].opParam = reinterpret_cast<uint64_t>(&opParam);
    return opResCtx;
}

class CcuMc2TestSuite : public testing::Test {
protected:
    void SetUp() override
    {
        StubCleanup();
        TopoMeta topoMeta = BuildTopoMeta(g_stubRankSize);
        HcclSim::SimWorld::Global()->Init(topoMeta, mc2_ops_hccl::g_stubDeviceType);

        HcclResult ret = HcclSim::Sim_HcclCommInitClusterInfo(topoMeta, g_stubRankId, &comm_);
        ASSERT_EQ(ret, HCCL_SUCCESS) << "Comm init failed";
        ASSERT_NE(comm_, nullptr) << "Comm is nullptr after init";

        // 堆上稳定地址替代栈局部变量：避免 TearDown 经 ~SimHcclThread 对已死栈地址回写（CI 段错误）。
        streamStorage_ = std::make_unique<int>(1);
        stream_ = reinterpret_cast<aclrtStream>(streamStorage_.get());
    }
    void TearDown() override
    {
        if (comm_ != nullptr) {
            auto simComm = static_cast<HcclSim::SimCommunicator*>(comm_);
            delete simComm;
            comm_ = nullptr;
        }
        HcclSim::SimWorld::Global()->Deinit();
        StubCleanup();
    }
    HcclComm comm_ = nullptr;
    HcclSim::SimStream dummyStream_{0U};
    aclrtStream stream_ = nullptr;
    std::unique_ptr<int> streamStorage_;
};

TEST_F(CcuMc2TestSuite, AllocCcuOpResCtx_BasicFields)
{
    OpResCtx resCtx{};
    u32 rankSize = 8;
    u32 userRank = 3;
    HcclResult ret = AllocCcuOpResCtx(comm_, "test_tag", rankSize, userRank, resCtx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(resCtx.rankId, userRank);
    EXPECT_EQ(resCtx.rankSize, rankSize);
    EXPECT_EQ(resCtx.workSpaceSize, Hccl::MC2_WORKSPACE_SIZE);
    EXPECT_NE(resCtx.res[0], 0U);
}

TEST_F(CcuMc2TestSuite, AllocCcuOpResCtx_FourAddrsDistinct)
{
    OpResCtx resCtx{};
    EXPECT_EQ(AllocCcuOpResCtx(comm_, "tag", 4, 0, resCtx), HCCL_SUCCESS);
    EXPECT_NE(resCtx.workSpace, resCtx.xnAddr);
    EXPECT_NE(resCtx.workSpace, resCtx.ckeAddr);
    EXPECT_NE(resCtx.xnAddr, resCtx.ckeAddr);
    EXPECT_NE(resCtx.res[0], resCtx.workSpace);
    EXPECT_NE(resCtx.res[0], resCtx.xnAddr);
    EXPECT_NE(resCtx.res[0], resCtx.ckeAddr);
}

TEST_F(CcuMc2TestSuite, AllocCcuOpResCtx_ReuseExistingMemory)
{
    OpResCtx firstResCtx{};
    OpResCtx secondResCtx{};
    ASSERT_EQ(AllocCcuOpResCtx(comm_, "reuse_tag", 4, 0, firstResCtx), HCCL_SUCCESS);
    ASSERT_EQ(AllocCcuOpResCtx(comm_, "reuse_tag", 4, 0, secondResCtx), HCCL_SUCCESS);

    EXPECT_EQ(secondResCtx.workSpace, firstResCtx.workSpace);
    EXPECT_EQ(secondResCtx.res[0], firstResCtx.res[0]);
    EXPECT_EQ(secondResCtx.xnAddr, firstResCtx.xnAddr);
    EXPECT_EQ(secondResCtx.ckeAddr, firstResCtx.ckeAddr);
}

TEST_F(CcuMc2TestSuite, CcuSelectAlg_AllGather)
{
    SetCommEngineEnv(COMM_ENGINE_CCU);

    OpResCtx resCtx{};
    ASSERT_EQ(AllocCcuOpResCtx(comm_, "allgather_ctx", g_stubRankSize, g_stubRankId, resCtx), HCCL_SUCCESS);

    Mc2CcTilingInner ccTiling{};
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    ccTiling.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    strcpy(ccTiling.algConfig, "CcuSchedAllGatherSoleMesh");
    const void* ccTilingList[] = {&ccTiling};
    std::string topoTag[] = {"tag0"};

    EXPECT_EQ(RunCcuSelectAlg(comm_, stream_, topoTag, ccTilingList, 1, resCtx), HCCL_SUCCESS);
    EXPECT_EQ(resCtx.opType[0], static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER));
    EXPECT_EQ(resCtx.algorithmType[0], static_cast<uint32_t>(AlgorithmType::CcuSchedAllGatherSoleMesh));
}

// EXTERNAL 通道端到端：合法外部名经 tiling algConfig 进入，resolver 定名
// CcuSchedAllGatherSoleMesh，algorithmType 回填正确
TEST_F(CcuMc2TestSuite, CcuSelectAlg_ExternalNameAccepted)
{
    SetCommEngineEnv(COMM_ENGINE_CCU);

    OpResCtx resCtx{};
    ASSERT_EQ(AllocCcuOpResCtx(comm_, "external_alg_ctx", g_stubRankSize, g_stubRankId, resCtx), HCCL_SUCCESS);

    Mc2CcTilingInner ccTiling{};
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    ccTiling.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    ccTiling.srcDataType = HCCL_DATA_TYPE_FP16;
    ccTiling.dstDataType = HCCL_DATA_TYPE_FP16;
    ccTiling.reduceType = HCCL_REDUCE_SUM;
    strcpy(ccTiling.algConfig, "sole[mesh]");
    const void* ccTilingList[] = {&ccTiling};
    std::string topoTag[] = {"external_alg_tag"};

    EXPECT_EQ(RunCcuSelectAlg(comm_, stream_, topoTag, ccTilingList, 1, resCtx), HCCL_SUCCESS);
    EXPECT_EQ(resCtx.opType[0], static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER));
    EXPECT_EQ(resCtx.algorithmType[0], static_cast<uint32_t>(AlgorithmType::CcuSchedAllGatherSoleMesh));
    EXPECT_TRUE(resCtx.isKfc[0]);
}

TEST_F(CcuMc2TestSuite, CcuSelectAlg_ParallelMeshNhrMultiLink)
{
    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));

    OpResCtx resCtx{};
    ASSERT_EQ(AllocCcuOpResCtx(comm_, "parallel_allgather_ctx", g_stubRankSize, g_stubRankId, resCtx), HCCL_SUCCESS);

    Mc2CcTilingInner ccTiling{};
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    ccTiling.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    ccTiling.srcDataType = HCCL_DATA_TYPE_FP16;
    ccTiling.dstDataType = HCCL_DATA_TYPE_FP16;
    strcpy(ccTiling.algConfig, "CcuSchedAllGatherParallelMeshNHRMultiLink");
    // 新契约：裸名不再是强制算法（归为 NONE），算法由默认 selector 决定；
    // 本用例聚焦 algName→algorithmType 映射，用 selector 输出桩固定选名结果
    mc2_ops_hccl::g_stubSelectorAlgName = "CcuSchedAllGatherParallelMeshNHRMultiLink";
    const void* ccTilingList[] = {&ccTiling};
    std::string topoTag[] = {"parallel_mesh_nhr_tag"};

    EXPECT_EQ(RunCcuSelectAlg(comm_, stream_, topoTag, ccTilingList, 1, resCtx), HCCL_SUCCESS);
    EXPECT_EQ(resCtx.opType[0], static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER));
    EXPECT_EQ(resCtx.algorithmType[0], static_cast<uint32_t>(AlgorithmType::CcuSchedAllGatherParallelMeshNHRMultiLink));
}

TEST_F(CcuMc2TestSuite, CcuSelectAlg_ReduceScatterKfcMesh1DMem2Mem)
{
    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));

    OpResCtx resCtx{};
    ASSERT_EQ(AllocCcuOpResCtx(comm_, "reduce_scatter_ctx", g_stubRankSize, g_stubRankId, resCtx), HCCL_SUCCESS);

    Mc2CcTilingInner ccTiling{};
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    ccTiling.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    ccTiling.srcDataType = HCCL_DATA_TYPE_FP16;
    ccTiling.dstDataType = HCCL_DATA_TYPE_FP16;
    ccTiling.reduceType = HCCL_REDUCE_SUM;
    strcpy(ccTiling.algConfig, "CcuSchedReduceScatterSoleMesh");
    const void* ccTilingList[] = {&ccTiling};
    std::string topoTag[] = {"tag0"};

    EXPECT_EQ(RunCcuSelectAlg(comm_, stream_, topoTag, ccTilingList, 1, resCtx), HCCL_SUCCESS);
    EXPECT_EQ(resCtx.opType[0], static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
    EXPECT_EQ(resCtx.algorithmType[0], static_cast<uint32_t>(AlgorithmType::CcuReduceScatterMeshMem2Mem1D));
}

TEST_F(CcuMc2TestSuite, CcuSelectAlg_ReduceScatterMesh1DMem2MemPeerOnly)
{
    auto simComm = static_cast<HcclSim::SimCommunicator*>(comm_);
    delete simComm;
    comm_ = nullptr;
    HcclSim::SimWorld::Global()->Deinit();
    g_stubRankSize = 2U;
    TopoMeta topoMeta = BuildTopoMeta(g_stubRankSize);
    HcclSim::SimWorld::Global()->Init(topoMeta, g_stubDeviceType);
    ASSERT_EQ(HcclSim::Sim_HcclCommInitClusterInfo(topoMeta, g_stubRankId, &comm_), HCCL_SUCCESS);

    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));
    OpResCtx resCtx{};
    ASSERT_EQ(AllocCcuOpResCtx(comm_, "reduce_scatter_peer_only_ctx", 2U, g_stubRankId, resCtx), HCCL_SUCCESS);

    Mc2CcTilingInner ccTiling{};
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    ccTiling.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    ccTiling.srcDataType = HCCL_DATA_TYPE_FP16;
    ccTiling.dstDataType = HCCL_DATA_TYPE_FP16;
    ccTiling.reduceType = HCCL_REDUCE_SUM;
    ASSERT_EQ(strcpy_s(ccTiling.algConfig, sizeof(ccTiling.algConfig), KFC_REDUCE_SCATTER_PEER_ONLY_ALG_NAME), EOK);
    // 新契约：裸名不再是强制算法（归为 NONE），算法由默认 selector 决定；
    // 本用例聚焦 algName→algorithmType 映射，用 selector 输出桩固定选名结果
    mc2_ops_hccl::g_stubSelectorAlgName = KFC_REDUCE_SCATTER_PEER_ONLY_ALG_NAME;
    const void* ccTilingList[] = {&ccTiling};
    std::string topoTag[] = {"tag0"};

    EXPECT_EQ(RunCcuSelectAlg(comm_, stream_, topoTag, ccTilingList, 1U, resCtx), HCCL_SUCCESS);
    EXPECT_EQ(resCtx.opType[0], static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
    EXPECT_EQ(resCtx.algorithmType[0], static_cast<uint32_t>(AlgorithmType::CcuReduceScatterMeshMem2Mem1DPeerOnly));
}

TEST_F(CcuMc2TestSuite, CcuSelectAlg_ReduceScatterKfcSoleNhrMultiLink)
{
    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));

    OpResCtx resCtx{};
    ASSERT_EQ(AllocCcuOpResCtx(comm_, "reduce_scatter_nhr_ml_ctx", g_stubRankSize, g_stubRankId, resCtx), HCCL_SUCCESS);

    Mc2CcTilingInner ccTiling{};
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    ccTiling.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    ccTiling.srcDataType = HCCL_DATA_TYPE_FP16;
    ccTiling.dstDataType = HCCL_DATA_TYPE_FP16;
    ccTiling.reduceType = HCCL_REDUCE_SUM;
    strcpy(ccTiling.algConfig, "CcuSchedReduceScatterSoleNHRMultiLink");
    // 新契约：裸名不再是强制算法（归为 NONE），算法由默认 selector 决定；
    // 本用例聚焦 algName→algorithmType 映射，用 selector 输出桩固定选名结果
    mc2_ops_hccl::g_stubSelectorAlgName = "CcuSchedReduceScatterSoleNHRMultiLink";
    const void* ccTilingList[] = {&ccTiling};
    std::string topoTag[] = {"tag0"};

    EXPECT_EQ(RunCcuSelectAlg(comm_, stream_, topoTag, ccTilingList, 1, resCtx), HCCL_SUCCESS);
    EXPECT_EQ(resCtx.opType[0], static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
    EXPECT_EQ(resCtx.algorithmType[0], static_cast<uint32_t>(AlgorithmType::CcuSchedReduceScatterSoleNHRMultiLink));
}

TEST_F(CcuMc2TestSuite, CcuSelectAlg_ReduceScatterRejectsInt8)
{
    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));

    OpResCtx resCtx{};
    Mc2CcTilingInner ccTiling{};
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    ccTiling.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    ccTiling.srcDataType = HCCL_DATA_TYPE_INT8;
    ccTiling.dstDataType = HCCL_DATA_TYPE_INT8;
    ccTiling.reduceType = HCCL_REDUCE_SUM;
    const void* ccTilingList[] = {&ccTiling};
    std::string topoTag[] = {"tag0"};

    EXPECT_EQ(RunCcuSelectAlg(comm_, stream_, topoTag, ccTilingList, 1, resCtx), HCCL_E_NOT_SUPPORT);
}

TEST_F(CcuMc2TestSuite, CcuSelectAlg_ReduceScatterRejects64BitTypes)
{
    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));
    const std::vector<uint8_t> unsupportedTypes = {HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64};

    for (const uint8_t dataType : unsupportedTypes) {
        OpResCtx resCtx{};
        Mc2CcTilingInner ccTiling{};
        ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
        ccTiling.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
        ccTiling.srcDataType = dataType;
        ccTiling.dstDataType = dataType;
        ccTiling.reduceType = HCCL_REDUCE_SUM;
        const void* ccTilingList[] = {&ccTiling};
        std::string topoTag[] = {"tag0"};

        EXPECT_EQ(RunCcuSelectAlg(comm_, stream_, topoTag, ccTilingList, 1, resCtx), HCCL_E_NOT_SUPPORT);
    }
}

TEST_F(CcuMc2TestSuite, CcuSelectAlg_ReduceScatterRejectsProd)
{
    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));

    OpResCtx resCtx{};
    Mc2CcTilingInner ccTiling{};
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    ccTiling.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    ccTiling.srcDataType = HCCL_DATA_TYPE_FP16;
    ccTiling.dstDataType = HCCL_DATA_TYPE_FP16;
    ccTiling.reduceType = HCCL_REDUCE_PROD;
    const void* ccTilingList[] = {&ccTiling};
    std::string topoTag[] = {"tag0"};

    EXPECT_EQ(RunCcuSelectAlg(comm_, stream_, topoTag, ccTilingList, 1, resCtx), HCCL_E_NOT_SUPPORT);
}

TEST_F(CcuMc2TestSuite, CheckCcuKfcFlow_AcceptsMixedOps)
{
    Mc2InitTilingInner initTiling{};
    initTiling.version = INIT_TILING_CCU_NEW_VERSION;
    initTiling.mc2HcommCnt = 2U;
    Mc2CcTilingInner allGather{};
    allGather.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    allGather.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    Mc2CcTilingInner allReduce = allGather;
    allReduce.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE);
    const void* ccTilingList[] = {&allGather, &allReduce};

    EXPECT_EQ(CheckCcuKfcFlow(&initTiling, ccTilingList, 2U, g_stubRankSize), HCCL_SUCCESS);
    EXPECT_EQ(CheckCcuKfcFlow(&initTiling, ccTilingList, 1U, g_stubRankSize), HCCL_E_NOT_SUPPORT);
    initTiling.mc2HcommCnt = 0U;
    EXPECT_EQ(CheckCcuKfcFlow(&initTiling, ccTilingList, 0U, g_stubRankSize), HCCL_E_NOT_SUPPORT);
}

TEST_F(CcuMc2TestSuite, CheckCcuKfcFlow_AcceptsReduceScatter)
{
    Mc2InitTilingInner initTiling{};
    initTiling.version = INIT_TILING_CCU_NEW_VERSION;
    initTiling.mc2HcommCnt = 2U;
    Mc2CcTilingInner reduceScatter0{};
    reduceScatter0.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    reduceScatter0.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    Mc2CcTilingInner reduceScatter1 = reduceScatter0;
    const void* ccTilingList[] = {&reduceScatter0, &reduceScatter1};

    EXPECT_EQ(CheckCcuKfcFlow(&initTiling, ccTilingList, 2U, g_stubRankSize), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, CheckCcuKfcFlow_AcceptsAllGatherReduceScatterMix)
{
    Mc2InitTilingInner initTiling{};
    initTiling.version = INIT_TILING_CCU_NEW_VERSION;
    initTiling.mc2HcommCnt = 2U;
    Mc2CcTilingInner allGather{};
    allGather.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    allGather.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    Mc2CcTilingInner reduceScatter = allGather;
    reduceScatter.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    reduceScatter.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    const void* ccTilingList[] = {&allGather, &reduceScatter};

    EXPECT_EQ(CheckCcuKfcFlow(&initTiling, ccTilingList, 2U, g_stubRankSize), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, CheckCcuKfcFlow_RejectsPeerOnlyOutsideTp2)
{
    Mc2InitTilingInner initTiling{};
    initTiling.version = INIT_TILING_CCU_NEW_VERSION;
    initTiling.mc2HcommCnt = 1U;
    Mc2CcTilingInner reduceScatter{};
    reduceScatter.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    reduceScatter.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    ASSERT_EQ(
        strcpy_s(reduceScatter.algConfig, sizeof(reduceScatter.algConfig), KFC_REDUCE_SCATTER_PEER_ONLY_ALG_NAME), EOK);
    const void* ccTilingList[] = {&reduceScatter};

    EXPECT_EQ(CheckCcuKfcFlow(&initTiling, ccTilingList, 1U, 2U), HCCL_SUCCESS);
    EXPECT_EQ(CheckCcuKfcFlow(&initTiling, ccTilingList, 1U, 8U), HCCL_E_NOT_SUPPORT);
}

TEST_F(CcuMc2TestSuite, ObtainCommEngine_AllCcu)
{
    Mc2CcTilingInner t0{}, t1{};
    t0.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    t1.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    const void* list[] = {&t0, &t1};
    uint8_t engine;
    EXPECT_EQ(ObtainCommEngine(list, 2, engine), HCCL_SUCCESS);
    EXPECT_EQ(engine, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));
}

TEST_F(CcuMc2TestSuite, ObtainCommEngine_AllAicpu)
{
    Mc2CcTilingInner t0{}, t1{};
    t0.commEngine = COMM_ENGINE_AICPU;
    t1.commEngine = COMM_ENGINE_AICPU;
    const void* list[] = {&t0, &t1};
    uint8_t engine;
    EXPECT_EQ(ObtainCommEngine(list, 2, engine), HCCL_SUCCESS);
    EXPECT_EQ(engine, static_cast<uint8_t>(COMM_ENGINE_AICPU));
}

TEST_F(CcuMc2TestSuite, ObtainCommEngine_MixedFail)
{
    Mc2CcTilingInner t0{}, t1{};
    t0.commEngine = COMM_ENGINE_CCU;
    t1.commEngine = COMM_ENGINE_AICPU;
    const void* list[] = {&t0, &t1};
    uint8_t engine;
    EXPECT_EQ(ObtainCommEngine(list, 2, engine), HCCL_E_NOT_SUPPORT);
}

TEST_F(CcuMc2TestSuite, ObtainCommEngine_UnsupportedEngine)
{
    Mc2CcTilingInner t0{};
    t0.commEngine = COMM_ENGINE_AIV;
    const void* list[] = {&t0};
    uint8_t engine;
    EXPECT_EQ(ObtainCommEngine(list, 1, engine), HCCL_E_NOT_SUPPORT);
}

TEST_F(CcuMc2TestSuite, algorithmMap_AllEntries)
{
    EXPECT_EQ(algorithmMap.at("CcuSchedAllGatherSoleMesh"), AlgorithmType::CcuSchedAllGatherSoleMesh);
    EXPECT_EQ(algorithmMap.at("CcuAllGatherMeshMem2Mem1D"), AlgorithmType::CcuAllGatherMeshMem2Mem1D);
    EXPECT_EQ(
        algorithmMap.at("CcuSchedAllGatherConcurMeshNHRMultiLink"),
        AlgorithmType::CcuSchedAllGatherConcurMeshNHRMultiLink);
    EXPECT_EQ(algorithmMap.at("CcuSchedReduceScatterSoleMesh"), AlgorithmType::CcuReduceScatterMeshMem2Mem1D);
    EXPECT_EQ(
        algorithmMap.at("CcuSchedReduceScatterSoleMeshPeerOnly"), AlgorithmType::CcuReduceScatterMeshMem2Mem1DPeerOnly);
    EXPECT_EQ(
        algorithmMap.at("CcuSchedReduceScatterSoleNHRMultiLink"), AlgorithmType::CcuSchedReduceScatterSoleNHRMultiLink);
    EXPECT_EQ(algorithmMap.at("CcuSchedAllToAllSoleMesh"), AlgorithmType::CcuSchedAllToAllSoleMesh);
    EXPECT_EQ(algorithmMap.at("CcuSchedAllToAllMesh1DMultiJetty"), AlgorithmType::CcuSchedAllToAllMesh1DMultiJetty);
    EXPECT_EQ(algorithmMap.at("CcuSchedAllToAllVSoleMesh"), AlgorithmType::CcuSchedAllToAllVSoleMesh);
    EXPECT_EQ(
        algorithmMap.at("CcuSchedAllToAllVSoleMeshMultiJetty"), AlgorithmType::CcuSchedAllToAllVSoleMeshMultiJetty);
    EXPECT_EQ(algorithmMap.at("CcuSchedAllToAllSoleMeshConcurrent"), AlgorithmType::CcuSchedAllToAllSoleMeshConcurrent);
    EXPECT_EQ(
        algorithmMap.at("CcuSchedAllToAllVSoleMeshConcurrent"), AlgorithmType::CcuSchedAllToAllVSoleMeshConcurrent);
    EXPECT_EQ(algorithmMap.at("CcuSchedAllReduceSoleMesh"), AlgorithmType::CcuAllReduceMeshMem2Mem1D);
    EXPECT_EQ(algorithmMap.at("CcuAllGatherMesh1DMem2Mem"), AlgorithmType::CcuAllGatherMeshMem2Mem1D);
    EXPECT_EQ(algorithmMap.at("CcuSchedAllGatherMesh1DMem2Mem"), AlgorithmType::CcuAllGatherMeshMem2Mem1D);
    EXPECT_EQ(
        algorithmMap.at("CcuSchedAllGatherParallelMeshNHRMultiLink"),
        AlgorithmType::CcuSchedAllGatherParallelMeshNHRMultiLink);
    EXPECT_EQ(
        algorithmMap.at("CcuSchedReduceScatterConcurMeshNHRMultiLink"),
        AlgorithmType::CcuSchedReduceScatterConcurMeshNHRMultiLink);
    EXPECT_EQ(algorithmMap.at("CcuSchedReduceScatterSoleNHR"), AlgorithmType::CcuSchedReduceScatterSoleNHR);
    EXPECT_EQ(algorithmMap.at("CcuSchedAllGatherSoleNHR"), AlgorithmType::CcuSchedAllGatherSoleNHR);
    EXPECT_EQ(algorithmMap.at("CcuSchedAllReduceSoleNHR"), AlgorithmType::CcuSchedAllReduceSoleNHR);
    EXPECT_EQ(algorithmMap.at("CcuHalfAll2AllVMesh1D"), AlgorithmType::CcuHalfAll2AllVMesh1D);
    EXPECT_EQ(algorithmMap.size(), 21U);
}

TEST_F(CcuMc2TestSuite, AlgorithmType_EnumValues)
{
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuAllGatherMesh1D), 0U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuAllGatherMeshMem2Mem1D), 1U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuAllGatherMesh2D), 2U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuSchedAllGatherConcurMeshNHRMultiLink), 3U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuReduceScatterMesh1D), 50U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuReduceScatterMeshMem2Mem1D), 51U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuReduceScatterMesh2D), 52U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuReduceScatterMeshMem2Mem1DPeerOnly), 53U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuAllReduceMesh1D), 100U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuAllReduceMeshMem2Mem1D), 101U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuAllReduceMesh2DOneShot), 102U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuSchedAllToAllSoleMesh), 150U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuSchedAllToAllVSoleMesh), 151U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuSchedAllGatherSoleMesh), 153U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuSchedAllToAllMesh1DMultiJetty), 157U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuSchedReduceScatterSoleNHRMultiLink), 154U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuSchedAllToAllSoleMeshConcurrent), 155U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuSchedReduceScatterConcurMeshNHRMultiLink), 156U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuSchedReduceScatterSoleNHR), 158U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuSchedAllGatherSoleNHR), 159U);
    // 161 已被 upstream 并发 A2AV（Concurrent）占用，MultiJetty 顺延 162
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuSchedAllToAllVSoleMeshConcurrent), 161U);
    EXPECT_EQ(static_cast<uint32_t>(AlgorithmType::CcuSchedAllToAllVSoleMeshMultiJetty), 162U);
}

TEST_F(CcuMc2TestSuite, HcclAllocComResourceByTiling_CcuPath)
{
    uint32_t opTypes[] = {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER)};
    Mc2TilingTestData tiling = BuildMc2Tiling(1, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), opTypes, nullptr);
    // 外部名=强制算法定名
    strcpy(tiling.ccTiling[0].algConfig, "sole[mesh]");
    void* opResCtx = nullptr;
    HcclResult ret = HcclAllocComResourceByTiling(comm_, stream_, &tiling, &opResCtx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_NE(opResCtx, nullptr);
    OpResCtx* ctx = static_cast<OpResCtx*>(opResCtx);
    EXPECT_EQ(ctx->rankId, g_stubRankId);
    EXPECT_EQ(ctx->rankSize, g_stubRankSize);
    EXPECT_EQ(ctx->workSpaceSize, Hccl::MC2_WORKSPACE_SIZE);
    EXPECT_NE(ctx->workSpace, 0U);
    EXPECT_NE(ctx->xnAddr, 0U);
    EXPECT_NE(ctx->ckeAddr, 0U);
    EXPECT_EQ(ctx->opType[0], static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER));
    EXPECT_EQ(ctx->algorithmType[0], static_cast<uint32_t>(AlgorithmType::CcuSchedAllGatherSoleMesh));
}

// 定名未注册：algorithmMap 未命中按 NOT_SUPPORT 直接失败。
TEST_F(CcuMc2TestSuite, HcclAllocComResourceByTiling_AlgNameNotInMapFails)
{
    uint32_t opTypes[] = {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER)};
    Mc2TilingTestData tiling = BuildMc2Tiling(1, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), opTypes, nullptr);
    // 老语法归 LEGACY 回退 selector，故改写使 selector 定名路径生效。
    strcpy(tiling.ccTiling[0].algConfig, "AllReduce=level0:doublering");
    // selector 定名结果未注册
    mc2_ops_hccl::g_stubSelectorAlgName = "NotExistAlg";

    void* opResCtx = nullptr;
    HcclResult ret = HcclAllocComResourceByTiling(comm_, stream_, &tiling, &opResCtx);

    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(opResCtx, nullptr);
}

// 申请路径不经注册闸门：裸名交由 selector 定名
TEST_F(CcuMc2TestSuite, HcclAllocComResourceByTiling_BareNameUsesSelector)
{
    uint32_t opTypes[] = {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER)};
    // 默认填入裸名 CcuSchedAllGatherSoleMesh
    Mc2TilingTestData tiling = BuildMc2Tiling(1, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), opTypes, nullptr);

    void* opResCtx = nullptr;
    EXPECT_EQ(HcclAllocComResourceByTiling(comm_, stream_, &tiling, &opResCtx), HCCL_SUCCESS);
    ASSERT_NE(opResCtx, nullptr);
    EXPECT_EQ(
        static_cast<OpResCtx*>(opResCtx)->algorithmType[0],
        static_cast<uint32_t>(AlgorithmType::CcuSchedAllGatherSoleMesh));
}

TEST_F(CcuMc2TestSuite, HcclAllocComResourceByTiling_ReduceScatterCcuPath)
{
    uint32_t opTypes[] = {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER)};
    Mc2TilingTestData tiling = BuildMc2Tiling(1, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), opTypes, nullptr);
    // 外部名=强制算法定名
    strcpy(tiling.ccTiling[0].algConfig, "sole[mesh]");
    void* opResCtx = nullptr;
    HcclResult ret = HcclAllocComResourceByTiling(comm_, stream_, &tiling, &opResCtx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_NE(opResCtx, nullptr);
    OpResCtx* ctx = static_cast<OpResCtx*>(opResCtx);
    EXPECT_NE(ctx->res[0], 0U);
    EXPECT_EQ(ctx->opType[0], static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
    EXPECT_EQ(ctx->algorithmType[0], static_cast<uint32_t>(AlgorithmType::CcuReduceScatterMeshMem2Mem1D));
}

// 强制算法不可用或校验失败则回退默认 selector。
TEST_F(CcuMc2TestSuite, CcuSelectAlg_ForcedAlgFallbackToSelector)
{
    SetCommEngineEnv(COMM_ENGINE_CCU);

    Mc2CcTilingInner ccTiling{};
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    ccTiling.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    ccTiling.srcDataType = HCCL_DATA_TYPE_FP16;
    ccTiling.dstDataType = HCCL_DATA_TYPE_FP16;
    ccTiling.reduceType = HCCL_REDUCE_SUM;
    // 合法强制算法外部名（裸名归 NONE）
    strcpy(ccTiling.algConfig, "sole[mesh]");

    // 场景1：执行器不可用→校验失败→回退默认 selector
    mc2_ops_hccl::g_stubCcuAlgExecNull = true;
    mc2_ops_hccl::g_stubCcuAlgExecNullName = "CcuSchedAllGatherSoleMesh";
    OpParam opParam{};
    ASSERT_EQ(InitOpParamByTiling(comm_, stream_, "allgather_tag", &ccTiling, opParam), HCCL_SUCCESS);
    std::string algName;
    auto topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    bool forcedAlgAccepted = true;
    HcclResult ret = TryForcedAlgAndPrepareEngine(comm_, &ccTiling, opParam, algName, topoInfo, forcedAlgAccepted);
    // 强制算法失败后回退，不向上报错，由调用方走默认selector重新选择
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(forcedAlgAccepted);
    EXPECT_TRUE(algName.empty());

    // 场景2：强制算法可用 → 接受强制算法
    mc2_ops_hccl::g_stubCcuAlgExecNull = false;
    OpParam opParam2{};
    ASSERT_EQ(InitOpParamByTiling(comm_, stream_, "allgather_tag", &ccTiling, opParam2), HCCL_SUCCESS);
    std::string algName2;
    auto topoInfo2 = std::make_unique<TopoInfoWithNetLayerDetails>();
    bool forcedAlgAccepted2 = false;
    HcclResult ret2 = TryForcedAlgAndPrepareEngine(comm_, &ccTiling, opParam2, algName2, topoInfo2, forcedAlgAccepted2);
    EXPECT_EQ(ret2, HCCL_SUCCESS);
    EXPECT_TRUE(forcedAlgAccepted2);
    EXPECT_EQ(algName2, "CcuSchedAllGatherSoleMesh");
}

TEST_F(CcuMc2TestSuite, ForcedBridgeAlgorithmUsesMetadataAndV2)
{
    using namespace mc2_ops_hccl;
    for (auto op : {HcclCMDType::HCCL_CMD_ALLREDUCE, HcclCMDType::HCCL_CMD_ALLTOALL, HcclCMDType::HCCL_CMD_ALLTOALLV}) {
        for (auto engine : {COMM_ENGINE_AICPU, COMM_ENGINE_AICPU_TS}) {
            g_cannBridgeTestState = {};
            g_cannBridgeTestState.enabled = true;
            OpParam param{};
            param.opType = op;
            param.engine = engine;
            TopoInfoWithNetLayerDetails topo;
            const std::string algName = "AicpuAllReduceParallelMeshNHR";
            EXPECT_EQ(param.algName[0], '\0');
            EXPECT_EQ(CheckForcedAlgResource(comm_, param, &topo, algName), HCCL_SUCCESS);
            EXPECT_EQ(g_cannBridgeTestState.metadataName, algName);
            EXPECT_EQ(g_cannBridgeTestState.legacyCalls, 0U);
            EXPECT_EQ(g_cannBridgeTestState.v2Calls, 1U);
            EXPECT_EQ(g_cannBridgeTestState.resourceCalls, 1U);
        }
    }
}

TEST_F(CcuMc2TestSuite, ForcedBridgeAlgorithmAcceptanceAndFallback)
{
    using namespace mc2_ops_hccl;
    // -1 succeeds; 0/1/2 fail executor lookup, topology matching, and resource calculation.
    for (int stage = -1; stage < 3; ++stage) {
        g_cannBridgeTestState = {};
        g_cannBridgeTestState.enabled = true;
        g_cannBridgeTestState.missingExecutor = stage == 0;
        g_cannBridgeTestState.hierarchyResult = stage == 1 ? HCCL_E_PARA : HCCL_SUCCESS;
        g_cannBridgeTestState.resourceResult = stage == 2 ? HCCL_E_UNAVAIL : HCCL_SUCCESS;
        Mc2CcTilingInner tiling{};
        // 新契约：强制算法只认外部名；AICPU 引擎下用夹具注册的 (ALLREDUCE, sole[mesh]) -> InsAllReduceMesh1D
        strcpy(tiling.algConfig, "sole[mesh]");
        OpParam param{};
        param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
        param.engine = COMM_ENGINE_AICPU_TS;
        std::string algName;
        auto topo = std::make_unique<TopoInfoWithNetLayerDetails>();
        bool accepted = true;
        ASSERT_EQ(TryForcedAlgAndPrepareEngine(comm_, &tiling, param, algName, topo, accepted), HCCL_SUCCESS);
        EXPECT_EQ(accepted, stage == -1);
        EXPECT_EQ(algName, stage == -1 ? "InsAllReduceMesh1D" : "");
        EXPECT_EQ(param.engine, COMM_ENGINE_AICPU_TS);
        EXPECT_EQ(g_cannBridgeTestState.lookupCalls, 1U);
        EXPECT_EQ(g_cannBridgeTestState.resourceCalls, stage == -1 || stage == 2 ? 1U : 0U);
    }
}

TEST_F(CcuMc2TestSuite, AutomaticAndLegacyConfigSkipForcedBridgePrecheck)
{
    using namespace mc2_ops_hccl;
    g_cannBridgeTestState.enabled = true;
    for (const char* config : {"", "AllReduce=level0:doublering"}) {
        Mc2CcTilingInner tiling{};
        strcpy(tiling.algConfig, config);
        OpParam param{};
        param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
        param.engine = COMM_ENGINE_AICPU_TS;
        std::string algName;
        auto topo = std::make_unique<TopoInfoWithNetLayerDetails>();
        bool accepted = true;
        ASSERT_EQ(TryForcedAlgAndPrepareEngine(comm_, &tiling, param, algName, topo, accepted), HCCL_SUCCESS);
        EXPECT_FALSE(accepted);
        EXPECT_EQ(g_cannBridgeTestState.lookupCalls, 0U);
        EXPECT_EQ(g_cannBridgeTestState.v2Calls, 0U);
    }
}

// EXTERNAL 坏配置：与裸名未注册同路——WARNING + false，调用方走默认 selector
TEST_F(CcuMc2TestSuite, TryForcedAlgExternalParseFailFallsBackToSelector)
{
    // 语法错：不报错中止，回退 selector
    Mc2CcTilingInner tiling{};
    tiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    tiling.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    strcpy(tiling.algConfig, "sole[ring]");
    OpParam opParam{};
    std::string algName;
    auto topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    bool accepted = true;
    EXPECT_EQ(TryForcedAlgAndPrepareEngine(comm_, &tiling, opParam, algName, topoInfo, accepted), HCCL_SUCCESS);
    EXPECT_FALSE(accepted);
    EXPECT_TRUE(algName.empty());
}

TEST_F(CcuMc2TestSuite, TryForcedAlgExternalResolveFailFallsBackToSelector)
{
    // 语义错（无 3 层注册候选）：WARNING + false 回退 selector，且 opParam 复位到入口现场
    Mc2CcTilingInner tiling{};
    tiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    tiling.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    strcpy(tiling.algConfig, "pipeline[mesh,nhr,nhr]");
    OpParam opParam{};
    opParam.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
    opParam.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    std::string algName;
    auto topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    bool accepted = true;
    EXPECT_EQ(TryForcedAlgAndPrepareEngine(comm_, &tiling, opParam, algName, topoInfo, accepted), HCCL_SUCCESS);
    EXPECT_FALSE(accepted);
    EXPECT_TRUE(algName.empty());
    // hcclComm 入口为 nullptr 而 PrepareTopoInfoForOp 会改写为 comm，
    // 此断言钉住失败路径 opParam 复位真正生效
    EXPECT_EQ(opParam.hcclComm, nullptr);
    EXPECT_EQ(opParam.engine, CommEngine::COMM_ENGINE_AICPU_TS);
}

TEST_F(CcuMc2TestSuite, TryForcedAlgExternalTopoFailFallsBackToSelector)
{
    // topo 准备失败（HcclCalcTopoInfo 打桩失败）：WARNING + false 回退 selector，且 opParam 复位到入口现场
    Mc2CcTilingInner tiling{};
    tiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    tiling.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    strcpy(tiling.algConfig, "sole[mesh]");
    OpParam opParam{};
    opParam.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
    opParam.engine = CommEngine::COMM_ENGINE_CCU; // 入口值与 topo 准备的改写值不同，钉住复位
    std::string algName;
    auto topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    bool accepted = true;

    mc2_ops_hccl::g_stubCalcTopoInfoResult = HcclResult::HCCL_E_INTERNAL;
    EXPECT_EQ(TryForcedAlgAndPrepareEngine(comm_, &tiling, opParam, algName, topoInfo, accepted), HCCL_SUCCESS);
    mc2_ops_hccl::g_stubCalcTopoInfoResult = HcclResult::HCCL_SUCCESS;

    EXPECT_FALSE(accepted);
    EXPECT_TRUE(algName.empty());
    // topo 准备会改写 hcclComm(→comm) 与 engine(→AICPU_TS)，复位后须回到入口值
    EXPECT_EQ(opParam.hcclComm, nullptr);
    EXPECT_EQ(opParam.engine, CommEngine::COMM_ENGINE_CCU);
}

// CCU 闸门语义：只放行外部名；裸名/缺省/老语法归 NONE 由 selector 回退。

TEST_F(CcuMc2TestSuite, GetOpParam_ExternalNameResourceFallbackToSelector)
{
    Mc2CcTilingInner tiling{};
    tiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    tiling.commEngine = static_cast<uint8_t>(OpExecuteConfig::AICPU_TS);
    tiling.srcDataType = HCCL_DATA_TYPE_FP16;
    tiling.dstDataType = HCCL_DATA_TYPE_FP16;
    tiling.reduceType = HCCL_REDUCE_SUM;
    strcpy(tiling.algConfig, "sole[mesh]");

    // 仅定名算法（夹具 AICPU 行 InsAllGatherMesh1D）资源不足；回退算法资源正常
    mc2_ops_hccl::g_stubCcuAlgResUnavailableName = "InsAllGatherMesh1D";
    mc2_ops_hccl::g_stubSelectorAlgName = "AicpuAllGatherSoleNHR";

    OpParam opParam{};
    HcclResult ret = GetOpParam(comm_, stream_, "external_res_fallback_tag", &tiling, opParam);

    // 回退后 selector 选定算法，作业继续（与裸名资源回退行为一致）
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(opParam.algName, "AicpuAllGatherSoleNHR");
}

TEST_F(CcuMc2TestSuite, LocalForcedAlgorithmKeepsLegacyHierarchy)
{
    using namespace mc2_ops_hccl;
    g_cannBridgeTestState.enabled = true;
    for (auto op : {HcclCMDType::HCCL_CMD_ALLGATHER, HcclCMDType::HCCL_CMD_REDUCE_SCATTER}) {
        OpParam param{};
        param.opType = op;
        param.engine = COMM_ENGINE_AICPU_TS;
        TopoInfoWithNetLayerDetails topo;
        EXPECT_EQ(CheckForcedAlgResource(comm_, param, &topo, "local_algorithm"), HCCL_SUCCESS);
    }
    OpParam param{};
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    param.engine = COMM_ENGINE_CCU;
    TopoInfoWithNetLayerDetails topo;
    EXPECT_EQ(CheckForcedAlgResource(comm_, param, &topo, "ccu_algorithm"), HCCL_SUCCESS);
    EXPECT_EQ(g_cannBridgeTestState.lookupCalls, 0U);
    EXPECT_EQ(g_cannBridgeTestState.v2Calls, 0U);
}

TEST_F(CcuMc2TestSuite, HcclAllocComResourceByTiling_AicpuPath)
{
    uint32_t opTypes[] = {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER)};
    Mc2TilingTestData tiling = BuildMc2Tiling(1, COMM_ENGINE_AICPU, opTypes, nullptr);
    void* opResCtx = nullptr;
    HcclResult ret = HcclAllocComResourceByTiling(comm_, stream_, &tiling, &opResCtx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_NE(opResCtx, nullptr);
}

TEST_F(CcuMc2TestSuite, InitOpParamByTiling_SetsExpansionMode)
{
    Mc2CcTilingInner ccTiling{};
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    ccTiling.srcDataType = HCCL_DATA_TYPE_FP16;
    ccTiling.dstDataType = HCCL_DATA_TYPE_FP16;
    ccTiling.reduceType = HCCL_REDUCE_SUM;

    ccTiling.commEngine = static_cast<uint8_t>(OpExecuteConfig::AICPU_TS);
    OpParam aicpuOpParam{};
    ASSERT_EQ(InitOpParamByTiling(comm_, stream_, "aicpu_tag", &ccTiling, aicpuOpParam), HCCL_SUCCESS);
    EXPECT_EQ(aicpuOpParam.commOpExpansionMode, HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AI_CPU);

    ccTiling.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    OpParam ccuOpParam{};
    ASSERT_EQ(InitOpParamByTiling(comm_, stream_, "ccu_tag", &ccTiling, ccuOpParam), HCCL_SUCCESS);
    EXPECT_EQ(ccuOpParam.commOpExpansionMode, HcclOpExpansionMode::HCCL_OP_EXPANSION_CCU_SCHED);
}

TEST_F(CcuMc2TestSuite, HcclAllocComResourceByTiling_SingleAndMultiTiling)
{
    uint32_t opTypes[] = {
        static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER), static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE)};
    // 申请路径不经注册闸门，单/多算子均应成功。
    for (uint32_t tilingNum : {1U, 2U}) {
        SCOPED_TRACE(::testing::Message() << "tilingNum=" << tilingNum);
        auto tiling = BuildMc2Tiling(tilingNum, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), opTypes, nullptr);
        for (auto& cc : tiling.ccTiling) {
            strcpy(cc.algConfig, "sole[mesh]");
        }
        void* opResCtx = nullptr;
        const HcclResult ret = HcclAllocComResourceByTiling(comm_, stream_, &tiling, &opResCtx);
        ASSERT_EQ(ret, HCCL_SUCCESS);
        ASSERT_NE(opResCtx, nullptr);
        const auto& ctx = *static_cast<OpResCtx*>(opResCtx);
        EXPECT_EQ(ctx.resCtx, 0U);
        const auto& first = *reinterpret_cast<const OpParam*>(ctx.algInfo[0].opParam);
        for (uint32_t i = 0; i < tilingNum; ++i) {
            const auto& param = *reinterpret_cast<const OpParam*>(ctx.algInfo[i].opParam);
            EXPECT_EQ(ctx.algInfo[i].offset, tiling.offset[i]);
            EXPECT_EQ(param.resCtx, first.resCtx);
            std::vector<char> seq(
                static_cast<const char*>(param.resCtx), static_cast<const char*>(param.resCtx) + param.ctxSize);
            AlgResourceCtxSerializable resources;
            resources.DeSerialize(seq);
            EXPECT_EQ(resources.kfcServerArgs[0], ctx.xnAddr);
            EXPECT_EQ(resources.kfcServerArgs[1], ctx.ckeAddr);
        }
        EXPECT_EQ(ctx.res[1], 1000U);
        const uint32_t launches = mc2_ops_hccl::g_stubHcommCcuKernelLaunchCallCount;
        ASSERT_EQ(CcuKernelLaunch(comm_, opResCtx), CCU_SUCCESS);
        EXPECT_EQ(mc2_ops_hccl::g_stubHcommCcuKernelLaunchCallCount, launches + 1U);
        ASSERT_EQ(mc2_ops_hccl::g_stubHcommCcuKernelLaunchArgs.size(), launches + 1U);
        EXPECT_EQ(mc2_ops_hccl::g_stubHcommCcuKernelLaunchArgs.back()[0], ctx.xnAddr);
        EXPECT_EQ(Mc2CcKernelLaunch(stream_, opResCtx, sizeof(OpResCtx)), HCCL_SUCCESS);
    }
}

TEST_F(CcuMc2TestSuite, HcclAllocComResourceByTiling_NullComm)
{
    uint32_t opTypes[] = {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER)};
    Mc2TilingTestData tiling = BuildMc2Tiling(1, COMM_ENGINE_CCU, opTypes, nullptr);
    void* opResCtx = nullptr;
    EXPECT_EQ(HcclAllocComResourceByTiling(nullptr, stream_, &tiling, &opResCtx), HCCL_E_PTR);
}

TEST_F(CcuMc2TestSuite, HcclAllocComResourceByTiling_NullStream)
{
    uint32_t opTypes[] = {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER)};
    Mc2TilingTestData tiling = BuildMc2Tiling(1, COMM_ENGINE_CCU, opTypes, nullptr);
    void* opResCtx = nullptr;
    EXPECT_EQ(HcclAllocComResourceByTiling(comm_, nullptr, &tiling, &opResCtx), HCCL_E_PTR);
}

TEST_F(CcuMc2TestSuite, HcclAllocComResourceByTiling_NullTiling)
{
    void* opResCtx = nullptr;
    EXPECT_EQ(HcclAllocComResourceByTiling(comm_, stream_, nullptr, &opResCtx), HCCL_E_PTR);
}

TEST_F(CcuMc2TestSuite, CcuSelectAlg_AicpuAlgNotSupported)
{
    // tiling标记为CCU引擎，但环境变量设为AICPU模式，
    // 导致SelectAlgAndPrepareEngine选出AICPU算法名（如"InsAllGatherMesh"），
    // 该算法名不在CCU的algorithmMap中，CcuSelectAlg应返回HCCL_E_NOT_SUPPORT
    SetCommEngineEnv(COMM_ENGINE_AICPU);
    Mc2CcTilingInner ccTiling{};
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    ccTiling.commEngine = COMM_ENGINE_CCU;
    const void* ccTilingList[] = {&ccTiling};
    std::string topoTag[] = {"tag0"};
    OpResCtx resCtx{};
    EXPECT_EQ(RunCcuSelectAlg(comm_, stream_, topoTag, ccTilingList, 1, resCtx), HCCL_E_NOT_SUPPORT);
}

TEST_F(CcuMc2TestSuite, HcclAllocComResourceByTiling_UnsupportedEngine)
{
    uint32_t opTypes[] = {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER)};
    Mc2TilingTestData tiling = BuildMc2Tiling(1, COMM_ENGINE_AIV, opTypes, nullptr);
    void* opResCtx = nullptr;
    EXPECT_EQ(HcclAllocComResourceByTiling(comm_, stream_, &tiling, &opResCtx), HCCL_E_NOT_SUPPORT);
}

TEST_F(CcuMc2TestSuite, GetCcuOpParamResCtx_AllGatherMesh1D_KfcServerArgs)
{
    SetCommEngineEnv(COMM_ENGINE_CCU);

    // 动态获取CCL buffer地址
    void* cclBufferPtr = nullptr;
    uint64_t cclBufferSize = 0;
    ASSERT_EQ(HcclGetHcclBuffer(comm_, &cclBufferPtr, &cclBufferSize), HCCL_SUCCESS);
    uint64_t cclBufferAddr = reinterpret_cast<uint64_t>(cclBufferPtr);

    // 1. 分配CCU资源，获取opResCtx（含xnAddr、ckeAddr）
    OpResCtx opResCtx{};
    ASSERT_EQ(AllocCcuOpResCtx(comm_, "ctx_tag", g_stubRankSize, g_stubRankId, opResCtx), HCCL_SUCCESS);
    ASSERT_NE(opResCtx.xnAddr, 0U);
    ASSERT_NE(opResCtx.ckeAddr, 0U);

    // 2. 构造OpParam，模拟CcuSelectAlg中的准备流程
    Mc2CcTilingInner ccTiling{};
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    ccTiling.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    ccTiling.srcDataType = HCCL_DATA_TYPE_FP16;
    ccTiling.dstDataType = HCCL_DATA_TYPE_FP16;
    ccTiling.reduceType = HCCL_REDUCE_SUM;
    strcpy(ccTiling.algConfig, "CcuSchedAllGatherSoleMesh");

    OpParam opParam{};
    ASSERT_EQ(InitOpParamByTiling(comm_, stream_, "allgather_tag", &ccTiling, opParam), HCCL_SUCCESS);

    std::string algName;
    auto topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    // 外部名定名（走 ResolveExternalAlgName 同构路径：解析 → topo → 消歧）
    algName = "CcuSchedAllGatherSoleMesh";
    ASSERT_EQ(PrepareTopoInfoForOp(comm_, opParam, topoInfo), HCCL_SUCCESS);
    ASSERT_EQ(PrepareEngineForAlg(opParam, algName), HCCL_SUCCESS);
    ASSERT_EQ(algName, "CcuSchedAllGatherSoleMesh");
    ASSERT_EQ(opParam.engine, COMM_ENGINE_CCU);

    int result = sprintf_s(opParam.algName, sizeof(opParam.algName), "%s", algName.c_str());
    ASSERT_GT(result, 0);

    bool skipGetRes = false;
    ASSERT_EQ(HandleSingleRankAndCommMode(comm_, opParam, skipGetRes), HCCL_SUCCESS);
    ASSERT_FALSE(skipGetRes);

    // 3. 调用GetCcuOpParamResCtx
    auto executor = CollAlgExecRegistryV2::Instance().GetAlgExec(opParam.opType, algName);
    ASSERT_NE(executor, nullptr);
    AlgHierarchyInfoForAllLevel hierarchy;
    AlgResourceRequest request;
    ASSERT_EQ(CalcAlgResourceRequest(comm_, opParam, executor, topoInfo.get(), hierarchy, request), HCCL_SUCCESS);
    void* resCtxOut = nullptr;
    EXPECT_EQ(
        GetCcuOpParamResCtx(comm_, algName, opParam, topoInfo.get(), opResCtx, &resCtxOut, &request), HCCL_SUCCESS);
    EXPECT_NE(resCtxOut, nullptr);
    EXPECT_EQ(opParam.resCtx, resCtxOut);

    // 4. 通过反序列化resCtxOut还原resourceCtx，验证kfcServerArgs各字段
    ASSERT_NE(opParam.ctxSize, 0U);
    auto* resCtxRaw = static_cast<char*>(resCtxOut);
    std::vector<char> seq(resCtxRaw, resCtxRaw + opParam.ctxSize);
    mc2_ops_hccl::AlgResourceCtxSerializable resourceCtx;
    resourceCtx.DeSerialize(seq);

    // kfcServerArgs应包含6个uint64_t元素（xnAddr, ckeAddr, dieNum, missionNum, missionIndex, token）
    // AcquireAlgResources 预填充所有6字段（token 占位符为0）
    // GetAlgResCcu 内的 HcclAllocAlgResourceCcu 更新占位符为真实 token
    ASSERT_EQ(resourceCtx.kfcServerArgs.size(), 6U);
    EXPECT_EQ(resourceCtx.kfcServerArgSize, 6U);

    // xnAddr
    EXPECT_EQ(resourceCtx.kfcServerArgs[0], opResCtx.xnAddr);
    // ckeAddr
    EXPECT_EQ(resourceCtx.kfcServerArgs[1], opResCtx.ckeAddr);
    // dieNum = 1
    EXPECT_EQ(resourceCtx.kfcServerArgs[2], 1U);
    // missionNum = 1
    EXPECT_EQ(resourceCtx.kfcServerArgs[3], 1U);
    // missionIndex = 0
    EXPECT_EQ(resourceCtx.kfcServerArgs[4], 0U);
    // token (由GetTokenFromBuffInfo获取，stub返回基于CCL buffer地址的非零值)
    uint64_t expectedToken = cclBufferAddr + 0x1234;
    EXPECT_EQ(resourceCtx.kfcServerArgs[5], expectedToken);
}

TEST_F(CcuMc2TestSuite, GetCcuOpParamResCtx_TokenUpdateOnReuse)
{
    SetCommEngineEnv(COMM_ENGINE_CCU);

    // 动态获取CCL buffer地址
    void* cclBufferPtr = nullptr;
    uint64_t cclBufferSize = 0;
    ASSERT_EQ(HcclGetHcclBuffer(comm_, &cclBufferPtr, &cclBufferSize), HCCL_SUCCESS);
    uint64_t cclBufferAddr = reinterpret_cast<uint64_t>(cclBufferPtr);
    uint64_t expectedToken = cclBufferAddr + 0x1234;

    // 1. 第一次调用：创建资源并写入token
    OpResCtx opResCtx1{};
    ASSERT_EQ(AllocCcuOpResCtx(comm_, "ctx_tag1", g_stubRankSize, g_stubRankId, opResCtx1), HCCL_SUCCESS);

    Mc2CcTilingInner ccTiling{};
    ccTiling.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    ccTiling.commEngine = static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED);
    ccTiling.srcDataType = HCCL_DATA_TYPE_FP16;
    ccTiling.dstDataType = HCCL_DATA_TYPE_FP16;
    ccTiling.reduceType = HCCL_REDUCE_SUM;
    strcpy(ccTiling.algConfig, "CcuSchedAllGatherSoleMesh");

    OpParam opParam1{};
    ASSERT_EQ(InitOpParamByTiling(comm_, stream_, "reuse_tag", &ccTiling, opParam1), HCCL_SUCCESS);

    std::string algName;
    auto topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    // 外部名定名（走 ResolveExternalAlgName 同构路径：解析 → topo → 消歧）
    algName = "CcuSchedAllGatherSoleMesh";
    ASSERT_EQ(PrepareTopoInfoForOp(comm_, opParam1, topoInfo), HCCL_SUCCESS);
    ASSERT_EQ(PrepareEngineForAlg(opParam1, algName), HCCL_SUCCESS);
    ASSERT_EQ(algName, "CcuSchedAllGatherSoleMesh");

    int result = sprintf_s(opParam1.algName, sizeof(opParam1.algName), "%s", algName.c_str());
    ASSERT_GT(result, 0);

    bool skipGetRes = false;
    ASSERT_EQ(HandleSingleRankAndCommMode(comm_, opParam1, skipGetRes), HCCL_SUCCESS);
    ASSERT_FALSE(skipGetRes);

    void* resCtxOut1 = nullptr;
    auto executor1 = CollAlgExecRegistryV2::Instance().GetAlgExec(opParam1.opType, algName);
    ASSERT_NE(executor1, nullptr);
    AlgHierarchyInfoForAllLevel hierarchy1;
    AlgResourceRequest request1;
    ASSERT_EQ(CalcAlgResourceRequest(comm_, opParam1, executor1, topoInfo.get(), hierarchy1, request1), HCCL_SUCCESS);
    EXPECT_EQ(
        GetCcuOpParamResCtx(comm_, algName, opParam1, topoInfo.get(), opResCtx1, &resCtxOut1, &request1), HCCL_SUCCESS);
    EXPECT_NE(resCtxOut1, nullptr);

    // 验证第一次的token
    auto* ctxRaw1 = static_cast<char*>(resCtxOut1);
    std::vector<char> seq1(ctxRaw1, ctxRaw1 + opParam1.ctxSize);
    mc2_ops_hccl::AlgResourceCtxSerializable resourceCtx1;
    resourceCtx1.DeSerialize(seq1);
    EXPECT_EQ(resourceCtx1.kfcServerArgs.size(), 6U);
    EXPECT_EQ(resourceCtx1.kfcServerArgs[5], expectedToken);

    // 2. 第二次调用相同配置（相同algTag）：触发资源复用
    // 注意：必须使用相同的topo tag ("reuse_tag") 才能生成相同的algTag
    OpResCtx opResCtx2{};
    ASSERT_EQ(AllocCcuOpResCtx(comm_, "ctx_tag2", g_stubRankSize, g_stubRankId, opResCtx2), HCCL_SUCCESS);

    OpParam opParam2{};
    ASSERT_EQ(InitOpParamByTiling(comm_, stream_, "reuse_tag", &ccTiling, opParam2), HCCL_SUCCESS); // 相同tag
    algName = "CcuSchedAllGatherSoleMesh";
    ASSERT_EQ(PrepareTopoInfoForOp(comm_, opParam2, topoInfo), HCCL_SUCCESS);
    ASSERT_EQ(PrepareEngineForAlg(opParam2, algName), HCCL_SUCCESS);
    result = sprintf_s(opParam2.algName, sizeof(opParam2.algName), "%s", algName.c_str());
    ASSERT_GT(result, 0);
    ASSERT_EQ(HandleSingleRankAndCommMode(comm_, opParam2, skipGetRes), HCCL_SUCCESS);
    ASSERT_FALSE(skipGetRes);

    void* resCtxOut2 = nullptr;
    auto executor2 = CollAlgExecRegistryV2::Instance().GetAlgExec(opParam2.opType, algName);
    ASSERT_NE(executor2, nullptr);
    AlgHierarchyInfoForAllLevel hierarchy2;
    AlgResourceRequest request2;
    ASSERT_EQ(CalcAlgResourceRequest(comm_, opParam2, executor2, topoInfo.get(), hierarchy2, request2), HCCL_SUCCESS);
    EXPECT_EQ(
        GetCcuOpParamResCtx(comm_, algName, opParam2, topoInfo.get(), opResCtx2, &resCtxOut2, &request2), HCCL_SUCCESS);
    EXPECT_NE(resCtxOut2, nullptr);

    // 验证资源复用（相同的ctx地址）
    EXPECT_EQ(resCtxOut1, resCtxOut2);

    // 验证第二次的token（资源复用后仍保持正确值）
    auto* ctxRaw2 = static_cast<char*>(resCtxOut2);
    std::vector<char> seq2(ctxRaw2, ctxRaw2 + opParam2.ctxSize);
    mc2_ops_hccl::AlgResourceCtxSerializable resourceCtx2;
    resourceCtx2.DeSerialize(seq2);
    EXPECT_EQ(resourceCtx2.kfcServerArgs.size(), 6U);
    EXPECT_EQ(resourceCtx2.kfcServerArgs[5], expectedToken);
}

// ---------- CheckOpResSufficient ----------
// checkOnly预检：走真实资源链探测资源是否充足，UNAVAIL统一翻译为HCCL_E_RES_NOT_SUFFICIENT

TEST_F(CcuMc2TestSuite, CheckOpResSufficient_CcuSufficient)
{
    uint32_t opTypes[] = {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER)};
    Mc2TilingTestData tiling = BuildMc2Tiling(1, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), opTypes, nullptr);
    // 新契约：CCU 路径只认外部名（裸名归为 NONE，闸门直接拒绝）
    strcpy(tiling.ccTiling[0].algConfig, "sole[mesh]");

    HcclResult ret = CheckOpResSufficient(comm_, stream_, &tiling);

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 端到端验证UNAVAIL→RES_NOT_SUFFICIENT映射：打桩HcclGetAlgRes返回UNAVAIL（模拟CCU通道/实例资源不足），
// checkOnly场景在AcquireAlgResources调用点统一翻译为HCCL_E_RES_NOT_SUFFICIENT；
// 非checkOnly场景（HcclAllocComResourceByTiling）保持UNAVAIL原样返回，不受翻译影响。
TEST_F(CcuMc2TestSuite, CheckOpResSufficient_ResourceInsufficient)
{
    uint32_t opTypes[] = {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER)};
    Mc2TilingTestData tiling = BuildMc2Tiling(1, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), opTypes, nullptr);
    // 新契约：CCU 路径只认外部名（裸名归为 NONE，闸门直接拒绝）
    strcpy(tiling.ccTiling[0].algConfig, "sole[mesh]");
    mc2_ops_hccl::g_stubCcuAlgResUnavailable = true;

    // checkOnly：UNAVAIL 统一翻译为 HCCL_E_RES_NOT_SUFFICIENT
    EXPECT_EQ(CheckOpResSufficient(comm_, stream_, &tiling), HCCL_E_RES_NOT_SUFFICIENT);

    // 非checkOnly：UNAVAIL 原样返回，保持与原逻辑一致
    void* opResCtx = nullptr;
    EXPECT_EQ(HcclAllocComResourceByTiling(comm_, stream_, &tiling, &opResCtx), HCCL_E_UNAVAIL);
}

TEST_F(CcuMc2TestSuite, CheckOpResSufficient_ReduceScatterCcu)
{
    // ReduceScatter 走 CCU_SCHED 校验分支：外部名 sole[mesh] 定名为 CcuSchedReduceScatterSoleMesh，
    // 资源计算成功即视为校验通过
    uint32_t opTypes[] = {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER)};
    Mc2TilingTestData tiling = BuildMc2Tiling(1, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), opTypes, nullptr);
    // 新契约：CCU 路径只认外部名（裸名归为 NONE，闸门直接拒绝）
    strcpy(tiling.ccTiling[0].algConfig, "sole[mesh]");

    HcclResult ret = CheckOpResSufficient(comm_, stream_, &tiling);

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 校验路径端到端回退：algConfig 为合法外部名，但强制算法执行器不可用
// （CheckForcedAlgResource校验失败）→ 回退到默认selector重新选择算法 → 资源校验通过。
// g_stubCcuAlgExecNullName限定仅定名结果返回nullptr，不影响回退后算法的资源计算。
TEST_F(CcuMc2TestSuite, CheckOpResSufficient_ForcedAlgFallbackToSelector)
{
    uint32_t opTypes[] = {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER)};
    Mc2TilingTestData tiling = BuildMc2Tiling(1, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), opTypes, nullptr);
    // 新契约：强制算法只认外部名（裸名归为 NONE）
    strcpy(tiling.ccTiling[0].algConfig, "sole[mesh]");
    mc2_ops_hccl::g_stubCcuAlgExecNull = true;
    mc2_ops_hccl::g_stubCcuAlgExecNullName = "CcuSchedAllGatherSoleMesh";
    // 回退后默认selector选择合法CCU算法（与强制算法不同名，避免再次命中执行器不可用桩）
    mc2_ops_hccl::g_stubSelectorAlgName = "CcuAllGatherMeshMem2Mem1D";

    HcclResult ret = CheckOpResSufficient(comm_, stream_, &tiling);

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, CheckOpResSufficient_AicpuBypass)
{
    // 非CCU_SCHED引擎不做CCU校验，直接通过
    uint32_t opTypes[] = {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER)};
    Mc2TilingTestData tiling = BuildMc2Tiling(1, COMM_ENGINE_AICPU, opTypes, nullptr);

    HcclResult ret = CheckOpResSufficient(comm_, stream_, &tiling);

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, CheckOpResSufficient_SingleRankBypass)
{
    // 单rank场景资源必然充足：入口层统一提前返回SUCCESS，不进入算法选择与资源校验
    // （rankSize为通信域属性，对所有tiling一致，此处对CCU_SCHED引擎验证该守卫）
    if (comm_ != nullptr) {
        auto simComm = static_cast<HcclSim::SimCommunicator*>(comm_);
        delete simComm;
        comm_ = nullptr;
    }
    HcclSim::SimWorld::Global()->Deinit();
    g_stubRankSize = 1;
    TopoMeta topoMeta = BuildTopoMeta(g_stubRankSize);
    HcclSim::SimWorld::Global()->Init(topoMeta, mc2_ops_hccl::g_stubDeviceType);
    HcclResult initRet = HcclSim::Sim_HcclCommInitClusterInfo(topoMeta, g_stubRankId, &comm_);
    ASSERT_EQ(initRet, HCCL_SUCCESS) << "Comm init failed";

    uint32_t opTypes[] = {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER)};
    Mc2TilingTestData tiling = BuildMc2Tiling(1, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), opTypes, nullptr);

    HcclResult ret = CheckOpResSufficient(comm_, stream_, &tiling);

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 闸门只守卫空条目，算法名内容不参与判定
TEST_F(CcuMc2TestSuite, CheckCcuAlgorithmsRegistered_NullEntryRejected)
{
    const void* ccTilingList[] = {nullptr};
    EXPECT_FALSE(CheckCcuAlgorithmsRegistered(ccTilingList, 1U));
}

TEST_F(CcuMc2TestSuite, CheckCcuAlgorithmsRegistered_AllEntriesNonNull)
{
    Mc2CcTilingInner ccTiling[2]{};
    const void* ccTilingList[] = {&ccTiling[0], &ccTiling[1]};
    EXPECT_TRUE(CheckCcuAlgorithmsRegistered(ccTilingList, 2U));
    // 0 条目视为通过
    EXPECT_TRUE(CheckCcuAlgorithmsRegistered(ccTilingList, 0U));
}

// 裸名归 NONE；checkOnly 由默认 selector 定名
TEST_F(CcuMc2TestSuite, CheckOpResSufficient_BareNameUsesSelector)
{
    uint32_t opTypes[] = {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER)};
    Mc2TilingTestData tiling = BuildMc2Tiling(1, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), opTypes, nullptr);

    EXPECT_EQ(CheckOpResSufficient(comm_, stream_, &tiling), HCCL_SUCCESS);
}

// 多 tiling 逐条走强制算法/默认 selector
TEST_F(CcuMc2TestSuite, CheckOpResSufficient_MultiTilingExternalAlg)
{
    uint32_t opTypes[] = {
        static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER), static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE)};
    Mc2TilingTestData tiling = BuildMc2Tiling(2, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), opTypes, nullptr);

    for (auto& cc : tiling.ccTiling) {
        strcpy(cc.algConfig, "sole[mesh]");
    }
    EXPECT_EQ(CheckOpResSufficient(comm_, stream_, &tiling), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, CheckOpResSufficient_NullComm)
{
    uint32_t opTypes[] = {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER)};
    Mc2TilingTestData tiling = BuildMc2Tiling(1, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), opTypes, nullptr);
    EXPECT_EQ(CheckOpResSufficient(nullptr, stream_, &tiling), HCCL_E_PTR);
}

TEST_F(CcuMc2TestSuite, CheckOpResSufficient_NullStream)
{
    uint32_t opTypes[] = {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER)};
    Mc2TilingTestData tiling = BuildMc2Tiling(1, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), opTypes, nullptr);
    EXPECT_EQ(CheckOpResSufficient(comm_, nullptr, &tiling), HCCL_E_PTR);
}

TEST_F(CcuMc2TestSuite, CheckOpResSufficient_NullTiling)
{
    EXPECT_EQ(CheckOpResSufficient(comm_, stream_, nullptr), HCCL_E_PTR);
}

TEST_F(CcuMc2TestSuite, CheckOpResSufficient_UnsupportedEngine)
{
    uint32_t opTypes[] = {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER)};
    Mc2TilingTestData tiling = BuildMc2Tiling(1, COMM_ENGINE_AIV, opTypes, nullptr);
    EXPECT_EQ(CheckOpResSufficient(comm_, stream_, &tiling), HCCL_E_NOT_SUPPORT);
}

TEST_F(CcuMc2TestSuite, CheckOpResSufficient_MixedEnginesRejected)
{
    // 两个tiling携带不同commEngine（CCU+AICPU），ObtainCommEngine混合引擎分支拒绝，
    // 验证HcclGetTilingList→ObtainCommEngine整条链路返回HCCL_E_NOT_SUPPORT
    uint32_t opTypes[] = {
        static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER),
        static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER)};
    Mc2TilingTestData tiling = BuildMc2Tiling(2, COMM_ENGINE_CCU, opTypes, nullptr);
    tiling.ccTiling[1].commEngine = COMM_ENGINE_AICPU;
    EXPECT_EQ(CheckOpResSufficient(comm_, stream_, &tiling), HCCL_E_NOT_SUPPORT);
}

TEST_F(CcuMc2TestSuite, CheckOpResSufficient_ReduceScatterRejectsUnsupportedCombos)
{
    // 端到端：强制算法校验失败→回退默认 selector，桩复刻 64 位/PROD 拒绝规则。
    uint32_t opTypes[] = {static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER)};
    mc2_ops_hccl::g_stubCcuAlgExecNull = true;
    mc2_ops_hccl::g_stubCcuAlgExecNullName = "CcuSchedReduceScatterSoleMesh";

    const std::vector<uint8_t> unsupportedTypes = {HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64};
    for (const uint8_t dataType : unsupportedTypes) {
        Mc2TilingTestData tiling =
            BuildMc2Tiling(1, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), opTypes, nullptr);
        strcpy(tiling.ccTiling[0].algConfig, "sole[mesh]");
        tiling.ccTiling[0].srcDataType = dataType;
        tiling.ccTiling[0].dstDataType = dataType;
        EXPECT_EQ(CheckOpResSufficient(comm_, stream_, &tiling), HCCL_E_NOT_SUPPORT);
    }

    Mc2TilingTestData tiling = BuildMc2Tiling(1, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), opTypes, nullptr);
    strcpy(tiling.ccTiling[0].algConfig, "sole[mesh]");
    tiling.ccTiling[0].reduceType = HCCL_REDUCE_PROD;
    EXPECT_EQ(CheckOpResSufficient(comm_, stream_, &tiling), HCCL_E_NOT_SUPPORT);
}

// ---------- MC2 builtin public APIs ----------

TEST_F(CcuMc2TestSuite, Mc2GetCcArgs_NullOutput) { EXPECT_EQ(Mc2GetCcArgs(nullptr), HCCL_E_PTR); }

TEST_F(CcuMc2TestSuite, Mc2GetAndFreeCcArgs)
{
    void* ccArgs = nullptr;
    ASSERT_EQ(Mc2GetCcArgs(&ccArgs), HCCL_SUCCESS);
    ASSERT_NE(ccArgs, nullptr);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, Mc2FreeCcArgs_Null) { EXPECT_EQ(Mc2FreeCcArgs(nullptr), HCCL_E_PTR); }

TEST_F(CcuMc2TestSuite, Mc2SetCcCommEngine_NullArgs)
{
    EXPECT_EQ(Mc2SetCcCommEngine(nullptr, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED)), HCCL_E_PTR);
}

TEST_F(CcuMc2TestSuite, Mc2SetCcCommEngine_RejectsUnsupportedEngine)
{
    void* ccArgs = nullptr;
    ASSERT_EQ(Mc2GetCcArgs(&ccArgs), HCCL_SUCCESS);
    ASSERT_NE(ccArgs, nullptr);

    EXPECT_EQ(Mc2SetCcCommEngine(ccArgs, static_cast<uint8_t>(OpExecuteConfig::AIV)), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, Mc2SetCcCommEngine_AcceptsCcuEngines)
{
    void* ccArgs = nullptr;
    ASSERT_EQ(Mc2GetCcArgs(&ccArgs), HCCL_SUCCESS);
    ASSERT_NE(ccArgs, nullptr);

    EXPECT_EQ(Mc2SetCcCommEngine(ccArgs, static_cast<uint8_t>(OpExecuteConfig::CCU_MS)), HCCL_SUCCESS);
    EXPECT_EQ(Mc2SetCcCommEngine(ccArgs, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED)), HCCL_SUCCESS);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

// ---------- CheckOpResSufficient(ccArgs) ----------
// ccArgs版checkOnly预检：由Mc2GetCcArgs/Mc2SetCc*构造参数，
// 走与tiling版相同的算法选择与资源校验链路，不落盘OpResCtx

TEST_F(CcuMc2TestSuite, CheckOpResSufficientByArgs_NullParameters)
{
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "CcuSchedAllGatherSoleMesh");
    ASSERT_NE(ccArgs, nullptr);

    EXPECT_EQ(CheckOpResSufficient(nullptr, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs), HCCL_E_PTR);
    EXPECT_EQ(CheckOpResSufficient(comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), nullptr), HCCL_E_PTR);

    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, CheckOpResSufficientByArgs_CcuSufficient)
{
    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "sole[mesh]");
    ASSERT_NE(ccArgs, nullptr);

    HcclResult ret = CheckOpResSufficient(comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

// 端到端验证UNAVAIL→RES_NOT_SUFFICIENT映射（ccArgs路径）：
// 打桩HcclGetAlgRes返回UNAVAIL（模拟CCU通道/实例资源不足），
// checkOnly场景统一翻译为HCCL_E_RES_NOT_SUFFICIENT
TEST_F(CcuMc2TestSuite, CheckOpResSufficientByArgs_ResourceInsufficient)
{
    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "sole[mesh]");
    ASSERT_NE(ccArgs, nullptr);
    mc2_ops_hccl::g_stubCcuAlgResUnavailable = true;

    EXPECT_EQ(
        CheckOpResSufficient(comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs),
        HCCL_E_RES_NOT_SUFFICIENT);

    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, CheckOpResSufficientByArgs_UnsupportedDevice)
{
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "CcuSchedAllGatherSoleMesh");
    ASSERT_NE(ccArgs, nullptr);
    mc2_ops_hccl::g_stubDeviceType = DevType::DEV_TYPE_910B;

    EXPECT_EQ(
        CheckOpResSufficient(comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs), HCCL_E_NOT_SUPPORT);

    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, CheckOpResSufficientByArgs_UnsupportedCcType)
{
    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "CcuSchedAllGatherSoleMesh");
    ASSERT_NE(ccArgs, nullptr);

    // 0xFF非Mc2支持的算子类型，BuildMc2CcTiling拒绝
    EXPECT_EQ(CheckOpResSufficient(comm_, 0xFFU, ccArgs), HCCL_E_NOT_SUPPORT);

    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, CheckOpResSufficientByArgs_SingleRankBypass)
{
    // 单rank场景资源必然充足：入口层提前返回SUCCESS，不进入算法选择与资源校验
    auto simComm = static_cast<HcclSim::SimCommunicator*>(comm_);
    delete simComm;
    comm_ = nullptr;
    HcclSim::SimWorld::Global()->Deinit();
    g_stubRankSize = 1;
    TopoMeta topoMeta = BuildTopoMeta(g_stubRankSize);
    HcclSim::SimWorld::Global()->Init(topoMeta, mc2_ops_hccl::g_stubDeviceType);
    ASSERT_EQ(HcclSim::Sim_HcclCommInitClusterInfo(topoMeta, g_stubRankId, &comm_), HCCL_SUCCESS);

    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "CcuSchedAllGatherSoleMesh");
    ASSERT_NE(ccArgs, nullptr);

    EXPECT_EQ(CheckOpResSufficient(comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs), HCCL_SUCCESS);

    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, CheckOpResSufficientByArgs_AicpuBypass)
{
    // 非CCU_SCHED引擎（AICPU等）不做CCU校验，直接返回SUCCESS
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::AICPU_TS), nullptr);
    ASSERT_NE(ccArgs, nullptr);

    EXPECT_EQ(CheckOpResSufficient(comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs), HCCL_SUCCESS);

    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

// ccArgs 校验走 checkOnly：裸名交由默认 selector 定名
TEST_F(CcuMc2TestSuite, CheckOpResSufficientByArgs_BareNameUsesSelector)
{
    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "CcuSchedAllGatherSoleMesh");
    ASSERT_NE(ccArgs, nullptr);

    // 裸名回退默认 selector，校验通过
    EXPECT_EQ(CheckOpResSufficient(comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs), HCCL_SUCCESS);

    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, CheckOpResSufficientByArgs_ForcedAlgFallbackToSelector)
{
    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "sole[mesh]");
    ASSERT_NE(ccArgs, nullptr);
    // 强制算法执行器不可用→回退默认selector；桩selector返回另一合法CCU算法
    mc2_ops_hccl::g_stubCcuAlgExecNull = true;
    mc2_ops_hccl::g_stubCcuAlgExecNullName = "CcuSchedAllGatherSoleMesh";
    mc2_ops_hccl::g_stubSelectorAlgName = "CcuAllGatherMeshMem2Mem1D";

    EXPECT_EQ(CheckOpResSufficient(comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs), HCCL_SUCCESS);

    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, Mc2SetCcAlgConfig_ParameterValidation)
{
    void* ccArgs = nullptr;
    ASSERT_EQ(Mc2GetCcArgs(&ccArgs), HCCL_SUCCESS);
    ASSERT_NE(ccArgs, nullptr);

    EXPECT_EQ(Mc2SetCcAlgConfig(nullptr, "CcuSchedAllGatherSoleMesh"), HCCL_E_PTR);
    EXPECT_EQ(Mc2SetCcAlgConfig(ccArgs, nullptr), HCCL_E_PTR);
    EXPECT_EQ(Mc2SetCcAlgConfig(ccArgs, "CcuSchedAllGatherSoleMesh"), HCCL_SUCCESS);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

// algConfig最终要装入Mc2CcTilingInner::algConfig[ALG_CONFIG_SIZE]（128字节），
// 长度>=128的algConfig应在设置阶段即以HCCL_E_PARA拒绝，不应静默截断
TEST_F(CcuMc2TestSuite, Mc2SetCcAlgConfig_RejectsOverlongConfig)
{
    void* ccArgs = nullptr;
    ASSERT_EQ(Mc2GetCcArgs(&ccArgs), HCCL_SUCCESS);
    ASSERT_NE(ccArgs, nullptr);

    const std::string maxValid(HcclApi::ALG_CONFIG_SIZE - 1U, 'a'); // 127字节：合法上限
    const std::string overlong(HcclApi::ALG_CONFIG_SIZE, 'a');      // 128字节：无NUL终止位，拒绝
    EXPECT_EQ(Mc2SetCcAlgConfig(ccArgs, maxValid.c_str()), HCCL_SUCCESS);
    EXPECT_EQ(Mc2SetCcAlgConfig(ccArgs, overlong.c_str()), HCCL_E_PARA);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, Mc2SetCcDataTypes_ParameterValidation)
{
    void* ccArgs = nullptr;
    ASSERT_EQ(Mc2GetCcArgs(&ccArgs), HCCL_SUCCESS);
    ASSERT_NE(ccArgs, nullptr);

    EXPECT_EQ(Mc2SetCcSrcDataType(nullptr, HCCL_DATA_TYPE_FP16), HCCL_E_PTR);
    EXPECT_EQ(Mc2SetCcDstDataType(nullptr, HCCL_DATA_TYPE_FP16), HCCL_E_PTR);

    mc2_ops_hccl::g_stubHcomCheckDataTypeResult = HCCL_E_NOT_SUPPORT;
    EXPECT_EQ(Mc2SetCcSrcDataType(ccArgs, HCCL_DATA_TYPE_FP16), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(Mc2SetCcDstDataType(ccArgs, HCCL_DATA_TYPE_FP16), HCCL_E_NOT_SUPPORT);

    mc2_ops_hccl::g_stubHcomCheckDataTypeResult = HCCL_SUCCESS;
    EXPECT_EQ(Mc2SetCcSrcDataType(ccArgs, HCCL_DATA_TYPE_FP32), HCCL_SUCCESS);
    EXPECT_EQ(Mc2SetCcDstDataType(ccArgs, HCCL_DATA_TYPE_FP32), HCCL_SUCCESS);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, Mc2SetCcReduceType_ParameterValidation)
{
    void* ccArgs = nullptr;
    ASSERT_EQ(Mc2GetCcArgs(&ccArgs), HCCL_SUCCESS);
    ASSERT_NE(ccArgs, nullptr);

    EXPECT_EQ(Mc2SetCcReduceType(nullptr, HCCL_REDUCE_SUM), HCCL_E_PTR);

    mc2_ops_hccl::g_stubHcomCheckReductionOpResult = HCCL_E_NOT_SUPPORT;
    EXPECT_EQ(Mc2SetCcReduceType(ccArgs, HCCL_REDUCE_SUM), HCCL_E_NOT_SUPPORT);

    mc2_ops_hccl::g_stubHcomCheckReductionOpResult = HCCL_SUCCESS;
    EXPECT_EQ(Mc2SetCcReduceType(ccArgs, HCCL_REDUCE_SUM), HCCL_SUCCESS);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

// dataLen 区间不做值域校验（除空窗）：(0,0) 表示查询时不校验长度，区间为单卡字节
// （count×dtypeSize），具体语义由消费方（Mc2GetSupportAlgList，不同窗名单不同）决定，
// setter 只管存；空窗（minLen > maxLen）拒绝；存储验证由消费方 UT 间接完成，本任务不提供 getter。
TEST_F(CcuMc2TestSuite, Mc2SetCcDataLenRange_Basics)
{
    EXPECT_EQ(Mc2SetCcDataLenRange(nullptr, 1, 2), HCCL_E_PTR);
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "sole[mesh]");
    ASSERT_NE(ccArgs, nullptr);
    EXPECT_EQ(Mc2SetCcDataLenRange(ccArgs, 8, 4), HCCL_E_PARA);  // minLen > maxLen 拒空窗
    EXPECT_EQ(Mc2SetCcDataLenRange(ccArgs, 0, 0), HCCL_SUCCESS); // 哨兵
    EXPECT_EQ(Mc2SetCcDataLenRange(ccArgs, 1, 1), HCCL_SUCCESS); // 单值退化
    EXPECT_EQ(Mc2SetCcDataLenRange(ccArgs, 0, UINT64_MAX), HCCL_SUCCESS);
    EXPECT_EQ(Mc2SetCcDataLenRange(ccArgs, UINT64_MAX, UINT64_MAX), HCCL_SUCCESS); // uint64 边界值
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

// D18 查询 helper：调 C ABI（JSON 输出）并解析 alg_list 回 vector，供既有集合语义断言复用。
// 名单为零转义直拼（字符集白名单已由注册面断言钉死），引号/逗号即完整词法，可机械解析。
// 解析失败以 EXPECT 记录后返回已解析前缀（ASSERT 不能用于非 void 函数），后续断言自然红。
static std::vector<std::string> QueryAndParseAlgList(HcclComm comm, uint8_t cmd, void* ccArgs)
{
    char buf[16384];
    const Mc2Result ret = Mc2GetSupportAlgList(comm, cmd, ccArgs, buf, sizeof(buf));
    EXPECT_EQ(ret, HCCL_SUCCESS);
    if (ret != HCCL_SUCCESS) {
        return {};
    }
    std::vector<std::string> out;
    const char* p = strstr(buf, "\"alg_list\":[");
    EXPECT_NE(p, nullptr);
    if (p == nullptr) {
        return out;
    }
    p += strlen("\"alg_list\":[");
    while (*p != ']') {
        if (*p != '"') {
            ADD_FAILURE() << "alg_list 元素未以引号开头: " << buf;
            return out;
        }
        ++p;
        const char* q = strchr(p, '"');
        if (q == nullptr) {
            ADD_FAILURE() << "alg_list 元素缺闭合引号: " << buf;
            return out;
        }
        out.emplace_back(p, static_cast<size_t>(q - p));
        p = q + 1;
        if (*p == ',') {
            ++p;
        }
    }
    return out;
}

// 查询接口参数校验面：空指针/非法 ccType 在 topo 获取之前拒绝；PTR 路径不写 buffer
// （错误码是兜底真值，预置哨兵验证未被写入），ccType 错误路径 best-effort 写失败态
// JSON（三字段两态：count=0/alg_list=[]/error_msg 非空）。
// 引擎拒绝分支中，setter 白名单外的非法引擎不可经公开 API 构造（Mc2SetCcCommEngine
// 拒绝非法值，ccArgs 不透明），由代码审覆盖（引擎允许集 = Mc2SetCcCommEngine 同款）；
// CCU_MS（D19，老链路引擎，查询拒）可经 setter 构造，由紧邻用例验证。
TEST_F(CcuMc2TestSuite, Mc2GetSupportAlgList_ParamValidation)
{
    char buf[16384];
    void* ccArgs = nullptr;
    ASSERT_EQ(Mc2GetCcArgs(&ccArgs), HCCL_SUCCESS);
    ASSERT_NE(ccArgs, nullptr);

    strcpy(buf, "sentinel");
    EXPECT_EQ(Mc2GetSupportAlgList(nullptr, HCCL_CMD_ALLGATHER, ccArgs, buf, sizeof(buf)), HCCL_E_PTR);
    EXPECT_STREQ(buf, "sentinel");
    strcpy(buf, "sentinel");
    EXPECT_EQ(Mc2GetSupportAlgList(comm_, HCCL_CMD_ALLGATHER, nullptr, buf, sizeof(buf)), HCCL_E_PTR);
    EXPECT_STREQ(buf, "sentinel");
    EXPECT_EQ(Mc2GetSupportAlgList(comm_, HCCL_CMD_KFC_SERVER, ccArgs, buf, sizeof(buf)), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(buf[0], '{');
    EXPECT_NE(strstr(buf, "\"count\":0"), nullptr);
    EXPECT_NE(strstr(buf, "\"error_msg\":[\""), nullptr);

    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

// D19：CCU_MS 是老链路引擎（算法面在 V1 注册表），V2 名单不覆盖——查询拒，error_msg 引导
// CCU_SCHED（错误码兜底真值 + error_msg 增强）
TEST_F(CcuMc2TestSuite, Mc2GetSupportAlgList_EngineCcuMsNotSupported)
{
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_MS), "sole[mesh]");
    ASSERT_NE(ccArgs, nullptr);

    char buf[16384];
    EXPECT_EQ(Mc2GetSupportAlgList(comm_, HCCL_CMD_ALLREDUCE, ccArgs, buf, sizeof(buf)), HCCL_E_NOT_SUPPORT);
    EXPECT_NE(strstr(buf, "CCU_SCHED"), nullptr);
    EXPECT_NE(strstr(buf, "\"error_msg\":[\""), nullptr);

    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

// 错误返回 → 出参不残留写入：topo 获取失败的错误码原样透传；非枚举错误路径不写 buffer
// （错误码是兜底真值），哨兵验证 buffer 未被触碰
TEST_F(CcuMc2TestSuite, Mc2GetSupportAlgList_ErrorPathLeavesBufferUntouched)
{
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), nullptr);
    ASSERT_NE(ccArgs, nullptr);

    char buf[16384];
    strcpy(buf, "sentinel");
    mc2_ops_hccl::g_stubCalcTopoInfoResult = HcclResult::HCCL_E_INTERNAL;
    EXPECT_EQ(
        Mc2GetSupportAlgList(comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs, buf, sizeof(buf)),
        HCCL_E_INTERNAL);
    EXPECT_STREQ(buf, "sentinel");
    mc2_ops_hccl::g_stubCalcTopoInfoResult = HcclResult::HCCL_SUCCESS;

    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

// veto 分支（spec §3.3 pass2）：AICPU_TS 引擎在 level0PcieMix && level0BigClosRange 拓扑上
// 入口否决 → SUCCESS + 空名单；CCU 引擎不受 veto 影响（名单与 flag=false 基线同构）；
// 单 flag 不构成 veto（谓词为合取）。
TEST_F(CcuMc2TestSuite, Mc2GetSupportAlgList_AicpuVetoYieldsEmptyList)
{
    VerifyQueryTagRegistrations();
    const uint8_t allGatherType = static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    void* ccArgsAicpu = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::AICPU_TS), nullptr);
    ASSERT_NE(ccArgsAicpu, nullptr);
    void* ccArgsCcu = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), nullptr);
    ASSERT_NE(ccArgsCcu, nullptr);

    // flag=false 基线：AICPU 名单非空（fixture 确定性），veto 反差才有效
    const std::vector<std::string> aicpuBaseline = QueryAndParseAlgList(comm_, allGatherType, ccArgsAicpu);
    ASSERT_FALSE(aicpuBaseline.empty());
    const std::vector<std::string> ccuBaseline = QueryAndParseAlgList(comm_, allGatherType, ccArgsCcu);

    // 两 flag 同时 true：AICPU 入口否决 → SUCCESS + 空名单（成功态空 JSON 钉形，错误码同为
    // SUCCESS 时 buffer 自解释的分态判据）
    mc2_ops_hccl::g_stubTopoPcieMix = true;
    mc2_ops_hccl::g_stubTopoBigClosRange = true;
    char vetoBuf[16384];
    EXPECT_EQ(Mc2GetSupportAlgList(comm_, allGatherType, ccArgsAicpu, vetoBuf, sizeof(vetoBuf)), HCCL_SUCCESS);
    EXPECT_STREQ(vetoBuf, "{\"count\":0,\"alg_list\":[],\"error_msg\":[]}");
    // CCU 名单不受 veto 影响：与基线同构；基线非空则 veto 下仍非空
    const std::vector<std::string> ccuUnvetoed = QueryAndParseAlgList(comm_, allGatherType, ccArgsCcu);
    EXPECT_EQ(ccuUnvetoed, ccuBaseline);
    if (!ccuBaseline.empty()) {
        EXPECT_FALSE(ccuUnvetoed.empty());
    }
    // 单 flag（仅 level0PcieMix）不构成 veto：AICPU 名单回到基线
    mc2_ops_hccl::g_stubTopoBigClosRange = false;
    const std::vector<std::string> aicpuSingleFlag = QueryAndParseAlgList(comm_, allGatherType, ccArgsAicpu);
    EXPECT_EQ(aicpuSingleFlag, aicpuBaseline);

    // 恢复全局注入量
    mc2_ops_hccl::g_stubTopoPcieMix = false;
    mc2_ops_hccl::g_stubTopoBigClosRange = false;

    EXPECT_EQ(Mc2FreeCcArgs(ccArgsAicpu), HCCL_SUCCESS);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgsCcu), HCCL_SUCCESS);
}

// 查询接口名单语义。fixture 确定性：本目标 HcclCalcTopoInfo 为桩（恒 SUCCESS；缺省不动
// topoInfo，veto 注入量见 AicpuVetoYieldsEmptyList），topo 保持 make_unique 预分配的默认
// 构造值——全 Any 模式必命中，LenGated 长度子句按 dataLen 窗判定，名单对上方注册表是确定性
// 的。断言锚定注册名集合与结构性质（元素均有该 cmd+引擎的注册元数据、去重、引擎隔离、
// 候选∩元数据），不依赖真实拓扑。
TEST_F(CcuMc2TestSuite, Mc2GetSupportAlgList_ListSemantics)
{
    VerifyQueryTagRegistrations();
    const uint8_t allGatherType = static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    const uint8_t reduceScatterType = static_cast<uint8_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);

    // CCU 引擎，dataLen 区间缺省 (0,0)（哨兵：跳过长度子句，命中集最宽）
    void* ccArgsCcu = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), nullptr);
    ASSERT_NE(ccArgsCcu, nullptr);
    const std::vector<std::string> ccuListZero = QueryAndParseAlgList(comm_, allGatherType, ccArgsCcu);
    const std::set<std::string> ccuUniqZero(ccuListZero.begin(), ccuListZero.end());
    EXPECT_EQ(ccuListZero.size(), ccuUniqZero.size()); // 去重
    // 名单元素为外部名：每个键在注册表中至少有一行该 cmd+引擎的注册行（经 GetAll 反查）
    const std::vector<AlgMetaRow> allRows = AlgMetaRegistry::Instance().GetAll();
    for (const std::string& extName : ccuListZero) {
        bool found = false;
        for (const AlgMetaRow& row : allRows) {
            if (row.cmd == HcclCMDType::HCCL_CMD_ALLGATHER && row.meta.engine == AlgEngine::CCU &&
                extName == row.meta.externalName) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "externalName[" << extName << "] 无对应注册行";
    }
    EXPECT_EQ(ccuUniqZero.count("sole[mesh]"), 1U); // SoleMesh/LenGated 同键去重后单次
    // CcuSchedAgExecutorOnly 无元数据被跳过；测试 tag 无 meta 行，不产生任何键

    // dataLen 区间语义：单值窗 [k4MB,k4MB] 名单 ⊆ 哨兵 (0,0) 名单（包含关系）；
    // k4MB 超 LenGated 的 LE k1MB 上界被拒，无长度子句的 SoleMesh 保留
    ASSERT_EQ(Mc2SetCcDataLenRange(ccArgsCcu, k4MB, k4MB), HCCL_SUCCESS);
    const std::vector<std::string> ccuListBig = QueryAndParseAlgList(comm_, allGatherType, ccArgsCcu);
    const std::set<std::string> ccuUniqBig(ccuListBig.begin(), ccuListBig.end());
    EXPECT_EQ(ccuUniqBig.count("sole[mesh]"), 1U); // SoleMesh 无长度子句保留（LenGated 同键不可单独观测）
    for (const std::string& name : ccuListBig) {
        EXPECT_EQ(ccuUniqZero.count(name), 1U);
    }

    // AICPU_TS 引擎：名单只含 AICPU 注册名（CCU 名单向 AICPU 侧隔离）
    void* ccArgsAicpu = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::AICPU_TS), nullptr);
    ASSERT_NE(ccArgsAicpu, nullptr);
    const std::vector<std::string> aicpuList = QueryAndParseAlgList(comm_, allGatherType, ccArgsAicpu);
    const std::set<std::string> aicpuUniq(aicpuList.begin(), aicpuList.end());
    EXPECT_EQ(aicpuUniq.count("sole[mesh]"), 1U); // AICPU 侧 InsAllGatherMesh1D 的外部名

    // RS + AICPU_TS：该 cmd 只有 CCU 注册名 → 空名单 + SUCCESS（空名单是合法结果；
    // 下方 RS + CCU 非空证明空名单归因于引擎对齐而非 tag 缺失）
    const std::vector<std::string> rsAicpuList = QueryAndParseAlgList(comm_, reduceScatterType, ccArgsAicpu);
    EXPECT_TRUE(rsAicpuList.empty());
    const std::vector<std::string> rsCcuList = QueryAndParseAlgList(comm_, reduceScatterType, ccArgsCcu);
    EXPECT_EQ(std::set<std::string>(rsCcuList.begin(), rsCcuList.end()).count("sole[mesh]"), 1U);

    EXPECT_EQ(Mc2FreeCcArgs(ccArgsCcu), HCCL_SUCCESS);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgsAicpu), HCCL_SUCCESS);
}

// dataLen 区间单调收窄（接口层，D17 存在量词）：名单随窗收窄只减不增——全量窗（哨兵
// (0,0)，跳长度子句）⊇ 单值窗 [k1MB,k1MB] ⊇ 窄窗 [k1MB,k1MB+8]；LenGated 的 LE k1MB 子句
// 在单值窗上界点含（探针=窗端点按 cmp 选端：LE 看 min），窄窗左端点同在界内——8 字节
// 未跨过任何阈值，名单不变。LenGated 条目自身的界上含/越界拒已由 MatchOne 单测在模式层验证。
TEST_F(CcuMc2TestSuite, Mc2GetSupportAlgList_DataLenMonotonicNarrowing)
{
    VerifyQueryTagRegistrations();
    const uint8_t allGatherType = static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    void* ccArgs0 = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "sole[mesh]");
    ASSERT_NE(ccArgs0, nullptr);
    ASSERT_EQ(Mc2SetCcDataLenRange(ccArgs0, 0, 0), HCCL_SUCCESS); // 全量：LenGated 在
    const std::vector<std::string> listFull = QueryAndParseAlgList(comm_, allGatherType, ccArgs0);
    void* ccArgs1 = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "sole[mesh]");
    ASSERT_NE(ccArgs1, nullptr);
    ASSERT_EQ(Mc2SetCcDataLenRange(ccArgs1, k1MB, k1MB), HCCL_SUCCESS); // 单值窗：界上含（LE）
    const std::vector<std::string> listPoint = QueryAndParseAlgList(comm_, allGatherType, ccArgs1);
    void* ccArgs2 = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "sole[mesh]");
    ASSERT_NE(ccArgs2, nullptr);
    ASSERT_EQ(Mc2SetCcDataLenRange(ccArgs2, k1MB, k1MB + 8U), HCCL_SUCCESS); // 窄窗：左端仍含界
    const std::vector<std::string> listNarrow = QueryAndParseAlgList(comm_, allGatherType, ccArgs2);

    const std::set<std::string> setFull(listFull.begin(), listFull.end());
    const std::set<std::string> setPoint(listPoint.begin(), listPoint.end());
    const std::set<std::string> setNarrow(listNarrow.begin(), listNarrow.end());
    // 单值窗/窄窗名单 ⊆ 全量名单（存在量词：区间收窄命中集只减不增）
    for (const std::string& name : setPoint) {
        EXPECT_EQ(setFull.count(name), 1U);
    }
    for (const std::string& name : setNarrow) {
        EXPECT_EQ(setFull.count(name), 1U);
    }
    EXPECT_FALSE(setFull.empty());
    EXPECT_EQ(listPoint, listNarrow); // 窄 8 字节未跨过任何阈值

    EXPECT_EQ(Mc2FreeCcArgs(ccArgs0), HCCL_SUCCESS);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs1), HCCL_SUCCESS);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs2), HCCL_SUCCESS);
}

// (0, hi>0) 含 0 真窗（非哨兵）：0 是合法数据点，过滤行为与全量/单值窗的包含关系
TEST_F(CcuMc2TestSuite, Mc2GetSupportAlgList_ZeroLowerBoundWindow)
{
    VerifyQueryTagRegistrations();
    const uint8_t allGatherType = static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    void* ccArgs0 = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "sole[mesh]");
    ASSERT_NE(ccArgs0, nullptr);
    ASSERT_EQ(Mc2SetCcDataLenRange(ccArgs0, 0, 0), HCCL_SUCCESS); // 哨兵（全量）
    const std::vector<std::string> listFull = QueryAndParseAlgList(comm_, allGatherType, ccArgs0);
    void* ccArgs1 = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "sole[mesh]");
    ASSERT_NE(ccArgs1, nullptr);
    ASSERT_EQ(Mc2SetCcDataLenRange(ccArgs1, 0, k1MB), HCCL_SUCCESS); // 含 0 真窗 [0, 1MB]
    const std::vector<std::string> listZeroLo = QueryAndParseAlgList(comm_, allGatherType, ccArgs1);

    const std::set<std::string> setFull(listFull.begin(), listFull.end());
    for (const std::string& name : listZeroLo) {
        EXPECT_EQ(setFull.count(name), 1U); // 含 0 窗 ⊆ 全量（哨兵窄化单调）
    }
    // 含 0 真窗不是哨兵：LenGated（LE 1MB）等低门槛算法在窗内，名单非空
    EXPECT_FALSE(listZeroLo.empty());

    EXPECT_EQ(Mc2FreeCcArgs(ccArgs0), HCCL_SUCCESS);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs1), HCCL_SUCCESS);
}

// D18：JSON 契约——三字段两态恒定、len 不足降级、error_msg 承载错误描述（含 D19 引导语）。
// 名单 JSON（本夹具 3 名单 ≈120B）> error JSON（含 need/len 约 110B）> 空串：三段降级逐级可观测
// （small[112] 落在 [110,119] 窗口：error JSON 含 need/len 实值约 110B，中档 buffer 须随之上调）
// 与任务文本的出入见报告）。
TEST_F(CcuMc2TestSuite, Mc2GetSupportAlgList_CAbiJsonContract)
{
    VerifyQueryTagRegistrations();
    const uint8_t allGatherType = static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    char buf[16384];

    // 1) 成功：三字段、error_msg 恒空数组
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), nullptr);
    ASSERT_NE(ccArgs, nullptr);
    EXPECT_EQ(Mc2GetSupportAlgList(comm_, allGatherType, ccArgs, buf, sizeof(buf)), HCCL_SUCCESS);
    EXPECT_EQ(buf[0], '{');
    EXPECT_NE(strstr(buf, "\"count\":"), nullptr);
    EXPECT_NE(strstr(buf, "\"error_msg\":[]"), nullptr);
    // count 一致性：alg_list 数组元素数（数引号对，名单零转义引号即完整词法）与 "count":
    // 字段整数值（strstr 后 atoi）相等——外部名含 '['/']'（如 sole[mesh]），逐对走引号
    // 才不会被名内 ']' 提前截断
    const char* p = strstr(buf, "\"alg_list\":[");
    ASSERT_NE(p, nullptr);
    p += strlen("\"alg_list\":[");
    size_t algNum = 0;
    while (*p != ']') {
        ASSERT_EQ(*p, '"');
        p = strchr(p + 1, '"');
        ASSERT_NE(p, nullptr);
        ++algNum;
        ++p;
        if (*p == ',') {
            ++p;
        }
    }
    const char* countField = strstr(buf, "\"count\":");
    ASSERT_NE(countField, nullptr);
    EXPECT_EQ(static_cast<size_t>(atoi(countField + strlen("\"count\":"))), algNum);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);

    // 2) len 极小：空串降级（连 error JSON 装不下）
    void* ccArgs2 = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), nullptr);
    ASSERT_NE(ccArgs2, nullptr);
    char tiny[8];
    EXPECT_EQ(Mc2GetSupportAlgList(comm_, allGatherType, ccArgs2, tiny, sizeof(tiny)), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(tiny[0], '\0');

    // 3) len 中等：名单 JSON 装不下、error JSON 装得下 → 完整 error JSON（含 16K 建议）
    char small[112];
    EXPECT_EQ(Mc2GetSupportAlgList(comm_, allGatherType, ccArgs2, small, sizeof(small)), HCCL_E_NOT_SUPPORT);
    EXPECT_NE(strstr(small, "\"error_msg\":[\""), nullptr);
    EXPECT_NE(strstr(small, "16K"), nullptr);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs2), HCCL_SUCCESS);

    // 4) 引擎 CCU_MS（D19）：error_msg 含引导语
    void* ccArgsMs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_MS), nullptr);
    ASSERT_NE(ccArgsMs, nullptr);
    EXPECT_EQ(Mc2GetSupportAlgList(comm_, allGatherType, ccArgsMs, buf, sizeof(buf)), HCCL_E_NOT_SUPPORT);
    EXPECT_NE(strstr(buf, "CCU_SCHED"), nullptr);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgsMs), HCCL_SUCCESS);

    // 5) algs=nullptr 且 len>0 → HCCL_E_PTR（无处可写）
    void* ccArgs3 = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), nullptr);
    ASSERT_NE(ccArgs3, nullptr);
    EXPECT_EQ(Mc2GetSupportAlgList(comm_, allGatherType, ccArgs3, nullptr, sizeof(buf)), HCCL_E_PTR);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs3), HCCL_SUCCESS);

    // 6) len==0：什么都不写（连 \0 都不写）；nullptr+0 组合不崩、纯返回码
    void* ccArgs4 = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), nullptr);
    ASSERT_NE(ccArgs4, nullptr);
    char untouched = 'X';
    EXPECT_EQ(Mc2GetSupportAlgList(comm_, allGatherType, ccArgs4, &untouched, 0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(untouched, 'X');
    EXPECT_EQ(Mc2GetSupportAlgList(comm_, allGatherType, ccArgs4, nullptr, 0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs4), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, Mc2AcquireCcResCtx_NullParameters)
{
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "CcuSchedAllGatherSoleMesh");
    ASSERT_NE(ccArgs, nullptr);

    void* ccResCtx = reinterpret_cast<void*>(0x1);
    uint32_t ccResCtxSize = 1U;
    EXPECT_EQ(
        Mc2AcquireCcResCtx(
            nullptr, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs, &ccResCtx, &ccResCtxSize),
        HCCL_E_PTR);
    EXPECT_EQ(
        Mc2AcquireCcResCtx(
            comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), nullptr, &ccResCtx, &ccResCtxSize),
        HCCL_E_PTR);
    EXPECT_EQ(
        Mc2AcquireCcResCtx(
            comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs, nullptr, &ccResCtxSize),
        HCCL_E_PTR);
    EXPECT_EQ(
        Mc2AcquireCcResCtx(comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs, &ccResCtx, nullptr),
        HCCL_E_PTR);

    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, Mc2AcquireCcResCtx_RejectsUnsupportedOperation)
{
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "CcuSchedAllGatherSoleMesh");
    ASSERT_NE(ccArgs, nullptr);

    void* ccResCtx = nullptr;
    uint32_t ccResCtxSize = 0U;
    EXPECT_EQ(Mc2AcquireCcResCtx(comm_, 0xFFU, ccArgs, &ccResCtx, &ccResCtxSize), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ccResCtx, nullptr);
    EXPECT_EQ(ccResCtxSize, 0U);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, Mc2AcquireCcResCtx_CcuOperations)
{
    struct CcuOperationCase {
        uint8_t ccType;
        const char* algConfig;
        uint32_t expectedOpType;
    };
    const CcuOperationCase operationCases[] = {
        {static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), "CcuSchedAllGatherSoleMesh",
         static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER)},
        {static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLREDUCE), "CcuSchedAllReduceSoleMesh",
         static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE)},
        {static_cast<uint8_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER), "CcuSchedReduceScatterSoleMesh",
         static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER)},
        {static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLTOALL), "CcuSchedAllToAllSoleMesh",
         static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLTOALL)},
        {static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLTOALLV), "CcuSchedAllToAllVSoleMesh",
         static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLTOALLV)},
    };

    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));
    for (const CcuOperationCase& operationCase : operationCases) {
        void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), operationCase.algConfig);
        ASSERT_NE(ccArgs, nullptr);

        void* ccResCtx = nullptr;
        uint32_t ccResCtxSize = 0U;
        ASSERT_EQ(Mc2AcquireCcResCtx(comm_, operationCase.ccType, ccArgs, &ccResCtx, &ccResCtxSize), HCCL_SUCCESS);
        ASSERT_NE(ccResCtx, nullptr);
        ASSERT_GT(ccResCtxSize, 0U);

        const auto* opResCtx = static_cast<const OpResCtx*>(ccResCtx);
        ASSERT_NE(opResCtx->algInfo[0].opParam, 0U);
        const auto* opParam = reinterpret_cast<const OpParam*>(opResCtx->algInfo[0].opParam);
        EXPECT_EQ(opParam->hcclComm, comm_);
        EXPECT_EQ(opParam->engine, COMM_ENGINE_CCU);
        EXPECT_EQ(opResCtx->version, static_cast<uint32_t>(HcclApi::Mc2LaunchVersion::MC2_CCU_LAUNCH_VERSION));
        EXPECT_EQ(opResCtx->opType[0], operationCase.expectedOpType);
        EXPECT_NE(opResCtx->workSpace, 0U);
        EXPECT_NE(opResCtx->workSpaceSize, 0U);
        EXPECT_EQ(opResCtx->algInfo[0].opParam, reinterpret_cast<uint64_t>(opParam));

        EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
    }
}

TEST_F(CcuMc2TestSuite, Mc2AcquireCcResCtx_RejectsUnsupportedDevice)
{
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "CcuSchedAllGatherSoleMesh");
    ASSERT_NE(ccArgs, nullptr);

    mc2_ops_hccl::g_stubDeviceType = DevType::DEV_TYPE_910B;
    void* ccResCtx = nullptr;
    uint32_t ccResCtxSize = 0U;
    EXPECT_EQ(
        Mc2AcquireCcResCtx(
            comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs, &ccResCtx, &ccResCtxSize),
        HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ccResCtx, nullptr);
    EXPECT_EQ(ccResCtxSize, 0U);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, Mc2AcquireCcResCtx_CcuMs)
{
    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_MS));
    // 新契约：CCU 引擎下算法只由外部名/默认selector决定，裸名不再生效
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_MS), "sole[mesh]");
    ASSERT_NE(ccArgs, nullptr);
    ASSERT_EQ(Mc2SetCcSrcDataType(ccArgs, HCCL_DATA_TYPE_FP32), HCCL_SUCCESS);
    ASSERT_EQ(Mc2SetCcDstDataType(ccArgs, HCCL_DATA_TYPE_FP32), HCCL_SUCCESS);
    ASSERT_EQ(Mc2SetCcReduceType(ccArgs, HCCL_REDUCE_SUM), HCCL_SUCCESS);

    void* ccResCtx = nullptr;
    uint32_t ccResCtxSize = 0U;
    ASSERT_EQ(
        Mc2AcquireCcResCtx(
            comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs, &ccResCtx, &ccResCtxSize),
        HCCL_SUCCESS);
    ASSERT_NE(ccResCtx, nullptr);
    ASSERT_GT(ccResCtxSize, 0U);

    const auto* opResCtx = static_cast<const OpResCtx*>(ccResCtx);
    ASSERT_NE(opResCtx->algInfo[0].opParam, 0U);
    const auto* opParam = reinterpret_cast<const OpParam*>(opResCtx->algInfo[0].opParam);
    EXPECT_EQ(opParam->hcclComm, comm_);
    EXPECT_EQ(opParam->engine, COMM_ENGINE_CCU);
    EXPECT_EQ(opResCtx->version, static_cast<uint32_t>(HcclApi::Mc2LaunchVersion::MC2_CCU_LAUNCH_VERSION));
    EXPECT_EQ(opParam->DataDes.dataType, HCCL_DATA_TYPE_FP32);
    EXPECT_EQ(opParam->DataDes.outputType, HCCL_DATA_TYPE_FP32);
    EXPECT_EQ(opParam->reduceType, HCCL_REDUCE_SUM);
    EXPECT_EQ(opResCtx->opType[0], static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER));
    Mc2CcKernelLaunch(stream_, ccResCtx, ccResCtxSize);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, Mc2AcquireCcResCtx_UsesSelectorWhenAlgConfigIsEmpty)
{
    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));
    mc2_ops_hccl::g_stubSelectorAlgName = "CcuAllGatherMeshMem2Mem1D";
    void* ccArgs = nullptr;
    ASSERT_EQ(Mc2GetCcArgs(&ccArgs), HCCL_SUCCESS);
    ASSERT_NE(ccArgs, nullptr);
    ASSERT_EQ(Mc2SetCcCommEngine(ccArgs, static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED)), HCCL_SUCCESS);

    void* ccResCtx = nullptr;
    uint32_t ccResCtxSize = 0U;
    EXPECT_EQ(
        Mc2AcquireCcResCtx(
            comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs, &ccResCtx, &ccResCtxSize),
        HCCL_SUCCESS);
    EXPECT_NE(ccResCtx, nullptr);
    EXPECT_GT(ccResCtxSize, 0U);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, Mc2AcquireCcResCtx_PropagatesAlgorithmFailure)
{
    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "CcuSchedAllGatherSoleMesh");
    ASSERT_NE(ccArgs, nullptr);

    mc2_ops_hccl::g_stubCcuAlgExecNull = true;
    mc2_ops_hccl::g_stubCcuAlgExecNullName = "CcuSchedAllGatherSoleMesh";
    void* ccResCtx = nullptr;
    uint32_t ccResCtxSize = 0U;
    EXPECT_EQ(
        Mc2AcquireCcResCtx(
            comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs, &ccResCtx, &ccResCtxSize),
        HCCL_E_PTR);
    EXPECT_EQ(ccResCtx, nullptr);
    EXPECT_EQ(ccResCtxSize, 0U);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, Mc2AcquireCcResCtx_PropagatesOpResCtxCopyFailure)
{
    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "CcuSchedAllGatherSoleMesh");
    ASSERT_NE(ccArgs, nullptr);

    // CcuSelectAlg copies OpParam first; Mc2AcquireCcResCtx copies the completed OpResCtx second.
    mc2_ops_hccl::g_stubAclrtMemcpyFailOnCall = 2U;
    void* ccResCtx = nullptr;
    uint32_t ccResCtxSize = 0U;
    EXPECT_EQ(
        Mc2AcquireCcResCtx(
            comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs, &ccResCtx, &ccResCtxSize),
        HCCL_E_RUNTIME);
    EXPECT_EQ(ccResCtx, nullptr);
    EXPECT_EQ(ccResCtxSize, 0U);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, CheckMc2CcDeviceType_Supports950And960)
{
    mc2_ops_hccl::g_stubDeviceType = DevType::DEV_TYPE_950;
    EXPECT_EQ(CheckMc2CcDeviceType(__func__), HCCL_SUCCESS);

    mc2_ops_hccl::g_stubDeviceType = DevType::DEV_TYPE_960;
    EXPECT_EQ(CheckMc2CcDeviceType(__func__), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, Mc2CcKernelLaunch_NullContext)
{
    EXPECT_EQ(Mc2CcKernelLaunch(nullptr, nullptr, 0U), HCCL_E_PTR);
}

TEST_F(CcuMc2TestSuite, Mc2CcKernelLaunch_RejectsInvalidContextSize)
{
    OpParam opParam{};
    OpResCtx opResCtx = BuildLaunchOpResCtx(opParam, comm_, COMM_ENGINE_CCU);
    EXPECT_EQ(Mc2CcKernelLaunch(nullptr, &opResCtx, 0U), HCCL_E_PARA);
}

TEST_F(CcuMc2TestSuite, Mc2CcKernelLaunch_HandlesOpResCopyFailure)
{
    OpParam opParam{};
    OpResCtx opResCtx = BuildLaunchOpResCtx(opParam, comm_, COMM_ENGINE_CCU);
    mc2_ops_hccl::g_stubAclrtMemcpyFail = true;

    EXPECT_EQ(Mc2CcKernelLaunch(nullptr, &opResCtx, sizeof(opResCtx)), HCCL_E_INTERNAL);
}

TEST_F(CcuMc2TestSuite, Mc2CcKernelLaunch_HandlesOpParamCopyFailure)
{
    OpParam opParam{};
    OpResCtx opResCtx = BuildLaunchOpResCtx(opParam, comm_, COMM_ENGINE_CCU);
    mc2_ops_hccl::g_stubAclrtMemcpyFailOnCall = 2U;

    EXPECT_EQ(Mc2CcKernelLaunch(nullptr, &opResCtx, sizeof(opResCtx)), HCCL_E_INTERNAL);
    EXPECT_EQ(mc2_ops_hccl::g_stubAclrtMemcpyCallCount, 2U);
}

TEST_F(CcuMc2TestSuite, Mc2CcKernelLaunch_RejectsNullCommInCcuContext)
{
    OpParam opParam{};
    OpResCtx opResCtx = BuildLaunchOpResCtx(opParam, nullptr, COMM_ENGINE_CCU);

    EXPECT_EQ(Mc2CcKernelLaunch(nullptr, &opResCtx, sizeof(opResCtx)), HCCL_E_INTERNAL);
    EXPECT_EQ(mc2_ops_hccl::g_stubAclrtMemcpyCallCount, 2U);
}

TEST_F(CcuMc2TestSuite, Mc2CcKernelLaunch_RejectsInvalidOpResCtx)
{
    OpParam opParam{};
    OpResCtx opResCtx = BuildLaunchOpResCtx(opParam, comm_, COMM_ENGINE_CCU);
    opResCtx.algInfo[0].opParam = 0U;
    opResCtx.workSpace = 0U;
    opResCtx.workSpaceSize = 0U;

    EXPECT_EQ(Mc2CcKernelLaunch(nullptr, &opResCtx, sizeof(opResCtx)), HCCL_E_INTERNAL);
    EXPECT_EQ(mc2_ops_hccl::g_stubAclrtMemcpyCallCount, 1U);
}

TEST_F(CcuMc2TestSuite, Mc2CcKernelLaunch_RejectsUnsupportedEngine)
{
    OpParam opParam{};
    OpResCtx opResCtx = BuildLaunchOpResCtx(opParam, comm_, static_cast<CommEngine>(0xFFU));

    EXPECT_EQ(Mc2CcKernelLaunch(nullptr, &opResCtx, sizeof(opResCtx)), HCCL_E_NOT_SUPPORT);
}

TEST_F(CcuMc2TestSuite, Mc2CcKernelLaunch_CcuPath)
{
    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "CcuSchedAllGatherSoleMesh");
    ASSERT_NE(ccArgs, nullptr);

    void* ccResCtx = nullptr;
    uint32_t ccResCtxSize = 0U;
    ASSERT_EQ(
        Mc2AcquireCcResCtx(
            comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs, &ccResCtx, &ccResCtxSize),
        HCCL_SUCCESS);
    ASSERT_NE(ccResCtx, nullptr);

    // CCU wraps the launch stream via HcclThreadAcquireWithStream instead of using the
    // pre-allocated thread handles in ccResCtx; the stub thread manager returns a valid handle.
    EXPECT_EQ(Mc2CcKernelLaunch(stream_, ccResCtx, ccResCtxSize), HCCL_SUCCESS);
    EXPECT_EQ(mc2_ops_hccl::g_stubAclrtMemcpyCallCount, 5U);
    EXPECT_EQ(mc2_ops_hccl::g_stubHcommCcuKernelLaunchCallCount, 1U);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, Mc2CcKernelLaunch_CcuPathLegacyNullStream)
{
    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "CcuSchedAllGatherSoleMesh");
    ASSERT_NE(ccArgs, nullptr);

    void* ccResCtx = nullptr;
    uint32_t ccResCtxSize = 0U;
    ASSERT_EQ(
        Mc2AcquireCcResCtx(
            comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs, &ccResCtx, &ccResCtxSize),
        HCCL_SUCCESS);
    ASSERT_NE(ccResCtx, nullptr);

    // Null stream falls back to the legacy path: the pre-allocated thread handles in ccResCtx
    // are used for launching, matching the behavior expected by existing callers.
    EXPECT_EQ(Mc2CcKernelLaunch(nullptr, ccResCtx, ccResCtxSize), HCCL_SUCCESS);
    EXPECT_EQ(mc2_ops_hccl::g_stubHcommCcuKernelLaunchCallCount, 1U);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, Mc2CcKernelLaunch_PropagatesCcuLaunchFailure)
{
    SetCommEngineEnv(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED));
    void* ccArgs = CreateMc2CcArgs(static_cast<uint8_t>(OpExecuteConfig::CCU_SCHED), "CcuSchedAllGatherSoleMesh");
    ASSERT_NE(ccArgs, nullptr);

    void* ccResCtx = nullptr;
    uint32_t ccResCtxSize = 0U;
    ASSERT_EQ(
        Mc2AcquireCcResCtx(
            comm_, static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLGATHER), ccArgs, &ccResCtx, &ccResCtxSize),
        HCCL_SUCCESS);
    ASSERT_NE(ccResCtx, nullptr);

    mc2_ops_hccl::g_stubHcommCcuKernelLaunchFail = true;
    EXPECT_EQ(Mc2CcKernelLaunch(stream_, ccResCtx, ccResCtxSize), HCCL_E_INTERNAL);
    EXPECT_EQ(mc2_ops_hccl::g_stubHcommCcuKernelLaunchCallCount, 1U);
    EXPECT_EQ(Mc2FreeCcArgs(ccArgs), HCCL_SUCCESS);
}

TEST_F(CcuMc2TestSuite, Mc2CcKernelLaunch_RejectsUnsupportedDevice)
{
    OpParam opParam{};
    OpResCtx opResCtx = BuildLaunchOpResCtx(opParam, comm_, COMM_ENGINE_CCU);

    mc2_ops_hccl::g_stubDeviceType = DevType::DEV_TYPE_910B;
    EXPECT_EQ(Mc2CcKernelLaunch(nullptr, &opResCtx, sizeof(opResCtx)), HCCL_E_NOT_SUPPORT);

    EXPECT_EQ(mc2_ops_hccl::g_stubAclrtMemcpyCallCount, 1U);
    EXPECT_EQ(mc2_ops_hccl::g_stubHcommCcuKernelLaunchCallCount, 0U);
}

TEST_F(CcuMc2TestSuite, CcuKernelLaunch_RejectsNullParameters)
{
    EXPECT_EQ(CcuKernelLaunch(nullptr, nullptr), CCU_E_PTR);
    EXPECT_EQ(CcuKernelLaunch(comm_, nullptr), CCU_E_PTR);
}

} // namespace

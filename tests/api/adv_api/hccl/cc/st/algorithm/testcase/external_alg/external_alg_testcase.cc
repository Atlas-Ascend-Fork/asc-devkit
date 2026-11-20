/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "alg_meta_registry.h"
#include "alg_param.h"
#include "auto_selector_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "external_alg_parser.h"
#include "external_alg_resolver.h"
#include "external_alg_rules.h"
#include "gtest/gtest.h"
#include "op_common.h"
#include "sim_communicator.h"
#include "sim_world.h"

#include "acl/acl_rt.h"

namespace mc2_ops_hccl {
// 本目标不链接 KFC server，实现桩仅用于补齐 op_common.cc 的链接依赖。
HcclResult BuildKfcServer(AlgResourceRequest&, std::unique_ptr<AlgResourceCtxSerializable>&, const OpParam&)
{
    return HCCL_E_NOT_SUPPORT;
}
} // namespace mc2_ops_hccl

using namespace mc2_ops_hccl;

namespace {
// 合法外部名样本（30 个规范名的代表集 + 空格/保序变体）
// 变体条目（带空格 / [order.strong] 后缀）解析必须成功，且规范化后与对应规范名相等
const struct {
    const char* name;              // 输入（可为带空格/后缀的变体）
    const char* expectedCanonical; // 期望的规范化名（剥空格；[order.strong] 后缀收编进主键）
} LEGAL_EXTERNAL_NAMES[] = {
    {"sole[mesh]", "sole[mesh]"},
    {"sole[nhr]", "sole[nhr]"},
    {"sole[mesh.chunk]", "sole[mesh.chunk]"},
    {"sole[mesh.one_shot]", "sole[mesh.one_shot]"},
    {"sole[mesh.two_shot]", "sole[mesh.two_shot]"},
    {"sole[mesh.two_shot.chunk]", "sole[mesh.two_shot.chunk]"},
    {"sole[mesh.single_channel]", "sole[mesh.single_channel]"},
    {"sole[mesh.multi_channel]", "sole[mesh.multi_channel]"},
    {"sole[mesh.pairwise]", "sole[mesh.pairwise]"},
    {"sole[nhr.multi_channel]", "sole[nhr.multi_channel]"},
    {"sequence[mesh,nhr]", "sequence[mesh,nhr]"},
    {"sequence[mesh,mesh]", "sequence[mesh,mesh]"},
    {"parallel[mesh,nhr]", "parallel[mesh,nhr]"},
    {"parallel[mesh,nhr.multi_channel]", "parallel[mesh,nhr.multi_channel]"},
    {"concur[mesh,nhr]", "concur[mesh,nhr]"},
    {"concur[mesh,nhr.multi_channel]", "concur[mesh,nhr.multi_channel]"},
    {"concur[mesh,mesh]", "concur[mesh,mesh]"},
    {"pipeline[mesh,nhr]", "pipeline[mesh,nhr]"},
    {"parallel[mesh, nhr]", "parallel[mesh,nhr]"},
    {"sole[mesh] [order.strong]", "sole[mesh][order.strong]"},
    {"sequence[mesh, nhr, nhr]", "sequence[mesh,nhr,nhr]"},
};
} // namespace

TEST(ExtAlgParser, LegalNamesParseAndCanonicalize)
{
    for (const auto& legal : LEGAL_EXTERNAL_NAMES) {
        ExternalAlgSpec spec;
        std::string errMsg;
        ASSERT_TRUE(ParseExternalAlg(legal.name, spec, errMsg)) << "name=" << legal.name << " err=" << errMsg;
        EXPECT_EQ(spec.canonical(), std::string(legal.expectedCanonical)) << "name=" << legal.name;
    }
}

TEST(ExtAlgParser, SpacesEverywhereAreStripped)
{
    ExternalAlgSpec spec;
    std::string errMsg;
    ASSERT_TRUE(ParseExternalAlg("  parallel [ mesh , nhr . multi_channel ]  [ order.strong ] ", spec, errMsg));
    EXPECT_EQ(spec.canonical(), "parallel[mesh,nhr.multi_channel][order.strong]");
    EXPECT_TRUE(spec.orderStrong);
    EXPECT_EQ(spec.executor, ExternalExecutor::PARALLEL);
    ASSERT_EQ(spec.layers.size(), 2U);
    EXPECT_EQ(spec.layers[0].atom, "mesh");
    EXPECT_TRUE(spec.layers[0].attrs.empty());
    EXPECT_EQ(spec.layers[1].atom, "nhr");
    ASSERT_EQ(spec.layers[1].attrs.size(), 1U);
    EXPECT_EQ(spec.layers[1].attrs[0], "multi_channel");
}

TEST(ExtAlgParser, OrderStrongIsPartOfCanonical)
{
    ExternalAlgSpec spec;
    std::string errMsg;
    ASSERT_TRUE(ParseExternalAlg("sole[mesh] [order.strong]", spec, errMsg));
    EXPECT_TRUE(spec.orderStrong);
    EXPECT_EQ(spec.canonical(), "sole[mesh][order.strong]");
}

TEST(ExtAlgParser, IllegalNamesAreRejectedWithReason)
{
    struct Case {
        const char* name;
        std::string expectedErrFragment;
    };
    const Case cases[] = {
        {"", "empty"},
        {"ring[mesh]", "executor"},
        {"sole[ring]", "atom"},
        {"sole[mesh.bigstep]", "attribute"},
        {"sole[mesh.muti_channel]", "attribute"}, // 需求文档笔误的旧拼写
        {"sole[mesh", "missing"},
        {"sole[]", "layer"},
        {"sole[mesh,]", "layer"},
        {"sole[mesh,nhr,nhr,nhr]", "layer"},
        {"sole[mesh] [order.weak]", "suffix"},
        {"parallel[mesh,nhr]]", "suffix"},
        {"sole[mesh.two_shot.chunk.chunk]", ""},
        {"sole[mesh.]", "empty token"}, // 点号结尾
        {"[mesh]", "executor"},         // 空 executor
        {"sole[,,]", "empty layer"},    // 重复逗号
        {"\t", "empty"},                // 纯空白
    };
    for (const Case& c : cases) {
        ExternalAlgSpec spec;
        std::string errMsg;
        const bool ok = ParseExternalAlg(c.name, spec, errMsg);
        if (c.expectedErrFragment.empty()) {
            EXPECT_TRUE(ok) << "name=" << c.name << " err=" << errMsg;
        } else {
            EXPECT_FALSE(ok) << "name=" << c.name;
            EXPECT_NE(errMsg.find(c.expectedErrFragment), std::string::npos) << "name=" << c.name << " err=" << errMsg;
        }
    }
}

TEST(ExtAlgClassify, ThreeStates)
{
    ExternalAlgSpec spec;
    std::string errMsg;

    EXPECT_EQ(ClassifyForcedAlgConfig("", spec, errMsg), ForcedAlgKind::NONE);
    EXPECT_EQ(ClassifyForcedAlgConfig("allgather=level0:fullmesh", spec, errMsg), ForcedAlgKind::LEGACY);
    // 裸算法名不再作为强制算法入口，归为 NONE
    EXPECT_EQ(ClassifyForcedAlgConfig("InsReduceScatterMesh1D", spec, errMsg), ForcedAlgKind::NONE);
    EXPECT_EQ(ClassifyForcedAlgConfig("parallel[mesh, nhr]", spec, errMsg), ForcedAlgKind::EXTERNAL);
    EXPECT_TRUE(errMsg.empty());
    EXPECT_EQ(spec.canonical(), "parallel[mesh,nhr]");
    EXPECT_EQ(ClassifyForcedAlgConfig("sole[ring]", spec, errMsg), ForcedAlgKind::EXTERNAL);
    EXPECT_FALSE(errMsg.empty());
}

TEST(ExtAlgV2Registry, EnumerateRegisteredTags)
{
    auto& registry = CollAlgExecRegistryV2::Instance();
    const std::vector<std::string> agTags = registry.GetRegisteredTags(HcclCMDType::HCCL_CMD_ALLGATHER);
    // 本目标全量编译（-DMC2_CLIENT_ENABLE_CCU=1，无 AICPU_COMPILE 守卫裁剪），AG 应为 15 条：
    // 10 条 AICPU 注册（含 omnipipe）+ 1 条 DPU 注册（InsAllGatherMeshNhrDPU）+ 4 条 CCU 注册
    // （CcuSchedAllGatherSoleMesh / CcuSchedAllGatherSoleNHR / CcuSchedAllGatherParallelMeshNHRMultiLink /
    // CcuSchedAllGatherConcurMeshNHRMultiLink）；InsAllGatherParallelMesh1DNHRUBX 注册已删除；
    // 此断言同时防止枚举接口漏实现
    EXPECT_FALSE(agTags.empty());
    for (const std::string& tag : agTags) {
        EXPECT_TRUE(registry.IsRegistered(HcclCMDType::HCCL_CMD_ALLGATHER, tag)) << tag;
    }
    // kfc_server/executor 未编入本目标且 AICPU_COMPILE/HCCL_CANN_COMPAT_850 均未定义，
    // 若意外编入将注册 CcuKfcServer——此断言是 GLOB 排除防线的测试侧副本
    EXPECT_TRUE(registry.GetRegisteredTags(HcclCMDType::HCCL_CMD_KFC_SERVER).empty());
}

TEST(ExtAlgMetaRegistry, RegisterAndQuery)
{
    // 用本目标已注册的 sidecar 行验证
    const AlgMeta* meta =
        AlgMetaRegistry::Instance().Get(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "InsReduceScatterMesh1D");
    ASSERT_NE(meta, nullptr);
    EXPECT_EQ(meta->engine, AlgEngine::AICPU);
    EXPECT_EQ(std::string(meta->externalName), "sole[mesh]");
}

// Register 空 patterns 防御：count==0 的行（零长数组或缺省聚合）经 MatchAnyPattern 对 0
// 条模式恒不命中——入库即永不可选的静默死行，须注册期拒绝且不入库（HCCL_E_PARA + Get 查无）。
// 宏路径由 PatternCountOf 的数组引用推导天然免疫零长数组（N=0 推导失败即编译错），此防御覆盖
// 绕过宏的直接 Register 调用。KFC cmd 不触碰满射断言的 5 个 COLL_CMDS；拒绝的行不入库，
// 54/55 行计数锚与候选表均不受影响
TEST(ExtAlgMetaRegistry, RegisterRejectsEmptyPatternArray)
{
    static const TopoPattern kEmptyPatterns[] = {}; // 零长数组：合法指针 + count 0
    const AlgMeta emptyArrMeta{AlgEngine::AICPU, "sole[mesh]", COND_NONE, kEmptyPatterns, 0U, 0};
    EXPECT_EQ(
        AlgMetaRegistry::Instance().Register(HcclCMDType::HCCL_CMD_KFC_SERVER, "TestEmptyPatternArray", emptyArrMeta),
        HCCL_E_PARA);
    // 缺省聚合（patterns nullptr / count 0，漏写 patterns 字段的直接构造形态）同拒
    const AlgMeta defaultMeta{AlgEngine::AICPU, "sole[mesh]", COND_NONE};
    EXPECT_EQ(
        AlgMetaRegistry::Instance().Register(HcclCMDType::HCCL_CMD_KFC_SERVER, "TestDefaultPatternMeta", defaultMeta),
        HCCL_E_PARA);
    // 拒绝的行不入库（防死行被 Get/候选表误消费）
    EXPECT_EQ(AlgMetaRegistry::Instance().Get(HcclCMDType::HCCL_CMD_KFC_SERVER, "TestEmptyPatternArray"), nullptr);
    EXPECT_EQ(AlgMetaRegistry::Instance().Get(HcclCMDType::HCCL_CMD_KFC_SERVER, "TestDefaultPatternMeta"), nullptr);
}

namespace {
const HcclCMDType COLL_CMDS[] = {
    HcclCMDType::HCCL_CMD_ALLGATHER, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclCMDType::HCCL_CMD_ALLREDUCE,
    HcclCMDType::HCCL_CMD_ALLTOALL,  HcclCMDType::HCCL_CMD_ALLTOALLV,
};
} // namespace

// 不变量：V2 注册表 ↔ AlgMetaRegistry 双向满射。本目标全量编译所有注册现场
// （18 个 executor 文件，-DMC2_CLIENT_ENABLE_CCU=1 且无 AICPU_COMPILE 守卫裁剪），可见全部
// 58 行 sidecar（COLL_CMDS 域内 57 行 + HALF_ALLTOALLV 域外 1 行）——COLL_CMDS 按 cmd 分布：
// AG 15 / RS 18 / AR 10 / A2A 7 / A2AV 7（含 upstream 新迁 CcuSchedAllToAllVSoleMeshMultiJetty）。 Mc2GetSupportAlgList
// TopoPattern 化
TEST(ExtAlgSurjectivity, V2AndMetaRegistryAreBijective)
{
    for (HcclCMDType cmd : COLL_CMDS) {
        std::vector<std::string> v2Tags = CollAlgExecRegistryV2::Instance().GetRegisteredTags(cmd);
        std::vector<std::string> metaNames;
        for (const AlgMetaRow& row : AlgMetaRegistry::Instance().GetAll()) {
            if (row.cmd == cmd) {
                metaNames.push_back(row.registeredName);
            }
        }
        std::sort(v2Tags.begin(), v2Tags.end());
        std::sort(metaNames.begin(), metaNames.end());
        EXPECT_EQ(v2Tags, metaNames) << "cmd=" << static_cast<u32>(cmd);
    }
    // 各 cmd 精确分布锚（AG 15 / RS 18 / AR 10 / A2A 7 / A2AV 5 + AR SoleNHR pattern 补挂 = 55），失败时定位更直接
    EXPECT_EQ(CollAlgExecRegistryV2::Instance().GetRegisteredTags(HcclCMDType::HCCL_CMD_ALLGATHER).size(), 15U);
    EXPECT_EQ(CollAlgExecRegistryV2::Instance().GetRegisteredTags(HcclCMDType::HCCL_CMD_REDUCE_SCATTER).size(), 18U);
    EXPECT_EQ(CollAlgExecRegistryV2::Instance().GetRegisteredTags(HcclCMDType::HCCL_CMD_ALLREDUCE).size(), 10U);
    EXPECT_EQ(CollAlgExecRegistryV2::Instance().GetRegisteredTags(HcclCMDType::HCCL_CMD_ALLTOALL).size(), 7U);
    EXPECT_EQ(CollAlgExecRegistryV2::Instance().GetRegisteredTags(HcclCMDType::HCCL_CMD_ALLTOALLV).size(), 7U);
    // 本目标全量编译（MC2_CLIENT_ENABLE_CCU=1，无 AICPU_COMPILE）：58 行精确锚（COLL_CMDS 57 + HALF 1）
    size_t total = 0;
    for (HcclCMDType cmd : COLL_CMDS) {
        for (const AlgMetaRow& row : AlgMetaRegistry::Instance().GetAll()) {
            if (row.cmd == cmd) {
                total++;
            }
        }
    }
    EXPECT_EQ(total, 57U);
    // 域外行（HALF_ALLTOALLV）双射锚：该 cmd 是 AllToAllvWrite 的辅助 cmd（Mc2AllToAllvWrite
    // 内部派生下发），非用户外部名域——COLL_CMDS 循环与 54 锚有意不含；其单注册行的
    // V2 tags ↔ meta registeredName 双射同样钉住（sidecar 与主注册脱离同生共死时先于计数漂移暴露）
    EXPECT_EQ(
        CollAlgExecRegistryV2::Instance().GetRegisteredTags(HcclCMDType::HCCL_CMD_HALF_ALLTOALLV),
        (std::vector<std::string>{"CcuHalfAll2AllVMesh1D"}));
    EXPECT_NE(AlgMetaRegistry::Instance().Get(HcclCMDType::HCCL_CMD_HALF_ALLTOALLV, "CcuHalfAll2AllVMesh1D"), nullptr);
}

// 不变量：注册行 externalName 字面量必须自身就是规范形（"与 parser 产出同构"的机器契约）。
// 手写字面量若带空格/词表外词/非规范拼写，将静默变死行——用户输入经 canonical() 规范化后永不相等，
// 计数锚（键/组合/行）与 registeredName 满射断言均无法发现。往返自检：parse(字面量) 成功且 canonical() == 字面量。
TEST(ExtAlgSurjectivity, ExternalNameLiteralsAreCanonical)
{
    const std::vector<AlgMetaRow> rows = AlgMetaRegistry::Instance().GetAll();
    ASSERT_FALSE(rows.empty()); // 全量编译目标：57 行产品 sidecar（COLL_CMDS 56 + HALF_ALLTOALLV 1）+ 测试夹具
    for (const AlgMetaRow& row : rows) {
        ExternalAlgSpec spec;
        std::string errMsg;
        EXPECT_TRUE(ParseExternalAlg(row.meta.externalName, spec, errMsg))
            << "registeredName=" << row.registeredName << " externalName=\"" << row.meta.externalName
            << "\" err=" << errMsg;
        EXPECT_EQ(spec.canonical(), std::string(row.meta.externalName))
            << "registeredName=" << row.registeredName << "（字面量非规范形，将成为死行）";
    }
}

// D18 JSON 零转义契约的机器验证：externalName 字符集白名单 = 标识符字符 + [ ] . , _——
// 逗号是 sequence[mesh,nhr] / concur[mesh,mesh] 等多层名的既成分隔符（实测注册面在用，
// 原设 [A-Za-z0-9_.\[\]] 漏列）。白名单外的任何字符（尤其需要 JSON 转义的 " 与 \）当场红：
// Mc2GetSupportAlgList 的 C ABI 输出按名裸拼 JSON 数组，无转义路径，字符集越界即契约破坏
TEST(ExtAlgSurjectivity, ExternalNameCharsetStaysJsonEscapeFree)
{
    for (const AlgMetaRow& row : AlgMetaRegistry::Instance().GetAll()) {
        for (const char ch : std::string(row.meta.externalName)) {
            const bool allowed = (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') ||
                                 ch == '_' || ch == '.' || ch == ',' || ch == '[' || ch == ']';
            EXPECT_TRUE(allowed) << "registeredName=" << row.registeredName << " externalName=\""
                                 << row.meta.externalName << "\" 含白名单外字符 '" << ch << "'";
        }
    }
}

namespace {
// ---- topo/opParam 夹具 ----

TopoInfoWithNetLayerDetails MakeFlatMesh1DTopo(u32 rankSize = 8U)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRank = 0U;
    topo.userRankSize = rankSize;
    topo.serverNum = 1U;
    topo.topoLevelNums = 1U;
    topo.level0Topo = Level0Shape::MESH_1D;
    topo.level0MeshType = Level0MeshType::SINGLE_DIE; // 真实 topo 对 MESH_1D 必填实际值（topo_host.cc）
    topo.netLayerDetails.netLayerNum = 1U;
    topo.netLayerDetails.netLayers = {0U};
    topo.netLayerDetails.localNetInsSizeOfLayer = {1U};
    topo.topoInstDetailsOfLayer.resize(1U);
    topo.topoInstDetailsOfLayer[0].rankNumForTopoType[COMM_TOPO_1DMESH] = {rankSize};
    return topo;
}

TopoInfoWithNetLayerDetails MakeUbxTopo(u32 rankSize = 16U)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRank = 0U;
    topo.userRankSize = rankSize;
    topo.serverNum = 1U;
    topo.topoLevelNums = 1U;
    topo.level0Topo = Level0Shape::MESH_1D_CLOS; // UBX: MESH_1D_CLOS 且非 pcieMix
    topo.level0PcieMix = false;
    topo.netLayerDetails.netLayerNum = 1U;
    topo.netLayerDetails.netLayers = {0U};
    // 真实 MESH_1D_CLOS 不变量：localNetInsSizeOfLayer[0] == CLOS rank 数（topo_host.cc
    // CalcLevel0TopoShape 校验），mesh 只覆盖 8 卡 < CLOS 域
    topo.netLayerDetails.localNetInsSizeOfLayer = {rankSize};
    topo.topoInstDetailsOfLayer.resize(1U);
    topo.topoInstDetailsOfLayer[0].rankNumForTopoType[COMM_TOPO_1DMESH] = {8U};
    topo.topoInstDetailsOfLayer[0].rankNumForTopoType[COMM_TOPO_CLOS] = {rankSize};
    return topo;
}

TopoInfoWithNetLayerDetails MakePcieMixTopo(u32 rankSize = 16U)
{
    TopoInfoWithNetLayerDetails topo = MakeUbxTopo(rankSize);
    topo.level0PcieMix = true;
    // level0 非全互联夹具（IsLayerAllConnetedWithTopo(0, 1DMESH) == false）。该函数比较
    // rankNumForTopoType[1DMESH] 与 localNetInsSizeOfLayer[0]（不是 rankSize），真实
    // MESH_1D_CLOS 不变量为 localNetInsSizeOfLayer[0] == CLOS rank 数（topo_host.cc
    // CalcLevel0TopoShape 校验），mesh 只覆盖 8 卡 < CLOS 域 16 卡即需跨 pcie 链路
    topo.netLayerDetails.localNetInsSizeOfLayer = {rankSize};
    return topo;
}

// UBX mesh == clos 形态（默认 4 卡）：concur 系算法要求 CheckMeshNumEqualToClosNum 命中且
// rank <= 4，MakeUbxTopo 的 mesh 恒 8 卡覆盖不到 meshEqClos 分支——在其基础上把 level0 mesh
// 卡数（rankNumForTopoType[COMM_TOPO_1DMESH]）改为与 CLOS 域一致（rankSize），其余字段沿用
// UBX 不变量（localNetInsSizeOfLayer[0] == CLOS rank 数，topoInstDetailsOfLayer[CLOS] = rankSize）
TopoInfoWithNetLayerDetails MakeUbxEqualTopo(u32 rankSize = 4U)
{
    TopoInfoWithNetLayerDetails topo = MakeUbxTopo(rankSize);
    topo.topoInstDetailsOfLayer[0].rankNumForTopoType[COMM_TOPO_1DMESH] = {rankSize};
    return topo;
}

TopoInfoWithNetLayerDetails MakeMultiLevelTopo(u32 rankSize = 16U)
{
    TopoInfoWithNetLayerDetails topo = MakeFlatMesh1DTopo(rankSize);
    topo.topoLevelNums = 2U;
    return topo;
}

// A2AV Pairwise 形态：IsSupportedTopology（ins_temp_all_to_all_v_pairwise.cc:51-58，与 selector
// 共用守卫）要求多级 MESH_1D + level0Symmetric + 每框 8 卡 + rank 16 的倍数；MakeMultiLevelTopo
// 未置 symmetric/deviceNumPerModule（各默认 false/0），在此补齐两字段
TopoInfoWithNetLayerDetails MakePairwiseTopo(u32 rankSize = 16U)
{
    TopoInfoWithNetLayerDetails topo = MakeMultiLevelTopo(rankSize);
    topo.level0Symmetric = true;
    topo.deviceNumPerModule = 8U;
    return topo;
}

// 单机纯 CLOS（大禹形态）：单层 level0 CLOS
TopoInfoWithNetLayerDetails MakeSingleLevelClosTopo(u32 rankSize = 8U)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRank = 0U;
    topo.userRankSize = rankSize;
    topo.serverNum = 1U;
    topo.topoLevelNums = 1U;
    topo.level0Topo = Level0Shape::CLOS;
    topo.netLayerDetails.netLayerNum = 1U;
    topo.netLayerDetails.netLayers = {0U};
    topo.netLayerDetails.localNetInsSizeOfLayer = {1U};
    topo.topoInstDetailsOfLayer.resize(1U);
    topo.topoInstDetailsOfLayer[0].rankNumForTopoType[COMM_TOPO_CLOS] = {rankSize};
    return topo;
}

// 两层网最小夹具（IsTwoLevelNetLayer 判定: netLayerNum>1 / level1 有 COMM_TOPO_CLOS / localNetInsSizeOfLayer[0]>1）
TopoInfoWithNetLayerDetails MakeTwoLevelNetLayerTopo(u32 rankSize = 16U)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRank = 0U;
    topo.userRankSize = rankSize;
    topo.topoLevelNums = 2U;
    topo.level0Topo = Level0Shape::MESH_1D;
    topo.netLayerDetails.netLayerNum = 2U;
    topo.netLayerDetails.netLayers = {0U, 1U};
    topo.netLayerDetails.localNetInsSizeOfLayer = {2U, 1U}; // level0 本实例卡数>1
    topo.topoInstDetailsOfLayer.resize(2U);
    topo.topoInstDetailsOfLayer[1].rankNumForTopoType[COMM_TOPO_CLOS] = {rankSize}; // level1 有 CLOS
    return topo;
}

// 单层 MESH_1D + 两层网（twoLevelNet 与 topoLevelNums 解耦形态）：IsTwoLevelNetLayer 读
// localNetInsSizeOfLayer[0]（本实例卡数）不读 netInstNumOfLayer（实例总数），L0 单实例覆盖
// 全卡（topoLevelNums=1）时两层网仍可为真——RS selector 的 twoLevel 分支
// （SelectMeshAlgoAicpuForMesh1D，仅单层路径 :470-471 调用）与 ZAxisDetour 强制变体均定义在
// 此形态上；MakeTwoLevelNetLayerTopo 写成 2 级拓扑（真实 selector 2 级走多级路径，永不读
// twoLevelNet），模式化后 golden 用例改用本夹具表达真语义
TopoInfoWithNetLayerDetails MakeSingleLevelTwoLevelNetTopo(u32 rankSize = 16U)
{
    TopoInfoWithNetLayerDetails topo = MakeTwoLevelNetLayerTopo(rankSize);
    topo.topoLevelNums = 1U;
    return topo;
}

OpParam MakeOpParam(
    HcclDataType dtype = HCCL_DATA_TYPE_FP32, u64 count = 1U, HcclReduceOp reduce = HcclReduceOp::HCCL_REDUCE_SUM)
{
    OpParam op{};
    op.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    op.DataDes.dataType = dtype;
    op.DataDes.count = count;
    op.reduceType = reduce;
    return op;
}
} // namespace

TEST(ExtAlgRules, CondZAxisAlignsWithHccl)
{
    // (2P 且 dataSize>=4MB 或 dataSize*rankSize>1536MB)；两层网拓扑面已由 ZAxisDetour 的
    // TopoPattern 承载（spec D2），谓词不查 IsTwoLevelNetLayer，数据量条件独立成立
    // dataSize = count × dtypeSize，FP32 为 4B/元素
    TopoInfoWithNetLayerDetails twoLevel = MakeTwoLevelNetLayerTopo();
    TopoInfoWithNetLayerDetails flat = MakeFlatMesh1DTopo();

    // 非两层网 + 大数据：总量 400MB×8=3200MB > 1536MB，数据量支独立命中（旧谓词被
    // IsTwoLevelNetLayer 前置拦截为 false，拓扑面移交 TopoPattern 后的谓词级预期变化；
    // 定名侧由模式 twoLevelNet=Yes 补齐，选名不变）
    EXPECT_TRUE(CondZAxis(&flat, MakeOpParam(HCCL_DATA_TYPE_FP32, 100U * 1024 * 1024)));
    EXPECT_FALSE(CondZAxis(&twoLevel, MakeOpParam(HCCL_DATA_TYPE_FP32, 1024U))); // 小数据
    // 2P + 4MB 边界：1M FP32 元素恰为 4MB
    TopoInfoWithNetLayerDetails twoRank = MakeTwoLevelNetLayerTopo(2U);
    EXPECT_TRUE(CondZAxis(&twoRank, MakeOpParam(HCCL_DATA_TYPE_FP32, 1024U * 1024)));
    EXPECT_FALSE(CondZAxis(&twoRank, MakeOpParam(HCCL_DATA_TYPE_FP32, 1024U * 1024 - 1)));
    // 多 rank + 总量 1536MB 边界：
    // 16 rank 时需 dataSize > 96MB；24M FP32 元素 = 96MB，×16 = 1536MB 恰不超阈值
    EXPECT_FALSE(CondZAxis(&twoLevel, MakeOpParam(HCCL_DATA_TYPE_FP32, 24U * 1024 * 1024)));
    EXPECT_TRUE(CondZAxis(&twoLevel, MakeOpParam(HCCL_DATA_TYPE_FP32, 24U * 1024 * 1024 + 1)));
}

TEST(ExtAlgRules, CondSpecialDtAlignsWithHccl)
{
    TopoInfoWithNetLayerDetails flat = MakeFlatMesh1DTopo();
    EXPECT_TRUE(CondSpecialDt(&flat, MakeOpParam(HCCL_DATA_TYPE_INT64)));
    EXPECT_TRUE(CondSpecialDt(&flat, MakeOpParam(HCCL_DATA_TYPE_UINT64)));
    EXPECT_TRUE(CondSpecialDt(&flat, MakeOpParam(HCCL_DATA_TYPE_FP64)));
    EXPECT_TRUE(CondSpecialDt(&flat, MakeOpParam(HCCL_DATA_TYPE_FP32, 1U, HcclReduceOp::HCCL_REDUCE_PROD)));
    EXPECT_FALSE(CondSpecialDt(&flat, MakeOpParam(HCCL_DATA_TYPE_FP32)));
}

TEST(ExtAlgRules, CondPeerOnlySemantics)
{
    TopoInfoWithNetLayerDetails twoRank = MakeFlatMesh1DTopo(2U);
    TopoInfoWithNetLayerDetails eightRank = MakeFlatMesh1DTopo(8U);
    OpParam op = MakeOpParam();                 // hcclComm 为空 -> 查询前提不满足，保守不命中
    EXPECT_FALSE(CondPeerOnly(&eightRank, op)); // rankSize!=2 直接 false
    EXPECT_FALSE(CondPeerOnly(&twoRank, op));   // 无 comm 环境保守 false（真实链路走 ST）
}

// ---- COND_PEER_ONLY 与建链同口径（sim 链路世界）----
// 本目标链接的 hccl_stub 将 HcclRankGraphGetLayers/GetLinks 委托给 sim 世界
// （SimCommunicator.topoModel_，TopoMeta+DevType 决定层序与链路协议），可构造谓词的
// 关键判定形态；"raw 链数"与"协议过滤后通道数"的差异即量纲修正的判定面。
namespace {
// Init/Deinit 成对（同 all_reduce/reduce_scatter 等 sim 用例范式）；comm 初始化失败路径
// 自清理，防止脏 SimWorld 单例（残留 TopoMeta/DevType 拓扑）影响同二进制内后续用例。
// aclrtSetDevice：桩内 curr_dev_id 默认 UINT32_MAX，谓词路径一旦引入设备类查询
// （GetCcuVersion/hrtGetDeviceType，见 parallel_migration_testcase 判例）会在 CommDomain
// 查询抛 InvalidParamsException——显式落 rank0 设备，异常失败防御。
HcclComm MakeSimComm(const TopoMeta& topoMeta, DevType devType)
{
    HcclSim::SimWorld::Global()->Init(topoMeta, devType);
    (void)aclrtSetDevice(0U);
    HcclComm comm = nullptr;
    if (HcclSim::Sim_HcclCommInitClusterInfo(topoMeta, 0U, &comm) != HCCL_SUCCESS) {
        HcclSim::SimWorld::Global()->Deinit();
        return nullptr;
    }
    return comm;
}

void DestroySimComm(HcclComm& comm)
{
    if (comm != nullptr) {
        delete static_cast<HcclSim::SimCommunicator*>(comm);
        comm = nullptr;
    }
    HcclSim::SimWorld::Global()->Deinit();
}

// 旧实现消费 topo->netLayerDetails.netLayers[0]，镜像 sim 世界层序（HcclRankGraphGetLayers）
TopoInfoWithNetLayerDetails MakeTwoRankTopo(const std::vector<u32>& netLayers)
{
    TopoInfoWithNetLayerDetails topo = MakeFlatMesh1DTopo(2U);
    topo.netLayerDetails.netLayerNum = static_cast<u32>(netLayers.size());
    topo.netLayerDetails.netLayers = netLayers;
    return topo;
}

OpParam MakeCcuOpParam(HcclComm comm)
{
    OpParam op = MakeOpParam();
    op.engine = CommEngine::COMM_ENGINE_CCU;
    op.hcclComm = comm;
    return op;
}
} // namespace

TEST(ExtAlgRules, CondPeerOnlySingleCtpChannelHits)
{
    // TP2 单 server 950（MESH_1D）：L0 0↔1 恰 1 条 UBC_CTP（同行设备直连）→ 恰 1 通道，命中
    HcclComm comm = MakeSimComm(TopoMeta{{{{0, 1}}}}, DevType::DEV_TYPE_950);
    ASSERT_NE(comm, nullptr);
    TopoInfoWithNetLayerDetails topo = MakeTwoRankTopo({0U});
    EXPECT_TRUE(CondPeerOnly(&topo, MakeCcuOpParam(comm)));
    DestroySimComm(comm);
}

TEST(ExtAlgRules, CondPeerOnlyCrossLayerSingleCtpChannelHits)
{
    // D2 跨层窗口：同 pod 跨 server（2 server × 1 rank）→ L0 无链、L1 恰 1 条 UBC_CTP，
    // 建链按层序落至 L1 产出 1 通道 → 命中；旧量纲只数 layer[0] raw 链（0 条）→ 漏检回落裸名
    HcclComm comm = MakeSimComm(TopoMeta{{{{0}}, {{1}}}}, DevType::DEV_TYPE_950);
    ASSERT_NE(comm, nullptr);
    TopoInfoWithNetLayerDetails topo = MakeTwoRankTopo({0U, 1U});
    OpParam op = MakeCcuOpParam(comm);
    EXPECT_TRUE(CondPeerOnly(&topo, op));
    op.hcclComm = nullptr; // 同一拓扑下无 comm 仍保守不命中
    EXPECT_FALSE(CondPeerOnly(&topo, op));
    DestroySimComm(comm);
}

TEST(ExtAlgRules, CondPeerOnlyRejectsNonCcuProtocolRawLink)
{
    // 量纲修正（误命中方向）：910B 同 server phyIds{0,9} → L0 恰 1 条 RESERVED 原始链。
    // raw 链数==1 是量纲错误：CCU 引擎协议集 [UBC_CTP, UBC_TP] 过滤后 0 通道
    // （模板守卫 channelCount==1 必拒，谓词须同口径不命中）
    HcclComm comm = MakeSimComm(TopoMeta{{{{0, 9}}}}, DevType::DEV_TYPE_910B);
    ASSERT_NE(comm, nullptr);
    TopoInfoWithNetLayerDetails topo = MakeTwoRankTopo({0U});
    OpParam op = MakeCcuOpParam(comm);
    EXPECT_FALSE(CondPeerOnly(&topo, op));
    op.engine = CommEngine::COMM_ENGINE_RESERVED; // 引擎协议集为空：查询前提不满足，保守不命中
    EXPECT_FALSE(CondPeerOnly(&topo, op));
    DestroySimComm(comm);
}

TEST(ExtAlgRules, CondPeerOnlyRequiresExactlyOneChannel)
{
    // 精确计数守卫：大禹纯 CLOS（既有 opt-in 旋钮）下 TP2 L0 有 die0/die1 两条 UBC_CTP
    // → 通道数 2 != 1 → 不命中（区分"恰 1 条"与"存在 CCU 协议链"两种实现）
    setenv("ENABLE_PURE_CLOS_TOPO_FOR_LLT", "1", 1); // TopoModel 构造期读取，须先于 comm 创建
    HcclComm comm = MakeSimComm(TopoMeta{{{{0, 1}}}}, DevType::DEV_TYPE_950);
    unsetenv("ENABLE_PURE_CLOS_TOPO_FOR_LLT");
    ASSERT_NE(comm, nullptr);
    TopoInfoWithNetLayerDetails topo = MakeTwoRankTopo({0U});
    EXPECT_FALSE(CondPeerOnly(&topo, MakeCcuOpParam(comm)));
    DestroySimComm(comm);
}

TEST(ExtAlgRules, CondPeerOnlyNoCcuChannelOnAnyLayerMisses)
{
    // 全层无 CCU 协议通道：910B 跨 server → L0 无链、L1 恰 1 条 ROCE（非 CCU 协议集）→ 不命中
    // （覆盖"该层有链但无引擎协议命中 → 下一层"分支）
    HcclComm comm = MakeSimComm(TopoMeta{{{{0}}, {{1}}}}, DevType::DEV_TYPE_910B);
    ASSERT_NE(comm, nullptr);
    TopoInfoWithNetLayerDetails topo = MakeTwoRankTopo({0U, 1U});
    EXPECT_FALSE(CondPeerOnly(&topo, MakeCcuOpParam(comm)));
    TopoInfoWithNetLayerDetails eightRank = MakeFlatMesh1DTopo(8U); // rankSize!=2：真实 comm 下仍不命中
    EXPECT_FALSE(CondPeerOnly(&eightRank, MakeCcuOpParam(comm)));
    DestroySimComm(comm);
}

TEST(ExtAlgRules, EngineMatchMatrix)
{
    OpParam op = MakeOpParam();
    op.engine = CommEngine::COMM_ENGINE_CPU; // DPU 行仅 CPU 引擎命中
    EXPECT_TRUE(EngineMatch(AlgEngine::DPU, op));
    EXPECT_FALSE(EngineMatch(AlgEngine::AICPU, op));
    EXPECT_FALSE(EngineMatch(AlgEngine::CCU, op));

    op.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    EXPECT_TRUE(EngineMatch(AlgEngine::AICPU, op));
    EXPECT_FALSE(EngineMatch(AlgEngine::DPU, op));
    op.engine = CommEngine::COMM_ENGINE_AICPU;
    EXPECT_TRUE(EngineMatch(AlgEngine::AICPU, op));

    op.engine = CommEngine::COMM_ENGINE_CCU;
    EXPECT_TRUE(EngineMatch(AlgEngine::CCU, op));
    EXPECT_FALSE(EngineMatch(AlgEngine::AICPU, op));
}

namespace {
OpParam MakeResolveParam(HcclCMDType cmd, CommEngine engine)
{
    OpParam op = MakeOpParam();
    op.opType = cmd;
    op.engine = engine;
    return op;
}
} // namespace

// golden：(引擎×op×外部名) 组合的定名全覆盖，期望值与注册现场 sidecar 表逐行核对
// （grep REGISTER_ALG_META 可复核）
TEST(ExtAlgResolver, GoldenAllEngines)
{
    struct Case {
        HcclCMDType cmd;
        const char* extName;
        const TopoInfoWithNetLayerDetails* topo;
        CommEngine engine;
        const char* expected;
    };
    TopoInfoWithNetLayerDetails flat = MakeFlatMesh1DTopo();
    TopoInfoWithNetLayerDetails ubx = MakeUbxTopo();
    TopoInfoWithNetLayerDetails pcie = MakePcieMixTopo();
    TopoInfoWithNetLayerDetails multi = MakeMultiLevelTopo();
    TopoInfoWithNetLayerDetails clos = MakeSingleLevelClosTopo();
    TopoInfoWithNetLayerDetails twoLevel = MakeTwoLevelNetLayerTopo();
    // 单层 twoLevelNet 形态（twoLevelNet 与 topoLevelNums 解耦，见夹具注释）：RS sole[mesh] 的
    // twoLevelNet=Yes 模式面 / ZAxisDetour 拓扑面定义在此形态上
    TopoInfoWithNetLayerDetails singleLevelTwoLevel = MakeSingleLevelTwoLevelNetTopo();
    // UBX 4 卡 mesh == clos 形态：concur 系算法要求 CheckMeshNumEqualToClosNum 命中且 rank <= 4
    // （MakeUbxTopo 的 mesh 恒 8 卡，覆盖不到 meshEqClos 分支）
    TopoInfoWithNetLayerDetails ubxEqual = MakeUbxEqualTopo();
    // A2AV Pairwise 形态：多级 MESH_1D + 对称 + 每框 8 卡 + 16 倍数 rank
    // （真实 selector 由 IsSupportedTopology 前置守卫）
    TopoInfoWithNetLayerDetails pairwise = MakePairwiseTopo();

    const Case cases[] = {
        // ---- 纯查表组合（AICPU/DPU）。AG 行模式化后拓扑须命中对应 TopoPattern
        // （sole[nhr] 至少两层非 NHR；sole[nhr.multi_channel] 单层 CLOS；concur 系 UBX mesh == clos 且 rank <= 4）
        {HcclCMDType::HCCL_CMD_ALLGATHER, "sole[mesh]", &flat, CommEngine::COMM_ENGINE_AICPU_TS, "InsAllGatherMesh1D"},
        {HcclCMDType::HCCL_CMD_ALLGATHER, "sole[nhr]", &multi, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuAllGatherSoleNHR"},
        {HcclCMDType::HCCL_CMD_ALLGATHER, "sole[nhr.multi_channel]", &clos, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuAllGatherSoleNHRMultiLink"},
        {HcclCMDType::HCCL_CMD_ALLGATHER, "parallel[mesh,nhr.multi_channel]", &ubx, CommEngine::COMM_ENGINE_AICPU_TS,
         "InsAllGatherParallelMesh1DNHRMultiJetty"},
        {HcclCMDType::HCCL_CMD_ALLGATHER, "concur[mesh,nhr]", &ubxEqual, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuAllGatherConcurMeshNHR"},
        {HcclCMDType::HCCL_CMD_ALLGATHER, "pipeline[mesh,nhr]", &pcie, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuAllGatherPipeLinePcie"},
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "sole[mesh.chunk]", &flat, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuReduceScatterSoleMeshChunk"},
        // RS·sequence[mesh,nhr]：模式化后两层 Mesh 需 level0 本实例卡数 > 1（GT 例外）；
        // multi 夹具 inst[0]==1 在真实 selector 走 :460 单实例支选 NHR，改 twoLevel 真命中
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "sequence[mesh,nhr]", &twoLevel, CommEngine::COMM_ENGINE_AICPU_TS,
         "InsReduceScatterSequenceMesh1DNhr"},
        // RS·concur[mesh,nhr]：模式面需全互联 mesh==clos（同 AG concur 先例用 ubxEqual；
        // MakeUbxTopo 的 mesh 恒 8 卡覆盖不到 meshEqClos 分支）
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "concur[mesh,nhr]", &ubxEqual, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuReduceScatterConcurMeshNHR"},
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "pipeline[mesh,nhr]", &pcie, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuReduceScatterPipeLinePcie"},
        {HcclCMDType::HCCL_CMD_ALLREDUCE, "sole[mesh.two_shot]", &flat, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuAllReduceSoleMeshTwoShot"},
        {HcclCMDType::HCCL_CMD_ALLREDUCE, "sole[mesh.one_shot]", &flat, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuAllReduceSoleMeshOneShot"},
        {HcclCMDType::HCCL_CMD_ALLREDUCE, "sole[mesh.two_shot.chunk]", &flat, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuAllReduceSoleMeshChunkTwoShot"},
        {HcclCMDType::HCCL_CMD_ALLTOALL, "sole[mesh.single_channel]", &flat, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuAllToAllSoleMeshSingleChannel"},
        // A2A·concur[mesh,mesh]：模式面需 mesh==clos 且 rank <= 4（AICPU 版 ELEMENT 子句在
        // 定名 len=0 下跳过，拓扑面同 CCU 版），ubx 形态（mesh 8 != clos 16 且 rank 16）两候选
        // 皆不命中——改 ubxEqual 真命中（同 AG concur 先例）
        {HcclCMDType::HCCL_CMD_ALLTOALL, "concur[mesh,mesh]", &ubxEqual, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuAllToAllSoleMeshConcurrent"},
        {HcclCMDType::HCCL_CMD_ALLTOALLV, "sole[mesh]", &flat, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuAllToAllVSoleMesh"},
        // A2AV·sole[mesh.pairwise]：Pairwise 模式面 = 多级 MESH_1D + 对称 + 每框 8 卡 + 16 倍数
        // rank（真实 selector 由 IsSupportedTopology 前置守卫），改 pairwise 夹具真命中
        {HcclCMDType::HCCL_CMD_ALLTOALLV, "sole[mesh.pairwise]", &pairwise, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuAllToAllVSolePairwise"},
        {HcclCMDType::HCCL_CMD_ALLTOALLV, "sole[mesh.multi_channel]", &ubx, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuAllToAllVSoleMeshMultiJetty"},
        // A2AV·concur[mesh,mesh]：mesh==clos 且 rank <= 4（A2AV 版不读长度，无 len 子句），改
        // ubxEqual 真命中
        {HcclCMDType::HCCL_CMD_ALLTOALLV, "concur[mesh,mesh]", &ubxEqual, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuAllToAllVSoleMeshConcurrent"},
        // ---- 纯查表组合（CCU）----
        {HcclCMDType::HCCL_CMD_ALLGATHER, "sole[mesh]", &flat, CommEngine::COMM_ENGINE_CCU,
         "CcuSchedAllGatherSoleMesh"},
        {HcclCMDType::HCCL_CMD_ALLGATHER, "parallel[mesh,nhr.multi_channel]", &ubx, CommEngine::COMM_ENGINE_CCU,
         "CcuSchedAllGatherParallelMeshNHRMultiLink"},
        {HcclCMDType::HCCL_CMD_ALLGATHER, "concur[mesh,nhr.multi_channel]", &ubxEqual, CommEngine::COMM_ENGINE_CCU,
         "CcuSchedAllGatherConcurMeshNHRMultiLink"},
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "sole[nhr.multi_channel]", &ubx, CommEngine::COMM_ENGINE_CCU,
         "CcuSchedReduceScatterSoleNHRMultiLink"},
        // CCU·RS·sole[nhr]：模式化后仅两层 Mesh（单实例 EQ 1）与单层 CLOS 命中；
        // flat/UBX 真实 selector 走 SoleMesh/SoleNHRMultiLink（sole[mesh]/sole[nhr.multi_channel]
        // 键），sole[nhr] 键全灭——见本测试尾部的 ERROR_INVALID 追加断言
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "sole[nhr]", &multi, CommEngine::COMM_ENGINE_CCU,
         "CcuSchedReduceScatterSoleNHR"},
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "sole[nhr]", &clos, CommEngine::COMM_ENGINE_CCU,
         "CcuSchedReduceScatterSoleNHR"},
        // AICPU·RS·sole[nhr.multi_channel]：一层 CLOS 面（flat 在真实 selector 走 Mesh1D/Chunk）
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "sole[nhr.multi_channel]", &clos, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuReduceScatterSoleNHRMultiLink"},
        // RS·concur[mesh,nhr.multi_channel]·CCU：模式面需全互联 mesh==clos 且 rank <= 4
        // （同 AG concur 先例用 ubxEqual；ubx 形态真实 selector 走 SoleNHRMultiLink）
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "concur[mesh,nhr.multi_channel]", &ubxEqual, CommEngine::COMM_ENGINE_CCU,
         "CcuSchedReduceScatterConcurMeshNHRMultiLink"},
        {HcclCMDType::HCCL_CMD_ALLREDUCE, "sole[mesh]", &flat, CommEngine::COMM_ENGINE_CCU,
         "CcuSchedAllReduceSoleMesh"},
        {HcclCMDType::HCCL_CMD_ALLTOALL, "sole[mesh]", &flat, CommEngine::COMM_ENGINE_CCU, "CcuSchedAllToAllSoleMesh"},
        {HcclCMDType::HCCL_CMD_ALLTOALL, "sole[mesh.multi_channel]", &ubx, CommEngine::COMM_ENGINE_CCU,
         "CcuSchedAllToAllMesh1DMultiJetty"},
        // A2A·CCU·concur[mesh,mesh]：模式面需 mesh==clos 且 rank <= 4（PER_RANK 字节子句在
        // 定名 len=0 下跳过），改 ubxEqual 真命中（同上 AICPU 版）
        {HcclCMDType::HCCL_CMD_ALLTOALL, "concur[mesh,mesh]", &ubxEqual, CommEngine::COMM_ENGINE_CCU,
         "CcuSchedAllToAllSoleMeshConcurrent"},
        {HcclCMDType::HCCL_CMD_ALLTOALLV, "sole[mesh]", &flat, CommEngine::COMM_ENGINE_CCU,
         "CcuSchedAllToAllVSoleMesh"},
        {HcclCMDType::HCCL_CMD_ALLTOALLV, "sole[mesh.multi_channel]", &ubx, CommEngine::COMM_ENGINE_CCU,
         "CcuSchedAllToAllVSoleMeshMultiJetty"},
        // ---- 8 处一对多 ----
        // AICPU·RS·sole[mesh]：两层网小数据走裸名（大数据触发在下一用例）。twoLevelNet 与
        // topoLevelNums 解耦（IsTwoLevelNetLayer 不读实例总数），改单层 twoLevelNet 夹具真命中
        // Mesh1D 的 twoLevelNet=Yes 面（旧 twoLevel 夹具是 2 级拓扑，真实 selector 走多级路径）
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "sole[mesh]", &singleLevelTwoLevel, CommEngine::COMM_ENGINE_AICPU_TS,
         "InsReduceScatterMesh1D"},
        // CCU·RS·sole[mesh]：UT 无真实 comm，COND_PEER_ONLY 保守不命中 -> 裸名兜底
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "sole[mesh]", &flat, CommEngine::COMM_ENGINE_CCU,
         "CcuSchedReduceScatterSoleMesh"},
        // AICPU·AG·parallel[mesh,nhr] 拓扑分流（模式化后；UBX 变体注册已删除，
        // 其单层形态不再有候选命中）。两层 Mesh 需 level0 本实例卡数 > 1（NE 1）
        {HcclCMDType::HCCL_CMD_ALLGATHER, "parallel[mesh,nhr]", &twoLevel, CommEngine::COMM_ENGINE_AICPU_TS,
         "InsAllGatherParallelMesh1DNHR"},
        {HcclCMDType::HCCL_CMD_ALLGATHER, "parallel[mesh,nhr]", &pcie, CommEngine::COMM_ENGINE_AICPU_TS,
         "InsAllGatherParallelMesh1DNHRPcie"},
        // AICPU·RS·parallel[mesh,nhr]：模式化后两层 Mesh 需 level0 本实例卡数 > 1（GT 例外），
        // multi 夹具 inst[0]==1 在真实 selector 走 :460 单实例支选 NHR，改 twoLevel 真命中（同 AR 先例）
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "parallel[mesh,nhr]", &twoLevel, CommEngine::COMM_ENGINE_AICPU_TS,
         "InsReduceScatterParallelMesh1DNHR"},
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "parallel[mesh,nhr]", &ubx, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuReduceScatterParallelMeshNHRUBX"},
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "parallel[mesh,nhr]", &pcie, CommEngine::COMM_ENGINE_AICPU_TS,
         "InsReduceScatterParallelMesh1DNHRPcie"},
        // AICPU·AR·parallel[mesh,nhr]（无 UBX 变体）。模式化后两层 Mesh 需 level0 本实例卡数 > 1
        // （NE 1）且 ExcludeE 已排除对称+8卡+uboe——multi 夹具 inst[0]==1 在真实 selector 走
        // SoleNHR 单实例支，改用 twoLevel（inst[0]==2）真命中基模式 7 ×E1
        {HcclCMDType::HCCL_CMD_ALLREDUCE, "parallel[mesh,nhr]", &twoLevel, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuAllReduceParallelMeshNHR"},
        {HcclCMDType::HCCL_CMD_ALLREDUCE, "parallel[mesh,nhr]", &pcie, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuAllreduceParallelMeshNHRPcie"},
        // AICPU·A2A·sole[mesh]
        {HcclCMDType::HCCL_CMD_ALLTOALL, "sole[mesh]", &flat, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuAllToAllSoleMesh"},
        {HcclCMDType::HCCL_CMD_ALLTOALL, "sole[mesh]", &ubx, CommEngine::COMM_ENGINE_AICPU_TS,
         "AicpuAllToAllSoleMeshUBX"},
        // ---- DPU 行只在 engine==COMM_ENGINE_CPU 命中 ----
        {HcclCMDType::HCCL_CMD_ALLGATHER, "sequence[mesh,nhr]", &multi, CommEngine::COMM_ENGINE_CPU,
         "InsAllGatherMeshNhrDPU"},
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "sequence[mesh,mesh]", &multi, CommEngine::COMM_ENGINE_CPU,
         "InsReduceScatterSequenceMeshMeshDPU"},
    };

    ExternalAlgSpec spec;
    std::string errMsg;
    for (const Case& c : cases) {
        ASSERT_TRUE(ParseExternalAlg(c.extName, spec, errMsg)) << c.extName;
        OpParam op = MakeResolveParam(c.cmd, c.engine);
        std::string algName;
        EXPECT_EQ(ResolveExternalAlg(op, c.topo, spec, algName, errMsg), ResolveStatus::RESOLVED)
            << "ext=" << c.extName << " engine=" << static_cast<int>(c.engine) << " err=" << errMsg;
        EXPECT_EQ(algName, std::string(c.expected)) << "ext=" << c.extName;
    }

    // CCU·RS·sole[nhr] 的 flat/UBX 形态：CcuSchedReduceScatterSoleNHR 的模式面无单层 Mesh
    // 与 UBX 条目（真实 selector 分别走 SoleMesh / SoleNHRMultiLink，属 sole[mesh] /
    // sole[nhr.multi_channel] 键），本键无候选命中 → 整键全灭（golden 适配：旧机型 flag 无拓扑闸时曾误命中）
    ASSERT_TRUE(ParseExternalAlg("sole[nhr]", spec, errMsg));
    {
        OpParam opCcu = MakeResolveParam(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CommEngine::COMM_ENGINE_CCU);
        std::string algName;
        EXPECT_EQ(ResolveExternalAlg(opCcu, &flat, spec, algName, errMsg), ResolveStatus::ERROR_INVALID);
        EXPECT_NE(errMsg.find("sole[nhr]"), std::string::npos);
        EXPECT_EQ(ResolveExternalAlg(opCcu, &ubx, spec, algName, errMsg), ResolveStatus::ERROR_INVALID);
        EXPECT_NE(errMsg.find("sole[nhr]"), std::string::npos);
    }
}

TEST(ExtAlgResolver, VariantPairDisambiguation)
{
    ExternalAlgSpec spec;
    std::string errMsg;
    std::string algName;
    TopoInfoWithNetLayerDetails twoLevel = MakeTwoLevelNetLayerTopo();
    // 单层 twoLevelNet + TP2 夹具（twoLevelNet 与 topoLevelNums 解耦，见夹具注释）：模式化后
    // ZAxisDetour 拓扑面 = 单层 MESH_1D + rankSize 2 + twoLevelNet，大数据（rank==2 且 >= 4MB）
    // 触发 COND_Z_AXIS → ZAxisDetour；小数据 → 裸 Mesh1D 的 twoLevelNet=Yes 面。
    // 旧用例以 2 级 twoLevel 夹具走 ZAxis 是旧机型 flag 无拓扑闸的误命中（真实 selector 2 级走多级路径）
    TopoInfoWithNetLayerDetails twoRankTwoLevel = MakeSingleLevelTwoLevelNetTopo(2U);

    // 两层网 + 大数据 -> ZAxisDetour；小数据 -> 裸 Mesh1D
    ASSERT_TRUE(ParseExternalAlg("sole[mesh]", spec, errMsg));
    OpParam big = MakeResolveParam(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CommEngine::COMM_ENGINE_AICPU_TS);
    big.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    big.DataDes.count = 24U * 1024 * 1024 + 1U; // 96MB+4B >= 4MB，经 rank==2 支触发
    EXPECT_EQ(ResolveExternalAlg(big, &twoRankTwoLevel, spec, algName, errMsg), ResolveStatus::RESOLVED);
    EXPECT_EQ(algName, "InsReduceScatterMesh1DZAxisDetour");

    OpParam small = big;
    small.DataDes.count = 1024U;
    EXPECT_EQ(ResolveExternalAlg(small, &twoRankTwoLevel, spec, algName, errMsg), ResolveStatus::RESOLVED);
    EXPECT_EQ(algName, "InsReduceScatterMesh1D");

    // specialDT -> AicpuReduceNHR；否则裸 NHR
    ASSERT_TRUE(ParseExternalAlg("sole[nhr]", spec, errMsg));
    OpParam int64 = MakeResolveParam(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CommEngine::COMM_ENGINE_AICPU_TS);
    int64.DataDes.dataType = HCCL_DATA_TYPE_INT64;
    EXPECT_EQ(ResolveExternalAlg(int64, &twoLevel, spec, algName, errMsg), ResolveStatus::RESOLVED);
    EXPECT_EQ(algName, "InsReduceScatterAicpuReduceNHR");

    OpParam fp32 = MakeResolveParam(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CommEngine::COMM_ENGINE_AICPU_TS);
    EXPECT_EQ(ResolveExternalAlg(fp32, &twoLevel, spec, algName, errMsg), ResolveStatus::RESOLVED);
    EXPECT_EQ(algName, "InsReduceScatterNHR");

    // AR sole[nhr]
    OpParam arProd = MakeResolveParam(HcclCMDType::HCCL_CMD_ALLREDUCE, CommEngine::COMM_ENGINE_AICPU_TS);
    arProd.reduceType = HcclReduceOp::HCCL_REDUCE_PROD;
    EXPECT_EQ(ResolveExternalAlg(arProd, &twoLevel, spec, algName, errMsg), ResolveStatus::RESOLVED);
    EXPECT_EQ(algName, "AicpuAllReduceSoleNHRAicpuReduce");
    OpParam arSum = MakeResolveParam(HcclCMDType::HCCL_CMD_ALLREDUCE, CommEngine::COMM_ENGINE_AICPU_TS);
    EXPECT_EQ(ResolveExternalAlg(arSum, &twoLevel, spec, algName, errMsg), ResolveStatus::RESOLVED);
    EXPECT_EQ(algName, "AicpuAllReduceSoleNHR");

    // AG sequence[mesh,nhr] 引擎消歧：AICPU 命中 AicpuAllGatherSequenceMeshConcurNHR（两层 Mesh1D
    // 且 level0 本实例卡数 > 1），DPU 行仅 CPU 引擎命中（CPU -> InsAllGatherMeshNhrDPU 由 GoldenAllEngines 覆盖）
    ASSERT_TRUE(ParseExternalAlg("sequence[mesh,nhr]", spec, errMsg));
    OpParam aicpu = MakeResolveParam(HcclCMDType::HCCL_CMD_ALLGATHER, CommEngine::COMM_ENGINE_AICPU_TS);
    EXPECT_EQ(ResolveExternalAlg(aicpu, &twoLevel, spec, algName, errMsg), ResolveStatus::RESOLVED);
    EXPECT_EQ(algName, "AicpuAllGatherSequenceMeshConcurNHR");
}

TEST(ExtAlgResolver, ErrorCasesCarryThreeElements)
{
    ExternalAlgSpec spec;
    std::string errMsg;
    std::string algName;
    OpParam op = MakeResolveParam(HcclCMDType::HCCL_CMD_ALLGATHER, CommEngine::COMM_ENGINE_AICPU_TS);

    // 候选为空（devkit 无 3 层注册）
    TopoInfoWithNetLayerDetails flat = MakeFlatMesh1DTopo();
    ASSERT_TRUE(ParseExternalAlg("pipeline[mesh,nhr,nhr]", spec, errMsg));
    EXPECT_EQ(ResolveExternalAlg(op, &flat, spec, algName, errMsg), ResolveStatus::ERROR_INVALID);
    EXPECT_NE(errMsg.find("pipeline[mesh,nhr,nhr]"), std::string::npos); // 三要素: 外部名原文

    // 规则全灭（AG parallel[mesh,nhr] 在单层 flat 拓扑：两条 pattern 均不命中，UBX 行已删除注册）
    ASSERT_TRUE(ParseExternalAlg("parallel[mesh,nhr]", spec, errMsg));
    EXPECT_EQ(ResolveExternalAlg(op, &flat, spec, algName, errMsg), ResolveStatus::ERROR_INVALID);
    EXPECT_NE(errMsg.find("parallel[mesh,nhr]"), std::string::npos);

    // 规则全灭（RS sole[nhr.multi_channel]·CCU：唯一 CCU 候选 CcuSchedReduceScatterSoleNHRMultiLink
    // 的模式面均为 MESH_1D_CLOS UBX 形态，单层 flat 拓扑不命中，无裸名兜底 -> 整键全灭回退）
    OpParam opRsCcu = MakeResolveParam(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CommEngine::COMM_ENGINE_CCU);
    ASSERT_TRUE(ParseExternalAlg("sole[nhr.multi_channel]", spec, errMsg));
    EXPECT_EQ(ResolveExternalAlg(opRsCcu, &flat, spec, algName, errMsg), ResolveStatus::ERROR_INVALID);
    EXPECT_NE(errMsg.find("sole[nhr.multi_channel]"), std::string::npos);

    // 规则全灭（A2AV sole[mesh.multi_channel]·CCU：唯一 CCU 候选 CcuSchedAllToAllVSoleMeshMultiJetty 带
    // FLAG_UBX，单层 flat 拓扑不命中，无裸名兜底 -> 整键全灭回退）
    OpParam opA2avCcu = MakeResolveParam(HcclCMDType::HCCL_CMD_ALLTOALLV, CommEngine::COMM_ENGINE_CCU);
    ASSERT_TRUE(ParseExternalAlg("sole[mesh.multi_channel]", spec, errMsg));
    EXPECT_EQ(ResolveExternalAlg(opA2avCcu, &flat, spec, algName, errMsg), ResolveStatus::ERROR_INVALID);
    EXPECT_NE(errMsg.find("sole[mesh.multi_channel]"), std::string::npos);
}

TEST(ExtAlgResolver, CandidateTableIsDeterministic)
{
    // 聚合确定性：同数据不同输入序（重跑 / 逆序）构建候选表，逐位一致（不依赖注册顺序）
    const std::vector<AlgMetaRow> rows = AlgMetaRegistry::Instance().GetAll();
    const std::map<ExtAlgKey, std::vector<AlgCandidate>> table1 = BuildCandidateTable(rows);
    const std::map<ExtAlgKey, std::vector<AlgCandidate>> table2 = BuildCandidateTable(rows);
    std::vector<AlgMetaRow> reversedRows = rows;
    std::reverse(reversedRows.begin(), reversedRows.end());
    const std::map<ExtAlgKey, std::vector<AlgCandidate>> table3 = BuildCandidateTable(reversedRows);
    const std::map<ExtAlgKey, std::vector<AlgCandidate>>* otherTables[] = {&table2, &table3};
    for (const auto* other : otherTables) {
        ASSERT_EQ(table1.size(), other->size());
        for (const auto& entry : table1) {
            const auto it = other->find(entry.first);
            ASSERT_NE(it, other->end());
            ASSERT_EQ(entry.second.size(), it->second.size());
            for (size_t i = 0; i < entry.second.size(); ++i) {
                EXPECT_EQ(entry.second[i].registeredName, it->second[i].registeredName);
                EXPECT_EQ(static_cast<int>(entry.second[i].engine), static_cast<int>(it->second[i].engine));
            }
        }
    }
    // 覆盖数锚（限 5 个集合通信 CMD）：34 个 (op,外部名) 键（57 行归并；MultiLink
    // 外部名修正为 sole[nhr.multi_channel]、MultiJetty 修正为 sole[mesh.multi_channel]
    // 后各自从原裸名键分立，新增这两个键；AllToAllV sole[mesh.multi_channel] 键
    // 由 AICPU 版（upstream d1c00ab999）与 CCU 版（本迁移）双引擎共享）；
    // 47 个 (引擎,op,外部名) 组合（键内引擎去重求和）。 限定 COLL_CMDS 范围使锚不受测试内注册行（如
    // KFC cmd 的宏验证行）影响。
    const std::set<HcclCMDType> collCmds(std::begin(COLL_CMDS), std::end(COLL_CMDS));
    size_t keyCount = 0;
    size_t comboCount = 0;
    for (const auto& entry : table1) {
        if (collCmds.find(entry.first.first) == collCmds.end()) {
            continue; // 集合通信域外的注册（KFC/测试行）不计入锚
        }
        keyCount++;
        std::vector<AlgEngine> engines;
        for (const AlgCandidate& cand : entry.second) {
            if (std::find(engines.begin(), engines.end(), cand.engine) == engines.end()) {
                engines.push_back(cand.engine);
            }
        }
        comboCount += engines.size();
    }
    EXPECT_EQ(keyCount, 34U);
    EXPECT_EQ(comboCount, 47U);
}

TEST(ExtAlgResolver, SpecificityOrdering)
{
    // 特异性排序：特化候选(带 cond)在前，裸名(无 cond)在后；同特异性内字典序
    const std::vector<AlgCandidate>& cands = GetExternalCandidates(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "sole[mesh]");
    // 本目标全量：4 候选 = ZAxisDetour(COND) / CcuSchedReduceScatterSoleMesh(无cond) /
    //              CcuSchedReduceScatterSoleMeshPeerOnly(COND) / InsReduceScatterMesh1D(无cond)
    ASSERT_EQ(cands.size(), 4U);
    int lastSpecificity = 1;
    for (const AlgCandidate& cand : cands) {
        const int specificity = (cand.variantCondId != COND_NONE ? 1 : 0); // 与 BuildCandidateTable 排序键镜像
        EXPECT_LE(specificity, lastSpecificity);                           // 降序
        lastSpecificity = specificity;
    }
    EXPECT_EQ(cands[0].registeredName, "CcuSchedReduceScatterSoleMeshPeerOnly"); // 特化字典序在前
    EXPECT_EQ(cands[1].registeredName, "InsReduceScatterMesh1DZAxisDetour");
    EXPECT_EQ(cands[2].registeredName, "CcuSchedReduceScatterSoleMesh"); // 裸名字典序
    EXPECT_EQ(cands[3].registeredName, "InsReduceScatterMesh1D");
}

// 显式 priority 注册可用性：REGISTER_ALG_META 第 7 参落库（KFC cmd 不触碰满射断言的 5 个
// COLL_CMDS）。externalName 须为规范形——ExternalNameLiteralsAreCanonical 往返自检覆盖所有注册行，
// 含测试夹具。模式面取全 Any 单条：本夹具只验证 priority 落库机制，不验证拓扑语义（定名与
// patternCount/patterns 语义由 TestPatternMacro 覆盖）
namespace {
const TopoPattern PRIO_ANY_PATTERNS[] = {{}};
} // namespace
REGISTER_ALG_META(
    HcclCMDType::HCCL_CMD_KFC_SERVER, TestPrioMacro, AlgEngine::AICPU, "sole[mesh]", COND_NONE, PRIO_ANY_PATTERNS, 7);

// 模式宏注册行：同样走 KFC cmd 豁免（CandidateTableIsDeterministic 的 34/44 锚限定 COLL_CMDS）。
// 独占 (KFC, sole[nhr]) 键——不与 TestPrioMacro 的 (KFC, sole[mesh]) 键混叠，保证下述
// "不匹配拓扑 → ERROR_INVALID" 不被键内其他候选兜底。两条 pattern 各钉一种单层形态：
// flat MESH_1D（MakeFlatMesh1DTopo 命中）与纯 CLOS（MakeSingleLevelClosTopo 命中），
// UBX 的 MESH_1D_CLOS 两不沾
namespace {
const TopoPattern TEST_PATTERNS[] = {
    {{1U, 1U}, Level0Shape::MESH_1D}, // level={1,1}（单层），level0=MESH_1D，其余默认任意
    {{1U, 1U}, Level0Shape::CLOS},    // level={1,1}（单层），level0=CLOS
};
} // namespace
REGISTER_ALG_META(
    HcclCMDType::HCCL_CMD_KFC_SERVER, TestPatternMacro, AlgEngine::AICPU, "sole[nhr]", COND_NONE, TEST_PATTERNS, 0);

// 三键排序语义（specificity 降序 > priority 降序 > name 字典序）——纯函数构建器直接验证，
// 测试内构造行（KFC cmd）+ 逆序传入（同时验证顺序无关性）。
// 产品行 priority 全 0 时三键退化为双键，此用例钉住非 0 语义
TEST(ExtAlgResolver, PriorityOrdering)
{
    // 宏注册行可查且 priority 落库
    const AlgMeta* macroMeta = AlgMetaRegistry::Instance().Get(HcclCMDType::HCCL_CMD_KFC_SERVER, "TestPrioMacro");
    ASSERT_NE(macroMeta, nullptr);
    EXPECT_EQ(macroMeta->priority, 7);

    std::vector<AlgMetaRow> rows = {
        {HcclCMDType::HCCL_CMD_KFC_SERVER,
         "TestPrioLow",
         {AlgEngine::AICPU, "test[prio]", COND_NONE, PRIO_ANY_PATTERNS, PatternCountOf(PRIO_ANY_PATTERNS), 1}},
        {HcclCMDType::HCCL_CMD_KFC_SERVER,
         "TestPrioHigh",
         {AlgEngine::AICPU, "test[prio]", COND_NONE, PRIO_ANY_PATTERNS, PatternCountOf(PRIO_ANY_PATTERNS), 5}},
        {HcclCMDType::HCCL_CMD_KFC_SERVER,
         "TestPrioSameA",
         {AlgEngine::AICPU, "test[prio]", COND_NONE, PRIO_ANY_PATTERNS, PatternCountOf(PRIO_ANY_PATTERNS), 5}},
        // priority=99 的无 cond 行：特异性第一级不可被 priority 越级，仍排带 cond 之后
        {HcclCMDType::HCCL_CMD_KFC_SERVER,
         "TestBareName",
         {AlgEngine::AICPU, "test[prio]", COND_NONE, PRIO_ANY_PATTERNS, PatternCountOf(PRIO_ANY_PATTERNS), 99}},
        // 带 cond：特异性 1 分最特化，排最前
        {HcclCMDType::HCCL_CMD_KFC_SERVER,
         "TestCondSlot",
         {AlgEngine::AICPU, "test[prio]", COND_SPECIAL_DT, PRIO_ANY_PATTERNS, PatternCountOf(PRIO_ANY_PATTERNS), 0}},
    };
    std::reverse(rows.begin(), rows.end()); // 逆序传入：排序与输入序无关
    const std::map<ExtAlgKey, std::vector<AlgCandidate>> table = BuildCandidateTable(rows);
    const auto it = table.find(std::make_pair(HcclCMDType::HCCL_CMD_KFC_SERVER, std::string("test[prio]")));
    ASSERT_NE(it, table.end());
    const std::vector<AlgCandidate>& cands = it->second;
    ASSERT_EQ(cands.size(), 5U);
    // 键1 特异性：带 cond(1) > 无 cond(0)
    EXPECT_EQ(cands[0].registeredName, "TestCondSlot");
    // 键2 priority：同特异性，99 > 5（大者优先）
    EXPECT_EQ(cands[1].registeredName, "TestBareName");
    EXPECT_EQ(cands[1].priority, 99);
    // 键3 字典序：同特异性同 priority(5)，"TestPrioHigh" < "TestPrioSameA"
    EXPECT_EQ(cands[2].registeredName, "TestPrioHigh");
    EXPECT_EQ(cands[3].registeredName, "TestPrioSameA");
    EXPECT_EQ(cands[4].registeredName, "TestPrioLow");
}

// 注册行最小闭环：①宏落库（PatternCountOf 数组计数 / patterns 指针即注册数组）
// ②模式匹配路径（匹配拓扑两形态命中定名；不匹配拓扑无兜底 → 全灭；topoInfo=nullptr 防御拒绝）
TEST(ExtAlgResolver, PatternMacroFallsThroughTopoFunnel)
{
    // ① 宏落库
    const AlgMeta* meta = AlgMetaRegistry::Instance().Get(HcclCMDType::HCCL_CMD_KFC_SERVER, "TestPatternMacro");
    ASSERT_NE(meta, nullptr);
    EXPECT_EQ(meta->patternCount, 2U);
    EXPECT_EQ(meta->patterns, TEST_PATTERNS); // 指针透传，指向静态数组本体

    // ② 模式匹配路径：两条 pattern 各覆盖一种单层形态
    ExternalAlgSpec spec;
    std::string errMsg;
    ASSERT_TRUE(ParseExternalAlg("sole[nhr]", spec, errMsg));
    OpParam op = MakeResolveParam(HcclCMDType::HCCL_CMD_KFC_SERVER, CommEngine::COMM_ENGINE_AICPU_TS);
    TopoInfoWithNetLayerDetails flat = MakeFlatMesh1DTopo();
    TopoInfoWithNetLayerDetails clos = MakeSingleLevelClosTopo();
    TopoInfoWithNetLayerDetails ubx = MakeUbxTopo();
    std::string algName;
    EXPECT_EQ(ResolveExternalAlg(op, &flat, spec, algName, errMsg), ResolveStatus::RESOLVED);
    EXPECT_EQ(algName, "TestPatternMacro");
    EXPECT_EQ(ResolveExternalAlg(op, &clos, spec, algName, errMsg), ResolveStatus::RESOLVED);
    EXPECT_EQ(algName, "TestPatternMacro");
    // 不匹配拓扑（MESH_1D_CLOS 两 pattern 均不命中）且键内无其他候选 → 全灭
    EXPECT_EQ(ResolveExternalAlg(op, &ubx, spec, algName, errMsg), ResolveStatus::ERROR_INVALID);
    EXPECT_NE(errMsg.find("sole[nhr]"), std::string::npos);
    // topoInfo=nullptr：模式分支 null 防御直接拒绝
    EXPECT_EQ(ResolveExternalAlg(op, nullptr, spec, algName, errMsg), ResolveStatus::ERROR_INVALID);
}

// extern 表转发初始化器的运行时绑定断言
TEST(ExtAlgResolver, RuleTableBinding)
{
    EXPECT_EQ(VARIANT_COND_TABLE[COND_Z_AXIS], &CondZAxis);
    EXPECT_EQ(VARIANT_COND_TABLE[COND_PEER_ONLY], &CondPeerOnly);
    EXPECT_EQ(VARIANT_COND_TABLE[COND_SPECIAL_DT], &CondSpecialDt);
    EXPECT_STREQ(VariantCondName(COND_Z_AXIS), "COND_Z_AXIS");
    EXPECT_STREQ(VariantCondName(-1), "n/a");
}

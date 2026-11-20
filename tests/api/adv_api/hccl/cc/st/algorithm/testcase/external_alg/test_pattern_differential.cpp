/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// 差分金标：oracle = 真实 selector 双 dtype（FP32+SUM / INT64+PROD）应选名并集；actual = 模式命中
// 集合（排除手写变体）。双向差集断言——onlyInSelector 为漏登，onlyInPattern 为陈旧。
// 新增算法须同步四处：模式数组（含 static_assert 计数）/ PatternArrayCountAnchors 计数锚 /
// ChainOrderInvariants 链序档 / 差分向量 + 文件尾覆盖清单登记。
// 注意：本套件预期产生 ERROR 级日志（degenerate 向量与未迁移向量的正常路过路径），非故障。
#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "alg_meta_registry.h"
#include "alg_param.h"
#include "all_gather_alg_patterns.h"
#include "all_gather_auto_selector.h"
#include "all_reduce_alg_patterns.h"
#include "all_reduce_auto_selector.h"
#include "all_to_all_v_alg_patterns.h"
#include "alltoall_auto_selector.h"
#include "alltoallv_auto_selector.h"
#include "auto_selector_base.h"
#include "gtest/gtest.h"
#include "half_alltoallv_auto_selector.h"
#include "reduce_scatter_alg_patterns.h"
#include "reduce_scatter_auto_selector.h"
#include "topo_pattern.h"

using namespace mc2_ops_hccl;

namespace {
// 手写三变体排除名单（spec §7.2）：COND_Z_AXIS / COND_PEER_ONLY / SingleChannel 的拓扑面为
// 人工登记，devkit selector 永不自动选中——无差分 oracle，模式侧跳过
const std::set<std::string> kHandwrittenVariants = {
    "InsReduceScatterMesh1DZAxisDetour",
    "CcuSchedReduceScatterSoleMeshPeerOnly",
    "AicpuAllToAllSoleMeshSingleChannel",
};

// 引擎→selector 函数分派表（六个 selector 头的 Select* 方法 grep 实证）。派生类的引擎
// override 均声明为 private，经 AutoSelectorBase 公有虚接口调用（访问控制按静态类型走
// 基类；selector 直实例化先例 half_alltoallv_selector_testcase.cc:65）：
//   ALLGATHER / REDUCE_SCATTER：SelectAicpuAlgo / SelectCcuScheduleAlgo / SelectDPUAlgo
//   ALLREDUCE / ALLTOALL / ALLTOALLV：SelectAicpuAlgo / SelectCcuScheduleAlgo
//   HALF_ALLTOALLV：仅 SelectCcuScheduleAlgo
// CCU 仅对账 SelectCcuScheduleAlgo（产品立场 CCU_SCHED-only，D19 后 CCU_MS 查询/定名均拒，
// 其 selector 行为不再作为 pattern 镜像义务）。RS CcuMs 死区裁决（selector 已修）：
// RS CcuMs 三个选名（CcuReduceScatterMesh1D/Mesh2Die/ConcurrentMeshNHRMs）无 sidecar 注册
// （注册随 ae71ab4ee7 ccu 切换逻辑清理删除，selector 未同步致 MATCH 断头）——selector 现已
// 对齐仓内"未注册"惯例（设名 + WARNING + NOT_MATCH），pick 不再进集合；"已注册名"过滤
// （SelectorPicks 内）保留为通用防御，将来 executor 迁移补注册后 selector 翻转 MATCH 即自动进 oracle。
// 分派表更新义务锚：新增 cmd/引擎注册时必须同步 MakeSelector/EngineFunctions，
// TEST DispatchTableCoversRegisteredDimensions 会在遗漏时报错。
using EngineFn = SelectorStatus (AutoSelectorBase::*)(
    const TopoInfoWithNetLayerDetails*, const OpParam&, const std::map<HcclCMDType, std::vector<HcclAlgoType>>&,
    std::string&) const;

std::vector<EngineFn> EngineFunctions(AlgEngine engine)
{
    switch (engine) {
        case AlgEngine::AICPU:
            return {&AutoSelectorBase::SelectAicpuAlgo};
        case AlgEngine::CCU:
            return {&AutoSelectorBase::SelectCcuScheduleAlgo};
        case AlgEngine::DPU:
            return {&AutoSelectorBase::SelectDPUAlgo};
        default:
            return {};
    }
}

const std::map<HcclCMDType, std::vector<HcclAlgoType>> kEmptyAlgConfigMap = {};

std::unique_ptr<AutoSelectorBase> MakeSelector(HcclCMDType cmd)
{
    switch (cmd) {
        case HcclCMDType::HCCL_CMD_ALLGATHER:
            return std::make_unique<AllGatherAutoSelector>();
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
            return std::make_unique<ReduceScatterAutoSelector>();
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
            return std::make_unique<AllReduceAutoSelector>();
        case HcclCMDType::HCCL_CMD_ALLTOALL:
            return std::make_unique<AlltoAllAutoSelector>();
        case HcclCMDType::HCCL_CMD_ALLTOALLV:
            return std::make_unique<AlltoAllVAutoSelector>();
        case HcclCMDType::HCCL_CMD_HALF_ALLTOALLV:
            return std::make_unique<HalfAlltoAllVAutoSelector>();
        default:
            return nullptr;
    }
}

// 跑分派表函数，MATCH 的名字进集合（再过滤为已注册名，死区裁决见上）。count×dtypeSize 即
// selector 侧 dataSize（调用方保证对 dataLen 整除，防截断破坏双 dtype 等价）
std::set<std::string> SelectorPicks(
    HcclCMDType cmd, AlgEngine engine, const TopoInfoWithNetLayerDetails& topo, u64 count, HcclDataType dtype,
    bool specialDtype)
{
    std::set<std::string> picks;
    const std::unique_ptr<AutoSelectorBase> selector = MakeSelector(cmd);
    if (selector == nullptr) {
        return picks;
    }
    OpParam param{}; // hcclComm / inputPtr / outputPtr 全空：IsInputOutputOverlap 恒 false（非原地）
    param.opType = cmd;
    param.DataDes.count = count;
    param.DataDes.dataType = dtype;
    param.reduceType = specialDtype ? HcclReduceOp::HCCL_REDUCE_PROD : HcclReduceOp::HCCL_REDUCE_SUM;
    for (const EngineFn fn : EngineFunctions(engine)) {
        std::string algName;
        if ((*selector.*fn)(&topo, param, kEmptyAlgConfigMap, algName) == SelectorStatus::MATCH) {
            // expected 集合裁剪为 AlgMetaRegistry 有 meta 的名字：未注册 pick（如 RS CcuMs 的
            // CcuReduceScatterMesh1D 系）是 executor 缺位而非模式漂移，模式侧无行可比对
            if (AlgMetaRegistry::Instance().Get(cmd, algName) != nullptr) {
                picks.insert(algName);
            }
        }
    }
    return picks;
}

// 注册表全扫：过滤 cmd + engine + 排除名单后 MatchAnyPattern（单值窗 (x,x)，x>0 时长度子句生效）。
// D17 差分挂校验：命中 pattern 的传播法幸存窗（NarrowByClause 逐子句收窄）须含本向量数据点——
// 点版 MatchLenClause 与区间传播在实网 pattern × 全矩阵向量上逐位一致；命中即窗非空且 x 落窗内。
// 注：不能反向对每条 pattern 断言 ValidatePattern==true——selector 链存在对特定 r 自然掏空的
// 忠实转录（AR TwoShot r=1 档 (8MB,4MB]、RS Chunk 相交支 r≥19 档（r³·256KB>1536MB 分界；
// r≥32 是 AR ChunkTwoShot 的 4GB 分界），见各数组注释），单点 topo
// 断真会误伤；"全 r 皆空"的病态合取由 RegisteredLenConjunctionsSatisfiableSomewhere 守门
std::set<std::string> PatternMatches(
    HcclCMDType cmd, AlgEngine engine, const TopoInfoWithNetLayerDetails& topo, u64 dataLen, u32 dtypeSize)
{
    std::set<std::string> hits;
    for (const AlgMetaRow& row : AlgMetaRegistry::Instance().GetAll()) {
        if (row.cmd != cmd || row.meta.engine != engine || kHandwrittenVariants.count(row.registeredName) != 0) {
            continue;
        }
        bool anyHit = false;
        for (u32 i = 0; i < row.meta.patternCount; ++i) {
            const TopoPattern& pat = row.meta.patterns[i];
            if (!MatchOne(pat, topo, dataLen, dataLen, dtypeSize)) {
                continue;
            }
            anyHit = true;
            uint64_t lo = 0;
            uint64_t hi = 0;
            EXPECT_TRUE(ValidatePattern(pat, topo, lo, hi, dtypeSize))
                << "registeredName=" << row.registeredName << " pattern[" << i << "] 命中但 len 子句传播解集为空";
            EXPECT_LE(lo, dataLen) << "registeredName=" << row.registeredName << " pattern[" << i
                                   << "] 幸存窗下端越过命中点";
            EXPECT_GE(hi, dataLen) << "registeredName=" << row.registeredName << " pattern[" << i
                                   << "] 幸存窗上端未达命中点";
        }
        if (anyHit) {
            hits.insert(row.registeredName);
        }
    }
    return hits;
}

std::string JoinNames(const std::set<std::string>& names)
{
    std::string joined = "{";
    for (const std::string& name : names) {
        joined += (joined.size() > 1U ? ", " : "") + name;
    }
    return joined + "}";
}

// 双向差集断言（打印合一）：onlyInSelector=漏登（selector 新分支无模式）、
// onlyInPattern=陈旧（已删分支的模式残留）；ExpectDifferential/ExpectA2ADifferential
// 共用，消息格式同源
void ExpectSetsEqual(const std::set<std::string>& expected, const std::set<std::string>& actual, const char* what)
{
    std::set<std::string> onlySelector;
    std::set<std::string> onlyPattern;
    std::set_difference(
        expected.begin(), expected.end(), actual.begin(), actual.end(),
        std::inserter(onlySelector, onlySelector.begin()));
    std::set_difference(
        actual.begin(), actual.end(), expected.begin(), expected.end(),
        std::inserter(onlyPattern, onlyPattern.begin()));
    EXPECT_TRUE(expected == actual) << what << "\n  onlyInSelector(漏登): " << JoinNames(onlySelector)
                                    << "\n  onlyInPattern(陈旧): " << JoinNames(onlyPattern);
}

// 差分断言：expected = SelectorPicks(正常) ∪ SelectorPicks(特殊)；actual = 模式命中集合。
// dataLen==0 时 selector 侧无法等价（selector 总有 dataSize=count×dtypeSize），本 harness
// 向量一律单值窗 (x,x) 且 x>0（哨兵 (0,0) 跳长度子句的语义由 test_topo_pattern.cpp 单测承载）。
// 模式侧 dtypeSize 取正常 dtype：ELEMENT 度量仅 A2A 族使用，AllToAll 向量走
// ExpectA2ADifferential（dtypeSize 参数化 + 元素计数 oracle）。
void ExpectDifferential(
    HcclCMDType cmd, AlgEngine engine, const TopoInfoWithNetLayerDetails& topo, u64 dataLen, const char* what)
{
    ASSERT_NE(dataLen, 0U) << what;
    ASSERT_EQ(dataLen % DATATYPE_SIZE_TABLE[HCCL_DATA_TYPE_INT64], 0U) << what;
    std::set<std::string> expected = SelectorPicks(
        cmd, engine, topo, dataLen / DATATYPE_SIZE_TABLE[HCCL_DATA_TYPE_FP32], HCCL_DATA_TYPE_FP32, false);
    const std::set<std::string> special = SelectorPicks(
        cmd, engine, topo, dataLen / DATATYPE_SIZE_TABLE[HCCL_DATA_TYPE_INT64], HCCL_DATA_TYPE_INT64, true);
    expected.insert(special.begin(), special.end());
    const std::set<std::string> actual =
        PatternMatches(cmd, engine, topo, dataLen, DATATYPE_SIZE_TABLE[HCCL_DATA_TYPE_FP32]);
    ExpectSetsEqual(expected, actual, what);
}

// ---- 拓扑夹具（字段形态照 external_alg_testcase.cc 抄，勿跨 TU 复用）----

// 两层 Mesh 底座（A/B/E/F）：level0 每实例 level0InstSize 卡，level1 单实例。
// instSizeListOfLayer 必填：两层 CCU AG 入口在分支前先算 frameNum（all_gather_auto_selector.cc:161），
// 空向量在 CalcFrameNum 内是越界读（前置契约：真实链路 HcclCalcTopoInfo 总是填充）
TopoInfoWithNetLayerDetails MakeTwoLevelMeshTopo(u32 rankSize, bool level1Nhr, u32 level0InstSize)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRank = 0U;
    topo.userRankSize = rankSize;
    topo.serverNum = 1U;
    topo.topoLevelNums = 2U;
    topo.level0Topo = Level0Shape::MESH_1D;
    topo.level0PcieMix = false;
    topo.Level0Nhr = false;
    topo.Level1Nhr = level1Nhr;
    topo.netLayerDetails.netLayerNum = 2U;
    topo.netLayerDetails.netLayers = {0U, 1U};
    topo.netLayerDetails.localNetInsSizeOfLayer = {level0InstSize, 1U};
    topo.netLayerDetails.instSizeListOfLayer = {{level0InstSize}};
    return topo;
}

// 一层 UBX 底座（C/D）：mesh==clos（§6.1 C）；degenerate=true 时不填 topoInstDetailsOfLayer，
// Check* 返回失败（§6.1 D 畸形形状）
TopoInfoWithNetLayerDetails MakeUbxTopo(u32 rankSize, bool degenerate = false)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRank = 0U;
    topo.userRankSize = rankSize;
    topo.serverNum = 1U;
    topo.topoLevelNums = 1U;
    topo.level0Topo = Level0Shape::MESH_1D_CLOS;
    topo.level0PcieMix = false;
    topo.netLayerDetails.netLayerNum = 1U;
    topo.netLayerDetails.netLayers = {0U};
    topo.netLayerDetails.localNetInsSizeOfLayer = {rankSize}; // MESH_1D_CLOS 不变量：== CLOS 域卡数
    if (!degenerate) {
        topo.topoInstDetailsOfLayer.resize(1U);
        topo.topoInstDetailsOfLayer[0].rankNumForTopoType[COMM_TOPO_1DMESH] = {rankSize}; // mesh==clos
        topo.topoInstDetailsOfLayer[0].rankNumForTopoType[COMM_TOPO_CLOS] = {rankSize};
    }
    return topo;
}

// 一层 Mesh1D 底座（引擎×len 组合）：meshType 保持缺省 NOT_MESH——AG 的 CcuSched 拒绝
// （要求 SINGLE_DIE）而 CcuMs 不查，正好钉 CCU 并集机制
TopoInfoWithNetLayerDetails MakeFlatMesh1DTopo(u32 rankSize = 8U)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRank = 0U;
    topo.userRankSize = rankSize;
    topo.serverNum = 1U;
    topo.topoLevelNums = 1U;
    topo.level0Topo = Level0Shape::MESH_1D;
    topo.netLayerDetails.netLayerNum = 1U;
    topo.netLayerDetails.netLayers = {0U};
    topo.netLayerDetails.localNetInsSizeOfLayer = {1U};
    topo.topoInstDetailsOfLayer.resize(1U);
    topo.topoInstDetailsOfLayer[0].rankNumForTopoType[COMM_TOPO_1DMESH] = {rankSize};
    return topo;
}

// 单层 MESH_1D + 两层网（twoLevelNet 与 topoLevelNums 解耦形态）：IsTwoLevelNetLayer 读
// localNetInsSizeOfLayer[0]（本实例卡数）不读 netInstNumOfLayer（实例数），L0 单实例覆盖全卡
// （topoLevelNums=1）时两层网仍可为真——AR/RS selector 的 twoLevel 分支恰定义在单层路径上
// （all_reduce/reduce_scatter 的 SelectMeshAlgoAicpu 仅单层入口调用）。字段形态照
// external_alg_testcase.cc 的 MakeSingleLevelTwoLevelNetTopo 抄（勿跨 TU 复用）
TopoInfoWithNetLayerDetails MakeSingleLevelTwoLevelNetTopo(u32 rankSize)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRank = 0U;
    topo.userRankSize = rankSize;
    topo.topoLevelNums = 1U;
    topo.level0Topo = Level0Shape::MESH_1D;
    topo.netLayerDetails.netLayerNum = 2U;
    topo.netLayerDetails.netLayers = {0U, 1U};
    topo.netLayerDetails.localNetInsSizeOfLayer = {2U, 1U}; // level0 本实例卡数 > 1
    topo.topoInstDetailsOfLayer.resize(2U);
    topo.topoInstDetailsOfLayer[1].rankNumForTopoType[COMM_TOPO_CLOS] = {rankSize}; // level1 有 CLOS
    return topo;
}

// 多层 CLOS 底座（levels=2/3/4）：[0]=2（NE 1 条目域），对称/uboe/每模块卡数由调用方按
// ×E/×F 路径覆写。instSizeListOfLayer 必填：两层 CCU AG 在分支前先算 frameNum
// （all_gather_auto_selector.cc:161），空向量在 CalcFrameNum 内是越界读
TopoInfoWithNetLayerDetails MakeMultiLevelClosTopo(u32 rankSize, u32 levels)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRank = 0U;
    topo.userRankSize = rankSize;
    topo.serverNum = 1U;
    topo.topoLevelNums = levels;
    topo.level0Topo = Level0Shape::CLOS;
    topo.Level0Nhr = false;
    topo.Level1Nhr = false;
    topo.netLayerDetails.netLayerNum = levels;
    for (u32 layer = 0; layer < levels; ++layer) {
        topo.netLayerDetails.netLayers.push_back(layer);
        topo.netLayerDetails.localNetInsSizeOfLayer.push_back(layer == 0U ? 2U : 1U);
    }
    topo.netLayerDetails.instSizeListOfLayer = {{2U}};
    return topo;
}

// 两层 CLOS + PCIe 混插：钉 b 修复（AR SoleNHR 两层CLOS selector 不读 pcieMix，pattern 曾多余
// pcieMix=No 致该组合漏报）——修复前此夹具上 pattern 不命中、selector 选，差分即红
TopoInfoWithNetLayerDetails MakeTwoLevelClosPcieMixTopo(u32 rankSize)
{
    TopoInfoWithNetLayerDetails topo = MakeMultiLevelClosTopo(rankSize, 2U);
    topo.level0PcieMix = true;
    return topo;
}

// UBX mesh == clos 底座（AllToAll 元素边界）：rank<=4 且 mesh 覆盖全卡（meshEqClos 命中），
// 字段形态照 external_alg_testcase.cc 的 MakeUbxEqualTopo 抄（勿跨 TU 复用）
TopoInfoWithNetLayerDetails MakeUbxEqualTopo(u32 rankSize = 4U)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRank = 0U;
    topo.userRankSize = rankSize;
    topo.serverNum = 1U;
    topo.topoLevelNums = 1U;
    topo.level0Topo = Level0Shape::MESH_1D_CLOS;
    topo.level0PcieMix = false;
    topo.netLayerDetails.netLayerNum = 1U;
    topo.netLayerDetails.netLayers = {0U};
    topo.netLayerDetails.localNetInsSizeOfLayer = {rankSize};
    topo.topoInstDetailsOfLayer.resize(1U);
    topo.topoInstDetailsOfLayer[0].rankNumForTopoType[COMM_TOPO_1DMESH] = {rankSize};
    topo.topoInstDetailsOfLayer[0].rankNumForTopoType[COMM_TOPO_CLOS] = {rankSize};
    return topo;
}

// 定制 UBX 底座：mesh/clos 的 rankNumForTopoType 显式可配，覆盖 selector 赋值行缺口——
// meshNum<closNum 且整除（如 2/8）钉 closMultiple=Yes 的矩形面，不整除（如 3/8）钉双 No 面，
// pcieMix 可配钉 PCIe 分流；localNetInsSizeOfLayer[0]=rankSize 使 meshNum<rankSize 时
// IsLayerAllConnetedWithTopo 为假（未全互联）
TopoInfoWithNetLayerDetails MakeUbxCountTopo(u32 rankSize, u32 meshNum, u32 closNum, bool pcieMix)
{
    TopoInfoWithNetLayerDetails topo = MakeUbxEqualTopo(rankSize);
    topo.level0PcieMix = pcieMix;
    topo.topoInstDetailsOfLayer[0].rankNumForTopoType[COMM_TOPO_1DMESH] = {meshNum};
    topo.topoInstDetailsOfLayer[0].rankNumForTopoType[COMM_TOPO_CLOS] = {closNum};
    return topo;
}

// 一层 CLOS 底座：CLOS 路径（各 selector 的单层 CLOS 分支）不调 Check*，无需
// topoInstDetailsOfLayer；rankSize 只入 rank 子句
TopoInfoWithNetLayerDetails MakeOneLevelClosTopo(u32 rankSize = 8U)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRank = 0U;
    topo.userRankSize = rankSize;
    topo.serverNum = 1U;
    topo.topoLevelNums = 1U;
    topo.level0Topo = Level0Shape::CLOS;
    topo.netLayerDetails.netLayerNum = 1U;
    topo.netLayerDetails.netLayers = {0U};
    topo.netLayerDetails.localNetInsSizeOfLayer = {rankSize};
    return topo;
}

// 多层 MESH_1D 底座（levels=2/3/4）：MakeMultiLevelClosTopo 的同构姊妹，唯 level0Topo
// 改 MESH_1D——[0]=2（实例数读点 NE/GT 侧面）、level0Symmetric 缺省 false（对称性由调用方覆写）
TopoInfoWithNetLayerDetails MakeMultiLevelMeshTopo(u32 rankSize, u32 levels)
{
    TopoInfoWithNetLayerDetails topo = MakeMultiLevelClosTopo(rankSize, levels);
    topo.level0Topo = Level0Shape::MESH_1D;
    return topo;
}

// AllToAll 差分（ELEMENT/字节双量纲，spec D11）：AICPU 侧 sendCounts[0] 与 512 按元素比
// （alltoall_auto_selector.cc:174-175）→ 模式 ELEMENT 子句；CCU 侧 dataSize = sendCounts[0]
// ×dtypeSize 与 512 按字节比（:80/:86）→ 模式 PER_RANK 子句。oracle 单 dtype 参数化
// （A2A selector 不读 DataDes.dataType/reduceType，无特殊类型并集需求）；sendCounts 数组
// 构造：AllToAll 各 rank 计数相同，selector 只读 [0]，单元素数组即可。union 布局注意：
// all2AllVDataDes 首两成员（sendType/recvType）与 all2AllDataDes 同型同偏移，写前者即
// 同步后者（CalcDataSize 读 all2AllDataDes.sendType 侧）
std::set<std::string> SelectorPicksA2A(
    HcclCMDType cmd, AlgEngine engine, const TopoInfoWithNetLayerDetails& topo, u64 elemCount, HcclDataType sendType)
{
    std::set<std::string> picks;
    const std::unique_ptr<AutoSelectorBase> selector = MakeSelector(cmd);
    if (selector == nullptr) {
        return picks;
    }
    u64 sendCounts[1] = {elemCount};
    OpParam param{}; // inputPtr/outputPtr 全空：非原地
    param.opType = cmd;
    param.all2AllVDataDes.sendType = sendType;
    param.all2AllVDataDes.recvType = sendType;
    param.all2AllVDataDes.sendCounts = sendCounts;
    for (const EngineFn fn : EngineFunctions(engine)) {
        std::string algName;
        if ((*selector.*fn)(&topo, param, kEmptyAlgConfigMap, algName) == SelectorStatus::MATCH) {
            if (AlgMetaRegistry::Instance().Get(cmd, algName) != nullptr) {
                picks.insert(algName);
            }
        }
    }
    return picks;
}

// A2A 版差分断言：expected = SelectorPicksA2A（单 dtype）；actual = PatternMatches，len =
// elemCount×dtypeSize（字节）、dtypeSize = 该 dtype 字节宽——ELEMENT 子句算出元素数、
// PER_RANK 子句算出字节数，两侧同源等价
void ExpectA2ADifferential(
    HcclCMDType cmd, AlgEngine engine, const TopoInfoWithNetLayerDetails& topo, u64 elemCount, HcclDataType sendType,
    const char* what)
{
    const u32 dtypeSize = DATATYPE_SIZE_TABLE[sendType];
    ASSERT_NE(dtypeSize, 0U) << what;
    const std::set<std::string> expected = SelectorPicksA2A(cmd, engine, topo, elemCount, sendType);
    const std::set<std::string> actual =
        PatternMatches(cmd, engine, topo, static_cast<u64>(elemCount) * dtypeSize, dtypeSize);
    ExpectSetsEqual(expected, actual, what);
}

// rankSize 边界扫档位：2=最小通信域/2P 绕行档；4/5=Concurrent rank 分界；
// 8/9=AG CCU SEQ 档与 ratio 基线；16/17=2Die 档（RS CCU 2*8p）/A2AV Pairwise 下界；32/33=
// AG CCU ccuSize 档；63/64/65=CCU maxSize（AR/RS CcuSched rank 分界，64 被 >=64 小数据支
// 抢走 NOT_MATCH）；1024=AG Parallel 门槛（entry 5/6 的 1023/1024 分界 + DPU/TOTAL 面）
constexpr u32 kScanRanks[] = {2U, 4U, 5U, 8U, 9U, 16U, 17U, 32U, 33U, 63U, 64U, 65U, 1024U};

// 登记门校验（D17 冷路径）代表 r 域：覆盖各声明 rankSize 段的端点/单点（{0,0}/{1,1}/{2,2}/
// {2,4}/{3,MAX} 等）与 RANK_SQ 饱和路径；声明域不含任何代表 r 的 pattern 由 Sweep 测试回退
// 到域端点自测（kGateRanks ∪ {range.min, range.max}）
constexpr u32 kGateRanks[] = {0U, 1U, 2U, 3U, 4U, 5U, 8U, 9U, 16U, 17U, 32U, 33U, 63U, 64U, 65U, 1024U};

// 登记门最小 topo：ValidatePattern/NarrowByClause 只读 userRankSize，其余字段不参与解集求解
TopoInfoWithNetLayerDetails MakeBareRankTopo(u32 rankSize)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRankSize = rankSize;
    return topo;
}

// ---- 链序不变量断言器（selector 链实读判据，非运行期向量推断）----

// 适用：读实例数的 selector 路径由 Level1Nhr 与 Level0Nhr 两道门把守的族——RS/AG 的 AICPU
// 多层链（RS :433 Level1Nhr → :436 Level0Nhr → :439/:460 实例数；AG :262 → :265 → :267）。
// 凡 level0Inst.cmp != ANY 的条目所镜像的 selector 路径，必已以 Level1Nhr==false 与
// Level0Nhr==false 通过两道门，条目须如实记录双 Tri::No。缺省 Any（Tri 默认）在 Level0Nhr=
// true 时条目仍可命中，而 selector 已在 :436/:265 改选 InsReduceScatterNHR /
// AicpuAllGatherSoleNHR——模式面多报（onlyInPattern 假阳性），正是 fb9930fa3a 勘误补回
// 65 条 level0Nhr=No 的病灶；本断言把该类错误从"矩阵向量恰好路过才暴露"升级为全量静态可检
void ExpectInstEntsBelowBothNhrGates(const TopoPattern* pats, size_t count, const char* what)
{
    for (size_t i = 0; i < count; ++i) {
        if (pats[i].level0Inst.cmp == InstCmp::ANY) {
            continue;
        }
        EXPECT_EQ(pats[i].level1Nhr, Tri::No) << what << " entry[" << i << "] 须记录 level1Nhr=No";
        EXPECT_EQ(pats[i].level0Nhr, Tri::No) << what << " entry[" << i << "] 须记录 level0Nhr=No";
    }
}

// 适用：链上无 Level0Nhr 读点、实例数读点仅由 Level1Nhr 门把守的族——RS/AG 的 CCU 链
// （RS :211 Level1Nhr → :220 实例数；AG :174 → :183）与 AR 全族（AICPU :102 → :106；
// CCU :348 → :357；AR 两引擎链 grep 无 Level0Nhr 读点）。level0Inst 条目只须记录
// level1Nhr=No；对它们断言 level0Nhr=No 反而是错——CCU/AR 链从不读该字段，Level0Nhr=true
// 的拓扑照常走到实例数读点（已实证：CCU 数组的 level0Inst 条目均不带
// level0Nhr，差分全绿）。RS/AG CCU 链的 is2DieFullMesh 门（:216/:179）同样先于实例数读点，
// 但 65 条漏写病灶是 Nhr 转写缺项，2die 门不在本断言面（如实记录见各数组条目）
void ExpectInstEntsBelowLevel1NhrGate(const TopoPattern* pats, size_t count, const char* what)
{
    for (size_t i = 0; i < count; ++i) {
        if (pats[i].level0Inst.cmp == InstCmp::ANY) {
            continue;
        }
        EXPECT_EQ(pats[i].level1Nhr, Tri::No) << what << " entry[" << i << "] 须记录 level1Nhr=No";
    }
}

// 适用：从不读实例数的链——A2A 全族（alltoall/alltoallv/half 三个 selector grep 无
// localNetInsSizeOfLayer 读点，all_to_all_v_alg_patterns.h:34-36 已入档）与 RS DPU
// （SelectDPUAlgo :541 只读 topoLevelNums）。level0Inst 条目会让模式面过滤一个 selector
// 从不读的维度 → 命中集缺名（onlyInSelector 假阳性），一律禁止出现
void ExpectNoInstConstraint(const TopoPattern* pats, size_t count, const char* what)
{
    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(pats[i].level0Inst.cmp, InstCmp::ANY) << what << " entry[" << i << "] 不得有实例数约束";
    }
}

#define INST_BELOW_BOTH_NHR_GATES(arr) ExpectInstEntsBelowBothNhrGates(arr, sizeof(arr) / sizeof(TopoPattern), #arr)
#define INST_BELOW_L1_NHR_GATE(arr) ExpectInstEntsBelowLevel1NhrGate(arr, sizeof(arr) / sizeof(TopoPattern), #arr)
#define NO_INST_CONSTRAINT(arr) ExpectNoInstConstraint(arr, sizeof(arr) / sizeof(TopoPattern), #arr)

// ---- 计数锚运行时复述（与各 *alg_patterns.h 的 static_assert 同值）----
#define EXPECT_PATTERN_COUNT(arr, n) EXPECT_EQ(sizeof(arr) / sizeof(TopoPattern), static_cast<size_t>(n)) << #arr
} // namespace

// 形状 A：两层 Mesh Level1Nhr，rank32——Level1Nhr 早返回族；AICPU 双 dtype 覆盖
// RS/AR 的 special 分支（InsReduceScatterAicpuReduceNHR / AicpuAllReduceSoleNHRAicpuReduce）
TEST(PatternDifferential, ShapeA_TwoLevelMeshLevel1Nhr)
{
    const TopoInfoWithNetLayerDetails topo = MakeTwoLevelMeshTopo(32U, true, 8U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, topo, k1MB, "A·AG·AICPU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::CCU, topo, k1MB, "A·AG·CCU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, topo, k4MB, "A·RS·AICPU·4MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, topo, k16MB, "A·AR·AICPU·16MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::DPU, topo, k1MB, "A·AG·DPU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::DPU, topo, k4MB, "A·RS·DPU·4MB");
}

// 形状 B：两层 Mesh rank2048，len=600KB——AG/RS 总量分档（600KB×2048 ≥ 1GB），
// CCU 侧 rank>64 全灭（空集 == NOT_MATCH）
TEST(PatternDifferential, ShapeB_TwoLevelMeshRank2048)
{
    const TopoInfoWithNetLayerDetails topo = MakeTwoLevelMeshTopo(2048U, false, 8U);
    constexpr u64 kLen600KB = 600ULL * 1024;
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, topo, kLen600KB, "B·AG·AICPU·600KB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::CCU, topo, kLen600KB, "B·AG·CCU·600KB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, topo, kLen600KB, "B·RS·AICPU·600KB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::DPU, topo, kLen600KB, "B·AG·DPU·600KB");
}

// 形状 C：一层 UBX mesh==clos rank4——512KB 边界两侧（AICPU：Concur/Mesh1D；
// CCU：ConcurMultiLink/SoleMesh）
TEST(PatternDifferential, ShapeC_OneLevelUbxMeshEqRank4)
{
    const TopoInfoWithNetLayerDetails topo = MakeUbxTopo(4U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, topo, k1MB, "C·AG·AICPU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, topo, k256KB, "C·AG·AICPU·256KB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::CCU, topo, k1MB, "C·AG·CCU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::CCU, topo, k256KB, "C·AG·CCU·256KB");
}

// 形状 D：一层 UBX 畸形（Check* 失败）——MatchChecked 否决 + selector 同点
// NOT_MATCH，双侧空集
TEST(PatternDifferential, ShapeD_OneLevelUbxDegenerate)
{
    const TopoInfoWithNetLayerDetails topo = MakeUbxTopo(8U, true);
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, topo, k1MB, "D·AG·AICPU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::CCU, topo, k1MB, "D·AG·CCU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, topo, k1MB, "D·RS·AICPU·1MB");
}

// 形状 E/F：两层 Mesh rank32 × (Level1Nhr, is2DieFullMesh 同真)——CCU 链上
// Level1Nhr 命中即返（不读 2die），F 依次被两守卫否决；F 的 AICPU 走单实例支
TEST(PatternDifferential, ShapeEF_TwoLevelMeshTwoDieFullMesh)
{
    TopoInfoWithNetLayerDetails shapeE = MakeTwoLevelMeshTopo(32U, true, 1U);
    shapeE.is2DieFullMesh = true;
    TopoInfoWithNetLayerDetails shapeF = MakeTwoLevelMeshTopo(32U, false, 1U);
    shapeF.is2DieFullMesh = true;
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::CCU, shapeE, k1MB, "E·AG·CCU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::CCU, shapeF, k1MB, "F·AG·CCU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, shapeF, k1MB, "F·AG·AICPU·1MB");
}

// 引擎×(cmd,len) 组合（一层 Mesh1D rank8，meshType 缺省）：AG·CCU 钉 CcuMs 并集机制
// （Sched 因 meshType 拒绝、MS 不查，并集仍等于模式面的 CCU_MS 分支）；AR·AICPU 16MB 落
// TwoShot 档（>8MB 且 ≤ r²·4MB）；AR·CCU 2MB 落 r²·1MB 公式档
TEST(PatternDifferential, EngineLenCombos_FlatMesh1D)
{
    const TopoInfoWithNetLayerDetails topo = MakeFlatMesh1DTopo(8U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, topo, k1MB, "flat·AG·AICPU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::CCU, topo, k1MB, "flat·AG·CCU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, topo, k4MB, "flat·RS·AICPU·4MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, topo, k16MB, "flat·AR·AICPU·16MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::CCU, topo, 2U * k1MB, "flat·AR·CCU·2MB");
}

// ---- 差分矩阵 ----

// 勘误定向边界：AR AICPU 一层 Mesh 公式条目。勘误后 T_ar = r²·4MB
// （selector :226 为 8/r²）。r=0 是退化通信域（rankSize=0 在真实
// 链路别处先拒），按裁决记录不测，用 r=1/2/3/8 覆盖公式条目；r=1 的 TwoShot 区间
// (8MB, 4MB] 自然掏空（OneShot 8MB 含、8MB+8 直落 Chunk）；r=2 twoLevel=No 的 TwoShot
// 区间 (8MB,16MB] 双边界；r=3 落公式上界内侧（36MB > 32MB+8）；r=8 双边界 8MB（OneShot|
// TwoShot）与 256MB（TwoShot|Chunk，64·4MB）。注：特殊 dtype 在 MESH_1D 上 >8MB 恒选
// TwoShot（无上界条目），故 TwoShot 正常分档上界经并集被遮蔽（见报告），可观测面是
// Chunk 下界与 OneShot 上界
TEST(PatternDifferential, ErrataArMeshFormulaBoundaries)
{
    const TopoInfoWithNetLayerDetails r1 = MakeFlatMesh1DTopo(1U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r1, k4MB, "AR·r1·4MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r1, k8MB, "AR·r1·8MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r1, k8MB + 8U, "AR·r1·8MB+8");

    const TopoInfoWithNetLayerDetails r2 = MakeFlatMesh1DTopo(2U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r2, k8MB, "AR·r2·8MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r2, k8MB + 8U, "AR·r2·8MB+8");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r2, k16MB, "AR·r2·16MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r2, k16MB + 8U, "AR·r2·16MB+8");

    const TopoInfoWithNetLayerDetails r3 = MakeFlatMesh1DTopo(3U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r3, k32MB + 8U, "AR·r3·32MB+8");

    const TopoInfoWithNetLayerDetails r8 = MakeFlatMesh1DTopo(8U);
    constexpr u64 kLen256MB = 256ULL * 1024 * 1024; // T_ar(8) = 8²·4MB
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r8, k8MB, "AR·r8·8MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r8, k8MB + 8U, "AR·r8·8MB+8");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r8, kLen256MB - 8U, "AR·r8·256MB-8");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r8, kLen256MB, "AR·r8·256MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r8, kLen256MB + 8U, "AR·r8·256MB+8");
}

// 勘误定向边界：AR twoLevel 绕行与 4GB 掏空。r=2+twoLevelNet=Yes：>=8MB 走 2P 绕行
// （SoleMeshConcur 未迁移 NOT_MATCH），16MB+8 处若无绕行 normal 将选 Chunk（(16MB+8)·2 >
// 32MB），并集里 Chunk 的缺失即绕行的可观测面；恰 8MB 点是 dtype 盲区伪边界（特殊 dtype
// 仍选 OneShot 而模式条目在 8MB-1 截断），按裁决规避取 8MB-8/8MB+8。r=31/r=32 是 Chunk
// twoLevel+4GB 上界支的掏空分界（4MB·31²=3844MB < 4GB 支仍存在，4MB·32²=4GB ≥ 4GB 公式
// 条目与上界相交掏空）——r=31 取 3800MB（TwoShot 侧）/3900MB（Chunk 侧，避开 3844MB 的
// double ratio 浮点临界）；r=32 的 4GB 是 LE 含点（4GB/128 恰 32MB 不大于）、4GB+8 处
// selector 走 overSequence 绕行 NOT_MATCH 而模式 Chunk 条目已掏空
TEST(PatternDifferential, ErrataArTwoLevelDetourAnd4gbBoundary)
{
    const TopoInfoWithNetLayerDetails r2 = MakeSingleLevelTwoLevelNetTopo(2U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r2, k8MB - 8U, "AR·r2·2L·8MB-8");
    // 恰 8MB 边界：双 dtype 并集含 OneShot（特殊 dtype :233 先于 2P 绕行），OneShot LE 8MB（含）
    // 与 TwoShot 特殊并入 GT 8MB（严格）在此点无重叠无空洞
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r2, k8MB, "AR·r2·2L·8MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r2, k16MB + 8U, "AR·r2·2L·16MB+8");

    constexpr u64 kLen3800MB = 3800ULL * 1024 * 1024;
    constexpr u64 kLen3900MB = 3900ULL * 1024 * 1024;
    const TopoInfoWithNetLayerDetails r31 = MakeSingleLevelTwoLevelNetTopo(31U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r31, kLen3800MB, "AR·r31·2L·3800MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r31, kLen3900MB, "AR·r31·2L·3900MB");

    const TopoInfoWithNetLayerDetails r32 = MakeSingleLevelTwoLevelNetTopo(32U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r32, k4GB, "AR·r32·2L·4GB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, r32, k4GB + 8U, "AR·r32·2L·4GB+8");
}

// 勘误定向边界：RS AICPU 一层 Mesh。T_rs = r²·256KB（r=8：16MB，Mesh1D|Chunk 分界，
// twoLevel=No 无总量子句）；twoLevel=Yes 时 Mesh1D 条目加总量 ≤ 1536MB 相与、Chunk 有
// 总量支（GT 1536MB，selector :561 先于 ratio 门槛 :563）；r=2 两层网的 1MB（Mesh1D 上界
// 静态）/4MB（2P 绕行下界）双边界；r=64 len=32MB 钉总量支优先序——ratio 侧 32MB/64=8KB
// 远小于 16MB，若无总量支优先 normal 将选 Mesh1D（此时 Chunk 只能来自总量支）。注：特殊
// dtype 在 MESH_1D 上恒选 Mesh1D（无长度条目），Mesh1D 正常分档公式经并集被遮蔽（见
// 报告），可观测面是 Chunk 的双侧边界与绕行窗口
TEST(PatternDifferential, ErrataRsMeshBoundaries)
{
    const TopoInfoWithNetLayerDetails r8Flat = MakeFlatMesh1DTopo(8U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, r8Flat, k16MB - 8U, "RS·r8·16MB-8");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, r8Flat, k16MB, "RS·r8·16MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, r8Flat, k16MB + 8U, "RS·r8·16MB+8");

    const TopoInfoWithNetLayerDetails r8TwoLevel = MakeSingleLevelTwoLevelNetTopo(8U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, r8TwoLevel, k16MB, "RS·r8·2L·16MB");
    ExpectDifferential(
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, r8TwoLevel, k16MB + 8U, "RS·r8·2L·16MB+8");
    ExpectDifferential(
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, r8TwoLevel, k1536MB / 8U, "RS·r8·2L·192MB");
    ExpectDifferential(
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, r8TwoLevel, k1536MB / 8U + 8U, "RS·r8·2L·192MB+8");

    const TopoInfoWithNetLayerDetails r2 = MakeSingleLevelTwoLevelNetTopo(2U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, r2, k1MB - 8U, "RS·r2·2L·1MB-8");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, r2, k1MB, "RS·r2·2L·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, r2, k1MB + 8U, "RS·r2·2L·1MB+8");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, r2, k4MB - 8U, "RS·r2·2L·4MB-8");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, r2, k4MB, "RS·r2·2L·4MB");

    const TopoInfoWithNetLayerDetails r64 = MakeSingleLevelTwoLevelNetTopo(64U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, r64, k32MB, "RS·r64·2L·32MB");
}

// 勘误定向边界：CCU 公式。AR T_ccu_ar = r²·1MB（代码
// :508 系 8/r² > 8MB）——r=8 上界 64MB（LE 含点），64MB+8 处 normal NOT_MATCH（64M 以上
// 切 aicpu）且 CCU 前置拒 PROD/64bit，expected 空；RS SoleMesh 一层上界 r·1MB（selector
// :381 单除 8/r 的互补）——r=8 的 8MB 是 LT 严格排他点（8MB·1 >= 8MB 即拒）。RS CCU 向量
// 同时实证死区过滤：CcuMs 在一层 Mesh 选 CcuReduceScatterMesh1D（未注册）被 expected
// 裁剪，Sched 的 SoleMesh 仍是可观测面
TEST(PatternDifferential, ErrataCcuFormulaBoundaries)
{
    const TopoInfoWithNetLayerDetails r8 = MakeFlatMesh1DTopo(8U);
    constexpr u64 kLen64MB = 64ULL * 1024 * 1024;
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::CCU, r8, kLen64MB - 8U, "AR·CCU·r8·64MB-8");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::CCU, r8, kLen64MB, "AR·CCU·r8·64MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::CCU, r8, kLen64MB + 8U, "AR·CCU·r8·64MB+8");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::CCU, r8, k8MB - 8U, "RS·CCU·r8·8MB-8");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::CCU, r8, k8MB, "RS·CCU·r8·8MB");
}

// rankSize 边界扫（边界值导向）：两层 Mesh [0]=8 底座 × 13 档 × (AG/RS/AR) ×
// (AICPU/CCU/DPU) × 各一条 len（AG/RS/AR 统一 256KB）。边界含义见 kScanRanks 注释；
// AICPU 面：全档 NHR/Parallel 分档的 rankSize 子句（如 AG entry 5/6 的 1023/1024 分界、
// RS entry 11/13 的 255/256 分界经 len=256KB 时 rank<4096 均落 NHR，但 63/64/65/1024 档
// 把 CCU rank 子句钉满）；CCU 面：AR SoleMesh entry1 的 {0,63} 分界（64/65 档被 >=64 小
// 数据支 NOT_MATCH 掏空）、AG SoleMesh entry2 的 {0,8} 分界、RS SoleNHR entry3 的
// rank>=64+frameNum>=17 支（1024 档 frameNum=128 命中）；DPU 面：AG/RS 多层恒 MATCH、
// AR 无 DPU 注册面（基类 NOT_MATCH == 模式空集）。RS CCU 另补 32KB 正向档（总量<512KB
// 支——256KB 时 2 档总量恰 512KB 不入，32KB 才进 SoleMesh）
TEST(PatternDifferential, RankSizeBoundaryScanTwoLevelMesh)
{
    for (const u32 rank : kScanRanks) {
        const TopoInfoWithNetLayerDetails topo = MakeTwoLevelMeshTopo(rank, false, 8U);
        const std::string tag = "scan·r" + std::to_string(rank) + "·";
        ExpectDifferential(
            HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, topo, k256KB, (tag + "AG·AICPU·256KB").c_str());
        ExpectDifferential(
            HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, topo, k256KB, (tag + "RS·AICPU·256KB").c_str());
        ExpectDifferential(
            HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, topo, k256KB, (tag + "AR·AICPU·256KB").c_str());
        ExpectDifferential(
            HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::CCU, topo, k256KB, (tag + "AG·CCU·256KB").c_str());
        ExpectDifferential(
            HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::CCU, topo, k256KB, (tag + "RS·CCU·256KB").c_str());
        ExpectDifferential(
            HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::CCU, topo, k256KB, (tag + "AR·CCU·256KB").c_str());
        ExpectDifferential(
            HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::DPU, topo, k256KB, (tag + "AG·DPU·256KB").c_str());
        ExpectDifferential(
            HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::DPU, topo, k256KB, (tag + "RS·DPU·256KB").c_str());
        ExpectDifferential(
            HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::DPU, topo, k256KB, (tag + "AR·DPU·256KB").c_str());
    }
    // RS CCU SoleMesh 两层正向档（总量 < 512KB）：2/8 档命中 SoleMesh；16 档总量恰 512KB
    // 严格排他（2*8p 与 SoleMesh 支均不进）；32/64 档落 Sequence 未迁移支双空
    constexpr u64 kLen32KB = 32ULL * 1024;
    for (const u32 rank : {2U, 8U, 16U, 32U, 64U}) {
        const TopoInfoWithNetLayerDetails topo = MakeTwoLevelMeshTopo(rank, false, 8U);
        ExpectDifferential(
            HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::CCU, topo, kLen32KB,
            ("scan·r" + std::to_string(rank) + "·RS·CCU·32KB").c_str());
    }
}

// [0]==0 形状（退化实例卡数）：RS 多层 MESH_1D 的 [0]==0 落 selector 最终 else NOT_MATCH
// （:465-468），normal 侧空集；特殊类型在多层第一支改选 AicpuReduceNHR，并集恰只剩该名
// ——Mesh 分档条目的 GT 例外（spec §7.1 规则 2）使 NHR 分档（InsReduceScatterNHR 的 Mesh
// 档、ParallelMesh1DNHR、SequenceMesh1DNhr）不进命中集，显式断言钉死。AG 多层无 GT 例外
// （NE 忠实）：[0]==0 与 [0]>1 命中集完全一致（AICPU 的 NE 1 条目、CCU 的 NE 1 条目、
// DPU 的不读实例数条目三面齐证）。RS+CCU 的 [0]==0 不测：SoleNHR 的 else 支（:263-266）
// 在 [0]==0 仍选名而模式 EQ 1 条目不收，spec D12 已裁决为接受语义（名单不因此多报）
TEST(PatternDifferential, ZeroInstNumDegenerateShapes)
{
    constexpr u32 dtypeSize = DATATYPE_SIZE_TABLE[HCCL_DATA_TYPE_FP32];
    const TopoInfoWithNetLayerDetails mesh00 = MakeTwoLevelMeshTopo(32U, false, 0U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, mesh00, k4MB, "RS·r32·[0]==0·4MB");
    const std::set<std::string> rsHits =
        PatternMatches(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, mesh00, k4MB, dtypeSize);
    EXPECT_EQ(rsHits.count("InsReduceScatterNHR"), 0U) << "NHR 分档（GT 例外）不得命中 [0]==0";
    EXPECT_EQ(rsHits.count("InsReduceScatterParallelMesh1DNHR"), 0U);
    EXPECT_EQ(rsHits.count("InsReduceScatterSequenceMesh1DNhr"), 0U);
    EXPECT_EQ(rsHits.count("InsReduceScatterAicpuReduceNHR"), 1U) << "特殊类型支是唯一存活面";

    const TopoInfoWithNetLayerDetails mesh08 = MakeTwoLevelMeshTopo(32U, false, 8U);
    const std::set<std::string> hits00 =
        PatternMatches(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, mesh00, k256KB, dtypeSize);
    const std::set<std::string> hits08 =
        PatternMatches(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, mesh08, k256KB, dtypeSize);
    EXPECT_EQ(hits00, hits08) << "AG 多层 [0]==0 与 [0]>1 命中集须一致（NE 忠实）";
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, mesh00, k256KB, "AG·r32·[0]==0·256KB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, mesh00, 2U * k1MB, "AG·r32·[0]==0·2MB");
    constexpr u64 kLen64KB = 64ULL * 1024;
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::CCU, mesh00, kLen64KB, "AG·r32·[0]==0·CCU·64KB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::DPU, mesh00, k256KB, "AG·r32·[0]==0·DPU·256KB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::DPU, mesh00, k256KB, "RS·r32·[0]==0·DPU·256KB");
}

// AllToAll 元素边界（ELEMENT 度量 dtypeSize 参数化，INT8/FP16/INT32 三态）：{400,512,513}
// 元素 × AICPU（元素分界：Concurrent 是 >512 元素、UBX 是 <=512）与 CCU（字节分界：
// Concurrent 是 >512 字节、MultiJetty 是 <=512 字节）两侧。INT8 时元素==字节两分界重合
// （400/512 → MultiJetty|UBX、513 → Concurrent|Concurrent）；FP16/INT32 时字节分界左移
// （400 元素 = 800/1600 字节 > 512 → CCU 侧 Concurrent 而 AICPU 侧仍 UBX）——正是
// ELEMENT 与 PER_RANK 量纲差的可观测面；rank=8（meshEq 且 >=5）钉 rank 子句（无长度面）
TEST(PatternDifferential, AllToAllElementByteBoundaries)
{
    const TopoInfoWithNetLayerDetails ubxEqual4 = MakeUbxEqualTopo(4U);
    const TopoInfoWithNetLayerDetails ubxEqual8 = MakeUbxEqualTopo(8U);
    constexpr HcclDataType kDtypes[] = {HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_INT32};
    for (const u64 elemCount : {400ULL, 512ULL, 513ULL}) {
        for (const HcclDataType dtype : kDtypes) {
            const std::string tag = "A2A·" + std::to_string(elemCount) + "·dt" + std::to_string(dtype);
            ExpectA2ADifferential(
                HcclCMDType::HCCL_CMD_ALLTOALL, AlgEngine::AICPU, ubxEqual4, elemCount, dtype,
                (tag + "·AICPU").c_str());
            ExpectA2ADifferential(
                HcclCMDType::HCCL_CMD_ALLTOALL, AlgEngine::CCU, ubxEqual4, elemCount, dtype, (tag + "·CCU").c_str());
        }
    }
    // meshEq 且 rank >= 5：AICPU 落 UBX（concurrentCondition 的 rank<=4 不满足）、CCU 落
    // MultiJetty（同因），均无长度子句——元素/字节边界在此形状上不应翻转选名
    ExpectA2ADifferential(
        HcclCMDType::HCCL_CMD_ALLTOALL, AlgEngine::AICPU, ubxEqual8, 513ULL, HCCL_DATA_TYPE_INT8, "A2A·r8·513·AICPU");
    ExpectA2ADifferential(
        HcclCMDType::HCCL_CMD_ALLTOALL, AlgEngine::CCU, ubxEqual8, 400ULL, HCCL_DATA_TYPE_INT8, "A2A·r8·400·CCU");
}

// §6.1 D 补齐（Check* 畸形——degenerate 拓扑不填 topoInstDetailsOfLayer）：已盖
// AG·AICPU/CCU、RS·AICPU；此处补 RS·CCU（Sched 的 UBX 支与 CcuMs 的 MESH_1D_CLOS 支均
// 先调 Check* → NOT_MATCH）、AR·AICPU（SelectMeshAlgoAicpuUBX 的 Check* 先于特殊类型支，
// 双 dtype 都被否决）、AR·CCU（UBX 支 Check* 先拒；AR 无 CcuMs override 走基类默认）、
// AG·DPU（单层不进 DPU 支，模式 level>=2 条目空集）。四向均双侧空集
TEST(PatternDifferential, ShapeDMoreDegenerateCheckShapes)
{
    const TopoInfoWithNetLayerDetails topo = MakeUbxTopo(8U, true);
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::CCU, topo, k1MB, "D·RS·CCU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, topo, k1MB, "D·AR·AICPU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::CCU, topo, k1MB, "D·AR·CCU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::DPU, topo, k1MB, "D·AG·DPU·1MB");
}

// 多层 CLOS（AG/RS MultiLink 的 ×E/×F 路径）：RS 两层 CLOS 的 ExcludeE 展开五形
// （E1 不对称/E2 仅 L0 对称/E3a 每模块卡数<8/E3b >8/E4 =8 且非 uboe）+ 对称+8+uboe 的
// pipeline 阻断形（normal NOT_MATCH，并集只剩特殊类型的 AicpuReduceNHR——×E 展开的完备
// 排除即其可观测面）；三层 CLOS 的 ExcludeF 两形 + 对称+uboe 分流（卡数≠8 →
// ParallelMesh1DNHR（该支不读 level0Topo）、卡数=8 → pipeline 阻断）；四层 CLOS 的 ×E1 +
// pipeline 阻断；AG 两/三/四层 CLOS（无对称支，恒 MultiLink）；CCU 侧两层 CLOS 的
// SoleNHR（AG/RS 各一）与三层的前置 >=3 拒绝双空
// b 修复回归钉：两层 CLOS + pcieMix 组合此前无夹具（覆盖空白让 pattern 多余约束漏网）
TEST(PatternDifferential, TwoLevelClosPcieMixSoleNhr)
{
    TopoInfoWithNetLayerDetails clos2Pcie = MakeTwoLevelClosPcieMixTopo(32U);
    // AR·CCU·两层CLOS+pcieMix·<8MB：selector :419-425 不读 pcieMix 选 SoleNHR，pattern 须命中
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::CCU, clos2Pcie, k1MB, "AR·CCU·clos2·pcie·1MB");
}

TEST(PatternDifferential, MultiLevelClosExcludeEFPaths)
{
    TopoInfoWithNetLayerDetails clos2E1 = MakeMultiLevelClosTopo(32U, 2U);
    TopoInfoWithNetLayerDetails clos2E2 = clos2E1;
    clos2E2.level0Symmetric = true;
    TopoInfoWithNetLayerDetails clos2E3a = clos2E1;
    clos2E3a.level0Symmetric = true;
    clos2E3a.level1Symmetric = true;
    clos2E3a.deviceNumPerModule = 4U;
    TopoInfoWithNetLayerDetails clos2E3b = clos2E1;
    clos2E3b.level0Symmetric = true;
    clos2E3b.level1Symmetric = true;
    clos2E3b.deviceNumPerModule = 9U;
    TopoInfoWithNetLayerDetails clos2E4 = clos2E1;
    clos2E4.level0Symmetric = true;
    clos2E4.level1Symmetric = true;
    clos2E4.deviceNumPerModule = 8U;
    TopoInfoWithNetLayerDetails clos2Pipe = clos2E4;
    clos2Pipe.topLevelUboe = true;
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, clos2E1, k1MB, "CLOS·2L·E1·RS");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, clos2E2, k1MB, "CLOS·2L·E2·RS");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, clos2E3a, k1MB, "CLOS·2L·E3a·RS");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, clos2E3b, k1MB, "CLOS·2L·E3b·RS");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, clos2E4, k1MB, "CLOS·2L·E4·RS");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, clos2Pipe, k1MB, "CLOS·2L·pipe·RS");

    TopoInfoWithNetLayerDetails clos3F1 = MakeMultiLevelClosTopo(32U, 3U);
    TopoInfoWithNetLayerDetails clos3F3 = clos3F1;
    clos3F3.level0Symmetric = true;
    clos3F3.level1Symmetric = true;
    TopoInfoWithNetLayerDetails clos3Par = clos3F3;
    clos3Par.topLevelUboe = true;
    clos3Par.deviceNumPerModule = 4U;
    TopoInfoWithNetLayerDetails clos3Pipe = clos3F3;
    clos3Pipe.topLevelUboe = true;
    clos3Pipe.deviceNumPerModule = 8U;
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, clos3F1, k1MB, "CLOS·3L·F1·RS");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, clos3F3, k1MB, "CLOS·3L·F3·RS");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, clos3Par, k1MB, "CLOS·3L·par·RS");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, clos3Pipe, k1MB, "CLOS·3L·pipe·RS");

    TopoInfoWithNetLayerDetails clos4E1 = MakeMultiLevelClosTopo(32U, 4U);
    TopoInfoWithNetLayerDetails clos4Pipe = clos4E1;
    clos4Pipe.level0Symmetric = true;
    clos4Pipe.level1Symmetric = true;
    clos4Pipe.deviceNumPerModule = 8U;
    clos4Pipe.topLevelUboe = true;
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, clos4E1, k1MB, "CLOS·4L·E1·RS");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, clos4Pipe, k1MB, "CLOS·4L·pipe·RS");

    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, clos2E1, k1MB, "CLOS·2L·AG");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, clos3F1, k1MB, "CLOS·3L·AG");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, clos4E1, k1MB, "CLOS·4L·AG");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::CCU, clos2E1, k1MB, "CLOS·2L·AG·CCU");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::CCU, clos2E1, k1MB, "CLOS·2L·RS·CCU");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::CCU, clos3F1, k1MB, "CLOS·3L·AG·CCU");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::CCU, clos3F1, k1MB, "CLOS·3L·RS·CCU");
}

// ---- 边界与组合补充向量 ----

// 链序不变量：实例数读点前的 Nhr 门序按族固化。判据全部来自六个
// selector 的链实读（文件:行号见各断言器注释），非运行期推断——矩阵向量只证"被路过的路径
// 不漂移"，本测试证"条目结构与 selector 链序一致"这一更强命题，漏写类错误无须向量恰好路过
// 即被捕获。族的取舍（不确定不立）：
//   立断言 level0Nhr+level1Nhr 双 No：RS AICPU 7 数组、AG AICPU 10 数组（两族 AICPU 多层链
//     在实例数读点前依次设 Level1Nhr/Level0Nhr 两道门）；
//   立断言 level1Nhr 单 No：RS CCU 5 数组、AG CCU 4 数组、AR 全 9 数组（三链无 Level0Nhr
//     读点，实例数读点只由 Level1Nhr 把守）；
//   立断言 禁止 level0Inst：A2A 13 数组 + RS DPU + RS UBX/PCIE 4 数组（链从不读实例数）；
//   不立断言：AG DPU（kInsAllGatherMeshNhrDPUPatterns）——SelectDPUAlgo :358 读 [0]==1 前无
//     任何 Nhr 门，EQ 条目合法且不带 level0Nhr 是链序的忠实镜像，强加 level0Nhr=No 断言反而
//     与链不符（该链的实例数读点不经 Nhr 门，属"从无门可漏写"形态）。
TEST(PatternDifferential, ChainOrderInvariants)
{
    // RS AICPU（reduce_scatter_auto_selector.cc :433 → :436 → :439/:460）
    INST_BELOW_BOTH_NHR_GATES(kInsReduceScatterNHRPatterns);
    INST_BELOW_BOTH_NHR_GATES(kInsReduceScatterParallelMesh1DNHRPatterns);
    INST_BELOW_BOTH_NHR_GATES(kInsReduceScatterSequenceMesh1DNhrPatterns);
    INST_BELOW_BOTH_NHR_GATES(kAicpuReduceScatterSoleNHRMultiLinkPatterns);
    INST_BELOW_BOTH_NHR_GATES(kInsReduceScatterMesh1DPatterns);
    INST_BELOW_BOTH_NHR_GATES(kAicpuReduceScatterSoleMeshChunkPatterns);
    INST_BELOW_BOTH_NHR_GATES(kInsReduceScatterAicpuReduceNHRPatterns);
    // RS AICPU·UBX/PCIE 四数组（并入禁实例数档）：镜像链 SelectMeshAlgoAicpuForMesh1DClos
    // （:580-:621 单层路径，grep 无 localNetInsSizeOfLayer 读点——从无门可漏写形态），当前
    // 条目均无 level0Inst 子句，并入防未来新条目误加（A2A 同款先例）
    NO_INST_CONSTRAINT(kAicpuReduceScatterConcurMeshNHRPatterns);
    NO_INST_CONSTRAINT(kAicpuReduceScatterParallelMeshNHRUBXPatterns);
    NO_INST_CONSTRAINT(kInsReduceScatterParallelMesh1DNHRPciePatterns);
    NO_INST_CONSTRAINT(kAicpuReduceScatterPipeLinePciePatterns);
    // AG AICPU（all_gather_auto_selector.cc :262 → :265 → :267）；单层数组现无 level0Inst 条目，
    // 断言对它们为空转，留作未来补条目的门禁
    INST_BELOW_BOTH_NHR_GATES(kAicpuAllGatherSoleNHRPatterns);
    INST_BELOW_BOTH_NHR_GATES(kInsAllGatherParallelMesh1DNHRPatterns);
    INST_BELOW_BOTH_NHR_GATES(kAicpuAllGatherSequenceMeshConcurNHRPatterns);
    INST_BELOW_BOTH_NHR_GATES(kAicpuAllGatherSoleNHRMultiLinkPatterns);
    INST_BELOW_BOTH_NHR_GATES(kInsAllGatherMesh1DPatterns);
    INST_BELOW_BOTH_NHR_GATES(kAicpuAllGatherConcurMeshNHRPatterns);
    INST_BELOW_BOTH_NHR_GATES(kInsAllGatherParallelMesh1DNHRMultiJettyPatterns);
    INST_BELOW_BOTH_NHR_GATES(kAicpuAllGatherPipeLineMeshNHRPatterns);
    INST_BELOW_BOTH_NHR_GATES(kInsAllGatherParallelMesh1DNHRPciePatterns);
    INST_BELOW_BOTH_NHR_GATES(kAicpuAllGatherPipeLinePciePatterns);

    // RS CCU（:211 → :220）；PeerOnly 手写数组同样过检（现无 level0Inst 条目，空转）
    INST_BELOW_L1_NHR_GATE(kCcuSchedReduceScatterSoleNHRPatterns);
    INST_BELOW_L1_NHR_GATE(kCcuSchedReduceScatterSoleMeshPatterns);
    INST_BELOW_L1_NHR_GATE(kCcuSchedReduceScatterConcurMeshNHRMultiLinkPatterns);
    INST_BELOW_L1_NHR_GATE(kCcuSchedReduceScatterSoleNHRMultiLinkPatterns);
    INST_BELOW_L1_NHR_GATE(kCcuSchedReduceScatterSoleMeshPeerOnlyPatterns);
    // AG CCU（:174 → :183）
    INST_BELOW_L1_NHR_GATE(kCcuSchedAllGatherSoleNHRPatterns);
    INST_BELOW_L1_NHR_GATE(kCcuSchedAllGatherSoleMeshPatterns);
    INST_BELOW_L1_NHR_GATE(kCcuSchedAllGatherConcurMeshNHRMultiLinkPatterns);
    INST_BELOW_L1_NHR_GATE(kCcuSchedAllGatherParallelMeshNHRMultiLinkPatterns);
    // AR 全族（AICPU :102 → :106；CCU :348 → :357；两链 grep 无 Level0Nhr 读点——AR 的
    // level0Inst 条目只带 level1Nhr=No 不带 level0Nhr，与链序一致，勿照抄 RS 双 No）
    INST_BELOW_L1_NHR_GATE(kAicpuAllReduceSoleNHRPatterns);
    INST_BELOW_L1_NHR_GATE(kAicpuAllReduceParallelMeshNHRPatterns);
    INST_BELOW_L1_NHR_GATE(kAicpuAllReduceSoleMeshOneShotPatterns);
    INST_BELOW_L1_NHR_GATE(kAicpuAllReduceSoleMeshTwoShotPatterns);
    INST_BELOW_L1_NHR_GATE(kAicpuAllReduceSoleMeshChunkTwoShotPatterns);
    INST_BELOW_L1_NHR_GATE(kAicpuAllreduceParallelMeshNHRPciePatterns);
    INST_BELOW_L1_NHR_GATE(kAicpuAllReducePipeLinePciePatterns);
    INST_BELOW_L1_NHR_GATE(kAicpuAllReduceSoleNHRAicpuReducePatterns);
    INST_BELOW_L1_NHR_GATE(kCcuSchedAllReduceSoleMeshPatterns);

    // A2A 全族 + RS DPU（链从不读实例数，level0Inst 条目一律禁止）
    NO_INST_CONSTRAINT(kAicpuAllToAllSoleMeshPatterns);
    NO_INST_CONSTRAINT(kAicpuAllToAllSoleMeshConcurrentPatterns);
    NO_INST_CONSTRAINT(kAicpuAllToAllSoleMeshUBXPatterns);
    NO_INST_CONSTRAINT(kCcuSchedAllToAllSoleMeshPatterns);
    NO_INST_CONSTRAINT(kCcuSchedAllToAllSoleMeshConcurrentPatterns);
    NO_INST_CONSTRAINT(kCcuSchedAllToAllMesh1DMultiJettyPatterns);
    NO_INST_CONSTRAINT(kAicpuAllToAllVSolePairwisePatterns);
    NO_INST_CONSTRAINT(kAicpuAllToAllVSoleMeshPatterns);
#if !defined(HCCL_CANN_COMPAT_850)
    NO_INST_CONSTRAINT(kAicpuAllToAllVSoleMeshConcurrentPatterns);
#endif
    NO_INST_CONSTRAINT(kAicpuAllToAllVSoleMeshMultiJettyPatterns);
    NO_INST_CONSTRAINT(kCcuSchedAllToAllVSoleMeshPatterns);
    NO_INST_CONSTRAINT(kCcuHalfAll2AllVMesh1DPatterns);
    NO_INST_CONSTRAINT(kAicpuAllToAllSoleMeshSingleChannelPatterns);
    NO_INST_CONSTRAINT(kInsReduceScatterSequenceMeshMeshDPUPatterns);
}

// 计数锚运行时复述：四族 55 数组（AG 15 / AR 9 / RS 18 / A2A 13，include 后实际清单
// 全列），N 与各 *alg_patterns.h 的 static_assert 一字不差。static_assert 已在本 TU 编译期
// 把关，运行期复述的增量价值：①单一清单面——新增数组漏挂 static_assert 时本清单可见缺口；
// ②条数与本文件模式遍历（PatternMatches 走注册表 patternCount）数值同源，防注册表侧
// patternCount 与数组实体漂移时静默越界/漏读
TEST(PatternDifferential, PatternArrayCountAnchors)
{
    // AG（15 数组）
    EXPECT_PATTERN_COUNT(kAicpuAllGatherSoleNHRPatterns, 9);
    EXPECT_PATTERN_COUNT(kInsAllGatherParallelMesh1DNHRPatterns, 2);
    EXPECT_PATTERN_COUNT(kAicpuAllGatherSequenceMeshConcurNHRPatterns, 1);
    EXPECT_PATTERN_COUNT(kAicpuAllGatherSoleNHRMultiLinkPatterns, 2);
    EXPECT_PATTERN_COUNT(kInsAllGatherMesh1DPatterns, 3);
    EXPECT_PATTERN_COUNT(kAicpuAllGatherConcurMeshNHRPatterns, 1);
    EXPECT_PATTERN_COUNT(kInsAllGatherParallelMesh1DNHRMultiJettyPatterns, 1);
    EXPECT_PATTERN_COUNT(kAicpuAllGatherPipeLineMeshNHRPatterns, 1);
    EXPECT_PATTERN_COUNT(kInsAllGatherParallelMesh1DNHRPciePatterns, 1);
    EXPECT_PATTERN_COUNT(kAicpuAllGatherPipeLinePciePatterns, 1);
    EXPECT_PATTERN_COUNT(kInsAllGatherMeshNhrDPUPatterns, 2);
    EXPECT_PATTERN_COUNT(kCcuSchedAllGatherSoleNHRPatterns, 7);
    EXPECT_PATTERN_COUNT(kCcuSchedAllGatherSoleMeshPatterns, 6);
    EXPECT_PATTERN_COUNT(kCcuSchedAllGatherConcurMeshNHRMultiLinkPatterns, 1);
    EXPECT_PATTERN_COUNT(kCcuSchedAllGatherParallelMeshNHRMultiLinkPatterns, 1);

    // AR（9 数组）
    EXPECT_PATTERN_COUNT(kAicpuAllReduceSoleNHRPatterns, 41);
    EXPECT_PATTERN_COUNT(kAicpuAllReduceParallelMeshNHRPatterns, 7);
    EXPECT_PATTERN_COUNT(kAicpuAllReduceSoleMeshOneShotPatterns, 7);
    EXPECT_PATTERN_COUNT(kAicpuAllReduceSoleMeshTwoShotPatterns, 11);
    EXPECT_PATTERN_COUNT(kAicpuAllReduceSoleMeshChunkTwoShotPatterns, 9);
    EXPECT_PATTERN_COUNT(kAicpuAllreduceParallelMeshNHRPciePatterns, 2);
    EXPECT_PATTERN_COUNT(kAicpuAllReducePipeLinePciePatterns, 2);
    EXPECT_PATTERN_COUNT(kAicpuAllReduceSoleNHRAicpuReducePatterns, 5);
    EXPECT_PATTERN_COUNT(kCcuSchedAllReduceSoleMeshPatterns, 10);

    // RS（18 数组）
    EXPECT_PATTERN_COUNT(kInsReduceScatterNHRPatterns, 63);
    EXPECT_PATTERN_COUNT(kInsReduceScatterParallelMesh1DNHRPatterns, 22);
    EXPECT_PATTERN_COUNT(kInsReduceScatterSequenceMesh1DNhrPatterns, 10);
    EXPECT_PATTERN_COUNT(kAicpuReduceScatterSoleNHRMultiLinkPatterns, 14);
    EXPECT_PATTERN_COUNT(kInsReduceScatterMesh1DPatterns, 29);
    EXPECT_PATTERN_COUNT(kAicpuReduceScatterSoleMeshChunkPatterns, 26);
    EXPECT_PATTERN_COUNT(kAicpuReduceScatterConcurMeshNHRPatterns, 2);
    EXPECT_PATTERN_COUNT(kAicpuReduceScatterParallelMeshNHRUBXPatterns, 1);
    EXPECT_PATTERN_COUNT(kInsReduceScatterParallelMesh1DNHRPciePatterns, 2);
    EXPECT_PATTERN_COUNT(kAicpuReduceScatterPipeLinePciePatterns, 2);
    EXPECT_PATTERN_COUNT(kInsReduceScatterAicpuReduceNHRPatterns, 4);
    EXPECT_PATTERN_COUNT(kInsReduceScatterSequenceMeshMeshDPUPatterns, 1);
    EXPECT_PATTERN_COUNT(kCcuSchedReduceScatterSoleNHRPatterns, 6);
    EXPECT_PATTERN_COUNT(kCcuSchedReduceScatterSoleMeshPatterns, 10);
    EXPECT_PATTERN_COUNT(kCcuSchedReduceScatterConcurMeshNHRMultiLinkPatterns, 1);
    EXPECT_PATTERN_COUNT(kCcuSchedReduceScatterSoleNHRMultiLinkPatterns, 3);
    EXPECT_PATTERN_COUNT(kInsReduceScatterMesh1DZAxisDetourPatterns, 1);
    EXPECT_PATTERN_COUNT(kCcuSchedReduceScatterSoleMeshPeerOnlyPatterns, 1);

    // A2A（13 数组，含 HalfA2AV 与手写 SingleChannel；850 宏守卫与头文件同款）
    EXPECT_PATTERN_COUNT(kAicpuAllToAllSoleMeshPatterns, 3);
    EXPECT_PATTERN_COUNT(kAicpuAllToAllSoleMeshConcurrentPatterns, 1);
    EXPECT_PATTERN_COUNT(kAicpuAllToAllSoleMeshUBXPatterns, 3);
    EXPECT_PATTERN_COUNT(kCcuSchedAllToAllSoleMeshPatterns, 4);
    EXPECT_PATTERN_COUNT(kCcuSchedAllToAllSoleMeshConcurrentPatterns, 1);
    EXPECT_PATTERN_COUNT(kCcuSchedAllToAllMesh1DMultiJettyPatterns, 3);
    EXPECT_PATTERN_COUNT(kAicpuAllToAllVSolePairwisePatterns, 8);
    EXPECT_PATTERN_COUNT(kAicpuAllToAllVSoleMeshPatterns, 36);
#if !defined(HCCL_CANN_COMPAT_850)
    EXPECT_PATTERN_COUNT(kAicpuAllToAllVSoleMeshConcurrentPatterns, 1);
#endif
    EXPECT_PATTERN_COUNT(kAicpuAllToAllVSoleMeshMultiJettyPatterns, 2);
    EXPECT_PATTERN_COUNT(kCcuSchedAllToAllVSoleMeshPatterns, 8);
    EXPECT_PATTERN_COUNT(kCcuHalfAll2AllVMesh1DPatterns, 2);
    EXPECT_PATTERN_COUNT(kAicpuAllToAllSoleMeshSingleChannelPatterns, 2);
}

// ---- selector 赋值行覆盖补向量（缺口见文件尾覆盖清单）----

// Nhr 门面与单实例面（基础矩阵只路过 Level1Nhr=true 的 A 面，Level0Nhr=true 与 [0]==1 的
// 多层面无向量）：Level0Nhr=true 钉 RS :437 / AG :266 两赋值行（Level1Nhr=false 前置，
// topoEq1 的 [0]==1 钉 RS :460/:461、RS CCU :263/:264、AG CCU :183/:184、AR :106/:107），
// topoL1nhr 的 RS·CCU 钉 :211/:212（A 面只跑过 AICPU 侧的 :433/:434）
TEST(PatternDifferential, NhrGateAndSingleInstanceFaces)
{
    // Level1Nhr=true 两层 Mesh × RS·CCU：CcuSched 链 :202(<32MB) → :211 → :212 SoleNHR
    const TopoInfoWithNetLayerDetails topoL1nhr = MakeTwoLevelMeshTopo(32U, true, 8U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::CCU, topoL1nhr, k4MB, "nhr·L1·RS·CCU·4MB");

    // Level0Nhr=true（Level1Nhr=false）：RS :436/:437 与 AG :265/:266 的 Level0Nhr 支
    TopoInfoWithNetLayerDetails topoL0nhr = MakeTwoLevelMeshTopo(32U, false, 8U);
    topoL0nhr.Level0Nhr = true;
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, topoL0nhr, k4MB, "nhr·L0·RS·AICPU·4MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, topoL0nhr, k256KB, "nhr·L0·AG·AICPU·256KB");

    // [0]==1 单实例两层 Mesh（无 2die）：四个族的实例数 == 1 支
    const TopoInfoWithNetLayerDetails topoEq1 = MakeTwoLevelMeshTopo(32U, false, 1U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, topoEq1, k4MB, "eq1·RS·AICPU·4MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::CCU, topoEq1, k4MB, "eq1·RS·CCU·4MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::CCU, topoEq1, k256KB, "eq1·AG·CCU·256KB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, topoEq1, k256KB, "eq1·AR·AICPU·256KB");

    // frameNum>16 回退支（rank 33..64 档）：AG CCU 链 frameNum = rank/gcd(instSize)，rank=64 且
    // [0]=2 → frameNum=32 > 16，依次否决 :204/:208/:214 三支后落 :220→:224 的 SoleNHR 回退；
    // rank<=64 过 :169 门（64 是含点）——rank 扫描的 1024 档被该门否决故本支需专用向量
    const TopoInfoWithNetLayerDetails topoFrame32 = MakeTwoLevelMeshTopo(64U, false, 2U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::CCU, topoFrame32, k256KB, "frame32·AG·CCU·256KB");
}

// 多层 MESH_1D 分档面（基础矩阵的多层向量均落 CLOS/两层数据档，3 层 else 支、≥4 层支与
// RS 4MB 分档 ternary 双臂无向量）：mesh3L 钉 RS :443→:447、AG :272→:273、AR :109→:117 的
// 3 层 else；mesh3LSym 钉 AR :100→:101（3 层对称 uboe 的 Parallel 支）；mesh4L 钉 AR :132→:133；
// 2 层 Mesh 的 8MB/256MB/64MB 分别钉 RS :453→:456（Parallel 臂）、RS :454→:455（Sequence 臂，
// 总量 256MB×32=8GB > 4GB）、AR :119→:128（>32MB 且总量 ≤ 4GB 的 Parallel 支）
TEST(PatternDifferential, MultiLevelMeshLevelBands)
{
    const TopoInfoWithNetLayerDetails mesh3L = MakeMultiLevelMeshTopo(32U, 3U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, mesh3L, k1MB, "mesh·3L·RS·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, mesh3L, k256KB, "mesh·3L·AG·256KB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, mesh3L, k256KB, "mesh·3L·AR·256KB");

    // 3 层两面对称 + uboe + 每模块 4 卡（≠8 不进 pipeline 未迁移支）：AR 的 ParallelMeshNHR 3 层支
    TopoInfoWithNetLayerDetails mesh3LSym = MakeMultiLevelMeshTopo(32U, 3U);
    mesh3LSym.level0Symmetric = true;
    mesh3LSym.level1Symmetric = true;
    mesh3LSym.topLevelUboe = true;
    mesh3LSym.deviceNumPerModule = 4U;
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, mesh3LSym, k256KB, "mesh·3L·sym·AR·256KB");

    const TopoInfoWithNetLayerDetails mesh4L = MakeMultiLevelMeshTopo(32U, 4U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, mesh4L, k256KB, "mesh·4L·AR·256KB");

    // 两层 Mesh（[0]=8>1）数据分档：RS 的 4MB 门槛两侧 ternary 双臂 + AR 的 32MB 大数据支
    const TopoInfoWithNetLayerDetails mesh2Lr8 = MakeTwoLevelMeshTopo(8U, false, 8U);
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, mesh2Lr8, k8MB, "mesh·2L·r8·RS·8MB");
    constexpr u64 kLen256MB = 256ULL * 1024 * 1024;
    constexpr u64 kLen64MB = 64ULL * k1MB;
    const TopoInfoWithNetLayerDetails mesh2Lr32 = MakeTwoLevelMeshTopo(32U, false, 8U);
    ExpectDifferential(
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, mesh2Lr32, kLen256MB, "mesh·2L·r32·RS·256MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, mesh2Lr32, kLen64MB, "mesh·2L·r32·AR·64MB");
}

// UBX（单层 MESH_1D_CLOS）引擎面（基础矩阵无 RS/AR 的 UBX 向量，AG 只盖 ShapeC 的 meshEq 两档）：
// ubxEqual4 钉 RS CCU :300→:303/:307、RS AICPU :597→:599-:606、AG CCU :85→:87、AR AICPU :164-:169
// 与 AR CCU :545→:547；ubxEqual1 钉 RS :603→:604（r=1 时 256KB..512KB 的 Chunk 区间，
// ratio=(8/1)²=64 使 300KB×64 > 16MB 而 IsSmallData 成立——r>=2 时两条件互斥不可达）；
// ubxNonEq（mesh=3/clos=8 不整除）钉 RS CCU :311→:319、RS AICPU :608/:609 与 :617→:618、
// AG :325→:326、AR :179→:180 与 :199→:201；ubxMult（mesh=2/clos=8 整除）钉 RS :610→:615、
// AG :321→:323 双臂（MultiJetty/PipeLineMeshNHR）、AG CCU :90→:91；PCIe 两态钉 RS :592→:593/
// :594→:595、AG :297→:298 与 :300/:301、AR :269→:270-:278（小/中/大覆盖 :274/:276/:278 与
// special ternary :271-:272）及 :280→:290/:291
TEST(PatternDifferential, UbxEngineFaces)
{
    const TopoInfoWithNetLayerDetails ubxEqual4 = MakeUbxEqualTopo(4U);
    const TopoInfoWithNetLayerDetails ubxEqual1 = MakeUbxEqualTopo(1U);
    const TopoInfoWithNetLayerDetails ubxNonEq = MakeUbxCountTopo(4U, 3U, 8U, false);
    const TopoInfoWithNetLayerDetails ubxMult = MakeUbxCountTopo(4U, 2U, 8U, false);
    const TopoInfoWithNetLayerDetails ubxPcieConn = MakeUbxCountTopo(4U, 4U, 4U, true);
    const TopoInfoWithNetLayerDetails ubxPcieNonConn = MakeUbxCountTopo(4U, 3U, 8U, true);
    constexpr u64 kLen300KB = 300ULL * 1024;
    constexpr u64 kLen8MBv = 8ULL * k1MB;
    constexpr u64 kLen16MBv = 16ULL * k1MB;
    constexpr u64 kLen32MBv = 32ULL * k1MB;
    constexpr u64 kLen64MBv = 64ULL * k1MB;
    constexpr u64 kLen128MBv = 128ULL * k1MB;

    // RS·CCU：4P meshEq 的小/大数据两支与矩形余支
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::CCU, ubxEqual4, k256KB, "ubx·eq4·RS·CCU·256KB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::CCU, ubxEqual4, k1MB, "ubx·eq4·RS·CCU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::CCU, ubxNonEq, k256KB, "ubx·neq·RS·CCU·256KB");
    // RS·AICPU：全互联面（小/大/r=1 区间）与未全互联面（特殊/矩形大数据/余支）与 PCIe 面
    ExpectDifferential(
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, ubxEqual4, k256KB, "ubx·eq4·RS·AICPU·256KB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, ubxEqual4, k1MB, "ubx·eq4·RS·AICPU·1MB");
    ExpectDifferential(
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, ubxEqual1, kLen300KB, "ubx·eq1·RS·AICPU·300KB");
    ExpectDifferential(
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, ubxNonEq, k256KB, "ubx·neq·RS·AICPU·256KB");
    ExpectDifferential(
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, ubxMult, 2U * k1MB, "ubx·mult·RS·AICPU·2MB");
    ExpectDifferential(
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, ubxPcieNonConn, k1MB, "ubx·pcnc·RS·AICPU·1MB");
    ExpectDifferential(
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, ubxPcieNonConn, kLen8MBv, "ubx·pcnc·RS·AICPU·8MB");

    // AG·CCU/AG·AICPU：sched 的 meshEq/矩形大数据两支、AICPU 的矩形分档双臂与余支、PCIe 双态
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::CCU, ubxEqual4, k1MB, "ubx·eq4·AG·CCU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::CCU, ubxMult, k1MB, "ubx·mult·AG·CCU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, ubxMult, k1MB, "ubx·mult·AG·AICPU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, ubxMult, kLen32MBv, "ubx·mult·AG·AICPU·32MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, ubxNonEq, k256KB, "ubx·neq·AG·AICPU·256KB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, ubxPcieConn, k1MB, "ubx·pcc·AG·AICPU·1MB");
    ExpectDifferential(
        HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, ubxPcieNonConn, k1MB, "ubx·pcnc·AG·AICPU·1MB");
    ExpectDifferential(
        HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, ubxPcieNonConn, kLen8MBv, "ubx·pcnc·AG·AICPU·8MB");

    // AR·AICPU：meshEq 4P 的 OneShot/TwoShot 双 dtype 双档、矩形余支双 dtype、PCIe 全互联三档
    // （256KB→:271/:274、16MB→:272/:278、128MB→:276 Chunk）与 PCIe 未全互联两档（Pcie/PipeLine）
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, ubxEqual4, k256KB, "ubx·eq4·AR·AICPU·256KB");
    ExpectDifferential(
        HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, ubxEqual4, kLen16MBv, "ubx·eq4·AR·AICPU·16MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, ubxNonEq, k256KB, "ubx·neq·AR·AICPU·256KB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, ubxMult, k256KB, "ubx·mult·AR·AICPU·256KB");
    ExpectDifferential(
        HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, ubxPcieConn, k256KB, "ubx·pcc·AR·AICPU·256KB");
    ExpectDifferential(
        HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, ubxPcieConn, kLen16MBv, "ubx·pcc·AR·AICPU·16MB");
    ExpectDifferential(
        HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, ubxPcieConn, kLen128MBv, "ubx·pcc·AR·AICPU·128MB");
    ExpectDifferential(
        HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, ubxPcieNonConn, k1MB, "ubx·pcnc·AR·AICPU·1MB");
    ExpectDifferential(
        HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, ubxPcieNonConn, kLen64MBv, "ubx·pcnc·AR·AICPU·64MB");
    // AR·CCU：UBX 4P 小数据 SoleMesh 支
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::CCU, ubxEqual4, k256KB, "ubx·eq4·AR·CCU·256KB");
    // PCIe 全互联分发行（非赋值行，下游赋值行已各向量覆盖，此处收口分发行本身）：
    // RS AICPU ForMesh1DClos :586→:587 复用 ForMesh1D（Mesh1D 主体条目已盖）、RS CCU :346→:348
    // 复用 SelectMeshAlgoCcuScheduleMesh1D（SoleMesh 一层公式条目已盖）、AR CCU :455→:456
    // 复用 SelectCcuScheduleLevel0AlgoMesh1D（CcuSoleMesh 一层公式条目已盖）
    ExpectDifferential(
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, ubxPcieConn, k1MB, "ubx·pcc·RS·AICPU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::CCU, ubxPcieConn, k1MB, "ubx·pcc·RS·CCU·1MB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::CCU, ubxPcieConn, k256KB, "ubx·pcc·AR·CCU·256KB");
}

// 单层 CLOS / 单层 Mesh 引擎面与 A2A/A2AV/HalfA2AV 的入口面（基础矩阵的 A2A 向量只盖
// MESH_1D_CLOS 非 PCIe 的 Concurrent/UBX 分叉，A2AV/Half 全族无向量）：clos1 钉各族单层
// CLOS 支（RS CCU :354→:359、RS AICPU :491→:493/:495、AG CCU :110→:115、AG AICPU :328→:329、
// AR AICPU :258→:260、A2A :146→:148 与 CCU :124→:126、A2AV :50→:51 与 CCU :181→:183）；
// flat8 钉 A2A/A2AV 的 MESH_1D 入口（:146/:50→SoleMesh；CCU :99→:117、:158→:177）与
// HalfA2AV :41-:55（rank<=8）；flatSd（SINGLE_DIE）钉 AG CCU :122→:129；pairwise32 钉
// A2AV :45→:46（多层对称 8 卡 16 倍数 rank）；mesh2Lr2 钉 A2AV CCU 多层 :145→:146；
// ubxEqual4/8 钉 A2AV :77→:78/:80；ubxPcieConn 钉 A2A :153→:154、A2A CCU :60→:69、
// A2AV :56→:57、A2AV CCU :190→:199
TEST(PatternDifferential, OneLevelClosAndA2AFaces)
{
    const TopoInfoWithNetLayerDetails clos1 = MakeOneLevelClosTopo(8U);
    const TopoInfoWithNetLayerDetails flat8 = MakeFlatMesh1DTopo(8U);
    TopoInfoWithNetLayerDetails flatSd = MakeFlatMesh1DTopo(8U);
    flatSd.level0MeshType = Level0MeshType::SINGLE_DIE;
    TopoInfoWithNetLayerDetails pairwise32 = MakeTwoLevelMeshTopo(32U, false, 8U);
    pairwise32.level0Symmetric = true;
    pairwise32.deviceNumPerModule = 8U;
    const TopoInfoWithNetLayerDetails mesh2Lr2 = MakeTwoLevelMeshTopo(2U, false, 8U);
    const TopoInfoWithNetLayerDetails ubxEqual4 = MakeUbxEqualTopo(4U);
    const TopoInfoWithNetLayerDetails ubxEqual8 = MakeUbxEqualTopo(8U);
    const TopoInfoWithNetLayerDetails ubxPcieConn = MakeUbxCountTopo(4U, 4U, 4U, true);

    // 单层 CLOS：四族 AICPU/CCU 的 CLOS 支（AR AICPU 正常类型 NOT_MATCH——SoleNHRMultiLink
    // 未迁移，双 dtype 并集只剩 special 的 AicpuReduce）
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::CCU, clos1, k256KB, "clos1·RS·CCU·256KB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AlgEngine::AICPU, clos1, k256KB, "clos1·RS·AICPU·256KB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::CCU, clos1, k256KB, "clos1·AG·CCU·256KB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::AICPU, clos1, k256KB, "clos1·AG·AICPU·256KB");
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLREDUCE, AlgEngine::AICPU, clos1, k256KB, "clos1·AR·AICPU·256KB");

    // 单层 Mesh：AG CCU 的 SINGLE_DIE 支（NOT_MESH 态被 :122 拒绝是已钉的并集机制）
    ExpectDifferential(HcclCMDType::HCCL_CMD_ALLGATHER, AlgEngine::CCU, flatSd, k256KB, "flatSd·AG·CCU·256KB");

    // A2A/A2AV/HalfA2AV 入口面：A2AV selector 不读长度，elemCount 只进 sendCounts[0]（无消费）
    ExpectA2ADifferential(
        HcclCMDType::HCCL_CMD_ALLTOALL, AlgEngine::AICPU, flat8, 400ULL, HCCL_DATA_TYPE_INT8, "a2a·flat8·AICPU");
    ExpectA2ADifferential(
        HcclCMDType::HCCL_CMD_ALLTOALL, AlgEngine::CCU, flat8, 400ULL, HCCL_DATA_TYPE_INT8, "a2a·flat8·CCU");
    ExpectA2ADifferential(
        HcclCMDType::HCCL_CMD_ALLTOALL, AlgEngine::AICPU, clos1, 400ULL, HCCL_DATA_TYPE_INT8, "a2a·clos1·AICPU");
    ExpectA2ADifferential(
        HcclCMDType::HCCL_CMD_ALLTOALL, AlgEngine::CCU, clos1, 400ULL, HCCL_DATA_TYPE_INT8, "a2a·clos1·CCU");
    ExpectA2ADifferential(
        HcclCMDType::HCCL_CMD_ALLTOALL, AlgEngine::AICPU, ubxPcieConn, 400ULL, HCCL_DATA_TYPE_INT8, "a2a·pcc·AICPU");
    ExpectA2ADifferential(
        HcclCMDType::HCCL_CMD_ALLTOALL, AlgEngine::CCU, ubxPcieConn, 400ULL, HCCL_DATA_TYPE_INT8, "a2a·pcc·CCU");
    ExpectA2ADifferential(
        HcclCMDType::HCCL_CMD_ALLTOALLV, AlgEngine::AICPU, pairwise32, 400ULL, HCCL_DATA_TYPE_FP16, "a2av·pw32·AICPU");
    ExpectA2ADifferential(
        HcclCMDType::HCCL_CMD_ALLTOALLV, AlgEngine::AICPU, flat8, 400ULL, HCCL_DATA_TYPE_FP16, "a2av·flat8·AICPU");
    ExpectA2ADifferential(
        HcclCMDType::HCCL_CMD_ALLTOALLV, AlgEngine::AICPU, clos1, 400ULL, HCCL_DATA_TYPE_FP16, "a2av·clos1·AICPU");
    ExpectA2ADifferential(
        HcclCMDType::HCCL_CMD_ALLTOALLV, AlgEngine::AICPU, ubxEqual4, 400ULL, HCCL_DATA_TYPE_FP16, "a2av·eq4·AICPU");
    ExpectA2ADifferential(
        HcclCMDType::HCCL_CMD_ALLTOALLV, AlgEngine::AICPU, ubxEqual8, 400ULL, HCCL_DATA_TYPE_FP16, "a2av·eq8·AICPU");
    ExpectA2ADifferential(
        HcclCMDType::HCCL_CMD_ALLTOALLV, AlgEngine::AICPU, ubxPcieConn, 400ULL, HCCL_DATA_TYPE_FP16, "a2av·pcc·AICPU");
    ExpectA2ADifferential(
        HcclCMDType::HCCL_CMD_ALLTOALLV, AlgEngine::CCU, mesh2Lr2, 400ULL, HCCL_DATA_TYPE_FP16, "a2av·2Lr2·CCU");
    ExpectA2ADifferential(
        HcclCMDType::HCCL_CMD_ALLTOALLV, AlgEngine::CCU, flat8, 400ULL, HCCL_DATA_TYPE_FP16, "a2av·flat8·CCU");
    ExpectA2ADifferential(
        HcclCMDType::HCCL_CMD_ALLTOALLV, AlgEngine::CCU, clos1, 400ULL, HCCL_DATA_TYPE_FP16, "a2av·clos1·CCU");
    ExpectA2ADifferential(
        HcclCMDType::HCCL_CMD_ALLTOALLV, AlgEngine::CCU, ubxPcieConn, 400ULL, HCCL_DATA_TYPE_FP16, "a2av·pcc·CCU");
    ExpectA2ADifferential(
        HcclCMDType::HCCL_CMD_HALF_ALLTOALLV, AlgEngine::CCU, flat8, 400ULL, HCCL_DATA_TYPE_FP16, "half·flat8·CCU");
}

// ---- 分派表完备性 ----

// 同二进制 external_alg_testcase.cc 的两条 KFC 测试夹具注册行（:921 TestPrioMacro /
// :935 TestPatternMacro——验证 priority 落库与模式宏机制，刻意借 KFC cmd 避开满射断言的
// COLL_CMDS）——非产品注册面、无差分矩阵语义，对账排除；夹具名变更时本 TEST 报错提示同步
const std::set<std::string> kTestFixtureNames = {
    "TestPrioMacro",
    "TestPatternMacro",
};

// 注册表驱动的分派对账：凡 AlgMetaRegistry 注册过的 (cmd, engine) 维度，harness 侧
// MakeSelector 必须能构造 selector、EngineFunctions 必须有可跑的引擎虚函数面。两个分派
// 表是手写 switch 映射，新增注册维度若漏改任一函数，差分矩阵该维度将静默双空集
// （expected == actual == {} 恒绿假象）——本 TEST 把遗漏从恒绿假象升级为对账报错，与
// 分派表注释区的更新义务锚互为印证
TEST(PatternDifferential, DispatchTableCoversRegisteredDimensions)
{
    std::set<HcclCMDType> cmds;
    std::set<AlgEngine> engines;
    for (const AlgMetaRow& row : AlgMetaRegistry::Instance().GetAll()) {
        if (kTestFixtureNames.count(row.registeredName) != 0) {
            continue;
        }
        cmds.insert(row.cmd);
        engines.insert(row.meta.engine);
    }
    for (const HcclCMDType cmd : cmds) {
        EXPECT_NE(MakeSelector(cmd), nullptr) << "cmd[" << cmd << "] 有注册算法，MakeSelector 缺分派";
    }
    for (const AlgEngine engine : engines) {
        EXPECT_FALSE(EngineFunctions(engine).empty())
            << "engine[" << static_cast<int>(engine) << "] 有注册算法，EngineFunctions 缺分派";
    }
}

// 登记门校验（D17）：每条注册 pattern 的 len 子句合取在"声明 rank 域内至少一个代表 r 下"
// 解集须非空——对所有可行 r 皆空（如手误 GT 4MB ∧ LT 4MB 的常数矛盾）即病态合取，当场红。
// 语义边界：selector 链存在对特定 r 自然掏空的忠实转录（AR TwoShot r=1 档、RS Chunk 相交支
// r≥19 档（r³·256KB>1536MB 分界；r≥32 是 AR ChunkTwoShot 的 4GB 分界）——分支死亡被如实
// 镜像，非病态），故断言取"∃r 非空"而非"∀r 非空"；dtypeSize 取
// 正值 4（现网 ELEMENT 子句均为单条无合取，任意正 dtypeSize 下解集非空，参与求解即可）
TEST(PatternDifferential, RegisteredLenConjunctionsSatisfiableSomewhere)
{
    const std::vector<AlgMetaRow> rows = AlgMetaRegistry::Instance().GetAll();
    ASSERT_FALSE(rows.empty());
    for (const AlgMetaRow& row : rows) {
        for (u32 i = 0; i < row.meta.patternCount; ++i) {
            const TopoPattern& pat = row.meta.patterns[i];
            bool satisfiable = false;
            // 候选 r = 声明域端点 ∪ 落入域内的代表 r（端点兜底：声明域不含任何代表 r 时仍可测）
            u32 candidates[2U + sizeof(kGateRanks) / sizeof(kGateRanks[0])];
            size_t candidateCount = 0;
            candidates[candidateCount++] = pat.rankSize.min;
            candidates[candidateCount++] = pat.rankSize.max;
            for (const u32 r : kGateRanks) {
                if (r >= pat.rankSize.min && r <= pat.rankSize.max) {
                    candidates[candidateCount++] = r;
                }
            }
            for (size_t j = 0; j < candidateCount && !satisfiable; ++j) {
                uint64_t lo = 0;
                uint64_t hi = 0;
                satisfiable = ValidatePattern(pat, MakeBareRankTopo(candidates[j]), lo, hi, 4U);
            }
            EXPECT_TRUE(satisfiable) << "registeredName=" << row.registeredName << " pattern[" << i
                                     << "] len 子句在声明 rank 域内无任何非空解（病态合取）";
        }
    }
}

// ============================================================================
// selector 赋值行覆盖清单（与差分向量对照的活账本）：
// 赋值行 + AlgoNotMigrated 名；覆盖义务 = "已注册算法名的 MATCH 赋值行"至少一个差分
// 向量命中；未注册名/未迁移名不属于模式义务——预期集合在 SelectorPicks 内被注册表
// 过滤，模式侧无行可比对。判定证据 = build_gcov 覆盖构建逐 TEST 过滤运行 + gcov 行
// 覆盖（63 用例全绿后采集；三元赋值语句的计数器可能落在条件行或字符串臂行，按
// 语句首行 ±2 行归并）。行号 = 各 selector .cc 文件当前行号。
//
// 向量标签对照（TEST 名 → 短名）：
//   ShapeA=ShapeA_TwoLevelMeshLevel1Nhr        ShapeB=ShapeB_TwoLevelMeshRank2048
//   ShapeC=ShapeC_OneLevelUbxMeshEqRank4       ShapeD=ShapeD_OneLevelUbxDegenerate
//   ShapeE/F=ShapeEF_TwoLevelMeshTwoDieFullMesh EngineLenCombos=EngineLenCombos_FlatMesh1D
//   ErrataArMesh=ErrataArMeshFormulaBoundaries ErrataAr2L=ErrataArTwoLevelDetourAnd4gbBoundary
//   ErrataRs=ErrataRsMeshBoundaries            ErrataCcu=ErrataCcuFormulaBoundaries
//   RankScan=RankSizeBoundaryScanTwoLevelMesh  ZeroInst=ZeroInstNumDegenerateShapes
//   A2A边界=AllToAllElementByteBoundaries       ShapeD+=ShapeDMoreDegenerateCheckShapes
//   ClosEF=MultiLevelClosExcludeEFPaths         NhrFaces/MeshBands/UbxFaces/ClosA2A=补充 TEST
//
// 末列图例（账本统一）：义务行与豁免行同轨——凡被差分向量路过（定向或顺带）一律
// 行内列出向量短名；"-" = 无任何向量覆盖（gcov 隔离运行逐 TEST 归因实证），豁免 "-"
// 现共 25 行、全部真未覆盖，原 5 行顺带路过（RS:231/240/244←RankScan、427/429←ClosEF）
// 已行内归位。L422 行向量 7 项超行宽，压缩列距保"一行一记录"。豁免行成因与
// 未覆盖原因（gcov 行覆盖实证，非推断）：
//   ① 未注册名（55 行 sidecar 无此行）：RS 的 CcuMs 死区四名（CcuReduceScatterMesh1D/
//     ConcurrentMeshNHRMs/Mesh2Die）、AR CCU 的 CcuSchedAllReduceSoleNHR/SequenceMeshMesh/
//     ParallelMeshNHR/SoleMesh2Die/ConcurMeshNHRMultiLink/SoleNHRMultiLink 系、A2A 的
//     CcuSchedAllToAllSoleMeshMultiLink/2Die/Concur、A2AV 的 CcuSchedAllToAllV* 未注册四名
//     ——selector 设名后 NOT_MATCH，ExecutorPicks 过滤后无模式义务（部分行被路过向量
//     覆盖属顺带，非义务）；
//   ② AlgoNotMigrated 未迁移名：RS 的 SoleMesh2Die/SequenceMeshMesh/ParallelMeshNHR/
//     ParallelMeshNHRMultiLink/PipeLineMeshNHR×3/StrictOrderedMesh/SequenceMeshConcurNHRNHR/
//     SoleMeshConcur/SequenceMeshNHRAicpuReduce（hccl 有分支、devkit 无注册面）；
//   ③ AIV 引擎行（RS:529 AivReduceScatterMesh1D / AG:347 AivAllGatherMesh1D）：名未注册
//     且差分 harness 的 EngineFunctions 无 AIV 面（AICPU/CCU/DPU 三引擎）；
//   ④ AG:66 CcuSchedAllGatherSoleMesh——SelectMeshAlgo 全仓无调用方（死方法，引擎分派
//     只走 SelectAicpuAlgo/SelectCcuScheduleAlgo/SelectCcuMsAlgo/SelectDPUAlgo）；
//   ⑤ AG:200 CcuSchedAllGatherSoleNHR——不可达支：:187 rank<=8 前置与 frameNum>16 互斥
//     （frameNum = rank/gcd ≤ rank ≤ 8 < 17；gcd 空时 CalcFrameNum 早退 0，auto_selector_base.cc
//     :380-402）；
//   ⑥ AG:200 之外的缺口面：A2AV 全族原先零向量（四个新 TEST 补齐 13 行义务）、
//     RS/AR 的 UBX 面与 PCIe 面、各族单层 CLOS 面、RS 2L >4MB ternary 双臂、3L/≥4L Mesh 面、
//     AG CCU frameNum>16 回退支（frame32 向量）与 AG UBX sched 双支、HalfA2AV。
//
// PeerOnly 正向确认（spec §7.2 手写变体，kHandwrittenVariants 无差分 oracle）：UT 层正向
// 已存在——external_alg_testcase.cc:496 CondPeerOnlySingleCtpChannelHits 与 :506
// CondPeerOnlyCrossLayerSingleCtpChannelHits（MakeSimComm 建链查询基建在该目标可跑，
// EXPECT_TRUE 正向钉死"恰 1 条 CCU 协议通道"量纲）+ :895-:912 SpecificityOrdering 的注册
// 排序面（cands[0] = CcuSchedReduceScatterSoleMeshPeerOnly）；ST 层两例已钉
// （reduce_scatter_testcase.cpp:363/:373 forced-alg 经 pattern+COND 定名解析到 PeerOnly）。
// 差分层维持排除（人工登记拓扑面，selector 永不自动选中，无漂移语义）。
// ============================================================================
// ---- RS（reduce_scatter_auto_selector.cc）----
// 豁免|L106 | CcuReduceScatterMesh2Die(未注册)                              | -
// 豁免|L111 | CcuReduceScatterMesh1D(未注册)                                | ErrataCcu
// 豁免|L127 | CcuReduceScatterMesh1D(未注册)                                | UbxFaces
// 豁免|L130 | CcuReduceScatterConcurrentMeshNHRMs(未注册)                   | UbxFaces
// 豁免|L136 | CcuReduceScatterMesh1D(未注册)                                | UbxFaces
// 义务|L212 | CcuSchedReduceScatterSoleNHR                               | NhrFaces
// 豁免|L231 | CcuSchedReduceScatterSoleMesh2Die(未迁移)                     | RankScan
// 义务|L235 | CcuSchedReduceScatterSoleMesh                              | RankScan
// 豁免|L240 | CcuSchedReduceScatterSequenceMeshMesh(未迁移)                 | RankScan
// 豁免|L244 | CcuSchedReduceScatterSequenceMeshMesh(未迁移)                 | RankScan
// 豁免|L248 | CcuSchedReduceScatterParallelMeshNHR(未迁移)                  | -
// 义务|L254 | CcuSchedReduceScatterSoleNHR                               | RankScan
// 义务|L264 | CcuSchedReduceScatterSoleNHR                               | NhrFaces
// 义务|L268 | CcuSchedReduceScatterSoleNHR                               | ClosEF
// 义务|L303 | CcuSchedReduceScatterSoleMesh                              | UbxFaces
// 义务|L307 | CcuSchedReduceScatterConcurMeshNHRMultiLink                | UbxFaces
// 义务|L319 | CcuSchedReduceScatterSoleNHRMultiLink                      | UbxFaces
// 义务|L359 | CcuSchedReduceScatterSoleNHR                               | ClosA2A
// 义务|L399 | CcuSchedReduceScatterSoleMesh                              | ErrataCcu,UbxFaces
// 义务|L422 | InsReduceScatterAicpuReduceNHR | ClosEF,MeshBands,NhrFaces,RankScan,ShapeA,ShapeB,ZeroInst
// 豁免|L427 | AicpuReduceScatterPipeLineMeshNHR(未迁移)                    | ClosEF
// 豁免|L429 | AicpuReduceScatterPipeLineMeshNHRMesh(未迁移)                 | ClosEF
// 义务|L432 | InsReduceScatterParallelMesh1DNHR                          | ClosEF,ShapeA
// 义务|L434 | InsReduceScatterNHR                                        | ShapeA
// 义务|L437 | InsReduceScatterNHR                                        | NhrFaces
// 豁免|L445 | AicpuReduceScatterSequenceMeshConcurNHRNHR(未迁移)            | -
// 义务|L447 | InsReduceScatterNHR                                        | MeshBands
// 义务|L452 | InsReduceScatterParallelMesh1DNHR                          | MeshBands,ShapeB
// 义务|L454 | InsReduceScatterSequenceMesh1DNhr|InsReduceScatterParallel | MeshBands
// 义务|L455 | InsReduceScatterSequenceMesh1DNhr                          | MeshBands
// 义务|L456 | InsReduceScatterParallelMesh1DNHR                          | MeshBands,RankScan
// 义务|L458 | InsReduceScatterNHR                                        | RankScan
// 义务|L461 | InsReduceScatterNHR                                        | ClosEF,NhrFaces
// 义务|L463 | AicpuReduceScatterSoleNHRMultiLink                         | ClosEF
// 义务|L493 | InsReduceScatterAicpuReduceNHR                             | ClosA2A
// 义务|L495 | AicpuReduceScatterSoleNHRMultiLink                         | ClosA2A
// 豁免|L529 | AivReduceScatterMesh1D(AIV,未注册)                            | -
// 义务|L542 | InsReduceScatterSequenceMeshMeshDPU                        | RankScan,ShapeA,ZeroInst
// 义务|L555 | InsReduceScatterMesh1D                                     | EngineLenCombos,ErrataRs,UbxFaces
// 豁免|L560 | AicpuReduceScatterSoleMeshConcur(未迁移)                      | ErrataRs
// 义务|L562 | AicpuReduceScatterSoleMeshChunk                            | ErrataRs
// 义务|L564 | AicpuReduceScatterSoleMeshChunk                            | ErrataRs
// 义务|L566 | InsReduceScatterMesh1D                                     | ErrataRs
// 义务|L570 | AicpuReduceScatterSoleMeshChunk                            | EngineLenCombos,ErrataRs,UbxFaces
// 义务|L572 | InsReduceScatterMesh1D                                     | EngineLenCombos,ErrataRs,UbxFaces
// 豁免|L590 | AicpuReduceScatterSequenceMeshNHRAicpuReduce(未迁移)          | UbxFaces
// 义务|L593 | InsReduceScatterParallelMesh1DNHRPcie                      | UbxFaces
// 义务|L595 | AicpuReduceScatterPipeLinePcie                             | UbxFaces
// 义务|L600 | InsReduceScatterMesh1D                                     | UbxFaces
// 义务|L602 | AicpuReduceScatterConcurMeshNHR                            | UbxFaces
// 义务|L604 | AicpuReduceScatterSoleMeshChunk                            | UbxFaces
// 义务|L606 | InsReduceScatterMesh1D                                     | UbxFaces
// 义务|L609 | InsReduceScatterAicpuReduceNHR                             | UbxFaces
// 豁免|L613 | AicpuReduceScatterPipeLineMeshNHR(未迁移)                     | -
// 义务|L615 | AicpuReduceScatterParallelMeshNHRUBX                       | UbxFaces
// 义务|L618 | InsReduceScatterNHR                                        | UbxFaces
// ---- AG（all_gather_auto_selector.cc）----
// 义务|L46  | CcuSchedAllGatherSoleMesh                                  | ClosA2A,EngineLenCombos
// 豁免|L66  | CcuSchedAllGatherSoleMesh(死方法)                             | -
// 义务|L87  | CcuSchedAllGatherConcurMeshNHRMultiLink                    | ShapeC,UbxFaces
// 义务|L91  | CcuSchedAllGatherParallelMeshNHRMultiLink                  | UbxFaces
// 义务|L99  | CcuSchedAllGatherSoleMesh                                  | ShapeC
// 义务|L115 | CcuSchedAllGatherSoleNHR                                   | ClosA2A
// 义务|L129 | CcuSchedAllGatherSoleMesh                                  | ClosA2A
// 义务|L175 | CcuSchedAllGatherSoleNHR                                   | ShapeA,ShapeE/F
// 义务|L184 | CcuSchedAllGatherSoleNHR                                   | NhrFaces
// 义务|L189 | CcuSchedAllGatherSoleMesh                                  | RankScan
// 豁免|L200 | CcuSchedAllGatherSoleNHR(不可达)                              | -
// 义务|L205 | CcuSchedAllGatherSoleMesh                                  | ZeroInst
// 义务|L224 | CcuSchedAllGatherSoleNHR                                   | NhrFaces
// 义务|L227 | CcuSchedAllGatherSoleNHR                                   | ClosEF
// 义务|L263 | AicpuAllGatherSoleNHR                                      | ShapeA
// 义务|L266 | AicpuAllGatherSoleNHR                                      | NhrFaces,ShapeE/F
// 义务|L268 | AicpuAllGatherSoleNHR                                      | ShapeE/F
// 义务|L273 | AicpuAllGatherSoleNHR                                      | MeshBands
// 义务|L277 | InsAllGatherParallelMesh1DNHR                              | ShapeB,ZeroInst
// 义务|L279 | AicpuAllGatherSequenceMeshConcurNHR|InsAllGatherParallelMe | ZeroInst
// 义务|L283 | AicpuAllGatherSoleNHR                                      | RankScan,ZeroInst
// 义务|L286 | AicpuAllGatherSoleNHRMultiLink                             | ClosEF
// 义务|L293 | InsAllGatherMesh1D                                         | EngineLenCombos
// 义务|L298 | InsAllGatherMesh1D                                         | UbxFaces
// 义务|L300 | InsAllGatherParallelMesh1DNHRPcie|AicpuAllGatherPipeLinePc | UbxFaces
// 义务|L317 | AicpuAllGatherConcurMeshNHR                                | ShapeC
// 义务|L319 | InsAllGatherMesh1D                                         | ShapeC
// 义务|L323 | InsAllGatherParallelMesh1DNHRMultiJetty|AicpuAllGatherPipe | UbxFaces
// 义务|L326 | AicpuAllGatherSoleNHR                                      | UbxFaces
// 义务|L329 | AicpuAllGatherSoleNHRMultiLink                             | ClosA2A
// 豁免|L347 | AivAllGatherMesh1D(AIV,未注册)                                | -
// 义务|L360 | InsAllGatherMeshNhrDPU                                     | RankScan,ShapeA,ShapeB,ZeroInst
// ---- AR（all_reduce_auto_selector.cc）----
// 义务|L89  | AicpuAllReduceSoleNHRAicpuReduce                           | MeshBands,NhrFaces,RankScan,ShapeA
// 义务|L101 | AicpuAllReduceParallelMeshNHR                              | MeshBands
// 义务|L104 | AicpuAllReduceSoleNHR                                      | ShapeA
// 义务|L107 | AicpuAllReduceSoleNHR                                      | NhrFaces
// 义务|L117 | AicpuAllReduceSoleNHR                                      | MeshBands
// 义务|L128 | AicpuAllReduceParallelMeshNHR                              | MeshBands,RankScan
// 义务|L130 | AicpuAllReduceSoleNHR                                      | RankScan
// 义务|L133 | AicpuAllReduceSoleNHR                                      | MeshBands
// 义务|L166 | AicpuAllReduceSoleMeshOneShot|AicpuAllReduceSoleMeshTwoSho | UbxFaces
// 义务|L169 | AicpuAllReduceSoleMeshOneShot                              | UbxFaces
// 义务|L180 | AicpuAllReduceSoleNHRAicpuReduce                           | UbxFaces
// 义务|L201 | AicpuAllReduceSoleNHR                                      | UbxFaces
// 义务|L233 | AicpuAllReduceSoleMeshOneShot|AicpuAllReduceSoleMeshTwoSho | EngineLenCombos,ErrataAr2L,ErrataArMesh
// 义务|L244 | AicpuAllReduceSoleMeshOneShot                              | ErrataAr2L,ErrataArMesh
// 义务|L254 | AicpuAllReduceSoleMeshChunkTwoShot                         | EngineLenCombos,ErrataAr2L,ErrataArMesh
// 义务|L256 | AicpuAllReduceSoleMeshTwoShot                              | EngineLenCombos,ErrataAr2L,ErrataArMesh
// 义务|L260 | AicpuAllReduceSoleNHRAicpuReduce                           | ClosA2A
// 义务|L271 | AicpuAllReduceSoleMeshOneShot|AicpuAllReduceSoleMeshTwoSho | UbxFaces
// 义务|L274 | AicpuAllReduceSoleMeshOneShot                              | UbxFaces
// 义务|L276 | AicpuAllReduceSoleMeshChunkTwoShot                         | UbxFaces
// 义务|L278 | AicpuAllReduceSoleMeshTwoShot                              | UbxFaces
// 义务|L290 | AicpuAllreduceParallelMeshNHRPcie|AicpuAllReducePipeLinePc | UbxFaces
// 豁免|L350 | CcuSchedAllReduceSoleNHR(未注册)                              | -
// 豁免|L360 | CcuSchedAllReduceSoleNHR(未注册)                              | -
// 豁免|L372 | CcuSchedAllReduceSequenceMeshMesh(未注册)                     | RankScan
// 豁免|L381 | CcuSchedAllReduceSequenceMeshMesh(未注册)                     | RankScan
// 义务|L391 | CcuSchedAllReduceSoleMesh                                  | RankScan
// 豁免|L398 | CcuSchedAllReduceSequenceMeshMesh(未注册)                     | -
// 豁免|L414 | CcuSchedAllReduceParallelMeshNHR(未注册)                      | -
// 豁免|L425 | CcuSchedAllReduceSoleNHR(未注册)                              | -
// 豁免|L479 | CcuSchedAllReduceSoleNHR(未注册)                              | -
// 豁免|L515 | CcuSchedAllReduceSoleMesh2Die|CcuSchedAllReduceSequenceMes | -
// 义务|L524 | CcuSchedAllReduceSoleMesh                                  | EngineLenCombos,ErrataCcu,UbxFaces
// 义务|L547 | CcuSchedAllReduceSoleMesh                                  | UbxFaces
// 豁免|L552 | CcuSchedAllReduceConcurMeshNHRMultiLink(未注册)               | -
// 豁免|L559 | CcuSchedAllReduceParallelMeshNHRMultiJetty|CcuSchedAllRedu | -
// 豁免|L566 | CcuSchedAllReduceSoleNHRMultiLink(未注册)                     | -
// ---- A2A（alltoall_auto_selector.cc）----
// 豁免|L44  | CcuSchedAllToAllSoleMeshMultiLink(未注册)                     | -
// 义务|L69  | CcuSchedAllToAllSoleMesh                                   | ClosA2A
// 义务|L87  | CcuSchedAllToAllSoleMeshConcurrent                         | A2A边界
// 义务|L90  | CcuSchedAllToAllMesh1DMultiJetty                           | A2A边界
// 豁免|L101 | CcuSchedAllToAllSoleMesh2Die(未注册)                          | -
// 豁免|L112 | CcuSchedAllToAllSoleMeshConcur(未注册)                        | -
// 义务|L117 | CcuSchedAllToAllSoleMesh                                   | ClosA2A
// 义务|L126 | CcuSchedAllToAllSoleMesh                                   | ClosA2A
// 义务|L148 | AicpuAllToAllSoleMesh                                      | ClosA2A
// 义务|L154 | AicpuAllToAllSoleMesh                                      | ClosA2A
// 义务|L176 | AicpuAllToAllSoleMeshConcurrent|AicpuAllToAllSoleMeshUBX   | A2A边界
// ---- A2AV（alltoallv_auto_selector.cc）----
// 义务|L46  | AicpuAllToAllVSolePairwise                                 | ClosA2A
// 义务|L51  | AicpuAllToAllVSoleMesh                                     | ClosA2A
// 义务|L57  | AicpuAllToAllVSoleMesh                                     | ClosA2A
// 义务|L78  | AicpuAllToAllVSoleMeshConcurrent                           | ClosA2A
// 义务|L80  | AicpuAllToAllVSoleMeshMultiJetty                           | ClosA2A
// 义务|L146 | CcuSchedAllToAllVSoleMesh                                  | ClosA2A
// 豁免|L150 | CcuSchedAllToAllVSoleMeshMultiLink(未注册)                    | -
// 豁免|L160 | CcuSchedAllToAllVSoleMesh2Die(未注册)                         | -
// 义务|L177 | CcuSchedAllToAllVSoleMesh                                  | ClosA2A
// 义务|L183 | CcuSchedAllToAllVSoleMesh                                  | ClosA2A
// 义务|L199 | CcuSchedAllToAllVSoleMesh                                  | ClosA2A
// 豁免|L211 | CcuSchedAllToAllVSoleMeshConcurrent(未注册)                   | -
// 豁免|L216 | CcuSchedAllToAllVSoleMeshMultiJetty(未注册)                   | -
// ---- Half（half_alltoallv_auto_selector.cc）----
// 义务|L55  | CcuHalfAll2AllVMesh1D                                      | ClosA2A

// 义务行合计 112，全部被差分向量覆盖 112

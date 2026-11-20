/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALL_TO_ALL_V_ALG_PATTERNS_H
#define ALL_TO_ALL_V_ALG_PATTERNS_H

#include "topo_pattern.h"

namespace mc2_ops_hccl {

// 不看层数
static const TopoPattern kAicpuAllToAllSoleMeshPatterns[] = {
    TopoPattern().SetLevel0(Level0Shape::MESH_1D).SetRankSize({2, kMaxU32}),
    TopoPattern().SetLevel0(Level0Shape::CLOS).SetRankSize({2, kMaxU32}),
    TopoPattern().SetLevel0(Level0Shape::MESH_1D_CLOS).SetLevel0PcieMix(Tri::Yes).SetRankSize({2, kMaxU32}),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_TO_ALL_SOLE_MESH = 3;
static_assert(
    sizeof(kAicpuAllToAllSoleMeshPatterns) / sizeof(TopoPattern) == PATTERN_NUM_AICPU_ALL_TO_ALL_SOLE_MESH,
    "A2A SoleMesh count drift");

// 的两半，
// 量纲为元素个数。Concurrent 留 >512 元素，UBX 的
// meshEq 且 rank 2..4 条目留 <=512 元素——数据量查询（传真 len）时两候选靠 ELEMENT 子句分界
static const TopoPattern kAicpuAllToAllSoleMeshConcurrentPatterns[] = {
    TopoPattern()
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetRankSize({2, 4})
        .AddLenClause({LenMetric::ELEMENT, LenCmp::GT, LenScale::NONE, 512}),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_TO_ALL_SOLE_MESH_CONCURRENT = 1;
static_assert(
    sizeof(kAicpuAllToAllSoleMeshConcurrentPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_AICPU_ALL_TO_ALL_SOLE_MESH_CONCURRENT,
    "A2A Concur count drift");

static const TopoPattern kAicpuAllToAllSoleMeshUBXPatterns[] = {
    TopoPattern()
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::No)
        .SetRankSize({2, kMaxU32}),
    TopoPattern()
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetRankSize({5, kMaxU32}),
    // mesh 相等且 rank 2..4：仅当 sendCounts[0] <= 512
    TopoPattern()
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetRankSize({2, 4})
        .AddLenClause({LenMetric::ELEMENT, LenCmp::LE, LenScale::NONE, 512}),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_TO_ALL_SOLE_MESH_UBX = 3;
static_assert(
    sizeof(kAicpuAllToAllSoleMeshUBXPatterns) / sizeof(TopoPattern) == PATTERN_NUM_AICPU_ALL_TO_ALL_SOLE_MESH_UBX,
    "A2A UBX count drift");

// MultiLink（未注册），只剩一层。Mesh 上 2die 两形态失败，剩 SINGLE_DIE/NOT_MESH，rank 上界
// 16；CLOS 上界 64；PCIe 全互联上界 16、未全互联失败
static const TopoPattern kCcuSchedAllToAllSoleMeshPatterns[] = {
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetMeshType(Level0MeshType::SINGLE_DIE)
        .SetLevel2UbRtp(Tri::No)
        .SetRankSize({0, 16}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetMeshType(Level0MeshType::NOT_MESH)
        .SetLevel2UbRtp(Tri::No)
        .SetRankSize({0, 16}),
    TopoPattern().SetLevel({1, 1}).SetLevel0(Level0Shape::CLOS).SetLevel2UbRtp(Tri::No).SetRankSize({0, 64}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({0, 16}),
};
constexpr size_t PATTERN_NUM_CCU_SCHED_ALL_TO_ALL_SOLE_MESH = 4;
static_assert(
    sizeof(kCcuSchedAllToAllSoleMeshPatterns) / sizeof(TopoPattern) == PATTERN_NUM_CCU_SCHED_ALL_TO_ALL_SOLE_MESH,
    "A2A CcuSoleMesh count drift");

// 同一 UBX 子函数非 PCIe 支的两半，量纲为字节（:76/:80
// dataSize = sendCounts[0]×dtypeSize 与 BIG_DATA_SIZE_LIMIT=512 比）。CheckClosNumMultipleOfMeshNum
// 未调用，closMultipleOfMesh 保持 Any
static const TopoPattern kCcuSchedAllToAllSoleMeshConcurrentPatterns[] = {
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetRankSize({0, 4})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, 512}),
};
constexpr size_t PATTERN_NUM_CCU_SCHED_ALL_TO_ALL_SOLE_MESH_CONCURRENT = 1;
static_assert(
    sizeof(kCcuSchedAllToAllSoleMeshConcurrentPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_CCU_SCHED_ALL_TO_ALL_SOLE_MESH_CONCURRENT,
    "A2A CcuConcur count drift");

// A2AV Concurrent（CcuSched）：镜像 upstream selector 单级 MESH_1D_CLOS 非混插分支——meshEqClos
// 且 rank≤4（>4 落 MultiJetty——A2AV 侧仍未注册，selector NOT_MATCH 兜底）；分支不读数据量，无 len 子句
static const TopoPattern kCcuSchedAllToAllVSoleMeshConcurrentPatterns[] = {
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetRankSize({0, 4}),
};
constexpr size_t PATTERN_NUM_CCU_SCHED_ALL_TO_ALL_V_SOLE_MESH_CONCURRENT = 1;
static_assert(
    sizeof(kCcuSchedAllToAllVSoleMeshConcurrentPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_CCU_SCHED_ALL_TO_ALL_V_SOLE_MESH_CONCURRENT,
    "A2AV CcuConcur count drift");

static const TopoPattern kCcuSchedAllToAllMesh1DMultiJettyPatterns[] = {
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshEqClos(Tri::No),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetRankSize({5, kMaxU32}),
    // mesh 相等且 rank <= 4：仅当单卡 <= 512 字节
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetRankSize({0, 4})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, 512}),
};
constexpr size_t PATTERN_NUM_CCU_SCHED_ALL_TO_ALL_MESH1_D_MULTI_JETTY = 3;
static_assert(
    sizeof(kCcuSchedAllToAllMesh1DMultiJettyPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_CCU_SCHED_ALL_TO_ALL_MESH1_D_MULTI_JETTY,
    "A2A CcuMultiJetty count drift");

// selector 共用守卫，ins_temp_all_to_all_v_pairwise.cc:51-58）——层数 > 1、level0 Mesh、每框
// 8 卡、level0 对称、rank 是 16 的倍数且落在 16..128。区间表达不了倍数，有限枚举保持静态
static const TopoPattern kAicpuAllToAllVSolePairwisePatterns[] = {
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({16, 16}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({32, 32}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({48, 48}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({64, 64}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({80, 80}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({96, 96}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({112, 112}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({128, 128}),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_TO_ALL_V_SOLE_PAIRWISE = 8;
static_assert(
    sizeof(kAicpuAllToAllVSolePairwisePatterns) / sizeof(TopoPattern) == PATTERN_NUM_AICPU_ALL_TO_ALL_V_SOLE_PAIRWISE,
    "A2AV Pairwise count drift");

// CLOS 1 + PCIe 1）：Pairwise 已返回的形状不能再命中（多级对称 8 卡的 16 倍数 rank 归
// sole[mesh.pairwise] 键）；Mesh 多级按区间掏空 8 个 Pairwise rank，每个 rank 按卡数/对称性
// 拆 3 条补回；CLOS 与 PCIe 混插与 Pairwise 不相交，任意层数
static const TopoPattern kAicpuAllToAllVSoleMeshPatterns[] = {
    // 单层 Mesh：Pairwise 需层数 > 1，全 rank 直收
    TopoPattern().SetLevel({1, 1}).SetLevel0(Level0Shape::MESH_1D).SetRankSize({2, kMaxU32}),
    // 多层 Mesh 非 Pairwise rank 的区间（掏空 16/32/48/64/80/96/112/128）
    TopoPattern().SetLevel({2, kMaxU32}).SetLevel0(Level0Shape::MESH_1D).SetRankSize({2, 15}),
    TopoPattern().SetLevel({2, kMaxU32}).SetLevel0(Level0Shape::MESH_1D).SetRankSize({17, 31}),
    TopoPattern().SetLevel({2, kMaxU32}).SetLevel0(Level0Shape::MESH_1D).SetRankSize({33, 47}),
    TopoPattern().SetLevel({2, kMaxU32}).SetLevel0(Level0Shape::MESH_1D).SetRankSize({49, 63}),
    TopoPattern().SetLevel({2, kMaxU32}).SetLevel0(Level0Shape::MESH_1D).SetRankSize({65, 79}),
    TopoPattern().SetLevel({2, kMaxU32}).SetLevel0(Level0Shape::MESH_1D).SetRankSize({81, 95}),
    TopoPattern().SetLevel({2, kMaxU32}).SetLevel0(Level0Shape::MESH_1D).SetRankSize({97, 111}),
    TopoPattern().SetLevel({2, kMaxU32}).SetLevel0(Level0Shape::MESH_1D).SetRankSize({113, 127}),
    TopoPattern().SetLevel({2, kMaxU32}).SetLevel0(Level0Shape::MESH_1D).SetRankSize({129, kMaxU32}),
    // 8 个 Pairwise rank：卡数不是 8（0..7 与 9..MAX 各一条）
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetDeviceNumPerModule({0, 7})
        .SetRankSize({16, 16}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetDeviceNumPerModule({9, kMaxU32})
        .SetRankSize({16, 16}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetDeviceNumPerModule({0, 7})
        .SetRankSize({32, 32}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetDeviceNumPerModule({9, kMaxU32})
        .SetRankSize({32, 32}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetDeviceNumPerModule({0, 7})
        .SetRankSize({48, 48}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetDeviceNumPerModule({9, kMaxU32})
        .SetRankSize({48, 48}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetDeviceNumPerModule({0, 7})
        .SetRankSize({64, 64}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetDeviceNumPerModule({9, kMaxU32})
        .SetRankSize({64, 64}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetDeviceNumPerModule({0, 7})
        .SetRankSize({80, 80}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetDeviceNumPerModule({9, kMaxU32})
        .SetRankSize({80, 80}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetDeviceNumPerModule({0, 7})
        .SetRankSize({96, 96}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetDeviceNumPerModule({9, kMaxU32})
        .SetRankSize({96, 96}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetDeviceNumPerModule({0, 7})
        .SetRankSize({112, 112}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetDeviceNumPerModule({9, kMaxU32})
        .SetRankSize({112, 112}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetDeviceNumPerModule({0, 7})
        .SetRankSize({128, 128}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetDeviceNumPerModule({9, kMaxU32})
        .SetRankSize({128, 128}),
    // 8 个 Pairwise rank：卡数是 8，但 level0 不对称
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Symmetric(Tri::No)
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({16, 16}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Symmetric(Tri::No)
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({32, 32}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Symmetric(Tri::No)
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({48, 48}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Symmetric(Tri::No)
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({64, 64}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Symmetric(Tri::No)
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({80, 80}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Symmetric(Tri::No)
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({96, 96}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Symmetric(Tri::No)
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({112, 112}),
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Symmetric(Tri::No)
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({128, 128}),
    // CLOS 与 PCIe 混插（与 Pairwise 不相交）
    TopoPattern().SetLevel0(Level0Shape::CLOS).SetRankSize({2, kMaxU32}),
    TopoPattern().SetLevel0(Level0Shape::MESH_1D_CLOS).SetLevel0PcieMix(Tri::Yes).SetRankSize({2, kMaxU32}),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_TO_ALL_V_SOLE_MESH = 36;
static_assert(
    sizeof(kAicpuAllToAllVSoleMeshPatterns) / sizeof(TopoPattern) == PATTERN_NUM_AICPU_ALL_TO_ALL_V_SOLE_MESH,
    "A2AV SoleMesh count drift");

// rank <= 4，不读长度。HCCL_CANN_COMPAT_850 打开时该支
// NOT_MATCH，数组与注册同守卫；clos 倍数检查未调用。多级非 PCIe MESH_1D_CLOS
// 被 topoLevelNums != 1 拒绝，不进 Concurrent 也不回 SoleMesh
#if !defined(HCCL_CANN_COMPAT_850)
static const TopoPattern kAicpuAllToAllVSoleMeshConcurrentPatterns[] = {
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetRankSize({2, 4}),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_TO_ALL_V_SOLE_MESH_CONCURRENT = 1;
static_assert(
    sizeof(kAicpuAllToAllVSoleMeshConcurrentPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_AICPU_ALL_TO_ALL_V_SOLE_MESH_CONCURRENT,
    "A2AV Concur count drift");
#endif

// 勘误补录（蓝本未收录，见文件头注）：AicpuAllToAllVSoleMeshConcurrent 的 else 支（selector
// :80 MATCH）——单层、非 PCIe、MESH_1D_CLOS、CheckMeshNumEqualToClosNum 成功、非
// （meshEq 且 rank <= 4）：mesh 不相等（任意 rank）或 mesh 相等且 rank >= 5
// A2AV CCU MultiJetty：镜像 selector MESH_1D_CLOS 非混插 else 支（非 Concurrent 剩余域）——
// meshEq 任意、rank <= 64（A2AV_CCU_MAX_RANK_SIZE 前置守卫）；该支不读数据量，无 len 子句
static const TopoPattern kCcuSchedAllToAllVSoleMeshMultiJettyPatterns[] = {
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshEqClos(Tri::No)
        .SetRankSize({0, 64}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetRankSize({5, 64}),
};
constexpr size_t PATTERN_NUM_CCU_SCHED_ALL_TO_ALL_V_SOLE_MESH_MULTI_JETTY = 2;
static_assert(
    sizeof(kCcuSchedAllToAllVSoleMeshMultiJettyPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_CCU_SCHED_ALL_TO_ALL_V_SOLE_MESH_MULTI_JETTY,
    "A2AV CcuMultiJetty count drift");

static const TopoPattern kAicpuAllToAllVSoleMeshMultiJettyPatterns[] = {
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::No)
        .SetRankSize({2, kMaxU32}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetRankSize({5, kMaxU32}),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_TO_ALL_V_SOLE_MESH_MULTI_JETTY = 2;
static_assert(
    sizeof(kAicpuAllToAllVSoleMeshMultiJettyPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_AICPU_ALL_TO_ALL_V_SOLE_MESH_MULTI_JETTY,
    "A2AV MultiJetty count drift");

// MESH_1D_CLOS」时 MATCH；一层 Mesh 上界
// 16、CLOS 64、PCIe 全互联 16。两层 PCIe 全互联按 meshType 拆 SINGLE_DIE/NOT_MESH（2die 已被
// :133-139 拒绝，Any 会多算）；一层 PCIe 全互联在 2die 判断前按全互联直选，不限 meshType。
// UBX 非 PCIe 的 Concurrent/MultiJetty 均未注册，单层非 PCIe MESH_1D_CLOS 无条目
static const TopoPattern kCcuSchedAllToAllVSoleMeshPatterns[] = {
    // 两层 Mesh
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetMeshType(Level0MeshType::SINGLE_DIE)
        .SetLevel2UbRtp(Tri::No)
        .SetRankSize({0, 4}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetMeshType(Level0MeshType::NOT_MESH)
        .SetLevel2UbRtp(Tri::No)
        .SetRankSize({0, 4}),
    // 两层 UBX，PCIe 全互联，非 2die
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetMeshType(Level0MeshType::SINGLE_DIE)
        .SetLevel0PcieMix(Tri::Yes)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({0, 4}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetMeshType(Level0MeshType::NOT_MESH)
        .SetLevel0PcieMix(Tri::Yes)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({0, 4}),
    // 一层
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetMeshType(Level0MeshType::SINGLE_DIE)
        .SetLevel2UbRtp(Tri::No)
        .SetRankSize({0, 16}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetMeshType(Level0MeshType::NOT_MESH)
        .SetLevel2UbRtp(Tri::No)
        .SetRankSize({0, 16}),
    TopoPattern().SetLevel({1, 1}).SetLevel0(Level0Shape::CLOS).SetLevel2UbRtp(Tri::No).SetRankSize({0, 64}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({0, 16}),
};
constexpr size_t PATTERN_NUM_CCU_SCHED_ALL_TO_ALL_V_SOLE_MESH = 8;
static_assert(
    sizeof(kCcuSchedAllToAllVSoleMeshPatterns) / sizeof(TopoPattern) == PATTERN_NUM_CCU_SCHED_ALL_TO_ALL_V_SOLE_MESH,
    "A2AV CcuSoleMesh count drift");

// Mesh 也不是 UBX 时失败；rank > 8 失败
// （PostSync 源端保序上限）。不读长度、不分 pcieMix
static const TopoPattern kCcuHalfAll2AllVMesh1DPatterns[] = {
    TopoPattern().SetLevel({1, 2}).SetLevel0(Level0Shape::MESH_1D).SetLevel2UbRtp(Tri::No).SetRankSize({0, 8}),
    TopoPattern().SetLevel({1, 2}).SetLevel0(Level0Shape::MESH_1D_CLOS).SetLevel2UbRtp(Tri::No).SetRankSize({0, 8}),
};
constexpr size_t PATTERN_NUM_CCU_HALF_ALL2_ALL_V_MESH1_D = 2;
static_assert(
    sizeof(kCcuHalfAll2AllVMesh1DPatterns) / sizeof(TopoPattern) == PATTERN_NUM_CCU_HALF_ALL2_ALL_V_MESH1_D,
    "HalfA2AV count drift");

// 手写变体（selector 从不自动选中——devkit tiling 期无计数恒选多通道；仅外部名强制可达）。
// 150MB 总量分界镜像 hccl 分支；level={1,1}：注册面为 TopoMatch1D 扁平单层
static const TopoPattern kAicpuAllToAllSoleMeshSingleChannelPatterns[] = {
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetRankSize({2, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k150MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::CLOS)
        .SetRankSize({2, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k150MB}),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_TO_ALL_SOLE_MESH_SINGLE_CHANNEL = 2;
static_assert(
    sizeof(kAicpuAllToAllSoleMeshSingleChannelPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_AICPU_ALL_TO_ALL_SOLE_MESH_SINGLE_CHANNEL,
    "A2A SingleChannel count drift");

} // namespace mc2_ops_hccl
#endif // ALL_TO_ALL_V_ALG_PATTERNS_H

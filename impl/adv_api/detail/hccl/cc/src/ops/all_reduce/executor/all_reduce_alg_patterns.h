/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALL_REDUCE_ALG_PATTERNS_H
#define ALL_REDUCE_ALG_PATTERNS_H

#include "topo_pattern.h"

namespace mc2_ops_hccl {

static const TopoPattern kAicpuAllReduceSoleNHRPatterns[] = {
    // Level1Nhr，层数 2
    TopoPattern().SetLevel({2, 2}).SetLevel1Nhr(Tri::Yes).SetLevel0Symmetric(Tri::No),
    TopoPattern().SetLevel({2, 2}).SetLevel1Nhr(Tri::Yes).SetLevel0Symmetric(Tri::Yes).SetLevel1Symmetric(Tri::No),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel1Nhr(Tri::Yes)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({0, 7}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel1Nhr(Tri::Yes)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({9, kMaxU32}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel1Nhr(Tri::Yes)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({8, 8}),
    // Level1Nhr，层数 >= 4
    TopoPattern().SetLevel({4, kMaxU32}).SetLevel1Nhr(Tri::Yes).SetLevel0Symmetric(Tri::No),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel1Nhr(Tri::Yes)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel1Nhr(Tri::Yes)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({0, 7}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel1Nhr(Tri::Yes)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({9, kMaxU32}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel1Nhr(Tri::Yes)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({8, 8}),
    // Level1Nhr，层数 3
    TopoPattern().SetLevel({3, 3}).SetLevel1Nhr(Tri::Yes).SetLevel0Symmetric(Tri::No),
    TopoPattern().SetLevel({3, 3}).SetLevel1Nhr(Tri::Yes).SetLevel0Symmetric(Tri::Yes).SetLevel1Symmetric(Tri::No),
    TopoPattern()
        .SetLevel({3, 3})
        .SetLevel1Nhr(Tri::Yes)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes),
    // 单实例，层数 2
    TopoPattern().SetLevel({2, 2}).SetLevel1Nhr(Tri::No).SetLevel0Symmetric(Tri::No).SetLevel0Inst({0, InstCmp::EQ, 1}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::EQ, 1}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::EQ, 1})
        .SetDeviceNumPerModule({0, 7}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::EQ, 1})
        .SetDeviceNumPerModule({9, kMaxU32}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::EQ, 1})
        .SetDeviceNumPerModule({8, 8}),
    // 单实例，层数 >= 4
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::EQ, 1}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::EQ, 1}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::EQ, 1})
        .SetDeviceNumPerModule({0, 7}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::EQ, 1})
        .SetDeviceNumPerModule({9, kMaxU32}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::EQ, 1})
        .SetDeviceNumPerModule({8, 8}),
    // 单实例，层数 3
    TopoPattern().SetLevel({3, 3}).SetLevel1Nhr(Tri::No).SetLevel0Symmetric(Tri::No).SetLevel0Inst({0, InstCmp::EQ, 1}),
    TopoPattern()
        .SetLevel({3, 3})
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::EQ, 1}),
    TopoPattern()
        .SetLevel({3, 3})
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::EQ, 1}),
    // 两层 Mesh，单卡 <= 32MB
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k32MB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k32MB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetDeviceNumPerModule({0, 7})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k32MB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetDeviceNumPerModule({9, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k32MB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetDeviceNumPerModule({8, 8})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k32MB}),
    // 四层及以上 Mesh，不读长度
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetDeviceNumPerModule({0, 7}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetDeviceNumPerModule({9, kMaxU32}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetDeviceNumPerModule({8, 8}),
    // 三层 Mesh，不对称两份拆分（对称在前面 NOT_MATCH，E/F 被不对称条件盖住）
    TopoPattern()
        .SetLevel({3, 3})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1}),
    TopoPattern()
        .SetLevel({3, 3})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1}),
    // 一层 UBX，mesh == clos，rank >= 5（rank <= 4 走 OneShot/TwoShot/特殊类型）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetRankSize({5, kMaxU32}),
    // 一层 UBX，clos 是 mesh 的倍数，单卡 < 1MB（>= 1MB 的 Parallel/PipeLine 未迁移）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::No)
        .SetClosMultipleOfMesh(Tri::Yes)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k1MB - 1}),
    // 一层 UBX，两种数量关系都不成立
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::No)
        .SetClosMultipleOfMesh(Tri::No),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_REDUCE_SOLE_NHR = 41;
static_assert(
    sizeof(kAicpuAllReduceSoleNHRPatterns) / sizeof(TopoPattern) == PATTERN_NUM_AICPU_ALL_REDUCE_SOLE_NHR,
    "AR SoleNHR count drift");

static const TopoPattern kAicpuAllReduceParallelMeshNHRPatterns[] = {
    // 三层，两层都对称，topLevelUboe，每模块卡数不是 8（卡数 8 走未迁移 pipeline）
    TopoPattern()
        .SetLevel({3, 3})
        .SetTopLevelUboe(Tri::Yes)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({0, 7}),
    TopoPattern()
        .SetLevel({3, 3})
        .SetTopLevelUboe(Tri::Yes)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({9, kMaxU32}),
    // 两层 Mesh，Level1Nhr 与单实例已离开，单卡 > 32MB 且 <= 4GB
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k32MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k32MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetDeviceNumPerModule({0, 7})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k32MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetDeviceNumPerModule({9, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k32MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetDeviceNumPerModule({8, 8})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k32MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k4GB}),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_REDUCE_PARALLEL_MESH_NHR = 7;
static_assert(
    sizeof(kAicpuAllReduceParallelMeshNHRPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_AICPU_ALL_REDUCE_PARALLEL_MESH_NHR,
    "AR Parallel count drift");

static const TopoPattern kAicpuAllReduceSoleMeshOneShotPatterns[] = {
    // 一层 Mesh，rank 不是 2（{0,1} 与 {3,MAX} 两段，rank==2 单列）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetRankSize({0, 1})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k8MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetRankSize({3, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k8MB}),
    // 一层 Mesh，rank == 2，不是两层网络
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::No)
        .SetRankSize({2, 2})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k8MB}),
    // 一层 Mesh，rank == 2，两层网络，单卡 <= 8MB（与 TwoShot 的"特殊类型并入"条目同款手法：
    // 对 dtype 存在量化并集——特殊 dtype 恰 8MB 仍选 OneShot（selector :233 先于 2P 绕行））
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({2, 2})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k8MB}),
    // 一层 UBX，PCIe 全互联，closMultiple 两态（Check* 已调用，失败则两条都不命中）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k8MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k8MB}),
    // 一层 UBX，mesh == clos，rank <= 4
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetRankSize({0, 4})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k8MB}),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_REDUCE_SOLE_MESH_ONE_SHOT = 7;
static_assert(
    sizeof(kAicpuAllReduceSoleMeshOneShotPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_AICPU_ALL_REDUCE_SOLE_MESH_ONE_SHOT,
    "AR OneShot count drift");

// 勘误：蓝本 :37 公式 64/r² 有误，代码为 8/r²，T_ar = r²·k4MB
static const TopoPattern kAicpuAllReduceSoleMeshTwoShotPatterns[] = {
    // r=0 静态条目：ratio=1 → T_ar=32MB（8MB, 32MB]
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k8MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k32MB}),
    // r>=1 非两层网（勘误：r=2 不掏空须拆 twoLevelNet；r=1 由 4MB·1 < 8MB+1 自然掏空）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::No)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k8MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::RANK_SQ, k4MB}),
    // r>=3 两层网（r=2 且 >= 8MB 被 SoleMeshConcur 绕行 NOT_MATCH，整段排除）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({3, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k8MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::RANK_SQ, k4MB}),
    // PCIe 全互联，closMultiple 两态，r=0 / r>=1 各一条（PCIe 无绕行，twoLevelNet 不读）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k8MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k32MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k8MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::RANK_SQ, k4MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k8MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k32MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k8MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::RANK_SQ, k4MB}),
    // 特殊类型并入：单卡 > 8MB 不再看 ratio，直接 TwoShot，无上界
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k8MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k8MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k8MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetRankSize({0, 4})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k8MB}),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_REDUCE_SOLE_MESH_TWO_SHOT = 11;
static_assert(
    sizeof(kAicpuAllReduceSoleMeshTwoShotPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_AICPU_ALL_REDUCE_SOLE_MESH_TWO_SHOT,
    "AR TwoShot count drift");

// 勘误：蓝本 :37 公式 64/r² 有误，代码为 8/r²，下界 r²·k4MB
static const TopoPattern kAicpuAllReduceSoleMeshChunkTwoShotPatterns[] = {
    // twoLevel = No：单卡 > max(8MB, r²·4MB)，无上界。r=0 静态（ratio=1 → 32MB）+ r>=1 公式
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::No)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k32MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::No)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k8MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::RANK_SQ, k4MB}),
    // twoLevel = Yes：上界 4GB（总量超 4GB 走未迁移 SoleMeshConcur）。r=0 静态 + r=1 + r>=3
    // （勘误：r>=32 由 4MB·r² >= 4GB 子句自然掏空，蓝本写 r>=91；r=2 被 >= 8MB 绕行排除）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k32MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({1, 1})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k8MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::RANK_SQ, k4MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({3, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k8MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::RANK_SQ, k4MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k4GB}),
    // PCIe 全互联：不看 twoLevelNet，无 4GB 上界。closMultiple 两态 × r=0/r>=1
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k32MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k8MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::RANK_SQ, k4MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k32MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k8MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::RANK_SQ, k4MB}),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_REDUCE_SOLE_MESH_CHUNK_TWO_SHOT = 9;
static_assert(
    sizeof(kAicpuAllReduceSoleMeshChunkTwoShotPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_AICPU_ALL_REDUCE_SOLE_MESH_CHUNK_TWO_SHOT,
    "AR ChunkTwoShot count drift");

static const TopoPattern kAicpuAllreduceParallelMeshNHRPciePatterns[] = {
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::No)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k32MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::No)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k32MB}),
};
constexpr size_t PATTERN_NUM_AICPU_ALLREDUCE_PARALLEL_MESH_NHR_PCIE = 2;
static_assert(
    sizeof(kAicpuAllreduceParallelMeshNHRPciePatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_AICPU_ALLREDUCE_PARALLEL_MESH_NHR_PCIE,
    "AR ParallelPcie count drift");

static const TopoPattern kAicpuAllReducePipeLinePciePatterns[] = {
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::No)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GE, LenScale::NONE, k32MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::No)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GE, LenScale::NONE, k32MB}),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_REDUCE_PIPE_LINE_PCIE = 2;
static_assert(
    sizeof(kAicpuAllReducePipeLinePciePatterns) / sizeof(TopoPattern) == PATTERN_NUM_AICPU_ALL_REDUCE_PIPE_LINE_PCIE,
    "AR PipeLinePcie count drift");

static const TopoPattern kAicpuAllReduceSoleNHRAicpuReducePatterns[] = {
    // 多层，进入 Mesh/CLOS 细分之前
    TopoPattern().SetLevel({2, kMaxU32}),
    // 一层 CLOS
    TopoPattern().SetLevel({1, 1}).SetLevel0(Level0Shape::CLOS),
    // 一层 UBX，不是「mesh 相等且 rank <= 4」（PCIe 未全互联特殊类型 NOT_MATCH，不含 pcieMix=Yes）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetRankSize({5, kMaxU32}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::No)
        .SetClosMultipleOfMesh(Tri::Yes),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::No)
        .SetClosMultipleOfMesh(Tri::No),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_REDUCE_SOLE_NHR_AICPU_REDUCE = 5;
static_assert(
    sizeof(kAicpuAllReduceSoleNHRAicpuReducePatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_AICPU_ALL_REDUCE_SOLE_NHR_AICPU_REDUCE,
    "AR AicpuReduce count drift");

// 勘误：蓝本 :39 的 T_ccu_ar = r²·128KB 有误，代码为 8/r² > 8MB，即 r²·k1MB
static const TopoPattern kCcuSchedAllReduceSoleMeshPatterns[] = {
    // 两层 Mesh，rank <= 63（rank 64 被 >= 64 小数据支抢走 NOT_MATCH），单卡 <= 512KB
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetTwoDieFullMesh(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetRankSize({0, 63})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k512KB}),
    // 一层 UBX，4 卡相等，单卡 < 512KB
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetRankSize({0, 4})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k512KB}),
    // 一层 Mesh：r=0 静态（ratio=1 → 8MB）与 r>=1 公式（T_ccu_ar = r²·k1MB），各 × meshType
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetMeshType(Level0MeshType::SINGLE_DIE)
        .SetLevel2UbRtp(Tri::No)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k8MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetMeshType(Level0MeshType::NOT_MESH)
        .SetLevel2UbRtp(Tri::No)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k8MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetMeshType(Level0MeshType::SINGLE_DIE)
        .SetLevel2UbRtp(Tri::No)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::RANK_SQ, k1MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetMeshType(Level0MeshType::NOT_MESH)
        .SetLevel2UbRtp(Tri::No)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::RANK_SQ, k1MB}),
    // 一层 UBX PCIe 全互联复用同一函数（SelectCcuScheduleLevel0AlgoMesh1D）：r=0 / r>=1 × meshType
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetMeshType(Level0MeshType::SINGLE_DIE)
        .SetLevel0PcieMix(Tri::Yes)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k8MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetMeshType(Level0MeshType::NOT_MESH)
        .SetLevel0PcieMix(Tri::Yes)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k8MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetMeshType(Level0MeshType::SINGLE_DIE)
        .SetLevel0PcieMix(Tri::Yes)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::RANK_SQ, k1MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetMeshType(Level0MeshType::NOT_MESH)
        .SetLevel0PcieMix(Tri::Yes)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::RANK_SQ, k1MB}),
};
constexpr size_t PATTERN_NUM_CCU_SCHED_ALL_REDUCE_SOLE_MESH = 10;
static_assert(
    sizeof(kCcuSchedAllReduceSoleMeshPatterns) / sizeof(TopoPattern) == PATTERN_NUM_CCU_SCHED_ALL_REDUCE_SOLE_MESH,
    "AR CcuSoleMesh count drift");

// CcuSchedAllReduceSoleNHR（upstream 新迁 0e4dba8235，selector 4 个 MATCH 分支转录：
// 两层 Mesh Level1Nhr / 两层 Mesh 单实例 / 两层 CLOS 小数据(<8MB) / 单层 CLOS 小数据(<8MB)；
// 引擎级前置：level2UbRtp/≥3 层/PROD/64bit dtype/inplace 的否决由 selector 入口与查询 pass 承担
// pattern 只承载拓扑+长度面）
static const TopoPattern kCcuSchedAllReduceSoleNHRPatterns[] = {
    // 两层 Mesh，Level1Nhr
    TopoPattern().SetLevel({2, 2}).SetLevel0(Level0Shape::MESH_1D).SetLevel1Nhr(Tri::Yes).SetLevel2UbRtp(Tri::No),
    // 两层 Mesh，layer0 单实例（Level1Nhr 已离开；层信息缺失时 selector 跳过本支，pattern 同拒）
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetLevel0Inst({0, InstCmp::EQ, 1}),
    // 两层 CLOS，单卡 < 8MB
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::CLOS)
        .SetLevel2UbRtp(Tri::No)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k8MB}),
    // 单层 CLOS，单卡 < 8MB（pcieMix 已在进入赋值前返回）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::CLOS)
        .SetLevel2UbRtp(Tri::No)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k8MB}),
};
constexpr size_t PATTERN_NUM_CCU_SCHED_ALL_REDUCE_SOLE_NHR = 4;
static_assert(
    sizeof(kCcuSchedAllReduceSoleNHRPatterns) / sizeof(TopoPattern) == PATTERN_NUM_CCU_SCHED_ALL_REDUCE_SOLE_NHR,
    "AR CcuSoleNHR count drift");

} // namespace mc2_ops_hccl
#endif // ALL_REDUCE_ALG_PATTERNS_H

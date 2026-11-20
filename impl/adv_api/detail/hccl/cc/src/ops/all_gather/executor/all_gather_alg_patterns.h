/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALL_GATHER_ALG_PATTERNS_H
#define ALL_GATHER_ALG_PATTERNS_H

#include "topo_pattern.h"

namespace mc2_ops_hccl {

static const TopoPattern kAicpuAllGatherSoleNHRPatterns[] = {
    // 至少两层 + Level1Nhr
    TopoPattern().SetLevel({2, kMaxU32}).SetLevel1Nhr(Tri::Yes).SetLevel0PcieMix(Tri::No),
    // 至少两层 + Level0Nhr
    TopoPattern().SetLevel({2, kMaxU32}).SetLevel0Nhr(Tri::Yes).SetLevel1Nhr(Tri::No).SetLevel0PcieMix(Tri::No),
    // 至少两层 + layer0 单实例
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel0Inst({0, InstCmp::EQ, 1}),
    // 三层及以上 Mesh，前三支都不成立
    TopoPattern()
        .SetLevel({3, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1}),
    // 两层 Mesh，rank < 1024，单卡 <= 1MB
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetRankSize({0, 1023})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k1MB}),
    // 两层 Mesh，rank >= 1024，总量 <= 1GB-1
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetRankSize({1024, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1GB - 1}),
    // 一层 UBX，mesh == clos，rank >= 5
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetRankSize({5, kMaxU32}),
    // 一层 UBX，clos 是 mesh 的倍数，单卡 <= 512KB
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::No)
        .SetClosMultipleOfMesh(Tri::Yes)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k512KB}),
    // 一层 UBX，两种数量关系都不成立
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::No)
        .SetClosMultipleOfMesh(Tri::No),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_GATHER_SOLE_NHR = 9;
static_assert(
    sizeof(kAicpuAllGatherSoleNHRPatterns) / sizeof(TopoPattern) == PATTERN_NUM_AICPU_ALL_GATHER_SOLE_NHR,
    "AG SoleNHR count drift");

static const TopoPattern kInsAllGatherParallelMesh1DNHRPatterns[] = {
    // 两层 Mesh，rank >= 1024，总量 >= 1GB（任意单卡长度）
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetRankSize({1024, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GE, LenScale::NONE, k1GB}),
    // 两层 Mesh，rank <= 1023，单卡 > 1MB 且总量 <= 4GB
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetRankSize({0, 1023})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k1MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k4GB}),
};
constexpr size_t PATTERN_NUM_INS_ALL_GATHER_PARALLEL_MESH1_DNHR = 2;
static_assert(
    sizeof(kInsAllGatherParallelMesh1DNHRPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_INS_ALL_GATHER_PARALLEL_MESH1_DNHR,
    "AG Parallel count drift");

static const TopoPattern kAicpuAllGatherSequenceMeshConcurNHRPatterns[] = {
    // 两层 Mesh，rank <= 1023，单卡 > 1MB 且总量 > 4GB
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetRankSize({0, 1023})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k1MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GT, LenScale::NONE, k4GB}),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_GATHER_SEQUENCE_MESH_CONCUR_NHR = 1;
static_assert(
    sizeof(kAicpuAllGatherSequenceMeshConcurNHRPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_AICPU_ALL_GATHER_SEQUENCE_MESH_CONCUR_NHR,
    "AG Sequence count drift");

static const TopoPattern kAicpuAllGatherSoleNHRMultiLinkPatterns[] = {
    // 多层 CLOS：排除 Level1Nhr、Level0Nhr 和 layer0 单实例
    TopoPattern()
        .SetLevel({2, kMaxU32})
        .SetLevel0(Level0Shape::CLOS)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1}),
    // 一层 CLOS 不读这些字段
    TopoPattern().SetLevel({1, 1}).SetLevel0(Level0Shape::CLOS),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_GATHER_SOLE_NHR_MULTI_LINK = 2;
static_assert(
    sizeof(kAicpuAllGatherSoleNHRMultiLinkPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_AICPU_ALL_GATHER_SOLE_NHR_MULTI_LINK,
    "AG MultiLink count drift");

static const TopoPattern kInsAllGatherMesh1DPatterns[] = {
    // 一层 Mesh
    TopoPattern().SetLevel({1, 1}).SetLevel0(Level0Shape::MESH_1D),
    // 一层 UBX，PCIe 混插且 layer0 全互联
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes),
    // 一层 UBX，mesh == clos，rank <= 4，单卡 <= 512KB
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetRankSize({0, 4})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k512KB}),
};
constexpr size_t PATTERN_NUM_INS_ALL_GATHER_MESH1_D = 3;
static_assert(
    sizeof(kInsAllGatherMesh1DPatterns) / sizeof(TopoPattern) == PATTERN_NUM_INS_ALL_GATHER_MESH1_D,
    "AG Mesh1D count drift");

static const TopoPattern kAicpuAllGatherConcurMeshNHRPatterns[] = {
    // 一层 UBX，mesh == clos，rank <= 4，单卡 > 512KB
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetRankSize({0, 4})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k512KB}),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_GATHER_CONCUR_MESH_NHR = 1;
static_assert(
    sizeof(kAicpuAllGatherConcurMeshNHRPatterns) / sizeof(TopoPattern) == PATTERN_NUM_AICPU_ALL_GATHER_CONCUR_MESH_NHR,
    "AG Concur count drift");

static const TopoPattern kInsAllGatherParallelMesh1DNHRMultiJettyPatterns[] = {
    // 一层 UBX，clos 是 mesh 的倍数，单卡 > 512KB 且 < 16MB
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::No)
        .SetClosMultipleOfMesh(Tri::Yes)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k512KB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k16MB}),
};
constexpr size_t PATTERN_NUM_INS_ALL_GATHER_PARALLEL_MESH1_DNHR_MULTI_JETTY = 1;
static_assert(
    sizeof(kInsAllGatherParallelMesh1DNHRMultiJettyPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_INS_ALL_GATHER_PARALLEL_MESH1_DNHR_MULTI_JETTY,
    "AG MultiJetty count drift");

static const TopoPattern kAicpuAllGatherPipeLineMeshNHRPatterns[] = {
    // 同 MultiJetty 拓扑，单卡 >= 16MB
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetMeshEqClos(Tri::No)
        .SetClosMultipleOfMesh(Tri::Yes)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GE, LenScale::NONE, k16MB}),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_GATHER_PIPE_LINE_MESH_NHR = 1;
static_assert(
    sizeof(kAicpuAllGatherPipeLineMeshNHRPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_AICPU_ALL_GATHER_PIPE_LINE_MESH_NHR,
    "AG PipeLine count drift");

static const TopoPattern kInsAllGatherParallelMesh1DNHRPciePatterns[] = {
    // 一层 UBX，PCIe 混插且 layer0 不是 1D Mesh 全互联，单卡 < 4MB
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetMeshFullConnect(Tri::No)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k4MB}),
};
constexpr size_t PATTERN_NUM_INS_ALL_GATHER_PARALLEL_MESH1_DNHR_PCIE = 1;
static_assert(
    sizeof(kInsAllGatherParallelMesh1DNHRPciePatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_INS_ALL_GATHER_PARALLEL_MESH1_DNHR_PCIE,
    "AG ParallelPcie count drift");

static const TopoPattern kAicpuAllGatherPipeLinePciePatterns[] = {
    // 同 PCIe 拓扑，单卡 >= 4MB
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetMeshFullConnect(Tri::No)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GE, LenScale::NONE, k4MB}),
};
constexpr size_t PATTERN_NUM_AICPU_ALL_GATHER_PIPE_LINE_PCIE = 1;
static_assert(
    sizeof(kAicpuAllGatherPipeLinePciePatterns) / sizeof(TopoPattern) == PATTERN_NUM_AICPU_ALL_GATHER_PIPE_LINE_PCIE,
    "AG PipeLinePcie count drift");

static const TopoPattern kInsAllGatherMeshNhrDPUPatterns[] = {
    // 层数大于 1 且 layer0 单实例（SelectDPUAlgo 两支「或」拆开，不读长度）
    TopoPattern().SetLevel({2, kMaxU32}).SetLevel0Inst({0, InstCmp::EQ, 1}),
    // 层数大于 1 且 level0 为 Mesh
    TopoPattern().SetLevel({2, kMaxU32}).SetLevel0(Level0Shape::MESH_1D),
};
constexpr size_t PATTERN_NUM_INS_ALL_GATHER_MESH_NHR_DPU = 2;
static_assert(
    sizeof(kInsAllGatherMeshNhrDPUPatterns) / sizeof(TopoPattern) == PATTERN_NUM_INS_ALL_GATHER_MESH_NHR_DPU,
    "AG DPU count drift");

static const TopoPattern kCcuSchedAllGatherSoleNHRPatterns[] = {
    // 两层 Mesh，Level1Nhr。2die、单实例尚未读取
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::Yes)
        .SetLevel2UbRtp(Tri::No)
        .SetRankSize({0, 64}),
    // 两层 Mesh，单实例。Level1Nhr 与 2die 已离开
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetTwoDieFullMesh(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetLevel0Inst({0, InstCmp::EQ, 1})
        .SetRankSize({0, 64}),
    // 两层 Mesh，rank <= 8，单卡 > 1MB，框数 >= 17
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetTwoDieFullMesh(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetRankSize({0, 8})
        .SetFrameNum({17, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k1MB}),
    // 两层 Mesh，rank 9..32，单卡 >= 128KB，框数 >= 17
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetTwoDieFullMesh(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetRankSize({9, 32})
        .SetFrameNum({17, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GE, LenScale::NONE, 128ULL * 1024}),
    // 两层 Mesh，rank 33..64，框数 >= 17，长度不限
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetTwoDieFullMesh(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetRankSize({33, 64})
        .SetFrameNum({17, kMaxU32}),
    // 两层 CLOS
    TopoPattern().SetLevel({2, 2}).SetLevel0(Level0Shape::CLOS).SetLevel2UbRtp(Tri::No),
    // 一层 CLOS，PCIe 混插在进入赋值前返回
    TopoPattern().SetLevel({1, 1}).SetLevel0(Level0Shape::CLOS).SetLevel0PcieMix(Tri::No).SetLevel2UbRtp(Tri::No),
};
constexpr size_t PATTERN_NUM_CCU_SCHED_ALL_GATHER_SOLE_NHR = 7;
static_assert(
    sizeof(kCcuSchedAllGatherSoleNHRPatterns) / sizeof(TopoPattern) == PATTERN_NUM_CCU_SCHED_ALL_GATHER_SOLE_NHR,
    "AG CcuSoleNHR count drift");

static const TopoPattern kCcuSchedAllGatherSoleMeshPatterns[] = {
    TopoPattern().SetLevel({1, 1}).SetLevel0(Level0Shape::MESH_1D).SetMeshType(Level0MeshType::SINGLE_DIE),
    // schedule：两层 Mesh，rank <= 8，单卡 <= 1MB
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetTwoDieFullMesh(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetRankSize({0, 8})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k1MB}),
    // schedule：两层 Mesh，rank 9..32，单卡 < 128KB
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetTwoDieFullMesh(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetRankSize({9, 32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, 128ULL * 1024}),
    // schedule：一层 UBX，单卡 <= 512KB。Check* 已执行，三种成立的数量关系各一条
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k512KB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshEqClos(Tri::No)
        .SetClosMultipleOfMesh(Tri::Yes)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k512KB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshEqClos(Tri::No)
        .SetClosMultipleOfMesh(Tri::No)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k512KB}),
};
constexpr size_t PATTERN_NUM_CCU_SCHED_ALL_GATHER_SOLE_MESH = 6;
static_assert(
    sizeof(kCcuSchedAllGatherSoleMeshPatterns) / sizeof(TopoPattern) == PATTERN_NUM_CCU_SCHED_ALL_GATHER_SOLE_MESH,
    "AG CcuSoleMesh count drift");

static const TopoPattern kCcuSchedAllGatherConcurMeshNHRMultiLinkPatterns[] = {
    // 一层 UBX，mesh == clos，rank <= 4，单卡 > 512KB
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetRankSize({0, 4})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k512KB}),
};
constexpr size_t PATTERN_NUM_CCU_SCHED_ALL_GATHER_CONCUR_MESH_NHR_MULTI_LINK = 1;
static_assert(
    sizeof(kCcuSchedAllGatherConcurMeshNHRMultiLinkPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_CCU_SCHED_ALL_GATHER_CONCUR_MESH_NHR_MULTI_LINK,
    "AG CcuConcur count drift");

static const TopoPattern kCcuSchedAllGatherParallelMeshNHRMultiLinkPatterns[] = {
    // 一层 UBX，clos 是 mesh 的倍数，单卡 > 512KB 且 < 16MB
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshEqClos(Tri::No)
        .SetClosMultipleOfMesh(Tri::Yes)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k512KB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k16MB}),
};
constexpr size_t PATTERN_NUM_CCU_SCHED_ALL_GATHER_PARALLEL_MESH_NHR_MULTI_LINK = 1;
static_assert(
    sizeof(kCcuSchedAllGatherParallelMeshNHRMultiLinkPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_CCU_SCHED_ALL_GATHER_PARALLEL_MESH_NHR_MULTI_LINK,
    "AG CcuParallel count drift");

} // namespace mc2_ops_hccl
#endif // ALL_GATHER_ALG_PATTERNS_H

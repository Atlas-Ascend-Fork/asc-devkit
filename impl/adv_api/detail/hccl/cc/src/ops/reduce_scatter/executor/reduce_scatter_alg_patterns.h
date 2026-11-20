/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef REDUCE_SCATTER_ALG_PATTERNS_H
#define REDUCE_SCATTER_ALG_PATTERNS_H

#include "topo_pattern.h"

namespace mc2_ops_hccl {

static const TopoPattern kInsReduceScatterNHRPatterns[] = {
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
    // Level0Nhr（Level1Nhr 已返回），层数 2
    TopoPattern().SetLevel({2, 2}).SetLevel0Nhr(Tri::Yes).SetLevel1Nhr(Tri::No).SetLevel0Symmetric(Tri::No),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0Nhr(Tri::Yes)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0Nhr(Tri::Yes)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({0, 7}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0Nhr(Tri::Yes)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({9, kMaxU32}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0Nhr(Tri::Yes)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({8, 8}),
    // Level0Nhr，层数 >= 4
    TopoPattern().SetLevel({4, kMaxU32}).SetLevel0Nhr(Tri::Yes).SetLevel1Nhr(Tri::No).SetLevel0Symmetric(Tri::No),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0Nhr(Tri::Yes)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0Nhr(Tri::Yes)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({0, 7}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0Nhr(Tri::Yes)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({9, kMaxU32}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0Nhr(Tri::Yes)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetDeviceNumPerModule({8, 8}),
    // Level0Nhr，层数 3
    TopoPattern().SetLevel({3, 3}).SetLevel0Nhr(Tri::Yes).SetLevel1Nhr(Tri::No).SetLevel0Symmetric(Tri::No),
    TopoPattern()
        .SetLevel({3, 3})
        .SetLevel0Nhr(Tri::Yes)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No),
    TopoPattern()
        .SetLevel({3, 3})
        .SetLevel0Nhr(Tri::Yes)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes),
    // layer0 单实例（EQ 1），层数 2
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::EQ, 1}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::EQ, 1}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::EQ, 1})
        .SetDeviceNumPerModule({0, 7}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::EQ, 1})
        .SetDeviceNumPerModule({9, kMaxU32}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::EQ, 1})
        .SetDeviceNumPerModule({8, 8}),
    // 单实例，层数 >= 4
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::EQ, 1}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::EQ, 1}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::EQ, 1})
        .SetDeviceNumPerModule({0, 7}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::EQ, 1})
        .SetDeviceNumPerModule({9, kMaxU32}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::EQ, 1})
        .SetDeviceNumPerModule({8, 8}),
    // 单实例，层数 3
    TopoPattern()
        .SetLevel({3, 3})
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::EQ, 1}),
    TopoPattern()
        .SetLevel({3, 3})
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::EQ, 1}),
    TopoPattern()
        .SetLevel({3, 3})
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::EQ, 1}),
    // 三层 Mesh，本实例卡数 > 1（GT 例外），不对称两份拆分
    TopoPattern()
        .SetLevel({3, 3})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1}),
    TopoPattern()
        .SetLevel({3, 3})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1}),
    // 层数 2 Mesh，本实例卡数 > 1（GT），rank <= 255，单卡 <= 4MB
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k4MB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k4MB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({0, 7})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k4MB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({9, kMaxU32})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k4MB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k4MB}),
    // 层数 >= 4 Mesh，rank <= 255，单卡 <= 4MB
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k4MB}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k4MB}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({0, 7})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k4MB}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({9, kMaxU32})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k4MB}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k4MB}),
    // 层数 2 Mesh，rank >= 256，总量 <= 1GB-1（已推出单卡 < 4MB）
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1GB - 1}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1GB - 1}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({0, 7})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1GB - 1}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({9, kMaxU32})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1GB - 1}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1GB - 1}),
    // 层数 >= 4 Mesh，rank >= 256，总量 <= 1GB-1
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1GB - 1}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1GB - 1}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({0, 7})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1GB - 1}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({9, kMaxU32})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1GB - 1}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1GB - 1}),
    // 一层 UBX，非 PCIe，未全互联，clos 不是 mesh 的倍数（MESH_1D_CLOS 入口已调 Check）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::No),
    // 一层 UBX，非 PCIe，未全互联，clos 是倍数，单卡 < 1MB（>= 1MB 走 ParallelMeshNHRUBX）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::No)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k1MB}),
};
constexpr size_t PATTERN_NUM_INS_REDUCE_SCATTER_NHR = 63;
static_assert(
    sizeof(kInsReduceScatterNHRPatterns) / sizeof(TopoPattern) == PATTERN_NUM_INS_REDUCE_SCATTER_NHR,
    "RS NHR count drift");

static const TopoPattern kInsReduceScatterParallelMesh1DNHRPatterns[] = {
    // 三层，两层都对称，topLevelUboe，每模块卡数不是 8
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
    // 层数 2 Mesh，本实例卡数 > 1（GT 例外），rank >= 256，总量 >= 1GB
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GE, LenScale::NONE, k1GB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GE, LenScale::NONE, k1GB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({0, 7})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GE, LenScale::NONE, k1GB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({9, kMaxU32})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GE, LenScale::NONE, k1GB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GE, LenScale::NONE, k1GB}),
    // 层数 >= 4 Mesh，rank >= 256，总量 >= 1GB
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GE, LenScale::NONE, k1GB}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GE, LenScale::NONE, k1GB}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({0, 7})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GE, LenScale::NONE, k1GB}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({9, kMaxU32})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GE, LenScale::NONE, k1GB}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({256, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GE, LenScale::NONE, k1GB}),
    // 层数 2 Mesh，rank <= 255，单卡 > 4MB 且总量 <= 4GB
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({0, 7})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({9, kMaxU32})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k4GB}),
    // 层数 >= 4 Mesh，rank <= 255，单卡 > 4MB 且总量 <= 4GB
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({0, 7})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({9, kMaxU32})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k4GB}),
};
constexpr size_t PATTERN_NUM_INS_REDUCE_SCATTER_PARALLEL_MESH1_DNHR = 22;
static_assert(
    sizeof(kInsReduceScatterParallelMesh1DNHRPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_INS_REDUCE_SCATTER_PARALLEL_MESH1_DNHR,
    "RS Parallel count drift");

// 只在 rank <= 255、单卡 > 4MB、总量 > 4GB；层数 2 与 >=4（三层对称 Mesh 的 Sequence 未迁移
// 层数 3 分支 :443-448 先行）。数组名保真注册名 Nhr 小写 hr
static const TopoPattern kInsReduceScatterSequenceMesh1DNhrPatterns[] = {
    // 层数 2 Mesh，本实例卡数 > 1（GT 例外），rank <= 255，单卡 > 4MB 且总量 > 4GB
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GT, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GT, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({0, 7})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GT, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({9, kMaxU32})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GT, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GT, LenScale::NONE, k4GB}),
    // 层数 >= 4 Mesh 同上
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GT, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GT, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({0, 7})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GT, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({9, kMaxU32})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GT, LenScale::NONE, k4GB}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetDeviceNumPerModule({8, 8})
        .SetRankSize({0, 255})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GT, LenScale::NONE, k4GB}),
};
constexpr size_t PATTERN_NUM_INS_REDUCE_SCATTER_SEQUENCE_MESH1_D_NHR = 10;
static_assert(
    sizeof(kInsReduceScatterSequenceMesh1DNhrPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_INS_REDUCE_SCATTER_SEQUENCE_MESH1_D_NHR,
    "RS Sequence count drift");

// 多层 CLOS 的 NE 1 判定依据：AICPU 链 :460 读 .at(0)==1 先选 NHR（该支不读 level0Topo）
// [0]==0 与 >1 都落到 :462 CLOS 分支选 MultiLink——两路合一即 != 1，NE 忠实（GT 例外不适用）
static const TopoPattern kAicpuReduceScatterSoleNHRMultiLinkPatterns[] = {
    // 一层 CLOS，普通类型
    TopoPattern().SetLevel({1, 1}).SetLevel0(Level0Shape::CLOS),
    // 层数 2 CLOS
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::CLOS)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::CLOS)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::CLOS)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetDeviceNumPerModule({0, 7}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::CLOS)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetDeviceNumPerModule({9, kMaxU32}),
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::CLOS)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetDeviceNumPerModule({8, 8}),
    // 层数 >= 4 CLOS
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::CLOS)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::CLOS)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::CLOS)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetDeviceNumPerModule({0, 7}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::CLOS)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetDeviceNumPerModule({9, kMaxU32}),
    TopoPattern()
        .SetLevel({4, kMaxU32})
        .SetLevel0(Level0Shape::CLOS)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::NE, 1})
        .SetDeviceNumPerModule({8, 8}),
    // 层数 3 CLOS
    TopoPattern()
        .SetLevel({3, 3})
        .SetLevel0(Level0Shape::CLOS)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1}),
    TopoPattern()
        .SetLevel({3, 3})
        .SetLevel0(Level0Shape::CLOS)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::No)
        .SetLevel0Inst({0, InstCmp::NE, 1}),
    TopoPattern()
        .SetLevel({3, 3})
        .SetLevel0(Level0Shape::CLOS)
        .SetLevel0Nhr(Tri::No)
        .SetLevel1Nhr(Tri::No)
        .SetTopLevelUboe(Tri::No)
        .SetLevel0Symmetric(Tri::Yes)
        .SetLevel1Symmetric(Tri::Yes)
        .SetLevel0Inst({0, InstCmp::NE, 1}),
};
constexpr size_t PATTERN_NUM_AICPU_REDUCE_SCATTER_SOLE_NHR_MULTI_LINK = 14;
static_assert(
    sizeof(kAicpuReduceScatterSoleNHRMultiLinkPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_AICPU_REDUCE_SCATTER_SOLE_NHR_MULTI_LINK,
    "RS MultiLink count drift");

// T_rs(r) = r²·k256KB。
// twoLevelNet 与 topoLevelNums 解耦：IsTwoLevelNetLayer 读 localNetInsSizeOfLayer[0]（本实例
// 卡数）不读 netInstNumOfLayer（实例数），单层拓扑可带两层网——SelectMeshAlgoAicpu 只在单层
// 路径被调，本组 twoLevelNet 条目定义在该形态上（golden 夹具同构，见 UT）
static const TopoPattern kInsReduceScatterMesh1DPatterns[] = {
    // --- normal（普通类型，长度分档生效） ---
    // 纯 Mesh，twoLevel = No：r=0 静态（ratio=1 → 16MB）+ r>=1 公式（T_rs = r²·256KB）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::No)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k16MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::No)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::RANK_SQ, k256KB}),
    // 纯 Mesh，twoLevel = Yes：r=0 静态（总量恒 0，无总量子句）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k16MB}),
    // r>=1 非 2：单卡 <= T_rs 且总量 <= 1536MB 相与。r=2 单列（
    // 公式在 r=2 处与静态条目数值重合，T_rs(2)=1MB，拆分是路径结构而非语义需要）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({1, 1})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::RANK_SQ, k256KB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1536MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({3, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::RANK_SQ, k256KB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1536MB}),
    // r == 2 两层：T_rs(2)=1MB（>= 4MB 绕行 NOT_MATCH，总量子句被 1MB 蕴含）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({2, 2})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k1MB}),
    // PCIe 全互联（复用 SelectMeshAlgoAicpuForMesh1D；Check* 已在 :498 调用，失败即 NOT_MATCH
    // → closMultipleOfMesh 两态），上面 6 形 × 2 态
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::No)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k16MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::No)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::RANK_SQ, k256KB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k16MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({1, 1})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::RANK_SQ, k256KB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1536MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({3, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::RANK_SQ, k256KB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1536MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({2, 2})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k1MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::No)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k16MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::No)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::RANK_SQ, k256KB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k16MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({1, 1})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::RANK_SQ, k256KB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1536MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({3, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::RANK_SQ, k256KB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1536MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({2, 2})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k1MB}),
    // 非 PCIe 全互联（dataSize >= 512KB 改选 Concur；twoLevelNet 在此函数不读留 Any）：
    // r=0 按 ratio=1（16MB > 512KB 松）取 512KB-1；r=1 取 T_rs(1)=256KB（更紧）；r>=2 的
    // T_rs >= 1MB 比 512KB 松，取 512KB-1。closMultipleOfMesh 两态（Check* 已调用）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k512KB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({1, 1})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k256KB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({2, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k512KB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k512KB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({1, 1})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k256KB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({2, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k512KB}),
    // --- special dtype（INT64/UINT64/FP64/PROD 在三个拓扑上不看长度直接 Mesh1D，蓝本 :734-744；
    // PCIe 未全互联的特殊类型 NOT_MATCH 不在表里） ---
    TopoPattern().SetLevel({1, 1}).SetLevel0(Level0Shape::MESH_1D),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes),
};
constexpr size_t PATTERN_NUM_INS_REDUCE_SCATTER_MESH1_D = 29;
static_assert(
    sizeof(kInsReduceScatterMesh1DPatterns) / sizeof(TopoPattern) == PATTERN_NUM_INS_REDUCE_SCATTER_MESH1_D,
    "RS Mesh1D count drift");

// r=0 两层支无「总量 > 1536MB」条目（rank 0 总量恒 0）；rank==2 的 [4MB,∞) 被 2P 绕行 NOT_MATCH 掏空
static const TopoPattern kAicpuReduceScatterSoleMeshChunkPatterns[] = {
    // 纯 Mesh，twoLevel = No：单卡 > T_rs_mesh(rank)。r=0 静态 + r>=1 公式
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::No)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k16MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::No)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::RANK_SQ, k256KB}),
    // 纯 Mesh，twoLevel = Yes：r=0（单卡 > 16MB）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k16MB}),
    // r == 2：单卡 > 1MB 且 < 4MB（>= 4MB 绕行失败；rank 2 总量超 1536MB 必然 >= 4MB，到不了）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({2, 2})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k1MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k4MB}),
    // r>=1 非 2，总量支：总量 > 1536MB
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({1, 1})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GT, LenScale::NONE, k1536MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({3, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GT, LenScale::NONE, k1536MB}),
    // r>=1 非 2，相交支：单卡 > T_rs(rank) 且总量 <= 1536MB
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({1, 1})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::RANK_SQ, k256KB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1536MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({3, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::RANK_SQ, k256KB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1536MB}),
    // PCIe 全互联（复用同函数，closMultipleOfMesh 两态），上面 8 形 × 2 态
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::No)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k16MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::No)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::RANK_SQ, k256KB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k16MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({2, 2})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k1MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k4MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({1, 1})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GT, LenScale::NONE, k1536MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({3, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GT, LenScale::NONE, k1536MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({1, 1})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::RANK_SQ, k256KB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1536MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({3, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::RANK_SQ, k256KB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1536MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::No)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k16MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::No)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::RANK_SQ, k256KB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k16MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({2, 2})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k1MB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k4MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({1, 1})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GT, LenScale::NONE, k1536MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({3, kMaxU32})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GT, LenScale::NONE, k1536MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({1, 1})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::RANK_SQ, k256KB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1536MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetTwoLevelNet(Tri::Yes)
        .SetRankSize({3, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::RANK_SQ, k256KB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, k1536MB}),
    // 非 PCIe 全互联：只有 r==1 有区间（256KB+1 .. 512KB-1；r>=2 时 T_rs >= 1MB 掏空
    // r=0 的 16MB > 512KB 掏空），closMultipleOfMesh 两态
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({1, 1})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k256KB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k512KB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({1, 1})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k256KB})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k512KB}),
};
constexpr size_t PATTERN_NUM_AICPU_REDUCE_SCATTER_SOLE_MESH_CHUNK = 26;
static_assert(
    sizeof(kAicpuReduceScatterSoleMeshChunkPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_AICPU_REDUCE_SCATTER_SOLE_MESH_CHUNK,
    "RS Chunk count drift");

static const TopoPattern kAicpuReduceScatterConcurMeshNHRPatterns[] = {
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::Yes)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GE, LenScale::NONE, k512KB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GE, LenScale::NONE, k512KB}),
};
constexpr size_t PATTERN_NUM_AICPU_REDUCE_SCATTER_CONCUR_MESH_NHR = 2;
static_assert(
    sizeof(kAicpuReduceScatterConcurMeshNHRPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_AICPU_REDUCE_SCATTER_CONCUR_MESH_NHR,
    "RS Concur count drift");

// 对称内存且 >= 16MB 那一截由执行前检查拿掉——非模式字段，spec D13）
static const TopoPattern kAicpuReduceScatterParallelMeshNHRUBXPatterns[] = {
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::No)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GE, LenScale::NONE, k1MB}),
};
constexpr size_t PATTERN_NUM_AICPU_REDUCE_SCATTER_PARALLEL_MESH_NHRUBX = 1;
static_assert(
    sizeof(kAicpuReduceScatterParallelMeshNHRUBXPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_AICPU_REDUCE_SCATTER_PARALLEL_MESH_NHRUBX,
    "RS ParallelUBX count drift");

// 特殊类型在此 NOT_MATCH）
static const TopoPattern kInsReduceScatterParallelMesh1DNHRPciePatterns[] = {
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::No)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k4MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::No)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k4MB}),
};
constexpr size_t PATTERN_NUM_INS_REDUCE_SCATTER_PARALLEL_MESH1_DNHR_PCIE = 2;
static_assert(
    sizeof(kInsReduceScatterParallelMesh1DNHRPciePatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_INS_REDUCE_SCATTER_PARALLEL_MESH1_DNHR_PCIE,
    "RS ParallelPcie count drift");

static const TopoPattern kAicpuReduceScatterPipeLinePciePatterns[] = {
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::No)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GE, LenScale::NONE, k4MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::No)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GE, LenScale::NONE, k4MB}),
};
constexpr size_t PATTERN_NUM_AICPU_REDUCE_SCATTER_PIPE_LINE_PCIE = 2;
static_assert(
    sizeof(kAicpuReduceScatterPipeLinePciePatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_AICPU_REDUCE_SCATTER_PIPE_LINE_PCIE,
    "RS PipeLinePcie count drift");

// 特殊类型（INT64/UINT64/FP64/PROD）专用：拓扑面与普通路径一致，meshFullConnect=No 排除全互联支
static const TopoPattern kInsReduceScatterAicpuReduceNHRPatterns[] = {
    TopoPattern().SetLevel({2, kMaxU32}),
    TopoPattern().SetLevel({1, 1}).SetLevel0(Level0Shape::CLOS),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetClosMultipleOfMesh(Tri::Yes)
        .SetMeshFullConnect(Tri::No),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetClosMultipleOfMesh(Tri::No)
        .SetMeshFullConnect(Tri::No),
};
constexpr size_t PATTERN_NUM_INS_REDUCE_SCATTER_AICPU_REDUCE_NHR = 4;
static_assert(
    sizeof(kInsReduceScatterAicpuReduceNHRPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_INS_REDUCE_SCATTER_AICPU_REDUCE_NHR,
    "RS AicpuReduce count drift");

static const TopoPattern kInsReduceScatterSequenceMeshMeshDPUPatterns[] = {
    TopoPattern().SetLevel({2, kMaxU32}),
};
constexpr size_t PATTERN_NUM_INS_REDUCE_SCATTER_SEQUENCE_MESH_MESH_DPU = 1;
static_assert(
    sizeof(kInsReduceScatterSequenceMeshMeshDPUPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_INS_REDUCE_SCATTER_SEQUENCE_MESH_MESH_DPU,
    "RS DPU count drift");

// 条目 3-4 的 No 按 GT 例外转写（:220 严格 >1 的三路链，[0]==0 走 else 不看框数但按 spec
// §7.1 规则 1/D12 不单列）；两层 CLOS与一层 CLOS分支不读本实例卡数 → Any
static const TopoPattern kCcuSchedReduceScatterSoleNHRPatterns[] = {
    // 两层 Mesh，Level1Nhr，单卡 < 32MB（2die、单实例尚未读取）
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::Yes)
        .SetLevel2UbRtp(Tri::No)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k32MB}),
    // 两层 Mesh，单实例（EQ 1），Level1Nhr 与 2die 已离开
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetTwoDieFullMesh(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetLevel0Inst({0, InstCmp::EQ, 1})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k32MB}),
    // 两层 Mesh，本实例卡数 > 1（GT），rank >= 64，框数 >= 17
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetTwoDieFullMesh(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({64, kMaxU32})
        .SetFrameNum({17, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k32MB}),
    // 两层 Mesh，本实例卡数 > 1（GT），rank <= 63，框数 >= 17，总量 >= 512KB（< 512KB 走 SoleMesh）
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetTwoDieFullMesh(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({0, 63})
        .SetFrameNum({17, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k32MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::GE, LenScale::NONE, k512KB}),
    // 两层 CLOS，无 32MB 门槛
    TopoPattern().SetLevel({2, 2}).SetLevel0(Level0Shape::CLOS).SetLevel2UbRtp(Tri::No),
    // 一层 CLOS，PCIe 混插在进入赋值前返回
    TopoPattern().SetLevel({1, 1}).SetLevel0(Level0Shape::CLOS).SetLevel0PcieMix(Tri::No).SetLevel2UbRtp(Tri::No),
};
constexpr size_t PATTERN_NUM_CCU_SCHED_REDUCE_SCATTER_SOLE_NHR = 6;
static_assert(
    sizeof(kCcuSchedReduceScatterSoleNHRPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_CCU_SCHED_REDUCE_SCATTER_SOLE_NHR,
    "RS CcuSoleNHR count drift");

// 一层 Mesh 上界 = r·k1MB（，dataSize·(8/r) >= 8MB 即 NOT_MATCH；
// r=0 时 ratio=1 → k8MB 静态）；meshType 排除两种 2die；
// PCIe 全互联复用 SelectMeshAlgoCcuScheduleMesh1D 且不调用 Check*→ closMultipleOfMesh Any
static const TopoPattern kCcuSchedReduceScatterSoleMeshPatterns[] = {
    // 两层 Mesh：Level1Nhr/2die 已离开，本实例卡数 > 1（GT），rank <= 63，单卡 < 32MB 且总量 < 512KB
    TopoPattern()
        .SetLevel({2, 2})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetLevel1Nhr(Tri::No)
        .SetTwoDieFullMesh(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetLevel0Inst({0, InstCmp::GT, 1})
        .SetRankSize({0, 63})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k32MB})
        .AddLenClause({LenMetric::TOTAL, LenCmp::LT, LenScale::NONE, k512KB}),
    // 一层 UBX，4 卡相等（meshEqClos），单卡 < 512KB
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetRankSize({0, 4})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k512KB}),
    // 一层 Mesh：r=0 静态 + r>=1 公式（上界 r·1MB-1），各 × meshType{SINGLE_DIE, NOT_MESH}
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetMeshType(Level0MeshType::SINGLE_DIE)
        .SetLevel2UbRtp(Tri::No)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k8MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetMeshType(Level0MeshType::NOT_MESH)
        .SetLevel2UbRtp(Tri::No)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k8MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetMeshType(Level0MeshType::SINGLE_DIE)
        .SetLevel2UbRtp(Tri::No)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::RANK, k1MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D)
        .SetMeshType(Level0MeshType::NOT_MESH)
        .SetLevel2UbRtp(Tri::No)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::RANK, k1MB}),
    // PCIe 全互联（复用同函数，不调 Check* → closMultipleOfMesh Any）× meshType × r
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetMeshType(Level0MeshType::SINGLE_DIE)
        .SetLevel0PcieMix(Tri::Yes)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k8MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetMeshType(Level0MeshType::NOT_MESH)
        .SetLevel0PcieMix(Tri::Yes)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({0, 0})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k8MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetMeshType(Level0MeshType::SINGLE_DIE)
        .SetLevel0PcieMix(Tri::Yes)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::RANK, k1MB}),
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetMeshType(Level0MeshType::NOT_MESH)
        .SetLevel0PcieMix(Tri::Yes)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshFullConnect(Tri::Yes)
        .SetRankSize({1, kMaxU32})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::RANK, k1MB}),
};
constexpr size_t PATTERN_NUM_CCU_SCHED_REDUCE_SCATTER_SOLE_MESH = 10;
static_assert(
    sizeof(kCcuSchedReduceScatterSoleMeshPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_CCU_SCHED_REDUCE_SCATTER_SOLE_MESH,
    "RS CcuSoleMesh count drift");

static const TopoPattern kCcuSchedReduceScatterConcurMeshNHRMultiLinkPatterns[] = {
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetRankSize({0, 4})
        .AddLenClause({LenMetric::PER_RANK, LenCmp::GE, LenScale::NONE, k512KB}),
};
constexpr size_t PATTERN_NUM_CCU_SCHED_REDUCE_SCATTER_CONCUR_MESH_NHR_MULTI_LINK = 1;
static_assert(
    sizeof(kCcuSchedReduceScatterConcurMeshNHRMultiLinkPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_CCU_SCHED_REDUCE_SCATTER_CONCUR_MESH_NHR_MULTI_LINK,
    "RS CcuConcurMultiLink count drift");

// >= 512KB」已返回，剩 3 块；矩形大数据 Parallel/PipeLine 未迁移 → 倍数为真时只留 < 512KB）
static const TopoPattern kCcuSchedReduceScatterSoleNHRMultiLinkPatterns[] = {
    // mesh == clos 且 rank >= 5（rank <= 4 已走 SoleMesh/Concur）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshEqClos(Tri::Yes)
        .SetClosMultipleOfMesh(Tri::No)
        .SetRankSize({5, kMaxU32}),
    // clos 倍数且单卡 < 512KB（>= 512KB 矩形大数据走未迁移 Parallel/PipeLine）
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshEqClos(Tri::No)
        .SetClosMultipleOfMesh(Tri::Yes)
        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k512KB}),
    // 两种数量关系都不成立
    TopoPattern()
        .SetLevel({1, 1})
        .SetLevel0(Level0Shape::MESH_1D_CLOS)
        .SetLevel0PcieMix(Tri::No)
        .SetLevel2UbRtp(Tri::No)
        .SetMeshEqClos(Tri::No)
        .SetClosMultipleOfMesh(Tri::No),
};
constexpr size_t PATTERN_NUM_CCU_SCHED_REDUCE_SCATTER_SOLE_NHR_MULTI_LINK = 3;
static_assert(
    sizeof(kCcuSchedReduceScatterSoleNHRMultiLinkPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_CCU_SCHED_REDUCE_SCATTER_SOLE_NHR_MULTI_LINK,
    "RS CcuSoleMultiLink count drift");

// 手写变体（selector 无自动分支，仅外部名强制可达；Z 轴条件由 COND_Z_AXIS 承载）。
// level={1,1} 依据：twoLevelNet 与 topoLevelNums 解耦——IsTwoLevelNetLayer 读本实例卡数，
// L0 单实例覆盖全卡时单层拓扑仍可为两层网
static const TopoPattern kInsReduceScatterMesh1DZAxisDetourPatterns[] = {
    TopoPattern().SetLevel({1, 1}).SetLevel0(Level0Shape::MESH_1D).SetTwoLevelNet(Tri::Yes).SetRankSize({2, 2}),
};
constexpr size_t PATTERN_NUM_INS_REDUCE_SCATTER_MESH1_DZ_AXIS_DETOUR = 1;
static_assert(
    sizeof(kInsReduceScatterMesh1DZAxisDetourPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_INS_REDUCE_SCATTER_MESH1_DZ_AXIS_DETOUR,
    "RS ZAxisDetour count drift");

// 手写变体（仅外部名强制可达；COND_PEER_ONLY 承载谓词）：实测可用域 1~2 层，3 层+ 保守不进
static const TopoPattern kCcuSchedReduceScatterSoleMeshPeerOnlyPatterns[] = {
    TopoPattern().SetLevel({1, 2}).SetLevel0(Level0Shape::MESH_1D).SetRankSize({2, 2}),
};
constexpr size_t PATTERN_NUM_CCU_SCHED_REDUCE_SCATTER_SOLE_MESH_PEER_ONLY = 1;
static_assert(
    sizeof(kCcuSchedReduceScatterSoleMeshPeerOnlyPatterns) / sizeof(TopoPattern) ==
        PATTERN_NUM_CCU_SCHED_REDUCE_SCATTER_SOLE_MESH_PEER_ONLY,
    "RS PeerOnly count drift");

} // namespace mc2_ops_hccl
#endif // REDUCE_SCATTER_ALG_PATTERNS_H

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TOPO_PATTERN_H
#define TOPO_PATTERN_H

#include <cstdint>
#include <vector>

#include "alg_param.h"

namespace mc2_ops_hccl {

template <typename E>
struct PatternSlot {
    E value;
    bool any;
    PatternSlot() : value(static_cast<E>(0)), any(true) {}
    PatternSlot(E v) : value(v), any(false) {} // NOLINT(google-explicit-constructor)
    bool Matches(E v) const { return any || value == v; }
};
using Level0ShapeSlot = PatternSlot<Level0Shape>;
using Level0MeshTypeSlot = PatternSlot<Level0MeshType>;

enum class Tri : uint8_t { Any, Yes, No };

struct U32Range {
    uint32_t min = 0;
    uint32_t max = 0xFFFFFFFFu; // 含
};

// 共享常量的单一事实源：各族 *alg_patterns.h 逐字引用，禁止本地重定义
static constexpr uint32_t kMaxU32 = 0xFFFFFFFFu;
static constexpr uint64_t k256KB = 256ULL * 1024; // 首用 RS：T_rs = r²·k256KB 的 RANK_SQ bound
static constexpr uint64_t k512KB = 512ULL * 1024;
static constexpr uint64_t k1MB = 1024ULL * 1024;
static constexpr uint64_t k4MB = 4ULL * k1MB;     // 首用 AR：T_ar = r²·k4MB 的 RANK_SQ bound
static constexpr uint64_t k8MB = 8ULL * k1MB;     // 首用 AR：OneShot/TwoShot 分界
static constexpr uint64_t k16MB = 16ULL * k1MB;   // 首用 RS：r=0 的 T_rs（ratio=1）
static constexpr uint64_t k32MB = 32ULL * k1MB;   // 首用 AR：Mesh 走 NHR 的分界 / r=0 的 T_ar
static constexpr uint64_t k150MB = 150ULL * k1MB; // 首用 A2A：SingleChannel 总量分界（镜像 hccl 150MB）
static constexpr uint64_t k1GB = 1024ULL * 1024 * 1024;
static constexpr uint64_t k1536MB = 1536ULL * k1MB; // 首用 RS：twoLevel 总量阈值（Chunk/总量分界）
static constexpr uint64_t k4GB = 4ULL * k1GB;       // 首用 AR：Parallel/ChunkTwoShot 上界

// 单调性契约：每个 metric 必须是 len 的单调不减变换（PER_RANK 恒等 / TOTAL 乘 userRankSize /
// ELEMENT 除 dtypeSize）。NarrowByClause 的反解公式依赖此单调性；新增度量若非单调
// （距离/取模等形态），反解失效，须重审（D15 触发器，spec D17）
enum class LenMetric : uint8_t { PER_RANK, TOTAL, ELEMENT };
enum class LenScale : uint8_t { NONE, RANK, RANK_SQ }; // 阈值 = bound · f(rank)
enum class LenCmp : uint8_t { LT, LE, GT, GE };

// 字段序 = 中缀表达式语序：左（metric 口径）- 符（cmp）- 右（scale 形状 + bound 常数），
// 读法如 {PER_RANK, LT, RANK_SQ, k4MB} = "单卡量 < r²·4MB"
struct LenClause {
    LenMetric metric = LenMetric::PER_RANK;
    LenCmp cmp = LenCmp::LE;
    LenScale scale = LenScale::NONE;
    uint64_t bound = 0;
};

enum class InstCmp : uint8_t { ANY, EQ, NE, GT, GE, LT, LE };

struct InstNumConstraint {
    uint32_t layer = 0;
    InstCmp cmp = InstCmp::ANY;
    uint32_t bound = 0;
};

struct TopoPattern {
    U32Range level;              // topoLevelNums
    Level0ShapeSlot level0;      // 默认不约束
    Level0MeshTypeSlot meshType; // 默认不约束

    Tri level0Nhr = Tri::Any;
    Tri level1Nhr = Tri::Any;
    Tri twoDieFullMesh = Tri::Any;
    Tri level0PcieMix = Tri::Any;
    Tri level0BigClosRange = Tri::Any;
    Tri topLevelUboe = Tri::Any;
    Tri level2UbRtp = Tri::Any;
    Tri level0Symmetric = Tri::Any;
    Tri level1Symmetric = Tri::Any;
    InstNumConstraint level0Inst;

    Tri meshEqClos = Tri::Any;
    Tri closMultipleOfMesh = Tri::Any;
    Tri meshFullConnect = Tri::Any;
    Tri twoLevelNet = Tri::Any;

    U32Range deviceNumPerModule;
    U32Range rankSize;
    U32Range frameNum;

    std::vector<LenClause> lenClauses; // 空 = 任意长度

    // C++14 兼容的指定初始化替代（designated initializers 是 C++20 特性，产品库按 C++14 编译）：
    // TopoPattern().SetLevel({2, kMaxU32}).SetLevel0Nhr(Tri::Yes)... 等价 {.level=..., .level0Nhr=...}
    TopoPattern& SetLevel(const U32Range& v)
    {
        level = v;
        return *this;
    }
    TopoPattern& SetLevel0(Level0Shape v)
    {
        level0 = v;
        return *this;
    }
    TopoPattern& SetMeshType(Level0MeshType v)
    {
        meshType = v;
        return *this;
    }
    TopoPattern& SetLevel0Nhr(Tri v)
    {
        level0Nhr = v;
        return *this;
    }
    TopoPattern& SetLevel1Nhr(Tri v)
    {
        level1Nhr = v;
        return *this;
    }
    TopoPattern& SetTwoDieFullMesh(Tri v)
    {
        twoDieFullMesh = v;
        return *this;
    }
    TopoPattern& SetLevel0PcieMix(Tri v)
    {
        level0PcieMix = v;
        return *this;
    }
    TopoPattern& SetLevel0BigClosRange(Tri v)
    {
        level0BigClosRange = v;
        return *this;
    }
    TopoPattern& SetTopLevelUboe(Tri v)
    {
        topLevelUboe = v;
        return *this;
    }
    TopoPattern& SetLevel2UbRtp(Tri v)
    {
        level2UbRtp = v;
        return *this;
    }
    TopoPattern& SetLevel0Symmetric(Tri v)
    {
        level0Symmetric = v;
        return *this;
    }
    TopoPattern& SetLevel1Symmetric(Tri v)
    {
        level1Symmetric = v;
        return *this;
    }
    TopoPattern& SetLevel0Inst(const InstNumConstraint& v)
    {
        level0Inst = v;
        return *this;
    }
    TopoPattern& SetMeshEqClos(Tri v)
    {
        meshEqClos = v;
        return *this;
    }
    TopoPattern& SetClosMultipleOfMesh(Tri v)
    {
        closMultipleOfMesh = v;
        return *this;
    }
    TopoPattern& SetMeshFullConnect(Tri v)
    {
        meshFullConnect = v;
        return *this;
    }
    TopoPattern& SetTwoLevelNet(Tri v)
    {
        twoLevelNet = v;
        return *this;
    }
    TopoPattern& SetDeviceNumPerModule(const U32Range& v)
    {
        deviceNumPerModule = v;
        return *this;
    }
    TopoPattern& SetRankSize(const U32Range& v)
    {
        rankSize = v;
        return *this;
    }
    TopoPattern& SetFrameNum(const U32Range& v)
    {
        frameNum = v;
        return *this;
    }
    TopoPattern& SetLenClauses(const std::vector<LenClause>& v)
    {
        lenClauses = v;
        return *this;
    }
    TopoPattern& AddLenClause(const LenClause& v)
    {
        lenClauses.push_back(v);
        return *this;
    }
};

// 区间匹配：模式命中当且仅当窗 [lenMin, lenMax] 与全部长度子句解集的交集
// 非空；lenMin/lenMax 为单卡字节数窗端点，(0,0) 哨兵跳过全部长度子句
// 区间传播：把可行窗 [lo, hi] 收窄到子句成立的 len 侧原像，空窗返回 false
bool NarrowByClause(
    const LenClause& c, uint64_t& lo, uint64_t& hi, const TopoInfoWithNetLayerDetails& topo, uint32_t dtypeSize);

bool ValidatePattern(
    const TopoPattern& p, const TopoInfoWithNetLayerDetails& topo, uint64_t& lo, uint64_t& hi, uint32_t dtypeSize = 0U);

bool MatchOne(
    const TopoPattern& pattern, const TopoInfoWithNetLayerDetails& topo, uint64_t lenMin, uint64_t lenMax,
    uint32_t dtypeSize);
bool MatchAnyPattern(
    const TopoPattern* patterns, uint32_t count, const TopoInfoWithNetLayerDetails& topo, uint64_t lenMin,
    uint64_t lenMax, uint32_t dtypeSize);

} // namespace mc2_ops_hccl
#endif // TOPO_PATTERN_H

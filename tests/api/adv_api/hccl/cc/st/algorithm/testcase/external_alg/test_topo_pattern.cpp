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
#include <vector>

#include "alg_param.h"
#include "gtest/gtest.h"
#include "topo_pattern.h"

using namespace mc2_ops_hccl;

namespace {
// 差分/单测复用的最小两层拓扑：MESH_1D 层 0 八实例 + 层 1 单实例
TopoInfoWithNetLayerDetails MakeTwoLevelMeshTopo(u32 rankSize)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.topoLevelNums = 2;
    topo.level0Topo = Level0Shape::MESH_1D;
    topo.userRankSize = rankSize;
    topo.netLayerDetails.localNetInsSizeOfLayer = {8, 1};
    return topo;
}
} // namespace

TEST(TopoPatternTest, TriAndRangeBasics)
{
    auto topo = MakeTwoLevelMeshTopo(8);
    TopoPattern p{};
    p.level = {2, 2};
    p.level0 = Level0Shape::MESH_1D;
    p.level0Nhr = Tri::No;
    p.rankSize = {2, 8};
    EXPECT_TRUE(MatchOne(p, topo, 0, 0, 0));
    p.level0Nhr = Tri::Yes;
    EXPECT_FALSE(MatchOne(p, topo, 0, 0, 0));
}

TEST(TopoPatternTest, InstNumConstraintFourStates)
{
    auto topo = MakeTwoLevelMeshTopo(8); // localNetInsSizeOfLayer = {8, 1}
    TopoPattern p{};
    p.level0Inst = {0, InstCmp::GT, 1};
    EXPECT_TRUE(MatchOne(p, topo, 0, 0, 0));
    p.level0Inst = {0, InstCmp::EQ, 1};
    EXPECT_FALSE(MatchOne(p, topo, 0, 0, 0));
    p.level0Inst = {1, InstCmp::EQ, 1};
    EXPECT_TRUE(MatchOne(p, topo, 0, 0, 0));
    p.level0Inst = {3, InstCmp::GT, 0}; // layer 越界
    EXPECT_FALSE(MatchOne(p, topo, 0, 0, 0));
    p.level0Inst = {};
    EXPECT_TRUE(MatchOne(p, topo, 0, 0, 0));
    topo.netLayerDetails.localNetInsSizeOfLayer.clear(); // 空向量：受约束不命中
    p.level0Inst = {0, InstCmp::NE, 1};
    EXPECT_FALSE(MatchOne(p, topo, 0, 0, 0));
}

TEST(TopoPatternTest, FormulaClauseExactBoundary)
{
    auto topo = MakeTwoLevelMeshTopo(8); // r=8，r·1MB = 8388608
    TopoPattern p{};
    p.rankSize = {1, UINT32_MAX};
    p.lenClauses = {{LenMetric::PER_RANK, LenCmp::LT, LenScale::RANK, 1024ULL * 1024}};
    EXPECT_TRUE(MatchOne(p, topo, 8388607, 8388607, 0));
    EXPECT_FALSE(MatchOne(p, topo, 8388608, 8388608, 0));
    EXPECT_TRUE(MatchOne(p, topo, 0, 0, 0)); // 哨兵 (0,0) 跳过长度子句
}

TEST(TopoPatternTest, RankSqAndElementAndSaturation)
{
    auto topo = MakeTwoLevelMeshTopo(8); // r²·512KB = 64·512KB = 32MB
    TopoPattern p{};
    p.rankSize = {1, UINT32_MAX};
    p.lenClauses = {{LenMetric::PER_RANK, LenCmp::LE, LenScale::RANK_SQ, 512ULL * 1024}};
    EXPECT_TRUE(MatchOne(p, topo, 32ULL * 1024 * 1024, 32ULL * 1024 * 1024, 0));
    EXPECT_FALSE(MatchOne(p, topo, 32ULL * 1024 * 1024 + 1, 32ULL * 1024 * 1024 + 1, 0));

    TopoPattern e{};
    e.lenClauses = {{LenMetric::ELEMENT, LenCmp::GT, LenScale::NONE, 512}};
    EXPECT_FALSE(MatchOne(e, topo, 800, 800, 2));   // 800B/2B=400 元素 ≤512 → 不命中
    EXPECT_FALSE(MatchOne(e, topo, 1025, 1025, 2)); // 1025B/2B=512（整型除法截断）→ 不命中
    EXPECT_TRUE(MatchOne(e, topo, 1027, 1027, 2));  // 1027B/2B=513 > 512 → 命中

    TopoPattern t{};
    t.lenClauses = {{LenMetric::TOTAL, LenCmp::GT, LenScale::NONE, 100}};
    EXPECT_TRUE(MatchOne(t, topo, UINT64_MAX / 2, UINT64_MAX / 2, 0)); // 饱和：不 UB、不崩溃
}

TEST(TopoPatternTest, MatchCheckedRejectsOnCheckFailure)
{
    TopoInfoWithNetLayerDetails topo{}; // 空拓扑：Check* 返回失败
    topo.level0Topo = Level0Shape::MESH_1D_CLOS;
    TopoPattern p{};
    p.meshEqClos = Tri::Yes;
    EXPECT_FALSE(MatchOne(p, topo, 0, 0, 0)); // MatchChecked 对 status!=SUCCESS 直接否决
}

TEST(TopoPatternTest, FrameNumConstraintWithValidInstSizeList)
{
    auto topo = MakeTwoLevelMeshTopo(8);              // topoLevelNums=2、userRankSize=8
    topo.netLayerDetails.instSizeListOfLayer = {{8}}; // 层0单实例8卡：GCD=8 → frameNum=8/8=1（8卡一框）
    TopoPattern p{};
    p.frameNum = {1, 1};
    EXPECT_TRUE(MatchOne(p, topo, 0, 0, 0));
    p.frameNum = {2, UINT32_MAX};
    EXPECT_FALSE(MatchOne(p, topo, 0, 0, 0));
}

TEST(TopoPatternTest, FrameNumConstraintRejectsOnEmptyInstSizeList)
{
    auto topo = MakeTwoLevelMeshTopo(8); // topoLevelNums>1 且 instSizeListOfLayer 外层为空
    TopoPattern p{};
    p.frameNum = {1, 1};
    EXPECT_FALSE(MatchOne(p, topo, 0, 0, 0)); // 无法计算 frameNum：保守否决，不崩溃
}

// D17 端点法：探针方向 = LT/LE 看 lenMin、GT/GE 看 lenMax——窗与子句解集（半区间）相交
// 即命中；窗内极值在端点（单调 metric 前提，见 topo_pattern.h 契约），探针构造性命中
// 宽窗 × 条件掏空档：r=32 时 GT r²·4MB(=4GB) ∧ LE 4GB 合取矛盾——传播法须判空，
// 端点法在此假阳性（逐条判交各自成立，无公共解）。回归自 GitCode 检视（ChunkTwoShot r>=3 档）
TEST(TopoPatternTest, MatchOneWideWindowEmptyConjunction)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRankSize = 32U;
    topo.topoLevelNums = 1U; // 拓扑面放行（level={1,1}），逼长度面执行
    topo.level0Topo = Level0Shape::MESH_1D;
    constexpr uint64_t k1G = 1024ULL * 1024 * 1024;
    TopoPattern p = TopoPattern()
                        .SetLevel({1, 1})
                        .SetLevel0(Level0Shape::MESH_1D)
                        .SetRankSize({3, kMaxU32})
                        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k8MB})
                        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::RANK_SQ, k4MB})
                        .AddLenClause({LenMetric::PER_RANK, LenCmp::LE, LenScale::NONE, k4GB});
    EXPECT_FALSE(MatchOne(p, topo, 1U, 8ULL * k1G, 0U));       // 宽窗 [1,8GB]：合取空，不命中
    EXPECT_FALSE(MatchOne(p, topo, k4GB, k4GB, 0U));           // 单值 4GB（GT 不含）
    EXPECT_FALSE(MatchOne(p, topo, k4GB + 1U, k4GB + 1U, 0U)); // 单值 4GB+1（LE 不含）
    // 对照：r=31 时 r²·4MB≈3.8GB < 4GB，合取非空——宽窗与单点一致命中
    TopoInfoWithNetLayerDetails topo31 = topo;
    topo31.userRankSize = 31U;
    EXPECT_TRUE(MatchOne(p, topo31, 1U, 8ULL * k1G, 0U));
    EXPECT_TRUE(MatchOne(p, topo31, 3900ULL * 1024 * 1024, 3900ULL * 1024 * 1024, 0U));
    EXPECT_FALSE(MatchOne(p, topo31, 3ULL * k1G, 3ULL * k1G, 0U)); // < 3.8GB 不命中
}

// 混合 metric 子句的宽窗传播：PER_RANK 下界 ∧ TOTAL 上界 ∧ ELEMENT 下界同时收窄
TEST(TopoPatternTest, MatchOneMixedMetricWideWindow)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRankSize = 8U; // TOTAL 阈值 = len×8
    topo.topoLevelNums = 1U;
    topo.level0Topo = Level0Shape::MESH_1D;
    TopoPattern p = TopoPattern()
                        .SetLevel({1, 1})
                        .SetLevel0(Level0Shape::MESH_1D)
                        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, 64}) // len > 64B
                        .AddLenClause({LenMetric::TOTAL, LenCmp::LE, LenScale::NONE, 1024}) // len×8 ≤ 1024 → len ≤ 128
                        .AddLenClause({LenMetric::ELEMENT, LenCmp::GE, LenScale::NONE, 10}); // len/4 ≥ 10 → len ≥ 40
    // 公共解 [65,128]（∩ [40,∞)=[65,128]）
    EXPECT_TRUE(MatchOne(p, topo, 1U, 4096U, 4U));   // 宽窗覆盖 [65,128]
    EXPECT_TRUE(MatchOne(p, topo, 100U, 100U, 4U));  // 单点 100 ∈ [65,128]
    EXPECT_TRUE(MatchOne(p, topo, 65U, 65U, 4U));    // 公共解下界含（GT 严格 >64）
    EXPECT_FALSE(MatchOne(p, topo, 64U, 64U, 4U));   // 64 不过 GT 64
    EXPECT_FALSE(MatchOne(p, topo, 129U, 129U, 4U)); // 129×8=1032 > 1024 不过 TOTAL
    EXPECT_FALSE(MatchOne(p, topo, 1U, 39U, 4U));    // 全窗 < 公共解下界
}

TEST(TopoPatternTest, MatchOneWindowEndpointDirection)
{
    auto topo = MakeTwoLevelMeshTopo(8); // r=8，既有夹具
    TopoPattern p1 =
        TopoPattern().SetRankSize({0, kMaxU32}).AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k16MB});
    EXPECT_TRUE(MatchOne(p1, topo, 0, 32 * k1MB, 0)); // lenMin=0 < 16MB：窗与解集相交
    TopoPattern p2 =
        TopoPattern().SetRankSize({0, kMaxU32}).AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k16MB});
    EXPECT_TRUE(MatchOne(p2, topo, 0, 32 * k1MB, 0));          // lenMax=32MB > 16MB：窗与解集相交
    EXPECT_FALSE(MatchOne(p1, topo, 24 * k1MB, 32 * k1MB, 0)); // lenMin=24MB 不小于 16MB
    EXPECT_FALSE(MatchOne(p2, topo, 0, 8 * k1MB, 0));          // lenMax=8MB 不大于 16MB
}

// [x,x] 单值窗逐位等于点匹配语义：双子句（GT k4MB + LT r·1MB=8MB）在边界点两侧扫描
TEST(TopoPatternTest, MatchOneSingleValueEqualsPointSemantics)
{
    auto topo = MakeTwoLevelMeshTopo(8); // r=8 → LT RANK k1MB 阈值 = 8MB
    TopoPattern p = TopoPattern()
                        .SetRankSize({0, kMaxU32})
                        .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
                        .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::RANK, k1MB}); // < 8MB（r=8）
    for (uint64_t x : {k4MB, k4MB + 1, k8MB - 1, k8MB, k8MB + 1, k4MB - 1}) {
        EXPECT_EQ(MatchOne(p, topo, x, x, 0), (x > k4MB && x < k8MB));
    }
}

// 哨兵 (0,0)：外部名定名（恒传 (0,0)）与查询不校验长度共用，长度面整体跳过
TEST(TopoPatternTest, MatchOneSentinelSkipsLen)
{
    auto topo = MakeTwoLevelMeshTopo(8);
    TopoPattern p =
        TopoPattern().SetRankSize({0, kMaxU32}).AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB});
    EXPECT_TRUE(MatchOne(p, topo, 0, 0, 0)); // 哨兵：跳过长度子句（点语义 0 > 4MB 本应不命中）
}

// D17：病态合取（解集交为空）被校验器拒绝；合法 pattern 给出幸存窗
TEST(TopoPatternTest, ValidatePatternRejectsEmptyDomain)
{
    auto topo = MakeTwoLevelMeshTopo(8); // 沿用本文件既有夹具（r=8），下同
    // 合法：len > 4MB 且 < 16MB
    TopoPattern ok = TopoPattern()
                         .SetRankSize({0, kMaxU32})
                         .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k4MB})
                         .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k16MB});
    uint64_t lo = 0;
    uint64_t hi = UINT64_MAX;
    EXPECT_TRUE(ValidatePattern(ok, topo, lo, hi));
    EXPECT_EQ(lo, k4MB + 1);  // 幸存窗下端
    EXPECT_EQ(hi, k16MB - 1); // 幸存窗上端
    // 病态：> 16MB 且 < 4MB
    TopoPattern bad = TopoPattern()
                          .SetRankSize({0, kMaxU32})
                          .AddLenClause({LenMetric::PER_RANK, LenCmp::GT, LenScale::NONE, k16MB})
                          .AddLenClause({LenMetric::PER_RANK, LenCmp::LT, LenScale::NONE, k4MB});
    EXPECT_FALSE(ValidatePattern(bad, topo, lo, hi));
}

// 40 万穷举（12 格 × 边界值域）：NarrowByClause 逐格反解与"小域逐点真值"对照（python 已验
// 401,280 组零分叉的 C++ 版；C++ 侧 scale 另作独立轴，实际跑 3×36 格）。r 域按 userRankSize
// 的 u32 字宽取 1<<16 与 0xFFFFFFFF（r²=2^32 / SatMul(r,r) 饱和），承接原设 1<<32/1<<40 的
// 饱和压测意图；真值侧阈值按 scale 同步饱和换算，与 MatchLenClause 同源
TEST(TopoPatternTest, NarrowByClauseExhaustiveAgainstGroundTruth)
{
    const uint64_t MAXv = UINT64_MAX;
    const uint32_t rs[] = {0, 1, 2, 3, 5, 8, 1U << 16, 0xFFFFFFFFu};
    const uint64_t ts[] = {0, 1, 2, 3, 7, 8, 15, 16, 1ull << 32, MAXv - 1, MAXv};
    const uint64_t ds[] = {1, 2, 3, 7, 16};
    std::vector<uint64_t> lens;
    for (uint64_t x = 0; x < 70; x++) {
        lens.push_back(x);
    }
    const uint64_t kBigLens[] = {1ull << 32, 1ull << 40, 1ull << 63, (1ull << 63) + 1, MAXv - 1, MAXv};
    for (const uint64_t x : kBigLens) {
        lens.push_back(x);
    }
    // 非负饱和参考：与产品 SatMul 同式（先判后乘）——直接 (a*b <= MAX) 判别对 uint64 恒真，
    // 退化为回绕乘，对照就不是同一道题
    const auto satMulRef = [](uint64_t a, uint64_t b) { return (a == 0U || b <= MAXv / a) ? a * b : MAXv; };
    for (int mi = 0; mi <= 2; mi++) {
        for (int ci = 0; ci <= 3; ci++) {
            for (int si = 0; si <= 2; si++) {
                for (const uint32_t r : rs) {
                    for (const uint64_t t0 : ts) {
                        for (const uint64_t d : ds) {
                            auto topo = MakeTwoLevelMeshTopo(r); // userRankSize=r（既有夹具）
                            const LenMetric metric = static_cast<LenMetric>(mi);
                            const LenCmp cmp = static_cast<LenCmp>(ci);
                            const LenScale scale = static_cast<LenScale>(si);
                            // 真值构造：先算 threshold（bound 经 scale 饱和换算）再比值；
                            // LenClause 里 bound 传 t0
                            uint64_t threshold = t0;
                            if (scale == LenScale::RANK) {
                                threshold = satMulRef(t0, r);
                            } else if (scale == LenScale::RANK_SQ) {
                                threshold = satMulRef(t0, satMulRef(r, r));
                            }
                            for (const uint64_t x : lens) {
                                const LenClause c{metric, cmp, scale, t0}; // C++14：成员按声明序
                                uint64_t value = 0;
                                if (metric == LenMetric::PER_RANK) {
                                    value = x;
                                } else if (metric == LenMetric::TOTAL) {
                                    value = satMulRef(x, r);
                                } else {
                                    if (d == 0) {
                                        continue; // d 恒非 0，防御
                                    }
                                    value = x / d;
                                }
                                const bool truth = (cmp == LenCmp::LT) ? (value < threshold) :
                                                   (cmp == LenCmp::LE) ? (value <= threshold) :
                                                   (cmp == LenCmp::GT) ? (value > threshold) :
                                                                         (value >= threshold);
                                uint64_t lo = x;
                                uint64_t hi = x; // 单值窗 [x,x]
                                EXPECT_EQ(NarrowByClause(c, lo, hi, topo, static_cast<uint32_t>(d)), truth)
                                    << "m=" << mi << " c=" << ci << " s=" << si << " r=" << r << " t=" << t0
                                    << " d=" << d << " x=" << x;
                            }
                        }
                    }
                }
            }
        }
    }
}

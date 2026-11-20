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
#include "topo_pattern.h"
#include "auto_selector_base.h"

namespace mc2_ops_hccl {
namespace {
const AutoSelectorBase kMatchSelectorBase;

constexpr U32Range kAnyRange{};

uint64_t SatMul(uint64_t a, uint64_t b)
{
    if (a != 0 && b > UINT64_MAX / a) {
        return UINT64_MAX;
    }
    return a * b;
}

uint64_t CeilDiv(uint64_t a, uint64_t b) { return (a == 0U) ? 0U : 1U + (a - 1U) / b; }

bool RealMulOver(uint64_t a, uint64_t b) { return a != 0U && b > UINT64_MAX / a; }

bool MatchTri(Tri pattern, bool value)
{
    if (pattern == Tri::Any) {
        return true;
    }
    return (pattern == Tri::Yes) == value;
}

bool MatchRange(const U32Range& range, uint32_t value) { return value >= range.min && value <= range.max; }

bool MatchChecked(Tri pattern, HcclResult status, bool value)
{
    if (pattern == Tri::Any) {
        return true;
    }
    if (status != HCCL_SUCCESS) {
        return false;
    }
    return MatchTri(pattern, value);
}

bool MatchInstNum(const InstNumConstraint& c, const TopoInfoWithNetLayerDetails& topo)
{
    if (c.cmp == InstCmp::ANY) {
        return true;
    }
    const std::vector<uint32_t>& layer = topo.netLayerDetails.localNetInsSizeOfLayer;
    if (c.layer >= layer.size()) {
        return false;
    }
    const uint32_t n = layer[c.layer];
    switch (c.cmp) {
        case InstCmp::EQ:
            return n == c.bound;
        case InstCmp::NE:
            return n != c.bound;
        case InstCmp::GT:
            return n > c.bound;
        case InstCmp::GE:
            return n >= c.bound;
        case InstCmp::LT:
            return n < c.bound;
        case InstCmp::LE:
            return n <= c.bound;
        default:
            return false;
    }
}

bool MatchLenClause(const LenClause& c, uint64_t len, const TopoInfoWithNetLayerDetails& topo, uint32_t dtypeSize)
{
    uint64_t value = 0;
    switch (c.metric) {
        case LenMetric::PER_RANK:
            value = len;
            break;
        case LenMetric::TOTAL:
            value = SatMul(len, topo.userRankSize);
            break;
        case LenMetric::ELEMENT:
            if (dtypeSize == 0U) {
                return false;
            }
            value = len / dtypeSize;
            break;
        default:
            return false;
    }
    uint64_t threshold = c.bound;
    switch (c.scale) {
        case LenScale::RANK:
            threshold = SatMul(c.bound, topo.userRankSize);
            break;
        case LenScale::RANK_SQ:
            threshold = SatMul(c.bound, SatMul(topo.userRankSize, topo.userRankSize));
            break;
        case LenScale::NONE:
        default:
            break;
    }
    switch (c.cmp) {
        case LenCmp::LT:
            return value < threshold;
        case LenCmp::LE:
            return value <= threshold;
        case LenCmp::GT:
            return value > threshold;
        case LenCmp::GE:
            return value >= threshold;
        default:
            return false;
    }
}
} // namespace

bool NarrowByClause(
    const LenClause& c, uint64_t& lo, uint64_t& hi, const TopoInfoWithNetLayerDetails& topo, uint32_t dtypeSize)
{
    const uint64_t r = topo.userRankSize;
    uint64_t t = c.bound;
    switch (c.scale) {
        case LenScale::RANK:
            t = SatMul(c.bound, r);
            break;
        case LenScale::RANK_SQ:
            t = SatMul(c.bound, SatMul(r, r));
            break;
        case LenScale::NONE:
        default:
            break;
    }
    switch (c.metric) {
        case LenMetric::PER_RANK:
            switch (c.cmp) {
                case LenCmp::LT:
                    if (t == 0U) {
                        return false;
                    }
                    hi = std::min(hi, t - 1U);
                    break;
                case LenCmp::LE:
                    hi = std::min(hi, t);
                    break;
                case LenCmp::GT:
                    if (t == UINT64_MAX) {
                        return false;
                    }
                    lo = std::max(lo, t + 1U);
                    break;
                case LenCmp::GE:
                default:
                    lo = std::max(lo, t);
                    break;
            }
            return lo <= hi;
        case LenMetric::TOTAL:
            if (r == 0U) { // value=SatMul(len,0) 恒 0，约束与 len 无关：真→全窗放行，假→空窗
                switch (c.cmp) {
                    case LenCmp::LT:
                        return t > 0U && lo <= hi;
                    case LenCmp::LE:
                        return lo <= hi;
                    case LenCmp::GT:
                        return false;
                    case LenCmp::GE:
                    default:
                        return t == 0U && lo <= hi;
                }
            }
            switch (c.cmp) {
                case LenCmp::LT:
                    if (t == 0U) {
                        return false;
                    }
                    hi = std::min(hi, CeilDiv(t, r) - 1U);
                    break;
                case LenCmp::LE:
                    if (t == UINT64_MAX) {
                        return lo <= hi; // 饱和复刻：钳位 MAX<=MAX 恒真
                    }
                    hi = std::min(hi, t / r);
                    break;
                case LenCmp::GT:
                    if (t == UINT64_MAX) {
                        return false; // 饱和复刻：MAX>MAX 恒假
                    }
                    lo = std::max(lo, t / r + 1U);
                    break;
                case LenCmp::GE:
                default:
                    lo = std::max(lo, CeilDiv(t, r));
                    break;
            }
            return lo <= hi;
        case LenMetric::ELEMENT:
            if (dtypeSize == 0U) {
                return false; // 防除零，同 MatchLenClause
            }
            switch (c.cmp) {
                case LenCmp::GT:
                    if (t == UINT64_MAX || RealMulOver(t + 1U, dtypeSize)) {
                        return false;
                    }
                    lo = std::max(lo, (t + 1U) * dtypeSize);
                    break;
                case LenCmp::GE:
                    if (RealMulOver(t, dtypeSize)) {
                        return false;
                    }
                    lo = std::max(lo, t * dtypeSize);
                    break;
                case LenCmp::LT:
                    if (t == 0U) {
                        return false;
                    }
                    if (RealMulOver(t, dtypeSize)) {
                        return lo <= hi; // len/d <= MAX/d < t 恒真
                    }
                    hi = std::min(hi, t * dtypeSize - 1U);
                    break;
                case LenCmp::LE:
                default:
                    if (t == UINT64_MAX || RealMulOver(t + 1U, dtypeSize)) {
                        return lo <= hi;
                    }
                    hi = std::min(hi, (t + 1U) * dtypeSize - 1U);
                    break;
            }
            return lo <= hi;
        default:
            return false;
    }
}

bool ValidatePattern(
    const TopoPattern& p, const TopoInfoWithNetLayerDetails& topo, uint64_t& lo, uint64_t& hi, uint32_t dtypeSize)
{
    lo = 0U;
    hi = UINT64_MAX;
    for (const LenClause& clause : p.lenClauses) {
        if (!NarrowByClause(clause, lo, hi, topo, dtypeSize)) {
            return false;
        }
    }
    return true;
}

bool MatchOne(
    const TopoPattern& p, const TopoInfoWithNetLayerDetails& topo, uint64_t lenMin, uint64_t lenMax, uint32_t dtypeSize)
{
    if (!MatchRange(p.level, topo.topoLevelNums)) {
        return false;
    }
    if (!p.level0.Matches(topo.level0Topo)) {
        return false;
    }
    if (!p.meshType.Matches(topo.level0MeshType)) {
        return false;
    }
    if (!MatchTri(p.level0Nhr, topo.Level0Nhr)) {
        return false;
    }
    if (!MatchTri(p.level1Nhr, topo.Level1Nhr)) {
        return false;
    }
    if (!MatchTri(p.twoDieFullMesh, topo.is2DieFullMesh)) {
        return false;
    }
    if (!MatchTri(p.level0PcieMix, topo.level0PcieMix)) {
        return false;
    }
    if (!MatchTri(p.level0BigClosRange, topo.level0BigClosRange)) {
        return false;
    }
    if (!MatchTri(p.topLevelUboe, topo.topLevelUboe)) {
        return false;
    }
    if (!MatchTri(p.level2UbRtp, topo.level2UbRtp)) {
        return false;
    }
    if (!MatchTri(p.level0Symmetric, topo.level0Symmetric)) {
        return false;
    }
    if (!MatchTri(p.level1Symmetric, topo.level1Symmetric)) {
        return false;
    }
    if (!MatchInstNum(p.level0Inst, topo)) {
        return false;
    }
    if (!MatchRange(p.deviceNumPerModule, topo.deviceNumPerModule)) {
        return false;
    }
    if (!MatchRange(p.rankSize, topo.userRankSize)) {
        return false;
    }

    if (p.meshEqClos != Tri::Any) {
        bool meshEq = false;
        const HcclResult meshEqStatus = kMatchSelectorBase.CheckMeshNumEqualToClosNum(&topo, meshEq);
        if (!MatchChecked(p.meshEqClos, meshEqStatus, meshEq)) {
            return false;
        }
    }
    if (p.closMultipleOfMesh != Tri::Any) {
        bool closMultiple = false;
        const HcclResult closMultipleStatus = kMatchSelectorBase.CheckClosNumMultipleOfMeshNum(&topo, closMultiple);
        if (!MatchChecked(p.closMultipleOfMesh, closMultipleStatus, closMultiple)) {
            return false;
        }
    }
    if (p.meshFullConnect != Tri::Any) {
        const bool full = kMatchSelectorBase.IsLayerAllConnetedWithTopo(&topo, 0U, CommTopo::COMM_TOPO_1DMESH);
        if (!MatchTri(p.meshFullConnect, full)) {
            return false;
        }
    }
    if (p.twoLevelNet != Tri::Any) {
        if (!MatchTri(p.twoLevelNet, kMatchSelectorBase.IsTwoLevelNetLayer(&topo))) {
            return false;
        }
    }

    const bool frameConstrained = p.frameNum.min != kAnyRange.min || p.frameNum.max != kAnyRange.max;
    if (frameConstrained) {
        uint32_t frameNum = 0;
        if (topo.topoLevelNums > 1U) {
            if (topo.netLayerDetails.instSizeListOfLayer.empty()) {
                return false;
            }
            frameNum = AutoSelectorBase::CalcFrameNum(&topo);
        }
        if (!MatchRange(p.frameNum, frameNum)) {
            return false;
        }
    }
    // 哨兵 (0,0)：外部名定名（恒传）与查询不校验长度共用，跳过全部长度子句
    if (lenMin == 0U && lenMax == 0U) {
        return true;
    }
    // 区间传播：窗作初值逐子句收窄（NarrowByClause 反解），空即短路——逐条判端点只能证明
    // 各子句与窗相交，证明不了子句合取有公共解（条件掏空档如 r>=32 的 r²·4MB>=4GB 会假阳性）
    uint64_t lo = lenMin;
    uint64_t hi = lenMax;
    for (const LenClause& clause : p.lenClauses) {
        if (!NarrowByClause(clause, lo, hi, topo, dtypeSize)) {
            return false;
        }
    }
    return true;
}

bool MatchAnyPattern(
    const TopoPattern* patterns, uint32_t count, const TopoInfoWithNetLayerDetails& topo, uint64_t lenMin,
    uint64_t lenMax, uint32_t dtypeSize)
{
    for (uint32_t i = 0; i < count; ++i) {
        if (MatchOne(patterns[i], topo, lenMin, lenMax, dtypeSize)) {
            return true;
        }
    }
    return false;
}

} // namespace mc2_ops_hccl

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALG_META_REGISTRY_H
#define ALG_META_REGISTRY_H

#include <cstddef>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "hccl_types.h"   // HcclCMDType / HcclResult
#include "topo_pattern.h" // TopoPattern（拓扑模式数组）

namespace mc2_ops_hccl {

// 候选引擎维度。显式声明，不按内部名前缀推导（前缀有历史包袱，如 InsAllGatherMeshNhrDPU 是
// AICPU sequence 编排 + DPU 卸载）。扩展位：CcuMS、AIV。
enum class AlgEngine { AICPU, CCU, DPU };

struct AlgMeta {
    AlgEngine engine = AlgEngine::AICPU;
    const char* externalName = ""; // 规范化外部名（与 parser 产出同构），如 "sole[mesh]"
    int variantCondId = -1;
    const TopoPattern* patterns = nullptr; // TopoPattern 数组（patternCount 计数）
    uint32_t patternCount = 0;
    int priority = 0; // 同键定序第二级（specificity > priority > name，见 BuildCandidateTable）
};

struct AlgMetaRow {
    HcclCMDType cmd;
    std::string registeredName;
    AlgMeta meta;
};

class AlgMetaRegistry {
public:
    static AlgMetaRegistry& Instance();
    HcclResult Register(HcclCMDType cmd, const std::string& registeredName, const AlgMeta& meta);
    const AlgMeta* Get(HcclCMDType cmd, const std::string& registeredName) const;
    std::vector<AlgMetaRow> GetAll() const;

private:
    AlgMetaRegistry() = default;
    std::map<std::pair<HcclCMDType, std::string>, AlgMeta> metas_;
    mutable std::mutex mu_;
};

// 宏配套的编译期计数：数组引用形参只接受数组实参——传指针时模板推导失败（编译错），
// 杜绝 sizeof(指针)/sizeof(元素) 静默归 0 的误用
template <typename T, std::size_t N>
constexpr uint32_t PatternCountOf(const T (&)[N])
{
    return static_cast<uint32_t>(N);
}

// sidecar 注册宏（与 REGISTER_EXEC_V2 同款 static-init 套路，见
// coll_alg_v2_exec_registry.h:46-48）。必须写在主注册同一条件编译块内，
// 保证 sidecar 与主注册同生共死（守卫天然同步）。
// patterns 须为静态数组（PatternCountOf 编译期计数）；priority 显式写出（无裁决意图恒填 0）；
// 加参须动全部现场，签名冻结由此自执行
#define REGISTER_ALG_META_HELPER(ctr, cmd, regName, engine, extName, condId, patterns, prio) \
    static HcclResult g_alg_meta_##regName##_##ctr = AlgMetaRegistry::Instance().Register(   \
        cmd, std::string(#regName), AlgMeta{engine, extName, condId, patterns, PatternCountOf(patterns), prio})

#define REGISTER_ALG_META_HELPER_1(ctr, cmd, regName, engine, extName, condId, patterns, prio) \
    REGISTER_ALG_META_HELPER(ctr, cmd, regName, engine, extName, condId, patterns, prio)

#define REGISTER_ALG_META(cmd, regName, engine, extName, condId, patterns, prio) \
    REGISTER_ALG_META_HELPER_1(__COUNTER__, cmd, regName, engine, extName, condId, patterns, prio)

} // namespace mc2_ops_hccl
#endif // ALG_META_REGISTRY_H

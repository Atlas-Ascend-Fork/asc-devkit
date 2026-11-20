/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef EXTERNAL_ALG_RULES_H
#define EXTERNAL_ALG_RULES_H

#include "alg_meta_registry.h" // AlgEngine
#include "alg_param.h"         // OpParam / TopoInfoWithNetLayerDetails

namespace mc2_ops_hccl {

// AlgMeta::variantCondId 取值；-1 = 无条件
enum VariantCondId {
    COND_NONE = -1,
    COND_Z_AXIS = 0, // 2P 大数据或总量超阈值；两层网拓扑面由 ZAxisDetour TopoPattern 承载 -> ZAxisDetour 候选
    COND_PEER_ONLY,  // rankSize==2 且对端仅 1 条链路 -> PeerOnly 候选
    COND_SPECIAL_DT, // 64bit dtype 或 PROD 归约 -> soft_reduce(AicpuReduce) 候选
    VARIANT_COND_COUNT
};

using VariantCond = bool (*)(const TopoInfoWithNetLayerDetails* topo, const OpParam& opParam);

bool CondZAxis(const TopoInfoWithNetLayerDetails* topo, const OpParam& opParam);
bool CondPeerOnly(const TopoInfoWithNetLayerDetails* topo, const OpParam& opParam);
bool CondSpecialDt(const TopoInfoWithNetLayerDetails* topo, const OpParam& opParam);

bool EngineMatch(AlgEngine candEngine, const OpParam& opParam);

// enum 下标函数指针表：候选只带 ID，ID->函数绑定集中于此
extern const VariantCond VARIANT_COND_TABLE[VARIANT_COND_COUNT];
const char* VariantCondName(int id); // 日志可打印规则名

} // namespace mc2_ops_hccl
#endif // EXTERNAL_ALG_RULES_H

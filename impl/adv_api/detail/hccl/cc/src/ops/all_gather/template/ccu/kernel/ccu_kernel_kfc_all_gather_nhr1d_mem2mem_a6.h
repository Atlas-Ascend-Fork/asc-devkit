/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_KFC_ALL_GATHER_NHR_1D_MEM2MEM_A6_H
#define HCCL_CCU_KERNEL_KFC_ALL_GATHER_NHR_1D_MEM2MEM_A6_H

#ifndef CCU_DATAPLANE_HCOMM
#define CCU_DATAPLANE_HCOMM
#endif

#include <map>
#include <vector>
#include "ccu_kernel_alg_base.h"
#include "kfc_server_protocol.h"

namespace mc2_ops_hccl {

// KFC 版 AllGather NHR 1D Mem2Mem kernel（CcuSchedAllGatherSoleNHR，A6/hcomm 数据面）：
// 逻辑与 A5 版 ccu_kernel_kfc_all_gather_nhr1d_mem2mem.cc 一致。常量参数（die0LastSize/
// repeatNum/三 stride）不占 xnData 槽，由 kernel 内联；isInputOutputEqual 经 goSize 全零表达。
// A6 无 D3 turn-end 提前：不接收 turnEndAddr，由 server 在 dispatch 后统一写完成信号。
CcuResult CcuKernelKfcAllGatherNHR1DMem2MemKernel(
    ccu::Variable inputAddr, ccu::Variable outputAddr, ccu::Variable tokenInfo, ccu::Variable die0Size,
    ccu::Variable die1Size, ccu::Variable outputSliceStride, ccu::Variable goSize0, ccu::Variable goSize1,
    ccu::Variable goSize2, ccu::Variable goSize3, const ChannelHandle channels[], uint32_t channelCount,
    uint32_t rankSize, uint32_t rankId, uint32_t axisId, const std::vector<KfcNhrStepInfo>& stepInfoVector,
    const std::map<uint32_t, uint32_t>& rank2ChannelIdx);
} // namespace mc2_ops_hccl

#endif // HCCL_CCU_KERNEL_KFC_ALL_GATHER_NHR_1D_MEM2MEM_A6_H

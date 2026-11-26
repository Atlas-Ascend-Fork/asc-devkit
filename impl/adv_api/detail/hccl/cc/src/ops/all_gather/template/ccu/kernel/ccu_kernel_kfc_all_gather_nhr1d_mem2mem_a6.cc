/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CCU_DATAPLANE_HCOMM
#define CCU_DATAPLANE_HCOMM
#endif

#include "ccu_kernel_kfc_all_gather_nhr1d_mem2mem_a6.h"

namespace mc2_ops_hccl {
namespace {
// XN 槽位与 CKE 信号位定义，与 A5 版（ccu_kernel_kfc_all_gather_nhr1d_mem2mem.cc）保持一致。
constexpr uint16_t AG_NHR_OUTPUT_XN_ID = 1;
constexpr uint16_t AG_NHR_TOKEN_XN_ID = 2;
constexpr uint16_t AG_NHR_POST_SYNC_BIT = 1U << 3U;
constexpr uint16_t AG_NHR_STEP_POST_SYNC_BIT = 1U << 5U;
constexpr uint16_t AG_NHR_CKE_IDX = 0;
constexpr uint16_t AG_NHR_BIT_NUM_PER_CKE = 16;

struct KfcAllGatherNhr1DContext : CcuKernelCtxBase {
    const ChannelHandle* channels = nullptr;
    uint32_t channelCount = 0;
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    const std::vector<KfcNhrStepInfo>* stepInfoVector = nullptr;
    const std::map<uint32_t, uint32_t>* rank2ChannelIdx = nullptr;

    ccu::Variable input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    ccu::Variable die0Size;
    ccu::Variable die0LastSize;
    ccu::Variable repeatNum;
    ccu::Variable inputSliceStride;
    ccu::Variable outputSliceStride;
    ccu::Variable inputRepeatStride;
    ccu::Variable outputRepeatStride;
    ccu::Event localEvent;
    ccu::Variable repeatTimeflag;
    std::vector<ccu::Variable> outputSliceOffset;
    ccu::Variable myrankInputSliceOffset;
    ccu::LocalAddr srcMem;
    ccu::RemoteAddr dstMem;
    ccu::LocalAddr localDst;
    ccu::Variable constVar1;
    GroupOpSizeVars goSize;
    ccu::Variable groupCopyRepeatNum;
};

CcuResult InitResource(KfcAllGatherNhr1DContext& ctx)
{
    if (ctx.rankSize == 0U || ctx.rankId >= ctx.rankSize || ctx.channelCount == 0U || ctx.rank2ChannelIdx == nullptr ||
        ctx.rank2ChannelIdx->size() != ctx.channelCount || ctx.stepInfoVector == nullptr) {
        HCCL_ERROR(
            "[CcuKernelKfcAllGatherNHR1DMem2Mem] invalid resource, rankSize[%u], rankId[%u], channels[%u]",
            ctx.rankSize, ctx.rankId, ctx.channelCount);
        return CCU_E_INTERNAL;
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.channelCount; ++channelIdx) {
        // 拷贝构造保存句柄共享的通道绑定变量，operator= 会在 PreSync 握手前快照旧槽位值
        ctx.output.push_back(ccu::GetResByChannel<ccu::Variable>(ctx.channels[channelIdx], AG_NHR_OUTPUT_XN_ID));
        ctx.token.push_back(ccu::GetResByChannel<ccu::Variable>(ctx.channels[channelIdx], AG_NHR_TOKEN_XN_ID));
    }
    // 本地槽 [channelCount]：值来自本轮参数帧，LoadNhr1DArgs 值赋值无竞争
    ctx.output.push_back(ccu::Variable());
    ctx.token.push_back(ccu::Variable());
    ctx.outputSliceOffset.resize(ctx.rankSize);
    ctx.myrankInputSliceOffset = 0;
    ctx.repeatTimeflag = 0;
    ctx.constVar1 = 1;
    return CCU_SUCCESS;
}

// sole-NHR 常量参数不占 xnData 槽，在此内联：die0LastSize=die0Size、repeatNum=1
// （UINT64_MAX-1）、三 stride=0；双 die 时 die0Size/die0LastSize 复用为“本 die 档”，
// axisId=1 将 die0Size 基偏移烘焙进 input/output[localIdx]。
void LoadNhr1DArgs(
    KfcAllGatherNhr1DContext& ctx, ccu::Variable input, ccu::Variable output, ccu::Variable token,
    ccu::Variable die0Size, ccu::Variable die1Size, ccu::Variable outputSliceStride, ccu::Variable goSize0,
    ccu::Variable goSize1, ccu::Variable goSize2, ccu::Variable goSize3, uint32_t axisId)
{
    const uint32_t localIdx = ctx.channelCount;
    ctx.input = input;
    ctx.output[localIdx] = output;
    ctx.token[localIdx] = token;
    if (axisId == 0U) {
        ctx.die0Size = die0Size;
        ctx.die0LastSize = die0Size;
    } else {
        ctx.die0Size = die1Size;
        ctx.die0LastSize = die1Size;
        ctx.input += die0Size;
        ctx.output[localIdx] += die0Size;
    }
    ctx.repeatNum = UINT64_MAX - 1ULL;
    ctx.inputSliceStride = 0;
    ctx.outputSliceStride = outputSliceStride;
    ctx.inputRepeatStride = 0;
    ctx.outputRepeatStride = 0;
    ctx.goSize.addrOffset = goSize0;
    ctx.goSize.loopParam = goSize1;
    ctx.goSize.parallelParam = goSize2;
    ctx.goSize.residual = goSize3;
}

CcuResult PreSync(KfcAllGatherNhr1DContext& ctx)
{
    const uint32_t localIdx = ctx.channelCount;
    for (uint32_t i = 0; i < ctx.channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            ctx.channels[i], ctx.output[localIdx], AG_NHR_OUTPUT_XN_ID, AG_NHR_CKE_IDX, 1U << AG_NHR_OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            ctx.channels[i], ctx.token[localIdx], AG_NHR_TOKEN_XN_ID, AG_NHR_CKE_IDX, 1U << AG_NHR_TOKEN_XN_ID));
    }
    const uint16_t allBit = (1U << AG_NHR_OUTPUT_XN_ID) | (1U << AG_NHR_TOKEN_XN_ID);
    for (uint32_t i = 0; i < ctx.channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.channels[i], AG_NHR_CKE_IDX, allBit));
    }
    return CCU_SUCCESS;
}

CcuResult PostSync(KfcAllGatherNhr1DContext& ctx)
{
    for (uint32_t i = 0; i < ctx.channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(ctx.channels[i], AG_NHR_CKE_IDX, AG_NHR_POST_SYNC_BIT));
    }
    for (uint32_t i = 0; i < ctx.channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.channels[i], AG_NHR_CKE_IDX, AG_NHR_POST_SYNC_BIT));
    }
    return CCU_SUCCESS;
}

// 只发射单个 slice 的 Write/占位 EventRecord，不等待；事件位由调用方分配
CcuResult IssueSendSlice(
    KfcAllGatherNhr1DContext& ctx, const u32& toRank, ccu::LocalAddr& src, ccu::RemoteAddr& dst, u32 signalIndex,
    bool islastSlice)
{
    const ChannelHandle sendChannel = ctx.channels[ctx.rank2ChannelIdx->at(toRank)];
    ccu::Variable& sliceSize = islastSlice ? ctx.die0LastSize : ctx.die0Size;
    const uint16_t signalMask = 1U << signalIndex;
    CCU_IF(sliceSize != 0) { CCU_CHK_RET(ccu::Write(sendChannel, dst, src, sliceSize, ctx.localEvent, signalMask)); }
    CCU_ELSE { CCU_CHK_RET(ccu::EventRecord(ctx.localEvent, signalMask)); }
    return CCU_SUCCESS;
}

// 串行回退路径：发射单个 slice 并立即等待其完成
CcuResult DoRepeatSendRecvSlices(
    KfcAllGatherNhr1DContext& ctx, const u32& toRank, ccu::LocalAddr& src, ccu::RemoteAddr& dst, u32 signalIndex,
    bool islastSlice)
{
    const ChannelHandle sendChannel = ctx.channels[ctx.rank2ChannelIdx->at(toRank)];
    ctx.repeatTimeflag = 0;
    ccu::Variable tmpRepeatNum;
    tmpRepeatNum = ctx.repeatNum;

    CCU_WHILE(tmpRepeatNum != UINT64_MAX)
    {
        tmpRepeatNum += ctx.constVar1;
        CCU_IF(ctx.repeatTimeflag == 1)
        {
            src.addr += ctx.inputRepeatStride;
            dst.addr += ctx.outputRepeatStride;
        }
        ccu::Variable& sliceSize = islastSlice ? ctx.die0LastSize : ctx.die0Size;

        const uint16_t signalMask = 1U << signalIndex;
        CCU_IF(sliceSize != 0)
        {
            CCU_CHK_RET(ccu::Write(sendChannel, dst, src, sliceSize, ctx.localEvent, signalMask));
            CCU_CHK_RET(ccu::EventWait(ctx.localEvent, signalMask));
        }
        ctx.repeatTimeflag = 1;
    }

    return CCU_SUCCESS;
}

// 本地拷贝自身分片；A6 需显式传 GetCcuVersion() 走 GroupCopyV2
CcuResult DoAllGatherGroupCopy(KfcAllGatherNhr1DContext& ctx)
{
    CCU_IF(ctx.groupCopyRepeatNum != UINT64_MAX)
    {
        ctx.repeatTimeflag = 0;
        CCU_WHILE(ctx.groupCopyRepeatNum != UINT64_MAX)
        {
            ctx.groupCopyRepeatNum += ctx.constVar1;
            CCU_IF(ctx.repeatTimeflag != 0)
            {
                ctx.localDst.addr += ctx.outputRepeatStride;
                ctx.srcMem.addr += ctx.inputRepeatStride;
            }
            CCU_CHK_RET(GroupCopy(ctx, ctx.localDst, ctx.srcMem, ctx.goSize, GetCcuVersion()));
            ctx.repeatTimeflag = 1;
        }
    }
    return CCU_SUCCESS;
}

// own slot 由 input 直发，不依赖 GroupCopy 落位，拷贝方可推迟至末步
void SetSendSrcAddr(KfcAllGatherNhr1DContext& ctx, const KfcNhrStepInfo& nhrStepInfo, u32 sendSliceIdx)
{
    if (nhrStepInfo.step == 0 || sendSliceIdx == ctx.rankId) {
        ctx.srcMem.addr = ctx.input;
        ctx.srcMem.addr += ctx.myrankInputSliceOffset;
    } else {
        ctx.srcMem.addr = ctx.output[ctx.channelCount];
        ctx.srcMem.addr += ctx.outputSliceOffset[sendSliceIdx];
    }
}

// 末批 Write 在飞时提交本地拷贝，拷贝被末步写耗时掩盖
CcuResult IssueLocalCopy(KfcAllGatherNhr1DContext& ctx)
{
    ctx.srcMem.addr = ctx.input;
    ctx.srcMem.addr += ctx.myrankInputSliceOffset;
    ctx.localDst.addr = ctx.output[ctx.channelCount];
    ctx.localDst.addr += ctx.outputSliceOffset[ctx.rankId];
    return DoAllGatherGroupCopy(ctx);
}

// 步内流水：本轮所有 slice 先全部发射（各占独立事件位），轮末一次 EventWait 批量等待
CcuResult DoPipelinedSendSlices(
    KfcAllGatherNhr1DContext& ctx, const KfcNhrStepInfo& nhrStepInfo, const std::vector<u32>& sendSliceIdxList,
    u32 toRankIdx, bool isLastStep)
{
    const uint32_t sliceNum = static_cast<uint32_t>(sendSliceIdxList.size());
    const uint16_t waitMask = static_cast<uint16_t>((1U << sliceNum) - 1U);
    ccu::Variable turnOffsetIn;
    ccu::Variable turnOffsetOut;
    turnOffsetIn = 0;
    turnOffsetOut = 0;
    ctx.repeatTimeflag = 0;
    ccu::Variable tmpRepeatNum;
    tmpRepeatNum = ctx.repeatNum;
    CCU_WHILE(tmpRepeatNum != UINT64_MAX)
    {
        tmpRepeatNum += ctx.constVar1;
        for (uint32_t sliceSeq = 0; sliceSeq < sliceNum; ++sliceSeq) {
            const u32 sendSliceIdx = sendSliceIdxList[sliceSeq];
            const bool islastSlice = (sendSliceIdx + 1 == ctx.rankSize);
            SetSendSrcAddr(ctx, nhrStepInfo, sendSliceIdx);
            ctx.srcMem.addr += turnOffsetIn;
            ctx.dstMem.addr = ctx.output[toRankIdx];
            ctx.dstMem.addr += ctx.outputSliceOffset[sendSliceIdx];
            ctx.dstMem.addr += turnOffsetOut;
            CCU_CHK_RET(IssueSendSlice(ctx, nhrStepInfo.toRank, ctx.srcMem, ctx.dstMem, sliceSeq, islastSlice));
        }
        if (isLastStep) {
            CCU_CHK_RET(IssueLocalCopy(ctx));
        }
        CCU_CHK_RET(ccu::EventWait(ctx.localEvent, waitMask));
        turnOffsetIn += ctx.inputRepeatStride;
        turnOffsetOut += ctx.outputRepeatStride;
        ctx.repeatTimeflag = 1;
    }
    return CCU_SUCCESS;
}

// 串行回退（slice 超 16 位事件预算或空步）：保持原“发一个等一个”结构
CcuResult DoSerialSendSlices(
    KfcAllGatherNhr1DContext& ctx, const KfcNhrStepInfo& nhrStepInfo, const std::vector<u32>& sendSliceIdxList,
    u32 toRankIdx, bool isLastStep)
{
    const uint32_t sliceNum = static_cast<uint32_t>(sendSliceIdxList.size());
    for (u32 i = 0; i < sliceNum; i++) {
        const u32 sendSliceIdx = sendSliceIdxList[i];
        SetSendSrcAddr(ctx, nhrStepInfo, sendSliceIdx);
        ctx.dstMem.addr = ctx.output[toRankIdx];
        ctx.dstMem.addr += ctx.outputSliceOffset[sendSliceIdx];
        const bool islastSlice = (sendSliceIdx + 1 == ctx.rankSize);
        CCU_CHK_RET(DoRepeatSendRecvSlices(
            ctx, nhrStepInfo.toRank, ctx.srcMem, ctx.dstMem, i % AG_NHR_BIT_NUM_PER_CKE, islastSlice));
    }
    if (isLastStep) {
        CCU_CHK_RET(IssueLocalCopy(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult DoRepeatAllGatherNHRSingleStep(KfcAllGatherNhr1DContext& ctx, const KfcNhrStepInfo& nhrStepInfo)
{
    const u32 toRankIdx = ctx.rank2ChannelIdx->at(nhrStepInfo.toRank);
    const u32 fromRankIdx = ctx.rank2ChannelIdx->at(nhrStepInfo.fromRank);
    const ChannelHandle sendChannel = ctx.channels[toRankIdx];
    const ChannelHandle recvChannel = ctx.channels[fromRankIdx];
    const std::vector<u32>& sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    const bool isLastStep = (nhrStepInfo.step + 1 == ctx.stepInfoVector->size());

    ctx.srcMem.token = ctx.token[ctx.channelCount];
    ctx.dstMem.token = ctx.token[toRankIdx];

    const uint32_t sliceNum = static_cast<uint32_t>(sendSliceIdxList.size());
    if (sliceNum != 0U && sliceNum <= AG_NHR_BIT_NUM_PER_CKE) {
        CCU_CHK_RET(DoPipelinedSendSlices(ctx, nhrStepInfo, sendSliceIdxList, toRankIdx, isLastStep));
    } else {
        CCU_CHK_RET(DoSerialSendSlices(ctx, nhrStepInfo, sendSliceIdxList, toRankIdx, isLastStep));
    }

    // A6 无 D3 turn-end 提前：末步也做步间同步，turn-end 由 server 在 dispatch 后统一写
    CCU_CHK_RET(ccu::NotifyRecord(sendChannel, AG_NHR_CKE_IDX, AG_NHR_STEP_POST_SYNC_BIT));
    CCU_CHK_RET(ccu::NotifyWait(recvChannel, AG_NHR_CKE_IDX, AG_NHR_STEP_POST_SYNC_BIT));

    return CCU_SUCCESS;
}

CcuResult DoRepeatAllGatherNHR(KfcAllGatherNhr1DContext& ctx)
{
    for (u64 i = 0; i < ctx.rankId; i++) {
        ctx.myrankInputSliceOffset += ctx.inputSliceStride;
    }

    ccu::Variable tmpSliceOffset;
    tmpSliceOffset = 0;
    for (u64 i = 0; i < ctx.rankSize; i++) {
        ctx.outputSliceOffset[i] = tmpSliceOffset;
        tmpSliceOffset += ctx.outputSliceStride;
    }

    const uint32_t localIdx = ctx.channelCount;
    ctx.srcMem.addr = ctx.input;
    ctx.srcMem.addr += ctx.myrankInputSliceOffset;
    ctx.srcMem.token = ctx.token[localIdx];
    ctx.localDst.addr = ctx.output[localIdx];
    ctx.localDst.addr += ctx.outputSliceOffset[ctx.rankId];
    ctx.localDst.token = ctx.token[localIdx];
    ctx.groupCopyRepeatNum = ctx.repeatNum;

    if (ctx.stepInfoVector->empty()) {
        // 空步骤向量（防御）：退化为仅本地拷贝
        CCU_CHK_RET(DoAllGatherGroupCopy(ctx));
        return CCU_SUCCESS;
    }

    for (const KfcNhrStepInfo& nhrStepInfo : *ctx.stepInfoVector) {
        CCU_CHK_RET(DoRepeatAllGatherNHRSingleStep(ctx, nhrStepInfo));
    }

    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuKernelKfcAllGatherNHR1DMem2MemKernel(
    ccu::Variable inputAddr, ccu::Variable outputAddr, ccu::Variable tokenInfo, ccu::Variable die0Size,
    ccu::Variable die1Size, ccu::Variable outputSliceStride, ccu::Variable goSize0, ccu::Variable goSize1,
    ccu::Variable goSize2, ccu::Variable goSize3, const ChannelHandle channels[], uint32_t channelCount,
    uint32_t rankSize, uint32_t rankId, uint32_t axisId, const std::vector<KfcNhrStepInfo>& stepInfoVector,
    const std::map<uint32_t, uint32_t>& rank2ChannelIdx)
{
    KfcAllGatherNhr1DContext ctx;
    ctx.channels = channels;
    ctx.channelCount = channelCount;
    ctx.rankSize = rankSize;
    ctx.rankId = rankId;
    ctx.stepInfoVector = &stepInfoVector;
    ctx.rank2ChannelIdx = &rank2ChannelIdx;
    InitCcuKernelCtxBase(ctx);
    CCU_CHK_RET(InitResource(ctx));

    LoadNhr1DArgs(
        ctx, inputAddr, outputAddr, tokenInfo, die0Size, die1Size, outputSliceStride, goSize0, goSize1, goSize2,
        goSize3, axisId);

    CCU_CHK_RET(PreSync(ctx));

    CCU_CHK_RET(DoRepeatAllGatherNHR(ctx));

    CCU_CHK_RET(PostSync(ctx));

    return CCU_SUCCESS;
}

} // namespace mc2_ops_hccl

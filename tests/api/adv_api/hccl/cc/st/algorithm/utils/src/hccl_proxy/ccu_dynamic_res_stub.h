/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_DYNAMIC_RES_STUB_H
#define CCU_DYNAMIC_RES_STUB_H

#include "hccl_res.h"
#include "hccl_ccu_res.h"
#include "ccu_res_defs_dl.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>

namespace mc2_ops_hccl {
constexpr uint32_t CCU_STUB_RES_TYPE_COUNT = 7U;
// 句柄空间须远大于单次用例内并存的 desc 数（4 rank 并发 + 每 rank 多个 desc），
// 否则不同线程会拿到同一句柄并互相覆盖 nums 表。
constexpr uint32_t CCU_STUB_MAX_DESC_HANDLES = 4096U;
// C++14 兼容的 header-only 全局（inline 变量是 C++17）：inline 函数内 static 跨 TU 唯一
struct CcuDynResStubState {
    std::atomic<uint32_t> resDescCounter{0U};
    std::mutex assignedMutex;
    std::map<HcclComm, CcuInsHandle> assignedInsByComm;
    // 实例创建时下发的资源容量，供复用路径 HcommCcuInsQueryResDesc 返回真实容量
    std::mutex insMutex;
    std::map<CcuInsHandle, std::array<uint32_t, CCU_STUB_RES_TYPE_COUNT>> insCapacity;
};
inline CcuDynResStubState& CcuDynResStub()
{
    static CcuDynResStubState state;
    return state;
}
struct CcuStubDescNums {
    std::mutex mutex;
    uint32_t nums[CCU_STUB_MAX_DESC_HANDLES][CCU_STUB_RES_TYPE_COUNT];
};
inline CcuStubDescNums& CcuStubDescNumsTable()
{
    static CcuStubDescNums table;
    return table;
}
} // namespace mc2_ops_hccl

extern "C" {
bool HcommIsSupportHcommCcuInsResDescCreate(void) { return true; }
bool HcommIsSupportHcommCcuInsResDescDestroy(void) { return true; }
bool HcommIsSupportHcommCcuInsResDescSetNum(void) { return true; }
bool HcommIsSupportHcommCcuInsResDescQueryNum(void) { return true; }
bool HcommIsSupportHcommCcuInsCreate(void) { return true; }
bool HcommIsSupportHcommCcuInsDestroy(void) { return true; }
bool HcommIsSupportHcommCcuInsQueryResDesc(void) { return true; }
bool HcommIsSupportHcommCcuQueryRemainResDesc(void) { return true; }
bool HcommIsSupportHcommCcuKernelQueryResReq(void) { return true; }
bool HcommIsSupportHcclCommAssignCcuIns(void) { return true; }
bool HcommIsSupportHcclCommQueryAssignedCcuIns(void) { return true; }

CcuResult HcommCcuInsResDescCreate(uint32_t dieId, HcommCcuResDescHandle* resDesc)
{
    (void)dieId;
    if (resDesc == nullptr) {
        return CCU_E_PTR;
    }
    const uint32_t counter = mc2_ops_hccl::CcuDynResStub().resDescCounter.fetch_add(1U, std::memory_order_relaxed);
    const uint32_t handle = (counter % (mc2_ops_hccl::CCU_STUB_MAX_DESC_HANDLES - 1U)) + 1U;
    auto& table = mc2_ops_hccl::CcuStubDescNumsTable();
    std::lock_guard<std::mutex> lock(table.mutex);
    for (uint32_t t = 0; t < mc2_ops_hccl::CCU_STUB_RES_TYPE_COUNT; t++) {
        table.nums[handle][t] = 0U;
    }
    *resDesc = static_cast<HcommCcuResDescHandle>(handle);
    return CCU_SUCCESS;
}

CcuResult HcommCcuInsResDescDestroy(HcommCcuResDescHandle resDesc)
{
    (void)resDesc;
    return CCU_SUCCESS;
}

CcuResult HcommCcuInsResDescSetNum(HcommCcuResDescHandle resDesc, HcommCcuResType resType, uint32_t num)
{
    const uint32_t handle = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(resDesc));
    const uint32_t type = static_cast<uint32_t>(resType);
    if (handle == 0U || handle >= mc2_ops_hccl::CCU_STUB_MAX_DESC_HANDLES ||
        type >= mc2_ops_hccl::CCU_STUB_RES_TYPE_COUNT) {
        return CCU_E_PARA;
    }
    auto& table = mc2_ops_hccl::CcuStubDescNumsTable();
    std::lock_guard<std::mutex> lock(table.mutex);
    table.nums[handle][type] = num;
    return CCU_SUCCESS;
}

CcuResult HcommCcuInsResDescQueryNum(HcommCcuResDescHandle resDesc, HcommCcuResType resType, uint32_t* num)
{
    if (num == nullptr) {
        return CCU_E_PTR;
    }
    const uint32_t handle = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(resDesc));
    const uint32_t type = static_cast<uint32_t>(resType);
    if (handle == 0U || handle >= mc2_ops_hccl::CCU_STUB_MAX_DESC_HANDLES ||
        type >= mc2_ops_hccl::CCU_STUB_RES_TYPE_COUNT) {
        return CCU_E_PARA;
    }
    auto& table = mc2_ops_hccl::CcuStubDescNumsTable();
    std::lock_guard<std::mutex> lock(table.mutex);
    *num = table.nums[handle][type];
    return CCU_SUCCESS;
}

CcuResult HcommCcuInsCreate(const HcommCcuResDescHandle* resDescs, uint32_t resDescNum, CcuInsHandle* insHandle)
{
    if (resDescNum == 0U || insHandle == nullptr || resDescs == nullptr) {
        return CCU_E_PARA;
    }
    static std::atomic<uint64_t> stubInsCounter{0U};
    const CcuInsHandle handle = static_cast<CcuInsHandle>(stubInsCounter.fetch_add(1U) + 1U);

    // 按创建时下发的 finalReqDesc 建模实例容量（多 die 取 max）：复用路径据此判定
    // 容量是否充足，替代原先恒 0 的假容量（会把正常复用误判为资源不足）。
    std::array<uint32_t, mc2_ops_hccl::CCU_STUB_RES_TYPE_COUNT> capacity{};
    auto& table = mc2_ops_hccl::CcuStubDescNumsTable();
    {
        std::lock_guard<std::mutex> lock(table.mutex);
        for (uint32_t i = 0; i < resDescNum; i++) {
            const uint32_t descHandle = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(resDescs[i]));
            if (descHandle == 0U || descHandle >= mc2_ops_hccl::CCU_STUB_MAX_DESC_HANDLES) {
                continue;
            }
            for (uint32_t t = 0; t < mc2_ops_hccl::CCU_STUB_RES_TYPE_COUNT; t++) {
                if (table.nums[descHandle][t] > capacity[t]) {
                    capacity[t] = table.nums[descHandle][t];
                }
            }
        }
    }
    {
        std::lock_guard<std::mutex> lock(mc2_ops_hccl::CcuDynResStub().insMutex);
        mc2_ops_hccl::CcuDynResStub().insCapacity[handle] = capacity;
    }
    *insHandle = handle;
    return CCU_SUCCESS;
}

CcuResult HcommCcuInsDestroy(CcuInsHandle insHandle)
{
    std::lock_guard<std::mutex> lock(mc2_ops_hccl::CcuDynResStub().insMutex);
    mc2_ops_hccl::CcuDynResStub().insCapacity.erase(insHandle);
    return CCU_SUCCESS;
}

HcclResult HcclCommAssignCcuIns(HcclComm comm, CcuInsHandle insHandle)
{
    std::lock_guard<std::mutex> lock(mc2_ops_hccl::CcuDynResStub().assignedMutex);
    mc2_ops_hccl::CcuDynResStub().assignedInsByComm[comm] = insHandle;
    return HCCL_SUCCESS;
}

HcclResult HcclCommQueryAssignedCcuIns(HcclComm comm, CcuInsHandle* insHandles, uint32_t* insNum)
{
    std::lock_guard<std::mutex> lock(mc2_ops_hccl::CcuDynResStub().assignedMutex);
    auto it = mc2_ops_hccl::CcuDynResStub().assignedInsByComm.find(comm);
    if (it == mc2_ops_hccl::CcuDynResStub().assignedInsByComm.end()) {
        return HCCL_E_UNAVAIL;
    }
    if (insHandles == nullptr || insNum == nullptr) {
        return HCCL_E_PTR;
    }
    insHandles[0] = it->second;
    *insNum = 1U;
    return HCCL_SUCCESS;
}

// 通信域销毁时清理 Assign 绑定：comm 地址被后续用例复用时若命中陈旧句柄，会误走复用路径
// 并对旧实例做容量比对，进而返回 HCCL_E_UNAVAIL。清理保证每个新通信域都从“未绑定”开始。
void CcuStubReleaseComm(HcclComm comm)
{
    auto& state = mc2_ops_hccl::CcuDynResStub();
    std::lock_guard<std::mutex> lock(state.assignedMutex);
    auto it = state.assignedInsByComm.find(comm);
    if (it == state.assignedInsByComm.end()) {
        return;
    }
    const CcuInsHandle insHandle = it->second;
    state.assignedInsByComm.erase(it);
    std::lock_guard<std::mutex> insLock(state.insMutex);
    state.insCapacity.erase(insHandle);
}

CcuResult HcommCcuInsQueryResDesc(CcuInsHandle insHandle, HcommCcuResDescHandle resDesc)
{
    const uint32_t handle = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(resDesc));
    if (handle == 0U || handle >= mc2_ops_hccl::CCU_STUB_MAX_DESC_HANDLES) {
        return CCU_E_PARA;
    }
    // 未登记的实例（如 legacy 通路句柄）按“容量充足”处理，避免桩世界产生假性不足。
    std::array<uint32_t, mc2_ops_hccl::CCU_STUB_RES_TYPE_COUNT> capacity;
    capacity.fill(UINT32_MAX);
    {
        std::lock_guard<std::mutex> lock(mc2_ops_hccl::CcuDynResStub().insMutex);
        auto it = mc2_ops_hccl::CcuDynResStub().insCapacity.find(insHandle);
        if (it != mc2_ops_hccl::CcuDynResStub().insCapacity.end()) {
            capacity = it->second;
        }
    }
    auto& table = mc2_ops_hccl::CcuStubDescNumsTable();
    std::lock_guard<std::mutex> lock(table.mutex);
    for (uint32_t t = 0; t < mc2_ops_hccl::CCU_STUB_RES_TYPE_COUNT; t++) {
        table.nums[handle][t] = capacity[t];
    }
    return CCU_SUCCESS;
}

CcuResult HcommCcuQueryRemainResDesc(HcommCcuResDescHandle resDesc)
{
    const uint32_t handle = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(resDesc));
    if (handle == 0U || handle >= mc2_ops_hccl::CCU_STUB_MAX_DESC_HANDLES) {
        return CCU_E_PARA;
    }
    // 桩世界不模拟资源消耗，剩余资源视为充足（仅用于回退诊断摘要）。
    auto& table = mc2_ops_hccl::CcuStubDescNumsTable();
    std::lock_guard<std::mutex> lock(table.mutex);
    for (uint32_t t = 0; t < mc2_ops_hccl::CCU_STUB_RES_TYPE_COUNT; t++) {
        table.nums[handle][t] = UINT32_MAX;
    }
    return CCU_SUCCESS;
}
} // extern "C"

#endif // CCU_DYNAMIC_RES_STUB_H

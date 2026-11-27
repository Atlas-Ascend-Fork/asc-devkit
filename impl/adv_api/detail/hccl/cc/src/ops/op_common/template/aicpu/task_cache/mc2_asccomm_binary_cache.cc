/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mc2_aicpu_task_cache.h"
#include "log.h"
#include <cstring>
#include <vector>
#include <atomic>
#include <cstdlib>
#include <dlfcn.h>
#include <utility>
#include "asccomm_primitives.h"
#include "asccomm_command_replay.h"
#include "mc2_server_stage_timing.h"

namespace mc2_ops_hccl {
namespace {
// Private v1 capability. counts[ASCCOMM_COUNT_SIZE] = command, SQE, WQE, DB counts of the sealed template.
using BinaryFn = int32_t (*)(uint32_t, void**, AsccommReplayInfoV1*, uint64_t*);
BinaryFn BinaryBackend()
{
    static BinaryFn fn = reinterpret_cast<BinaryFn>(dlsym(RTLD_DEFAULT, "asccomm_binary_replay_v1"));
    return fn;
}
using BinaryV2Fn = int32_t (*)(uint32_t, void**, AsccommReplayInfoV1*, uint64_t*, const uint64_t*);
BinaryV2Fn BinaryV2Backend()
{
    static auto fn = reinterpret_cast<BinaryV2Fn>(dlsym(RTLD_DEFAULT, "asccomm_binary_replay_v2"));
    return fn;
}
int32_t BinaryCall(
    uint32_t action, void** handle, AsccommReplayInfoV1* info, uint64_t* counts, const Mc2AicpuTaskCachePlan& plan)
{
    const uint64_t ranges[] = {
        reinterpret_cast<uint64_t>(plan.addrs[0]), plan.sizes[0], reinterpret_cast<uint64_t>(plan.addrs[1]),
        plan.sizes[1]};
    return BinaryV2Backend() ? BinaryV2Backend()(action, handle, info, counts, ranges) :
                               BinaryBackend()(action, handle, info, counts);
}
constexpr uint64_t MAX_BINARY_CACHE_BYTES = 64U * 1024U * 1024U;
constexpr uint64_t MAX_DOMAIN_BYTES = ASCCOMM_BINARY_ENTRY_BUDGET;
constexpr size_t MAX_DOMAINS = 256U;
constexpr size_t MAX_ENTRIES_PER_DOMAIN = 1024U;
std::atomic<uint64_t> binaryCacheBytes{0};
thread_local std::shared_ptr<const AlgResourceCtxSerializable> currentHolder;
thread_local std::shared_ptr<Mc2BinaryCacheDomain> currentDomain;
thread_local uint64_t scopeSequence = 0, currentScope = 0;
thread_local size_t currentRepeat = 0;
thread_local Mc2BinaryCacheDomain* lockedDomain = nullptr;

struct BinaryCacheEntry {
    void* handle = nullptr;
    uint64_t bytes = 0;
    ~BinaryCacheEntry()
    {
        if (handle != nullptr) {
            (void)BinaryBackend()(ASCCOMM_REPLAY_DESTROY, &handle, nullptr, nullptr);
        }
        binaryCacheBytes.fetch_sub(bytes, std::memory_order_relaxed);
    }
};
} // namespace
struct Mc2BinaryCacheDomain {
    void* comm;
    std::shared_ptr<const AlgResourceCtxSerializable> holder;
    std::mutex mutex;
    uint64_t bytes = 0;
    std::unordered_map<std::string, std::unique_ptr<BinaryCacheEntry>> entries;
    Mc2AicpuTaskCachePlan plan;
    const std::string* recentKey = nullptr;
    BinaryCacheEntry* recent = nullptr;
    struct Recent {
        const std::string* key = nullptr;
        BinaryCacheEntry* entry = nullptr;
    };
    std::vector<Recent> repeats;
    Mc2RepeatShape repeatShape;
    bool repeatBound = false;
    uint64_t shapeScope = 0;
    const void* shapeWorker = nullptr;
    // Cold admission metadata follows the existing hit-path fields.
    uint64_t identity = 0;
    std::weak_ptr<void> lifetime;
};
namespace {
std::mutex domainsMutex;
uint64_t nextDomainIdentity = 0;
std::unordered_map<const AlgResourceCtxSerializable*, std::shared_ptr<Mc2BinaryCacheDomain>> domains;

std::shared_ptr<Mc2BinaryCacheDomain> GetDomain(void* comm, const AlgResourceCtxSerializable& resCtx)
{
    if (currentHolder.get() != &resCtx) {
        return nullptr;
    }
    if (currentDomain && currentDomain->comm == comm && currentDomain->holder.get() == &resCtx) {
        return currentDomain;
    }
    std::lock_guard<std::mutex> lock(domainsMutex);
    auto found = domains.find(&resCtx);
    if (found != domains.end()) {
        return found->second->comm == comm ? found->second : nullptr;
    }
    if (domains.size() >= MAX_DOMAINS) {
        // Pressure path only. A holder retained by OpenResCtx or any live Server
        // is not retired. Active cache scopes also retain their domain.
        for (auto it = domains.begin(); it != domains.end();) {
            if (it->second.use_count() == 1 && it->second->holder.use_count() == 1) {
                it = domains.erase(it);
            } else {
                ++it;
            }
        }
    }
    if (domains.size() >= MAX_DOMAINS || nextDomainIdentity == UINT64_MAX || !ValidateMc2BinaryCacheResource(resCtx)) {
        return nullptr;
    }
    auto domain = std::make_shared<Mc2BinaryCacheDomain>();
    domain->identity = ++nextDomainIdentity;
    domain->lifetime = domain;
    domain->comm = comm;
    domain->holder = currentHolder;
    domains.emplace(&resCtx, domain);
    return domain;
}
bool ReserveBinaryCacheBytes(uint64_t bytes)
{
    uint64_t previous = binaryCacheBytes.load(std::memory_order_relaxed);
    do {
        if (previous > MAX_BINARY_CACHE_BYTES || bytes > MAX_BINARY_CACHE_BYTES - previous) {
            return false;
        }
    } while (!binaryCacheBytes.compare_exchange_weak(previous, previous + bytes, std::memory_order_relaxed));
    return true;
}

HcclResult ExecuteBatch(const OpParam& param, const std::function<HcclResult()>& execute)
{
    CHK_RET(static_cast<HcclResult>(asccomm_batch_mode_start(param.algTag)));
    HcclResult ret = HCCL_SUCCESS;
    try {
        ret = execute();
    } catch (...) {
        ret = HCCL_E_INTERNAL;
    }
    const auto end = static_cast<HcclResult>(asccomm_batch_mode_end(param.algTag));
    return ret != HCCL_SUCCESS ? ret : end;
}
} // namespace

std::shared_ptr<Mc2BinaryCacheDomain> BindMc2BinaryCacheDomain(
    void* comm, std::shared_ptr<const AlgResourceCtxSerializable> holder)
{
    if (!holder || !comm) {
        return nullptr;
    }
    Mc2BinaryCacheResourceScope scope(holder);
    try {
        return GetDomain(comm, *holder);
    } catch (...) {
        return nullptr;
    }
}

namespace {
using UbDomainExchange = uint64_t (*)(uint64_t, const void*);
UbDomainExchange UbDomainBackend()
{
    static auto fn = reinterpret_cast<UbDomainExchange>(dlsym(RTLD_DEFAULT, "asccomm_ub_domain_exchange_v2"));
    return fn;
}
} // namespace

Mc2BinaryCacheResourceScope::Mc2BinaryCacheResourceScope(
    std::shared_ptr<const AlgResourceCtxSerializable> holder, std::shared_ptr<Mc2BinaryCacheDomain> domain,
    bool regularRepeatInvariant)
    : previous_(std::move(currentHolder)),
      previousDomain_(std::move(currentDomain)),
      previousScope_(currentScope),
      previousRepeat_(currentRepeat)
{
    currentHolder = std::move(holder);
    currentDomain = std::move(domain);
    currentScope = regularRepeatInvariant ? ++scopeSequence : 0;
    currentRepeat = 0;
    previousLockedDomain_ = lockedDomain;
    if (currentDomain && lockedDomain != currentDomain.get()) {
        domainLock_ = std::unique_lock<std::mutex>(currentDomain->mutex);
        lockedDomain = currentDomain.get();
    }
    if (auto exchange = UbDomainBackend()) {
        const auto* domain = currentDomain.get();
        previousUbDomain_ = exchange(domain ? domain->identity : 0, domain ? &domain->lifetime : nullptr);
    }
}
Mc2BinaryCacheResourceScope::~Mc2BinaryCacheResourceScope()
{
    if (auto exchange = UbDomainBackend()) {
        exchange(previousUbDomain_, nullptr);
    }
    currentHolder = std::move(previous_);
    currentDomain = std::move(previousDomain_);
    currentScope = previousScope_;
    currentRepeat = previousRepeat_;
    lockedDomain = previousLockedDomain_;
}

bool Mc2AsccommRelocationSupported() { return BinaryV2Backend() != nullptr; }

bool Mc2AsccommBinaryCaptureSupported() { return BinaryBackend() != nullptr; }

HcclResult RunWithAsccommBinaryCache(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, const std::function<HcclResult()>& execute)
{
    std::shared_ptr<Mc2BinaryCacheDomain> domain;
    std::unique_lock<std::mutex> lock;
    const size_t repeatIndex = currentScope ? currentRepeat++ : MAX_ENTRIES_PER_DOMAIN;
    bool boundRepeat = false;
    try {
        if (param.aicpuCacheEnable && Mc2AsccommBinaryCaptureSupported()) {
            domain = GetDomain(param.hcclComm, resCtx);
            if (domain) {
                if (lockedDomain != domain.get()) {
                    lock = std::unique_lock<std::mutex>(domain->mutex);
                }
                if (repeatIndex == 0 || domain->shapeScope != currentScope || domain->shapeWorker != &scopeSequence) {
                    domain->shapeWorker = &scopeSequence;
                    domain->shapeScope = currentScope;
                    domain->repeatShape.ready = false;
                    domain->repeatBound = false;
                }
                if (repeatIndex < MAX_ENTRIES_PER_DOMAIN && domain->repeats.size() <= repeatIndex) {
                    domain->repeats.resize(repeatIndex + 1);
                }
                // The Server scope guarantees unchanged shape/resources within one
                // message. Independent Prepare messages and A2AV retain full keys.
                boundRepeat = currentScope && repeatIndex > 0 && domain->repeatBound &&
                              param.opType != HCCL_CMD_ALLTOALLV && Mc2AsccommRelocationSupported();
                if (boundRepeat) {
                    auto& plan = domain->plan;
                    const auto input = reinterpret_cast<uintptr_t>(param.inputPtr);
                    const auto output = reinterpret_cast<uintptr_t>(param.outputPtr);
                    if (!input || !output || plan.sizes[0] > UINTPTR_MAX - input ||
                        plan.sizes[1] > UINTPTR_MAX - output ||
                        (input < output + plan.sizes[1] && output < input + plan.sizes[0])) {
                        domain->repeatBound = false;
                        return ExecuteBatch(param, execute);
                    }
                    plan.addrs = {{param.inputPtr, param.outputPtr}};
                }
                auto* repeatShape = currentScope ? &domain->repeatShape : nullptr;
                const bool planUnavailable =
                    !boundRepeat &&
                    (BuildMc2AicpuTaskCachePlan(param, resCtx, domain->plan, true, true, repeatShape) != HCCL_SUCCESS ||
                     !domain->plan.enabled);
                if (planUnavailable) {
                    if (lock.owns_lock()) {
                        lock.unlock();
                    }
                    domain.reset();
                }
            }
        }
    } catch (...) {
        if (lock.owns_lock()) {
            lock.unlock();
        }
        domain.reset(); // No submission yet; ordinary path remains available.
    }
    if (!domain) {
        return ExecuteBatch(param, execute);
    }
    auto& plan = domain->plan;
    // Pointers into unordered_map elements survive rehash. Clear/error invalidate
    // them under this same mutex. Full equality includes restored a2av arrays.
    auto* repeatRecent = repeatIndex < domain->repeats.size() ? &domain->repeats[repeatIndex] : nullptr;
    BinaryCacheEntry* hit = boundRepeat ? domain->recent :
                            repeatRecent && repeatRecent->key && *repeatRecent->key == plan.exactKey ?
                                          repeatRecent->entry :
                                          nullptr;
    if (!hit) {
        hit = domain->recentKey && *domain->recentKey == plan.exactKey ? domain->recent : nullptr;
    }
    if (!hit) {
        const auto found = domain->entries.find(plan.exactKey);
        if (found != domain->entries.end()) {
            domain->recentKey = &found->first;
            domain->recent = hit = found->second.get();
        }
    }
    if (hit) {
        if (!boundRepeat && repeatRecent && repeatRecent->entry == hit) {
            domain->recentKey = repeatRecent->key;
        }
        domain->recent = hit;
        domain->repeatBound = true;
        if (repeatRecent && (!repeatRecent->entry || repeatRecent->entry != hit)) {
            *repeatRecent = {domain->recentKey, hit};
        }
        Mc2ServerStageTiming::Cache(true);
        CHK_RET(static_cast<HcclResult>(asccomm_batch_mode_start(param.algTag)));
        auto& audit = Mc2ServerStageTiming::Audit();
        static_assert(
            sizeof(audit.reasons) / sizeof(audit.reasons[0]) == ASCCOMM_REJECT_REASON_COUNT, "replay reason ABI");
        AsccommReplayInfoV1 replayInfo{};
        uint64_t counts[ASCCOMM_COUNT_SIZE]{};
        HcclResult ret;
        {
            ret = static_cast<HcclResult>(BinaryCall(ASCCOMM_REPLAY_EXECUTE, &hit->handle, &replayInfo, counts, plan));
            if (replayInfo.reserved == ASCCOMM_BINARY_FALLBACK) {
                ++audit.preflight;
                const uint32_t reason = (counts[ASCCOMM_COUNT_COMMAND] & ~ASCCOMM_REJECT_MASK) == ASCCOMM_REJECT_MAGIC ?
                                            static_cast<uint32_t>(counts[ASCCOMM_COUNT_COMMAND] & ASCCOMM_REJECT_MASK) :
                                            0;
                ++audit.reasons[reason < ASCCOMM_REJECT_REASON_COUNT ? reason : 0];
                domain->bytes -= hit->bytes;
                domain->recent = nullptr;
                domain->recentKey = nullptr;
                domain->repeats.clear();
                domain->repeatShape.ready = false;
                domain->repeatBound = false;
                domain->entries.erase(plan.exactKey);
                Mc2ServerStageTiming::Cache(false);
                // v2 guarantees that no taskId/PI/fence/publication has occurred.
                return ExecuteBatch(param, execute);
            }
            if (replayInfo.reserved == ASCCOMM_BINARY_READY) {
                ++audit.binary;
                audit.sqe += counts[ASCCOMM_COUNT_SQE];
                audit.wqe += counts[ASCCOMM_COUNT_WQE];
                audit.db += counts[ASCCOMM_COUNT_DB];
            }
        }
        if (ret != HCCL_SUCCESS) {
            ++audit.errors;
        }
        // Successful replay includes the final batch end recorded on the miss.
        // Only failures need the existing best-effort batch cleanup; preserve
        // the first error and never rerun a partially submitted collective.
        if (ret != HCCL_SUCCESS) {
            (void)asccomm_batch_mode_end(param.algTag);
            domain->bytes -= hit->bytes;
            domain->recent = nullptr;
            domain->recentKey = nullptr;
            domain->repeats.clear();
            domain->repeatShape.ready = false;
            domain->repeatBound = false;
            domain->entries.erase(plan.exactKey);
        }
        return ret; // A replay may have partially submitted. Never execute again.
    }
    if (domain->entries.size() >= MAX_ENTRIES_PER_DOMAIN || domain->bytes >= MAX_DOMAIN_BYTES) {
        return ExecuteBatch(param, execute);
    }
    std::unique_ptr<BinaryCacheEntry> entry;
    try {
        entry.reset(new BinaryCacheEntry);
    } catch (...) {
        return ExecuteBatch(param, execute);
    }
    if (BinaryBackend()(ASCCOMM_REPLAY_CREATE, &entry->handle, nullptr, nullptr) != 0) {
        return ExecuteBatch(param, execute);
    }
    // Start batch before recording. Its final flush is captured; replay has the same boundaries.
    CHK_RET(static_cast<HcclResult>(asccomm_batch_mode_start(param.algTag)));
    const bool capturing = BinaryCall(ASCCOMM_REPLAY_BEGIN, &entry->handle, nullptr, nullptr, plan) == 0;
    if (capturing) {
        Mc2ServerStageTiming::Cache(false);
    }
    HcclResult ret = HCCL_SUCCESS;
    try {
        ret = execute();
    } catch (...) {
        ret = HCCL_E_INTERNAL;
    }
    const auto end = static_cast<HcclResult>(asccomm_batch_mode_end(param.algTag));
    if (ret == HCCL_SUCCESS) {
        ret = end;
    }
    if (!capturing || ret != HCCL_SUCCESS) {
        return ret;
    }
    AsccommReplayInfoV1 info{};
    const auto cacheRet = BinaryBackend()(ASCCOMM_REPLAY_END, &entry->handle, &info, nullptr);
    ++Mc2ServerStageTiming::Audit().captures;
    if (info.reserved == ASCCOMM_BINARY_READY) {
        ++Mc2ServerStageTiming::Audit().ready;
    }
    if (cacheRet != 0 || info.ready == 0 ||
        info.storageBytes + plan.exactKey.size() > MAX_DOMAIN_BYTES - domain->bytes) {
        return HCCL_SUCCESS; // Communication completed, but its capture is not admitted.
    }
    info.storageBytes += plan.exactKey.size();
    if (!ReserveBinaryCacheBytes(info.storageBytes)) {
        return HCCL_SUCCESS;
    }
    entry->bytes = info.storageBytes;
    try {
        auto inserted = domain->entries.emplace(plan.exactKey, std::move(entry));
        domain->recentKey = &inserted.first->first;
        domain->recent = inserted.first->second.get();
        domain->repeatBound = true;
        if (repeatRecent) {
            *repeatRecent = {domain->recentKey, domain->recent};
        }
        domain->bytes += info.storageBytes;
    } catch (...) { /* Drop metadata; the completed communication is never rerun. */
    }
    return HCCL_SUCCESS;
}

HcclResult ClearAsccommBinaryCapture(void* comm)
{
    std::vector<std::shared_ptr<Mc2BinaryCacheDomain>> matching;
    {
        std::lock_guard<std::mutex> lock(domainsMutex);
        for (const auto& item : domains) {
            if (item.second->comm == comm) {
                matching.push_back(item.second);
            }
        }
    }
    for (auto& domain : matching) {
        std::lock_guard<std::mutex> lock(domain->mutex);
        domain->recent = nullptr;
        domain->recentKey = nullptr;
        domain->repeats.clear();
        domain->repeatShape.ready = false;
        domain->repeatBound = false;
        domain->entries.clear();
        domain->bytes = 0;
    }
    return HCCL_SUCCESS;
}
} // namespace mc2_ops_hccl

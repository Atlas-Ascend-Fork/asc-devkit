/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MC2_BINARY_CACHE_STORAGE_H
#define MC2_BINARY_CACHE_STORAGE_H

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mc2_ops_hccl {
namespace binary_cache {
class ExactKeyBuilder {
public:
    explicit ExactKeyBuilder(const char* domain, std::string* storage = nullptr)
    {
        if (domain) {
            if (storage) {
                exactBytes_.swap(*storage);
            }
            exactBytes_.clear();
            exactBytes_.reserve(sizeof(pending_));
            AddString(domain, std::strlen(domain));
        }
    }
    void AddU8(uint8_t value) { AddBytes(&value, sizeof(value)); }

    void AddU32(uint32_t value)
    {
        uint8_t bytes[sizeof(value)];
        for (size_t i = 0; i < sizeof(value); ++i) {
            bytes[i] = static_cast<uint8_t>((value >> (i * 8U)) & 0xffU);
        }
        AddBytes(bytes, sizeof(bytes));
    }

    void AddU64(uint64_t value)
    {
        uint8_t bytes[sizeof(value)];
        for (size_t i = 0; i < sizeof(value); ++i) {
            bytes[i] = static_cast<uint8_t>((value >> (i * 8U)) & 0xffU);
        }
        AddBytes(bytes, sizeof(bytes));
    }

    void AddBool(bool value) { AddU8(value ? 1U : 0U); }

    void AddDouble(double value)
    {
        static_assert(sizeof(value) == sizeof(uint64_t), "unexpected double size");
        // The cache is process-local on little-endian A5, so its object representation preserves the exact key.
        AddBytes(&value, sizeof(value));
    }

    void AddPointer(const void* value) { AddU64(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(value))); }

    void AddString(const char* value, size_t maxLen)
    {
        const size_t len = value == nullptr ? 0U : strnlen(value, maxLen);
        AddU64(static_cast<uint64_t>(len));
        if (len > 0U) {
            AddBytes(value, len);
        }
    }

    const std::string& ExactBytes()
    {
        FlushExact();
        return exactBytes_;
    }
    void LoadExactBytes(const std::string& value)
    {
        pendingSize_ = 0;
        exactBytes_.assign(value);
    }

    std::string TakeExactBytes()
    {
        FlushExact();
        return std::move(exactBytes_);
    }

private:
    void AddBytes(const void* data, size_t size)
    {
        if (size > sizeof(pending_) - pendingSize_) {
            FlushExact();
        }
        if (size > sizeof(pending_)) {
            exactBytes_.append(static_cast<const char*>(data), size);
        } else {
            (void)std::memcpy(pending_ + pendingSize_, data, size);
            pendingSize_ += size;
        }
    }
    void FlushExact()
    {
        if (pendingSize_ != 0) {
            exactBytes_.append(pending_, pendingSize_);
            pendingSize_ = 0;
        }
    }
    char pending_[512];
    size_t pendingSize_ = 0;
    std::string exactBytes_;
};

inline bool DisjointRanges(uintptr_t input, uint64_t inputSize, uintptr_t output, uint64_t outputSize)
{
    return input && output && inputSize <= UINTPTR_MAX - input && outputSize <= UINTPTR_MAX - output &&
           !(input < output + outputSize && output < input + inputSize);
}

inline bool ReserveBytes(std::atomic<uint64_t>& used, uint64_t bytes, uint64_t limit)
{
    uint64_t previous = used.load(std::memory_order_relaxed);
    do {
        if (previous > limit || bytes > limit - previous) {
            return false;
        }
    } while (!used.compare_exchange_weak(previous, previous + bytes, std::memory_order_relaxed));
    return true;
}

// The caller holds the domain mutex while using an entry or its repeat binding.
template <class Entry>
struct EntryTable {
    struct Recent {
        const std::string* key = nullptr;
        Entry* entry = nullptr;
    };
    std::unordered_map<std::string, std::unique_ptr<Entry>> entries;
    const std::string* recentKey = nullptr;
    Entry* recent = nullptr;
    std::vector<Recent> repeats;
    // Only for a caller that has already proved scope and repeat shape invariance.
    Entry* BoundRepeatEntry() const { return recent; }

    Entry* Lookup(const std::string& key, size_t repeatIndex)
    {
        auto* repeatRecent = repeatIndex < repeats.size() ? &repeats[repeatIndex] : nullptr;
        Entry* hit = repeatRecent && repeatRecent->key && *repeatRecent->key == key ? repeatRecent->entry : nullptr;
        if (!hit) {
            hit = recentKey && *recentKey == key ? recent : nullptr;
        }
        if (!hit) {
            const auto found = entries.find(key);
            if (found != entries.end()) {
                recentKey = &found->first;
                recent = hit = found->second.get();
            }
        }
        if (hit) {
            if (repeatRecent && repeatRecent->entry == hit) {
                recentKey = repeatRecent->key;
            }
            recent = hit;
            if (repeatRecent && (!repeatRecent->entry || repeatRecent->entry != hit)) {
                *repeatRecent = {recentKey, hit};
            }
        }
        return hit;
    }

    void Reset()
    {
        recent = nullptr;
        recentKey = nullptr;
        repeats.clear();
        entries.clear();
    }
};

template <class Resource, class Domain>
class DomainRegistry {
public:
    template <class Validate>
    std::shared_ptr<Domain> Bind(
        void* comm, const std::shared_ptr<const Resource>& holder, size_t limit, Validate validate)
    {
        if (!comm || !holder) {
            return nullptr;
        }
        const auto& resCtx = *holder;
        std::lock_guard<std::mutex> lock(domainsMutex);
        auto found = domains.find(&resCtx);
        if (found != domains.end()) {
            return found->second->comm == comm ? found->second : nullptr;
        }
        if (domains.size() >= limit) {
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
        if (domains.size() >= limit || nextDomainIdentity == UINT64_MAX || !validate(resCtx)) {
            return nullptr;
        }
        auto domain = std::make_shared<Domain>();
        domain->identity = ++nextDomainIdentity;
        domain->lifetime = domain;
        domain->comm = comm;
        domain->holder = holder;
        domains.emplace(&resCtx, domain);
        return domain;
    }

    std::vector<std::shared_ptr<Domain>> Matching(void* comm)
    {
        std::vector<std::shared_ptr<Domain>> matching;
        std::lock_guard<std::mutex> lock(domainsMutex);
        for (const auto& item : domains) {
            if (item.second->comm == comm) {
                matching.push_back(item.second);
            }
        }
        return matching;
    }

private:
    std::mutex domainsMutex;
    uint64_t nextDomainIdentity = 0;
    std::unordered_map<const Resource*, std::shared_ptr<Domain>> domains;
};
} // namespace binary_cache
} // namespace mc2_ops_hccl

#endif // MC2_BINARY_CACHE_STORAGE_H

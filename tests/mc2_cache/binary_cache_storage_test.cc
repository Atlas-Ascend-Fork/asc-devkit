/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../impl/adv_api/detail/hccl/cc/src/ops/op_common/template/aicpu/task_cache/mc2_binary_cache_storage.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

using namespace mc2_ops_hccl::binary_cache;
struct Domain {
    void* comm;
    uint64_t identity;
    std::weak_ptr<void> lifetime;
    std::shared_ptr<const int> holder;
};

int main()
{
    ExactKeyBuilder key("x");
    key.AddU32(0x01020304);
    assert(key.ExactBytes() == std::string("\1\0\0\0\0\0\0\0x\4\3\2\1", 13));
    const auto prefix = key.ExactBytes();
    key.AddString("ignored", 7);
    key.LoadExactBytes(prefix);
    std::string large(600, 'x');
    key.AddString(large.data(), large.size());
    assert(key.TakeExactBytes() == prefix + std::string("\x58\x02\0\0\0\0\0\0", 8) + large);
    assert(DisjointRanges(10, 4, 14, 8));
    assert(!DisjointRanges(10, 5, 14, 8));
    assert(!DisjointRanges(UINTPTR_MAX, 1, 1, 1));
    std::atomic<uint64_t> bytes{0};
    assert(ReserveBytes(bytes, 4, 4) && !ReserveBytes(bytes, 1, 4));
    EntryTable<int> table;
    table.entries["x"].reset(new int(7));
    table.repeats.resize(1);
    assert(*table.Lookup("x", 0) == 7);
    assert(table.Lookup("missing", 0) == nullptr);
    assert(*table.BoundRepeatEntry() == 7);
    assert(!table.Lookup("different", 1));
    table.entries["y"].reset(new int(9));
    assert(*table.Lookup("y", 0) == 9);
    assert(*table.BoundRepeatEntry() == 9);
    assert(*table.Lookup("x", 0) == 7);
    table.Reset();
    assert(table.BoundRepeatEntry() == nullptr);
    assert(!table.Lookup("x", 0));
    DomainRegistry<int, Domain> registry;
    int comm = 0;
    auto holder = std::make_shared<const int>(1);
    auto accept = [](int) { return true; };
    auto domain = registry.Bind(&comm, holder, 1, accept);
    assert(domain && registry.Bind(&comm, holder, 1, accept) == domain);
    int otherComm = 1;
    assert(!registry.Bind(&otherComm, holder, 1, accept));
    auto other = std::make_shared<const int>(2);
    assert(!registry.Bind(&comm, other, 1, accept));
    assert(registry.Matching(&comm).size() == 1);
    domain.reset();
    holder.reset();
    assert(registry.Bind(&comm, other, 1, accept));
}

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Run the production binary-cache control flow with a stub native backend.

This portable Host test does not validate SQE/WQE or NPU execution.
"""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


class HitPathTest(unittest.TestCase):
    def test_lazy_executor_and_batch_ownership(self):
        root = next(p for p in Path(__file__).resolve().parents if (p / "impl").is_dir())
        src = root / "impl/adv_api/detail/hccl/cc/src"
        cache = (src / "ops/op_common/template/aicpu/task_cache/mc2_asccomm_binary_cache.cc").read_text()
        cache = re.sub(r"^#include.*$", "", cache, flags=re.M)
        adapter = (src / "aicpu_kfc/framework/device/aicpu_kfc/decoupler/comm_kfc_open_kernel_executor.cc").read_text()
        # Compile the actual lazy callback rather than a handwritten imitation.
        begin = adapter.index("HcclResult ret = mc2_ops_hccl::RunWithAsccommBinaryCache(*param")
        end = adapter.index("\n    CHK_PRT_RET", begin)
        callback = adapter[begin:end]
        program = r"""
#include <cassert>
#include <array>
#include <cstring>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
using HcclResult = int;
constexpr int HCCL_SUCCESS=0, HCCL_E_INTERNAL=4;
#define CHK_RET(x) do { int r=(x); if(r) return r; } while(0)
#define CHK_SMART_PTR_NULL(x) do { if(!(x)) return 9; } while(0)
enum AsccommReplayActionV1 {
    ASCCOMM_REPLAY_CREATE = 0,
    ASCCOMM_REPLAY_BEGIN = 1,
    ASCCOMM_REPLAY_END = 2,
    ASCCOMM_REPLAY_EXECUTE = 3,
    ASCCOMM_REPLAY_DESTROY = 4,
    ASCCOMM_REPLAY_ABORT = 5,
    ASCCOMM_REPLAY_QUERY = 6
};
// Stable numeric values are consumed by devkit and diagnostic CSV parsers.
enum AsccommReplayStatus : uint32_t {
    ASCCOMM_BINARY_READY = 1,
    ASCCOMM_BINARY_UNAVAILABLE = 2,
    ASCCOMM_BINARY_FAILED = 4,
    ASCCOMM_BINARY_FALLBACK = 5
};
enum AsccommReplayCount : uint32_t {
    ASCCOMM_COUNT_COMMAND = 0,
    ASCCOMM_COUNT_SQE,
    ASCCOMM_COUNT_WQE,
    ASCCOMM_COUNT_DB,
    ASCCOMM_COUNT_SIZE
};
enum class AsccommRejectReason : uint32_t {
    Unknown = 0,
    EntryNotReady = 1,
    EntryBounds = 2,
    AdapterRejected = 3,
    PrepareException = 4,
    ScheduleCompile = 5,
    ApplyException = 6,
    Layout = 7,
    BatchEnd = 8,
    NotifyEncoding = 9,
    NotifyBinding = 10,
    NotifyChanged = 11,
    RtsqOrder = 12,
    RtsqEncoding = 13,
    ThreadBinding = 14,
    PendingSqe = 15,
    QueueChanged = 16,
    NestedTransport = 17,
    TransportResource = 18,
    UbOrder = 19,
    RdmaOrder = 20,
    UbPatchEncoding = 21,
    UbOpcode = 22,
    NotifyIndex = 23,
    RemoteNotifyRange = 24,
    RemoteBufferMissing = 25,
    RemoteBufferRange = 26,
    InlineProtection = 27,
    RemoteProtection = 28,
    InlinePayload = 29,
    LocalNotifyMissing = 30,
    LocalNotifyRange = 31,
    LocalProtection = 32,
    WriteNotifyProtection = 33,
    TransportEncoding = 34,
    ChannelQueue = 35,
    TransportBinding = 36,
    ThreadQueue = 37,
    PendingTransport = 38,
    UbQueue = 39,
    RdmaQueue = 40,
    ChannelNotifyEncoding = 41,
    ChannelNotifyIndex = 42,
    FenceEncoding = 43,
    FencePlan = 44,
    RdmaWqeEncoding = 45,
    RdmaPatchEncoding = 46,
    RdmaNotifyIndex = 47,
    RdmaNotifyRange = 48,
    RdmaProtection = 49,
    TransportEndEncoding = 50,
    UnsupportedEvent = 51,
    ResourceChanged = 52,
    ChannelNotifyChanged = 53,
    FenceChanged = 54,
    UbProtocol = 55,
    TransportEndBinding = 56,
    Count = 57
};
constexpr uint32_t ASCCOMM_REJECT_REASON_COUNT = static_cast<uint32_t>(AsccommRejectReason::Count);
constexpr uint64_t ASCCOMM_REJECT_MAGIC = 0x50460000ULL;
constexpr uint64_t ASCCOMM_REJECT_MASK = 0xffffULL;
constexpr uint64_t ASCCOMM_BINARY_ENTRY_BUDGET = 10ULL * 1024 * 1024;
typedef struct {
    uint64_t commandCount;
    uint64_t storageBytes;
    uint32_t ready;
    uint32_t reserved;
} AsccommReplayInfoV1;
struct NativeEntry { bool finalEnd=false; };
NativeEntry* active=nullptr;
int starts=0,ends=0,executes=0,creates=0,replays=0,replayError=0,executeError=0,endError=0;
bool nullExecutor=false, rejectReplay=false, rejectCapture=false;
int asccomm_batch_mode_start(const char*) { ++starts; return 0; }
int asccomm_batch_mode_end(const char*) { ++ends; if(active) active->finalEnd=true; return endError; }
int backend(uint32_t action, void** handle, AsccommReplayInfoV1* info, uint64_t*) {
 auto* entry=static_cast<NativeEntry*>(*handle);
 switch(action) {
 case ASCCOMM_REPLAY_CREATE: *handle=new NativeEntry; return 0;
 case ASCCOMM_REPLAY_BEGIN: active=entry; return 0;
 case ASCCOMM_REPLAY_END: active=nullptr; *info={1,64,entry->finalEnd && !rejectCapture?1U:0U,1}; return 0;
 case ASCCOMM_REPLAY_EXECUTE:
   ++replays; if(replayError) { info->reserved=4; return replayError; }
   if(rejectReplay) { info->reserved=5; return 0; }
   info->reserved=1;
   assert(entry->finalEnd); return asccomm_batch_mode_end(nullptr);
 case ASCCOMM_REPLAY_DESTROY: if(active==entry) active=nullptr; delete entry; *handle=nullptr; return 0;
 }
 return 8;
}
#define RTLD_DEFAULT nullptr
void* dlsym(void*, const char* name) { return std::strcmp(name,"asccomm_binary_replay_v1")==0 ? reinterpret_cast<void*>(&backend) : nullptr; }
namespace Mc2CostBridge { struct Scope { explicit Scope(int) {} void Stop() {} }; }
namespace mc2_ops_hccl {
struct OpParam { bool aicpuCacheEnable=true; void* hcclComm=reinterpret_cast<void*>(1); const char* algTag="test"; void* inputPtr=reinterpret_cast<void*>(4096); void* outputPtr=reinterpret_cast<void*>(8192); int opType=0; const char* algName="test"; };
struct AlgResourceCtxSerializable {};
struct Mc2AicpuTaskCachePlan { bool enabled=false; std::string exactKey; std::array<void*,2> addrs{}; std::array<uint64_t,2> sizes{}; };
struct Mc2RepeatShape { bool ready=false; };
constexpr int HCCL_CMD_ALLTOALLV=5;
bool ValidateMc2BinaryCacheResource(const AlgResourceCtxSerializable&) { return true; }
int BuildMc2AicpuTaskCachePlan(const OpParam&,const AlgResourceCtxSerializable&,Mc2AicpuTaskCachePlan& p,bool,bool,Mc2RepeatShape*) { p.enabled=true; p.exactKey="shape"; return 0; }
struct Mc2ServerStageTiming {
 struct AuditData { uint64_t preflight=0,binary=0,sqe=0,wqe=0,db=0,errors=0,captures=0,ready=0,reasons[57]{}; };
 static AuditData& Audit() { static AuditData a; return a; }
 static void Cache(bool) {}
};
struct Mc2BinaryCacheDomain;
class Mc2BinaryCacheResourceScope {
public:
    explicit Mc2BinaryCacheResourceScope(std::shared_ptr<const AlgResourceCtxSerializable> holder,
        std::shared_ptr<Mc2BinaryCacheDomain> domain = nullptr, bool regularRepeatInvariant = false);
    ~Mc2BinaryCacheResourceScope();
    Mc2BinaryCacheResourceScope(const Mc2BinaryCacheResourceScope&) = delete;
    Mc2BinaryCacheResourceScope& operator=(const Mc2BinaryCacheResourceScope&) = delete;

private:
    std::shared_ptr<const AlgResourceCtxSerializable> previous_;
    std::shared_ptr<Mc2BinaryCacheDomain> previousDomain_;
    uint64_t previousUbDomain_ = 0;
    uint64_t previousScope_ = 0;
    size_t previousRepeat_ = 0;
    Mc2BinaryCacheDomain* previousLockedDomain_ = nullptr;
    std::unique_lock<std::mutex> domainLock_;
};

int RestoreVarDataIfNeeded(const OpParam&,const AlgResourceCtxSerializable&) { return 0; }
struct Executor { int Orchestrate(const OpParam&,const AlgResourceCtxSerializable&) { ++executes; return executeError; } };
struct CollAlgExecRegistryV2 {
 static CollAlgExecRegistryV2& Instance() { static CollAlgExecRegistryV2 r; return r; }
 std::unique_ptr<Executor> GetAlgExec(int,const std::string&) {
   ++creates; return nullExecutor?nullptr:std::unique_ptr<Executor>(new Executor);
 }
};
}
"""
        program += cache
        program += "\nint run(mc2_ops_hccl::OpParam* param, const mc2_ops_hccl::AlgResourceCtxSerializable& resCtx) {\n"
        program += callback + "\nreturn ret;\n}\n"
        program += r"""
int main() {
 using namespace mc2_ops_hccl;
 OpParam p;
 auto holder=std::make_shared<const AlgResourceCtxSerializable>();
 Mc2BinaryCacheResourceScope scope(holder);
 assert(run(&p,*holder)==0); // miss
 assert(creates==1 && executes==1 && starts==1 && ends==1);
 assert(run(&p,*holder)==0); // hit: recorded final end, no outer end or executor
 assert(creates==1 && executes==1 && replays==1 && starts==2 && ends==2);
 replayError=7;
 assert(run(&p,*holder)==7); // cleanup once, no executor fallback, entry invalidated
 assert(creates==1 && executes==1 && ends==3);
 replayError=0;
 assert(run(&p,*holder)==0); // miss again after failed replay
 assert(creates==2 && executes==2 && ends==4);
 endError=6;
 assert(run(&p,*holder)==6); // recorded final end failure still returned
 assert(creates==2 && executes==2 && ends==6); // failed end + best-effort cleanup
 endError=0;
 assert(ClearAsccommBinaryCapture(p.hcclComm)==0);
 p.aicpuCacheEnable=false;
 assert(run(&p,*holder)==0); // bypass
 assert(creates==3 && executes==3 && ends==7);
 nullExecutor=true;
 assert(run(&p,*holder)==9); // factory error and batch cleanup
 assert(creates==4 && executes==3 && ends==8);
 nullExecutor=false;
 p.aicpuCacheEnable=true; executeError=5;
 assert(run(&p,*holder)==5); // failed miss is not published
 assert(ends==9 && active==nullptr);
 executeError=0;
 assert(run(&p,*holder)==0);
 assert(creates==6 && executes==5 && ends==10);
 assert(ClearAsccommBinaryCapture(p.hcclComm)==0);
 // Unsupported capture: every call expands once; no command template admitted.
 rejectCapture=true;
 const auto before=executes;
 assert(run(&p,*holder)==0); assert(run(&p,*holder)==0);
 assert(executes==before+2);
 rejectCapture=false;
 assert(run(&p,*holder)==0);
 // Pre-publication rejection: expand once, then next invocation captures anew.
 rejectReplay=true;
 const auto beforeReject=executes;
 assert(run(&p,*holder)==0); assert(executes==beforeReject+1);
 rejectReplay=false;
 assert(run(&p,*holder)==0); assert(executes==beforeReject+2);
 assert(run(&p,*holder)==0); assert(executes==beforeReject+2);
 assert(ClearAsccommBinaryCapture(p.hcclComm)==0);
 // Capacity pressure retires only unowned domains; the live holder stays pinned.
 auto live = BindMc2BinaryCacheDomain(p.hcclComm, holder);
 const auto liveId = live->identity;
 std::weak_ptr<void> retired;
 for (int i=0; i<600; ++i) {
   auto resource=std::make_shared<AlgResourceCtxSerializable>();
   auto domain=BindMc2BinaryCacheDomain(p.hcclComm, resource);
   assert(domain && domain->identity != liveId);
   if(i==0) retired=domain;
 }
 assert(retired.expired());
 assert(BindMc2BinaryCacheDomain(p.hcclComm,holder)==live);
 assert(domains.size()<=MAX_DOMAINS);
}
"""
        with tempfile.TemporaryDirectory() as temp:
            source, binary = Path(temp) / "test.cc", Path(temp) / "test"
            source.write_text(program)
            subprocess.run(
                [
                    os.environ.get("CXX", "c++"),
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    str(source),
                    "-o",
                    str(binary),
                ],
                check=True,
            )
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()

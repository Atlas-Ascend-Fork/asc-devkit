/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// libmc2_client.so 硬依赖 libasccomm_aicpu.so（设备侧 aarch64 ELF）；x86 host 上
// loader 拒绝异构 ELF（561000），本 stub 以 host 编译器提供同名同符号面占位。
// host 侧不执行任何实现，误调用返回错误码响亮失败。

#include "asccomm_primitives.h"
#include "asccomm_result.h"

extern "C" {
static int32_t HostStubNotAvailable() { return ASCCOMM_E_NOT_SUPPORT; }

int32_t asccomm_batch_mode_start(const char* batchTag)
{
    (void)batchTag;
    return HostStubNotAvailable();
}

int32_t asccomm_batch_mode_end(const char* batchTag)
{
    (void)batchTag;
    return HostStubNotAvailable();
}

int32_t asccomm_local_copy_on_thread(ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    return HostStubNotAvailable();
}

int32_t asccomm_local_reduce_on_thread(
    ThreadHandle thread, void* dst, const void* src, uint64_t count, asccomm_data_type dataType,
    asccomm_reduce_op reduceOp)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)count;
    (void)dataType;
    (void)reduceOp;
    return HostStubNotAvailable();
}

int32_t asccomm_thread_notify_record_on_thread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx)
{
    (void)thread;
    (void)dstThread;
    (void)dstNotifyIdx;
    return HostStubNotAvailable();
}

int32_t asccomm_thread_notify_wait_on_thread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut)
{
    (void)thread;
    (void)notifyIdx;
    (void)timeOut;
    return HostStubNotAvailable();
}

int32_t asccomm_write_on_thread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    (void)thread;
    (void)channel;
    (void)dst;
    (void)src;
    (void)len;
    return HostStubNotAvailable();
}

int32_t asccomm_write_reduce_on_thread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, asccomm_data_type dataType,
    asccomm_reduce_op reduceOp)
{
    (void)thread;
    (void)channel;
    (void)dst;
    (void)src;
    (void)count;
    (void)dataType;
    (void)reduceOp;
    return HostStubNotAvailable();
}

int32_t asccomm_write_with_notify_on_thread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    (void)thread;
    (void)channel;
    (void)dst;
    (void)src;
    (void)len;
    (void)remoteNotifyIdx;
    return HostStubNotAvailable();
}

int32_t asccomm_write_reduce_with_notify_on_thread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, asccomm_data_type dataType,
    asccomm_reduce_op reduceOp, uint32_t remoteNotifyIdx)
{
    (void)thread;
    (void)channel;
    (void)dst;
    (void)src;
    (void)count;
    (void)dataType;
    (void)reduceOp;
    (void)remoteNotifyIdx;
    return HostStubNotAvailable();
}

int32_t asccomm_read_on_thread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    (void)thread;
    (void)channel;
    (void)dst;
    (void)src;
    (void)len;
    return HostStubNotAvailable();
}

int32_t asccomm_read_reduce_on_thread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, asccomm_data_type dataType,
    asccomm_reduce_op reduceOp)
{
    (void)thread;
    (void)channel;
    (void)dst;
    (void)src;
    (void)count;
    (void)dataType;
    (void)reduceOp;
    return HostStubNotAvailable();
}

int32_t asccomm_batch_transfer_on_thread(
    ThreadHandle thread, ChannelHandle channel, const asccomm_batch_transfer_desc* transferDescs,
    uint32_t transferDescNum)
{
    (void)thread;
    (void)channel;
    (void)transferDescs;
    (void)transferDescNum;
    return HostStubNotAvailable();
}

int32_t asccomm_channel_notify_record_on_thread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx)
{
    (void)thread;
    (void)channel;
    (void)remoteNotifyIdx;
    return HostStubNotAvailable();
}

int32_t asccomm_channel_notify_wait_on_thread(
    ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut)
{
    (void)thread;
    (void)channel;
    (void)localNotifyIdx;
    (void)timeOut;
    return HostStubNotAvailable();
}

int32_t asccomm_channel_fence_on_thread(ThreadHandle thread, ChannelHandle channel)
{
    (void)thread;
    (void)channel;
    return HostStubNotAvailable();
}

int32_t asccomm_channel_notify_wait_on_thread_with_default_timeout(
    ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx)
{
    (void)thread;
    (void)channel;
    (void)localNotifyIdx;
    return HostStubNotAvailable();
}

int32_t asccomm_thread_notify_wait_on_thread_with_default_timeout(ThreadHandle thread, uint32_t notifyIdx)
{
    (void)thread;
    (void)notifyIdx;
    return HostStubNotAvailable();
}

int32_t asccomm_set_notify_wait_timeout(float timeOut)
{
    (void)timeOut;
    return HostStubNotAvailable();
}

int32_t asccomm_thread_res_acquire_timeout(float timeOut)
{
    (void)timeOut;
    return HostStubNotAvailable();
}

int32_t asccomm_channel_drain_on_thread(ThreadHandle thread, ChannelHandle channel)
{
    (void)thread;
    (void)channel;
    return HostStubNotAvailable();
}

// devkit AICPU KFC server 侧消费的内部入口（asccomm_primitives_internal.h），同款 stub。
ASCCOMM_API int32_t
asccomm_addr_notify_wait_on_thread(ThreadHandle thread, uint64_t waitAddr, uint64_t valueAddr, uint8_t last)
{
    (void)thread;
    (void)waitAddr;
    (void)valueAddr;
    (void)last;
    return HostStubNotAvailable();
}

ASCCOMM_API int32_t asccomm_addr_notify_record_on_thread(ThreadHandle thread, uint64_t recordAddr, uint64_t valueAddr)
{
    (void)thread;
    (void)recordAddr;
    (void)valueAddr;
    return HostStubNotAvailable();
}
} // extern "C"

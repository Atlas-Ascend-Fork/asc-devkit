# 通过Unified Buffer融合实现连续vector计算

【优先级】高

> [!NOTE] 说明
> 本文介绍的优化建议不依赖具体API。“优化原理”中的Exp、Abs及图示中的VECIN/VECOUT用于说明Basic API下的计算和资源流转；不同API下的具体实现请参见[不同API下的实现](#different_api_implementations)。

## 优化原理

【描述】算子实现中涉及多次vector计算，且前一次计算输出是后一次计算输入的情况下，可将前一次计算输出暂存在Unified Buffer（UB）上直接作为下一次计算的输入，不需要将前一次的计算输出从UB搬运到GM后再从GM搬运到UB。这种UB融合的方式可以减少搬入搬出次数，实现连续vector计算，提升内存使用效率。数据流图对比如下：

**图1**  数据流图对比    
<img src="../../../figures/flow_compare.png" alt="数据流图对比" width="920px" height="300px">

【反例】

该算子的计算逻辑为进行Exp计算后再进行Abs计算。计算过程中先把源操作数从GM搬运到UB进行Exp计算，Exp计算完成后将Exp的结果从UB搬运到GM；再从GM中把Exp的结果搬运到UB上作为Abs计算的输入，Abs计算完成后将目的操作数结果从UB搬运到GM。整个过程从GM搬进搬出共4次。当需要进行的vector计算为n次时，从GM搬进搬出共需要2n次。

【正例】

使用UB融合方式后，在UB上进行连续vector计算时，前一次的结果可直接作为后一次计算的输入，继续在UB上进行计算，不需要中间的搬进搬出，只需在开始计算时将源操作数搬运到UB，以及全部计算结束后将最终结果从UB搬运到GM，共2次搬进搬出。

## 不同API下的实现<a id="different_api_implementations"></a>

<!-- npu="950" id1 -->

> [!NOTE] 说明
> C API实现部分仅适用于Ascend 950PR&950DT系列产品。

### C API

使用GM和UB指针搬运数据，通过950支持的寄存器[asc_exp](../../../../api/SIMD-API/c_api/reg_compute/reg_arith/asc_exp.md)和[asc_abs](../../../../api/SIMD-API/c_api/reg_compute/reg_arith/asc_abs.md)计算。公共实现同样将Exp结果写入UB，随后由Abs读取。

```cpp
#include "c_api/asc_simd.h"

template <bool IsExp>
__simd_vf__ inline void UnaryC(__ubuf__ float* src, __ubuf__ float* dst)
{
    uint32_t remaining = 1024;
    for (uint32_t i = 0; i < 1024; i += 64) {
        vector_bool mask = asc_update_mask_b32(remaining);
        vector_float value, result;
        asc_loadalign(value, src + i);
        if constexpr (IsExp) {
            asc_exp(result, value, mask);
        } else {
            asc_abs(result, value, mask);
        }
        asc_storealign(dst + i, result, mask);
    }
}

template <bool Fused>
__aicore__ inline void ExpAbsC(__gm__ float* src, __gm__ float* dst)
{
    constexpr uint32_t length = 1024;
    __ubuf__ float input[length], intermediate[length], output[length];
    asc_copy_gm2ub_align(input, src, length * sizeof(float));
    asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_vf_call<UnaryC<true>>(input, intermediate);
    if constexpr (Fused) {
        // Exp结果写入UB后，Abs直接读取该UB结果。
        asc_vf_call<UnaryC<false>>(intermediate, output);
    } else {
        asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
        asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
        asc_copy_ub2gm_align(dst, intermediate, length * sizeof(float));
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        asc_copy_gm2ub_align(input, dst, length * sizeof(float));
        asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
        asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
        asc_vf_call<UnaryC<false>>(input, output);
    }
    asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_copy_ub2gm_align(dst, output, length * sizeof(float));
    asc_sync_pipe(PIPE_ALL);
}
```

【反例】

`Fused=false`时，将Exp结果写回GM后再搬入UB执行Abs。

```cpp
// 由设备函数调用，src、dst分别指向1024个float的GM输入和输出。
ExpAbsC<false>(src, dst);
```

【正例】

`Fused=true`时，将Exp结果保留在UB中，供Abs直接读取。

```cpp
ExpAbsC<true>(src, dst);
```

采用该方式时，需要注意以下事项：

- `src`和`dst`分别指向1024个float的GM输入和输出；UB中的`input`、`intermediate`和`output`分别容纳1024个float，且各缓冲区满足接口对齐要求并互不重叠。
- asc_exp和asc_abs使用寄存器重载，通过VF调用；搬运长度以字节传入。
- 搬入、计算和搬出之间通过事件保证依赖，950上连续VF之间的执行顺序由硬件保证。

<!-- end id1 -->

### Basic API

使用TPipe/TQue管理UB空间和搬运依赖，调用Exp和Abs计算。以下保留原例的KernelSample类，使用`kernel_operator.h`和`AscendC`命名空间。

【反例】

将Exp结果写回GM后，再搬入UB执行Abs。

```cpp
class KernelSample {
public:
    __aicore__ inline KernelSample() {}
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* dstGm)
    {
        src0Global.SetGlobalBuffer((__gm__ float*)src0Gm);
        dstGlobal.SetGlobalBuffer((__gm__ float*)dstGm);
        pipe.InitBuffer(inQueueSrc0, 1, 1024 * sizeof(float));
        pipe.InitBuffer(outQueueDst, 1, 1024 * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
        CopyIn1();
        Compute1();
        CopyOut1();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<float> src0Local = inQueueSrc0.AllocTensor<float>();
        DataCopy(src0Local, src0Global, 1024);
        inQueueSrc0.EnQue(src0Local);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<float> src0Local = inQueueSrc0.DeQue<float>();
        LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
        Exp(dstLocal, src0Local, 1024);
        outQueueDst.EnQue<float>(dstLocal);
        inQueueSrc0.FreeTensor(src0Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<float> dstLocal = outQueueDst.DeQue<float>();
        DataCopy(dstGlobal, dstLocal, 1024);
        outQueueDst.FreeTensor(dstLocal);
    }
    __aicore__ inline void CopyIn1()
    {
        LocalTensor<float> src0Local = inQueueSrc0.AllocTensor<float>();
        DataCopy(src0Local, dstGlobal, 1024);
        inQueueSrc0.EnQue(src0Local);
    }
    __aicore__ inline void Compute1()
    {
        LocalTensor<float> src0Local = inQueueSrc0.DeQue<float>();
        LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
        Abs(dstLocal, src0Local, 1024);
        outQueueDst.EnQue<float>(dstLocal);
        inQueueSrc0.FreeTensor(src0Local);
    }
    __aicore__ inline void CopyOut1()
    {
        LocalTensor<float> dstLocal = outQueueDst.DeQue<float>();
        DataCopy(dstGlobal, dstLocal, 1024);
        outQueueDst.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrc0;
    TQue<TPosition::VECOUT, 1> outQueueDst;
    GlobalTensor<float> src0Global, dstGlobal;
};
```

【正例】

在Compute中连续执行Exp和Abs，Exp结果保留在UB中。

```cpp
class KernelSample {
public:
    __aicore__ inline KernelSample() {}
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* dstGm)
    {
        src0Global.SetGlobalBuffer((__gm__ float*)src0Gm);
        dstGlobal.SetGlobalBuffer((__gm__ float*)dstGm);
        pipe.InitBuffer(inQueueSrc0, 1, 1024 * sizeof(float));
        pipe.InitBuffer(outQueueDst, 1, 1024 * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<float> src0Local = inQueueSrc0.AllocTensor<float>();
        DataCopy(src0Local, src0Global, 1024);
        inQueueSrc0.EnQue(src0Local);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<float> src0Local = inQueueSrc0.DeQue<float>();
        LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
        Exp(dstLocal, src0Local, 1024);
        Abs(dstLocal, dstLocal, 1024);
        outQueueDst.EnQue<float>(dstLocal);
        inQueueSrc0.FreeTensor(src0Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<float> dstLocal = outQueueDst.DeQue<float>();
        DataCopy(dstGlobal, dstLocal, 1024);
        outQueueDst.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrc0;
    TQue<TPosition::VECOUT, 1> outQueueDst;
    GlobalTensor<float> src0Global, dstGlobal;
};
```

采用该方式时，需要注意以下事项：

- 输入、输出均为1024个float，队列空间和LocalTensor应满足计算与搬运要求。
- 通过队列管理搬入和搬出的数据依赖。
- 连续计算保持原例Exp后Abs的顺序，Abs原地读取并更新UB中的Exp结果。

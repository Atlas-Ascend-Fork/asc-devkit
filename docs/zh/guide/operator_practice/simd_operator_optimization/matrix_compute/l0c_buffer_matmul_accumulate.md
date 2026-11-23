# 通过L0C Buffer数据暂存实现高效的矩阵乘结果累加

【优先级】高

> [!NOTE] 说明
> 本文介绍的优化建议不依赖具体API。为便于理解，“优化原理”小节沿用Basic API的接口及术语进行说明；不同API下的具体实现请参见[不同API下的实现](#different_api_implementations)。

## 优化原理

【描述】算子实现中对矩阵乘的结果进行累加时（比如矩阵A1 \* B1 + A2 \* B2...结果的累加），可将前一次矩阵乘的结果暂存在L0C Buffer（CO1）上，调用Mmad接口实现矩阵乘结果累加。相比于每次矩阵乘的结果从L0C Buffer（CO1）搬运到GM上，再搬运到Unified Buffer（UB）上进行累加计算，可减少数据搬运的次数，提升内存使用效率。

**图1**  反例数据流图    
<img src="../../../figures/bad_flow_80.png" alt="反例数据流图-80" width="920px" height="264px">

**图2**  正例数据流图    
<img src="../../../figures/good_dataflow_81.png" alt="正例数据流图-81" width="920px" height="181px">

【反例】

优化前，算子进行2次矩阵乘结果累加的过程如下：

-   将前一次矩阵乘的计算结果从L0C Buffer（CO1）搬运到workspace上，再从workspace搬运到UB上；
-   下一次矩阵乘计算重复完成上述步骤将结果搬运到UB上；
-   在UB上将2次矩阵乘的结果相加。

当需要累加n次矩阵乘时，分别增加了n次L0C Buffer（CO1）-\>workspace、workspace-\>UB搬运以及n次Add运算。

【正例】

通过优化，算子对矩阵乘结果累加时，可将前一次矩阵乘的结果暂存在L0C Buffer上，并通过Mmad接口的`cmatrixInitVal`和`cmatrixSource`参数控制每次矩阵乘的累加初值：

- 第一次调用Mmad时，将`cmatrixInitVal`配置为`true`，把C矩阵的初始值设置为0，计算得到`C = A1 × B1`，结果保留在L0C Buffer中。
- 第二次调用Mmad时，将`cmatrixInitVal`配置为`false`、`cmatrixSource`配置为`false`，表示不清零C矩阵，并以L0C Buffer中已有的结果作为C矩阵初始值，计算得到`C = C + A2 × B2`。

这样只需调用2次Mmad接口，即可在L0C Buffer中得到`C = A1 × B1 + A2 × B2`，无需将两次矩阵乘结果搬运到UB后再执行Add运算。

## 不同API下的实现<a id="different_api_implementations"></a>

<!-- npu="950" id20 -->

> [!NOTE] 说明
> C API实现部分仅适用于Ascend 950PR&950DT系列产品。


### C API

使用[asc_mmad](../../../../api/SIMD-API/c_api/cube_compute/mmad_compute/asc_mmad.md)的c_matrix_init_val控制是否清零；第二次计算选择L0C Buffer为初始值来源。

四个输入均为32×32的half矩阵，每个输入分别需要2048字节的L1 Buffer和L0A或L0B Buffer；L0C Buffer及每个GM结果需要4096字节。反例Vector阶段的三个UB数组各预留1024个float，满足完整VL读写范围和32字节地址对齐要求。LoadAccumulateInputsC、AccumulateC在Cube核函数中执行，FinishAccumulateC在独立的Vector核函数中执行；Host需保证Cube核函数完成后再启动Vector核函数，例如在同一Stream中依次下发，跨Stream时使用事件建立依赖。核内同步不能替代Cube与Vector核函数之间的依赖。

```cpp
#include "c_api/asc_simd.h"
__aicore__ inline void LoadAccumulateInputsC(__gm__ half* a1Gm, __gm__ half* b1Gm,
    __gm__ half* a2Gm, __gm__ half* b2Gm, __cbuf__ half* a1L1, __cbuf__ half* b1L1,
    __cbuf__ half* a2L1, __cbuf__ half* b2L1, __ca__ half* a1, __cb__ half* b1,
    __ca__ half* a2, __cb__ half* b2)
{
    constexpr auto cache = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM;
    asc_set_gm2l1_nz_para(1, 1, 32, 0);
    asc_copy_gm2l1_nd2nz(a1L1, a1Gm, 64, cache, 32, 32, 0, false);
    asc_copy_gm2l1_nd2nz(b1L1, b1Gm, 64, cache, 32, 32, 0, false);
    asc_copy_gm2l1_nd2nz(a2L1, a2Gm, 64, cache, 32, 32, 0, false);
    asc_copy_gm2l1_nd2nz(b2L1, b2Gm, 64, cache, 32, 32, 0, false);
    asc_sync_notify(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    asc_copy_l12l0a(a1, a1L1, 0, 0, 2, 2, 2, 2);
    asc_copy_l12l0b_transpose(b1, b1L1, 0, 0, 2, 2, 2, 2);
    asc_copy_l12l0a(a2, a2L1, 0, 0, 2, 2, 2, 2);
    asc_copy_l12l0b_transpose(b2, b2L1, 0, 0, 2, 2, 2, 2);
    asc_sync_notify(PIPE_MTE1, PIPE_M, EVENT_ID0);
    asc_sync_wait(PIPE_MTE1, PIPE_M, EVENT_ID0);
}

__aicore__ inline void StoreMatrixC(__gm__ float* dst, __cc__ float* src, uint16_t m, uint16_t n)
{
    asc_set_l0c_copy_nz_para(1, 0, 0);
    asc_copy_l0c2gm(dst, src, n, m, n, m,
        asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, asc_unit_flag_mode::DISABLE,
        asc_quant_mode::NoQuant, asc_relu_pre_mode::NONE, false, true, false, false);
}

__simd_vf__ inline void AddMatrixResultsC(__ubuf__ float* src0, __ubuf__ float* src1, __ubuf__ float* dst)
{
    uint32_t remaining = 64;
    auto mask = asc_update_mask_b32(remaining);
    for (uint16_t row = 0; row < 16; ++row) {
        vector_float x, y, z;
        asc_loadalign(x, src0 + row * 64);
        asc_loadalign(y, src1 + row * 64);
        asc_add(z, x, y, mask);
        asc_storealign(dst + row * 64, z, mask);
    }
}

__aicore__ inline void FinishAccumulateC(__gm__ float* first, __gm__ float* second,
    __gm__ float* output, __ubuf__ float* firstUb, __ubuf__ float* secondUb, __ubuf__ float* outputUb)
{
    asc_copy_gm2ub_align(firstUb, first, 32 * 32 * sizeof(float));
    asc_copy_gm2ub_align(secondUb, second, 32 * 32 * sizeof(float));
    asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_vf_call<AddMatrixResultsC>(firstUb, secondUb, outputUb);
    asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_copy_ub2gm_align(output, outputUb, 32 * 32 * sizeof(float));
    asc_sync();
}
template <bool KeepL0C>
__aicore__ inline void AccumulateC(__ca__ half* a1, __cb__ half* b1,
    __ca__ half* a2, __cb__ half* b2, __cc__ float* c,
    __gm__ float* first, __gm__ float* second)
{
    constexpr uint16_t m = 32, n = 32, k = 32;
    asc_mmad(c, a1, b1, m, k, n, asc_unit_flag_mode::DISABLE, true, false, true);
    asc_sync_pipe(PIPE_M);
    if constexpr (!KeepL0C) {
        asc_sync_notify(PIPE_M, PIPE_FIX, EVENT_ID0);
        asc_sync_wait(PIPE_M, PIPE_FIX, EVENT_ID0);
        StoreMatrixC(first, c, m, n);
        asc_sync_notify(PIPE_FIX, PIPE_M, EVENT_ID0);
        asc_sync_wait(PIPE_FIX, PIPE_M, EVENT_ID0);
    }
    asc_mmad(c, a2, b2, m, k, n, asc_unit_flag_mode::DISABLE, true, false, !KeepL0C);
    asc_sync_notify(PIPE_M, PIPE_FIX, EVENT_ID0);
    asc_sync_wait(PIPE_M, PIPE_FIX, EVENT_ID0);
    StoreMatrixC(second, c, m, n);
    asc_sync(); // 等待搬出完成后，调用者才可复用L0C Buffer。
}
```

【反例】

```cpp
LoadAccumulateInputsC(a1Gm, b1Gm, a2Gm, b2Gm, a1L1, b1L1, a2L1, b2L1, a1, b1, a2, b2);
AccumulateC<false>(a1, b1, a2, b2, c, first, second);
// Cube核函数结束；Host确认其完成后，在Vector核函数中调用：
FinishAccumulateC(first, second, output, firstUb, secondUb, outputUb);
```

【正例】

```cpp
LoadAccumulateInputsC(a1Gm, b1Gm, a2Gm, b2Gm, a1L1, b1L1, a2L1, b2L1, a1, b1, a2, b2);
AccumulateC<true>(a1, b1, a2, b2, c, first, second);
```

反例将四个输入矩阵从GM依次搬入L1 Buffer和L0A/L0B Buffer，将两次矩阵乘结果分别搬至first和second，再在Vector侧搬入UB相加并写回GM。正例使用相同的输入搬运，第二次矩阵乘直接累加L0C Buffer中的第一次结果，随后搬回GM。

<!-- end id20 -->

### Basic API

【反例】

```cpp
...
// 该样例仅做示例说明，非完整代码，省略了部分同步控制代码。
public:
    __aicore__ inline KernelSample()
    {
        aSize = m * k;
        bSize = k * n;
        cSize = m * n;
    }
    __aicore__ inline void Init(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *c)
    {
        aGM.SetGlobalBuffer((__gm__ half *)a);
        bGM.SetGlobalBuffer((__gm__ half *)b);
        cGM.SetGlobalBuffer((__gm__ float *)c);
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(half));
        pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(half));
        pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(float));
        pipe.InitBuffer(inQueueSrc0, 1, cSize * sizeof(float));
        pipe.InitBuffer(inQueueSrc1, 1, cSize * sizeof(float));
        pipe.InitBuffer(outQueueDst, 1, cSize * sizeof(float));

    }
    __aicore__ inline void Process()
    {
        // 第一次矩阵乘计算。
        CopyIn();
        SplitA();
        SplitB();
        Compute();
        // 将第一次矩阵乘的结果搬出。
        CopyOut();
        // 将第一次矩阵乘的结果搬运到UB。
        CopyIn1();
        // 第二次矩阵乘计算。
        Compute1();
        // 将第一次矩阵乘的结果搬出。
        CopyOut1();
        // 将第二次矩阵乘的结果搬运到UB。
        CopyIn1();
        // 将两次矩阵乘的结果累加。
        Compute2();
        CopyOut2();
    }
private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();

        Nd2NzParams dataCopyA1Params;
        dataCopyA1Params.ndNum = 1;
        dataCopyA1Params.nValue = m;
        dataCopyA1Params.dValue = k;
        dataCopyA1Params.srcNdMatrixStride = 0;
        dataCopyA1Params.srcDValue = k;
        dataCopyA1Params.dstNzC0Stride = m;
        dataCopyA1Params.dstNzNStride = 1;
        dataCopyA1Params.dstNzMatrixStride = 0;
        DataCopy(a1Local, aGM, dataCopyA1Params);

        Nd2NzParams dataCopyB1Params;
        dataCopyB1Params.ndNum = 1;
        dataCopyB1Params.nValue = k;
        dataCopyB1Params.dValue = n;
        dataCopyB1Params.srcNdMatrixStride = 0;
        dataCopyB1Params.srcDValue = n;
        dataCopyB1Params.dstNzC0Stride = k;
        dataCopyB1Params.dstNzNStride = 1;
        dataCopyB1Params.dstNzMatrixStride = 0;
        DataCopy(b1Local, bGM, dataCopyB1Params);

        inQueueA1.EnQue<half>(a1Local);
        inQueueB1.EnQue<half>(b1Local);
    }
    __aicore__ inline void SplitA()
    {
        ...
    }
    __aicore__ inline void SplitB()
    {
        ...
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
        LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
        LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
        MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        // 矩阵乘。
        Mmad(c1Local, a2Local, b2Local, mmadParams);
        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.EnQue<half>(a2Local);
        inQueueB2.EnQue<half>(b2Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        __gm__ uint8_t* usrWorkspace = AscendC::GetUserWorkspace(workspace);
        xGm.SetGlobalBuffer((__gm__ float *)(usrWorkspace));
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = n;
        fixpipeParams.mSize = m;
        fixpipeParams.srcStride = m;
        fixpipeParams.dstStride = n;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        // 将矩阵乘的计算结果从L0C Buffer（CO1）搬运到workspace。
        Fixpipe(xGm, c1Local, fixpipeParams);
        outQueueCO1.EnQue<float>(c1Local);
    }
    __aicore__ inline void CopyIn1()
    {
        LocalTensor<float> src0Local = inQueueSrc0.AllocTensor<float>();
        // 将矩阵乘的计算结果从workspace搬运到UB。
        DataCopy(src0Local, xGm, cSize);
        inQueueSrc0.EnQue<float>(src0Local);
    }
    __aicore__ inline void Compute1()
    {
        LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
        LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        // 矩阵乘。
        Mmad(c1Local, a2Local, b2Local, mmadParams);
        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }
    __aicore__ inline void CopyOut1()
    {
        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = n;
        fixpipeParams.mSize = m;
        fixpipeParams.srcStride = m;
        fixpipeParams.dstStride = n;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        // 将矩阵乘的计算结果从L0C Buffer（CO1）搬运到workspace。
        Fixpipe(xGm, c1Local, fixpipeParams);
        outQueueCO1.FreeTensor(c1Local);
    }
    __aicore__ inline void CopyIn2()
    {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<float> src1Local = inQueueSrc1.AllocTensor<float>();
        // 将矩阵乘的计算结果从workspace搬运到UB。
        DataCopy(src1Local, xGm, cSize);
        inQueueSrc1.EnQue<float>(src1Local);
    }
    __aicore__ inline void Compute2()
    {
        LocalTensor<float> src0Local = inQueueSrc0.DeQue<float>();
        LocalTensor<float> src1Local = inQueueSrc1.DeQue<float>();
        LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
        // 两次矩阵乘的结果相加。
        Add(dstLocal, src0Local, src1Local, cSize);
        outQueueDst.EnQue<float>(dstLocal);
        inQueueSrc0.FreeTensor(src0Local);
        inQueueSrc1.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut2()
    {
        ...
    }
private:
    TPipe pipe;
    TQue<TPosition::A1, 1> inQueueA1;
    TQue<TPosition::A2, 1> inQueueA2;
    TQue<TPosition::B1, 1> inQueueB1;
    TQue<TPosition::B2, 1> inQueueB2;
    TQue<TPosition::CO1, 1> outQueueCO1;
    TQue<TPosition::VECIN, 1> inQueueSrc0;
    TQue<TPosition::VECIN, 1> inQueueSrc1;
    TQue<TPosition::VECOUT, 1> outQueueDst;

    GlobalTensor<half> aGM;
    GlobalTensor<half> bGM;
    GlobalTensor<float> cGM;
    uint16_t m = 32, k = 32, n = 32;
    uint16_t aSize, bSize, cSize;
...
```

【正例】

```cpp
...
// 该样例仅做示例说明，非完整代码，省略了部分同步控制代码。
public:
    __aicore__ inline KernelSample()
    {
        aSize = m * k;
        bSize = k * n;
        cSize = m * n;
    }
    __aicore__ inline void Init(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *c)
    {
        aGM.SetGlobalBuffer((__gm__ half *)a);
        bGM.SetGlobalBuffer((__gm__ half *)b);
        cGM.SetGlobalBuffer((__gm__ float *)c);
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(half));
        pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(half));
        pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        SplitA();
        SplitB();
        Compute();
        CopyOut();
    }
private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();

        Nd2NzParams dataCopyA1Params;
        dataCopyA1Params.ndNum = 1;
        dataCopyA1Params.nValue = m;
        dataCopyA1Params.dValue = k;
        dataCopyA1Params.srcNdMatrixStride = 0;
        dataCopyA1Params.srcDValue = k;
        dataCopyA1Params.dstNzC0Stride = m;
        dataCopyA1Params.dstNzNStride = 1;
        dataCopyA1Params.dstNzMatrixStride = 0;
        DataCopy(a1Local, aGM, dataCopyA1Params);

        Nd2NzParams dataCopyB1Params;
        dataCopyB1Params.ndNum = 1;
        dataCopyB1Params.nValue = k;
        dataCopyB1Params.dValue = n;
        dataCopyB1Params.srcNdMatrixStride = 0;
        dataCopyB1Params.srcDValue = n;
        dataCopyB1Params.dstNzC0Stride = k;
        dataCopyB1Params.dstNzNStride = 1;
        dataCopyB1Params.dstNzMatrixStride = 0;
        DataCopy(b1Local, bGM, dataCopyB1Params);

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
    }
    __aicore__ inline void SplitA()
    {
        ...
    }
    __aicore__ inline void SplitB()
    {
        ...
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
        LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
        LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
        MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        // 第一次矩阵乘。
        Mmad(c1Local, a2Local, b2Local, mmadParams);
        PipeBarrier<PIPE_M>();
        // 第二次矩阵乘累加第一次矩阵乘的结果。
        mmadParams.cmatrixInitVal = false;
        Mmad(c1Local, a2Local, b2Local, c1Local, mmadParams);
        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = n;
        fixpipeParams.mSize = m;
        fixpipeParams.srcStride = m;
        fixpipeParams.dstStride = n;

        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        Fixpipe(cGM, c1Local, fixpipeParams);
        outQueueCO1.FreeTensor(c1Local);
    }
private:
    TPipe pipe;
    TQue<TPosition::A1, 1> inQueueA1;
    TQue<TPosition::A2, 1> inQueueA2;
    TQue<TPosition::B1, 1> inQueueB1;
    TQue<TPosition::B2, 1> inQueueB2;
    TQue<TPosition::CO1, 1> outQueueCO1;

    GlobalTensor<half> aGM;
    GlobalTensor<half> bGM;
    GlobalTensor<dst_T> cGM;
    uint16_t m = 32, k = 32, n = 32;
    uint16_t aSize, bSize, cSize;
```

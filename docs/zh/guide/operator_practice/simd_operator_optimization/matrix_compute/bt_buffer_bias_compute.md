# 通过BT Buffer实现高效的bias计算

【优先级】高

## 优化原理

【描述】算子中进行带bias的矩阵乘计算时，可将bias数据搬运至Bias Table Buffer（C2）上，调用一次Mmad接口实现矩阵乘加bias的计算，或者直接调用Matmul高阶API完成功能。相比于先将矩阵乘的结果从L0C Buffer（CO1）搬运到GM上，再搬运到Unified Buffer（UB）上进行加bias的过程，减少了数据搬运的次数，可提升内存使用效率。数据流图对比如下：

**图1**  反例数据流图    
<img src="../../../figures/bad_flow.png" alt="反例数据流图" width="600px" height="200px">

**图2**  正例数据流图    
<img src="../../../figures/good_dataflow.png" alt="正例数据流图" width="600px" height="147px">

【反例】

该算子进行带bias的矩阵乘计算时，过程如下：

-   将矩阵乘的计算结果从L0C Buffer（CO1）搬运到workspace上；
-   从workspace搬运到UB上；
-   在UB上进行加bias的运算；
-   最后将结果搬运到GM。

当循环n次该计算过程，则分别增加了n次L0C Buffer（CO1）-\>workspace、workspace-\>UB的搬运。

【正例】

该算子进行带bias的矩阵乘计算时，先将bias搬运到BT上，调用一次Mmad接口实现矩阵乘加bias的计算。

## 不同API下的实现<a id="different_api_implementations"></a>

<!-- npu="950" id20 -->

> [!NOTE] 说明
> C API实现部分仅适用于Ascend 950PR&950DT系列产品。


### C API

反例中的srcUb、biasUb、dstUb分别预留至少1056、64、1024个float，起始地址32字节对齐。整VL加载会读入有效矩阵、bias之外的补齐区，搬入后应将补齐区初始化为0；搬出仅覆盖1024个有效元素。Cube与Vector代码分别运行在对应类型的核中，Host必须等待Cube任务完成后才能启动读取workspace的Vector任务。

使用[asc_copy_l12bt](../../../../api/SIMD-API/c_api/cube_datamove/cube_compute_load/asc_copy_l12bt_arch_3510.md)搬运完整32个float（128B），再调用带bias起始地址的asc_mmad。

```cpp
#include "c_api/asc_simd.h"
__aicore__ inline void LoadCubeInputsC(__gm__ half* aGm, __gm__ half* bGm,
    __cbuf__ half* aL1, __cbuf__ half* bL1, __ca__ half* aL0, __cb__ half* bL0)
{
    constexpr auto cache = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM;
    asc_set_gm2l1_nz_para(1, 1, 32, 0);
    asc_copy_gm2l1_nd2nz(aL1, aGm, 64, cache, 32, 32, 0, false);
    asc_copy_gm2l1_nd2nz(bL1, bGm, 64, cache, 32, 32, 0, false);
    asc_sync_notify(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    asc_copy_l12l0a(aL0, aL1, 0, 0, 2, 2, 2, 2);
    asc_copy_l12l0b_transpose(bL0, bL1, 0, 0, 2, 2, 2, 2);
    asc_sync_notify(PIPE_MTE1, PIPE_M, EVENT_ID0);
    asc_sync_wait(PIPE_MTE1, PIPE_M, EVENT_ID0);
}

__aicore__ inline void LoadBiasC(__gm__ float* biasGm, __cbuf__ float* biasL1)
{
    constexpr auto cache = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM;
    // bias在BT中按连续float向量存放，不执行ND到NZ转换。
    asc_copy_gm2l1_align(biasL1, biasGm, 1, 128, 0, 0, false, cache, 128, 128);
    asc_sync_notify(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
}

__aicore__ inline void StoreMatrixC(__gm__ float* dst, __cc__ float* src, uint16_t m, uint16_t n)
{
    asc_set_l0c_copy_nz_para(1, 0, 0);
    asc_copy_l0c2gm(dst, src, n, m, n, m,
        asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, asc_unit_flag_mode::DISABLE,
        asc_quant_mode::NoQuant, asc_relu_pre_mode::NONE, false, true, false, false);
}
template <bool WithBias>
__aicore__ inline void BiasC(__ca__ half* a, __cb__ half* b, __cc__ float* c,
    __cbuf__ float* biasL1, __gm__ float* output)
{
    if constexpr (WithBias) {
        asc_copy_l12bt(0, biasL1, 128);
        asc_sync_notify(PIPE_MTE1, PIPE_M, EVENT_ID0);
        asc_sync_wait(PIPE_MTE1, PIPE_M, EVENT_ID0);
        asc_mmad(c, a, b, uint64_t(0), 32, 32, 32, asc_unit_flag_mode::DISABLE, true);
    } else {
        asc_mmad(c, a, b, 32, 32, 32, asc_unit_flag_mode::DISABLE, true, false, true);
    }
    asc_sync_notify(PIPE_M, PIPE_FIX, EVENT_ID0);
    asc_sync_wait(PIPE_M, PIPE_FIX, EVENT_ID0);
    StoreMatrixC(output, c, 32, 32);
}

__simd_vf__ inline void AddBiasC(__ubuf__ float* src, __ubuf__ float* bias, __ubuf__ float* dst)
{
    uint32_t remaining = 32;
    auto mask = asc_update_mask_b32(remaining);
    vector_float biasValue;
    asc_loadalign(biasValue, bias);
    for (uint16_t row = 0; row < 32; ++row) {
        vector_float value;
        asc_loadalign(value, src + row * 32);
        asc_add(value, value, biasValue, mask);
        asc_storealign(dst + row * 32, value, mask);
    }
}

__aicore__ inline void FinishBiasC(__gm__ float* workspace, __gm__ float* biasGm,
    __gm__ float* output, __ubuf__ float* srcUb, __ubuf__ float* biasUb, __ubuf__ float* dstUb)
{
    asc_copy_gm2ub_align(srcUb, workspace, 32 * 32 * sizeof(float));
    asc_copy_gm2ub_align(biasUb, biasGm, 32 * sizeof(float));
    asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_vf_call<AddBiasC>(srcUb, biasUb, dstUb);
    asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_copy_ub2gm_align(output, dstUb, 32 * 32 * sizeof(float));
    asc_sync();
}

```

【反例】

```cpp
LoadCubeInputsC(aGm, bGm, aL1, bL1, aL0, bL0);
BiasC<false>(aL0, bL0, cL0, biasL1, workspace);
FinishBiasC(workspace, biasGm, output, srcUb, biasUb, dstUb);
```

【正例】

```cpp
LoadCubeInputsC(aGm, bGm, aL1, bL1, aL0, bL0);
LoadBiasC(biasGm, biasL1);
BiasC<true>(aL0, bL0, cL0, biasL1, output);
```

反例依次执行A/B矩阵搬入、普通矩阵乘、L0C Buffer到workspace搬出、workspace和bias搬入UB、Vector Add以及UB到GM搬出。正例执行相同的A/B矩阵搬入，并将bias从GM搬入L1 Buffer后，由`BiasC<true>`完成L1 Buffer到BiasTable Buffer搬运、带bias矩阵乘及结果搬出。

<!-- end id20 -->

### Basic API

【反例】

```cpp
// 该样例仅做示例说明，非完整代码，省略了部分同步控制代码
public:
    __aicore__ inline KernelSample()
    {
        aSize = m * k;
        bSize = k * n;
        cSize = m * n;
    }
    __aicore__ inline void Init(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *bias, __gm__ uint8_t *c)
    {
        aGM.SetGlobalBuffer((__gm__ half *)a);
        bGM.SetGlobalBuffer((__gm__ half *)b);
        cGM.SetGlobalBuffer((__gm__ float *)c);
        biasGM.SetGlobalBuffer((__gm__ float *)bias);
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(half));
        pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(half));
        pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(float));
        pipe.InitBuffer(inQueueBias, 1, n * sizeof(float));
        pipe.InitBuffer(inQueueSrc0, 1, cSize * sizeof(float));
        pipe.InitBuffer(outQueueDst, 1, cSize * sizeof(float));

    }
    __aicore__ inline void Process()
    {
        CopyIn();
        SplitA();
        SplitB();
        Compute();
        CopyOut();
        CopyIn1();
        Compute1();
        CopyOut1();
    }
private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();
        LocalTensor<float> biasLocal = inQueueBias.AllocTensor<float>();

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
        // 将bias搬运到UB
        DataCopy(biasLocal, biasGM, n);

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
        inQueueBias.EnQue(biasLocal);
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
        // 矩阵乘
        Mmad(c1Local, a2Local, b2Local, mmadParams); // m*n
        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
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
        // 将矩阵乘的计算结果从L0C Buffer（CO1）搬运到workspace
        Fixpipe(xGm, c1Local, fixpipeParams);
        outQueueCO1.FreeTensor(c1Local);
    }
    __aicore__ inline void CopyIn1()
    {
        PipeBarrier<PIPE_ALL>();
        // 将矩阵乘的计算结果从workspace搬运到UB
        LocalTensor<float> src0Local = inQueueSrc0.AllocTensor<float>();
        DataCopy(src0Local, xGm, cSize);
        inQueueSrc0.EnQue(src0Local);
    }
    __aicore__ inline void Compute1()
    {
        LocalTensor<float> src0Local = inQueueSrc0.DeQue<float>();
        LocalTensor<float> biasLocal = inQueueBias.DeQue<float>();
        LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
        BinaryRepeatParams addRepeatParams;
        addRepeatParams.dstRepStride = 8;
        addRepeatParams.src0RepStride = 8;
        addRepeatParams.src1RepStride = 0;
        // 加bias的运算
        Add(dstLocal, src0Local, biasLocal, 32, m, addRepeatParams);
        outQueueDst.EnQue<float>(dstLocal);
        inQueueSrc0.FreeTensor(src0Local);
        inQueueBias.FreeTensor(biasLocal);
    }
    __aicore__ inline void CopyOut1()
    {
        ...
    }
private:
    TPipe pipe;
    TQue<TPosition::A1, 1> inQueueA1;
    TQue<TPosition::A2, 1> inQueueA2;
    TQue<TPosition::B1, 1> inQueueB1;
    TQue<TPosition::B2, 1> inQueueB2;
    TQue<TPosition::VECIN, 1> inQueueBias;
    TQue<TPosition::VECIN, 1> inQueueSrc0;
    TQue<TPosition::VECOUT, 1> outQueueDst;

    GlobalTensor<half> aGM;
    GlobalTensor<half> bGM;
    GlobalTensor<float> cGM;
    GlobalTensor<float> biasGM;
    uint16_t m = 32, k = 32, n = 32;
    uint16_t aSize, bSize, cSize;
...

```

【正例】

```cpp
...
// 该样例仅做示例说明，非完整代码，省略了部分同步控制代码
public:
    __aicore__ inline KernelSample()
    {
        aSize = m * k;
        bSize = k * n;
        cSize = m * n;
    }
    __aicore__ inline void Init(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *bias, __gm__ uint8_t *c)
    {
        aGM.SetGlobalBuffer((__gm__ half *)a);
        bGM.SetGlobalBuffer((__gm__ half *)b);
        cGM.SetGlobalBuffer((__gm__ float *)c);
        biasGM.SetGlobalBuffer((__gm__ float *)bias);
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(half));
        pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(half));
        pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(float));
        pipe.InitBuffer(inQueueC1, 1, n * sizeof(float));
        pipe.InitBuffer(outQueueC2, 1, n * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        SplitA();
        SplitB();
        SplitBias();
        Compute();
        CopyOut();
    }
private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();
        LocalTensor<float> bias1Local = inQueueC1.AllocTensor<float>();

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
        // 将bias从GM搬运到L1
        DataCopy(bias1Local, biasGM, n);

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
        inQueueC1.EnQue(bias1Local);
    }
    __aicore__ inline void SplitA()
    {
        ...
    }
    __aicore__ inline void SplitB()
    {
        ...
    }
    __aicore__ inline void SplitBias()
    {
        LocalTensor<float> bias1Local = inQueueC1.DeQue<float>();
        LocalTensor<float> bias2Local = outQueueC2.AllocTensor<float>();
        // 将bias从L1搬运到BT
        DataCopy(bias2Local, bias1Local, { 1, (uint16_t)(n * sizeof(float) / 64), 0, 0 });
        outQueueC2.EnQue<float>(bias2Local);
        inQueueC1.FreeTensor(bias1Local);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
        LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
        LocalTensor<float> bias2Local = outQueueC2.DeQue<float>();
        LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
        MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.cmatrixInitVal = false;
        // 矩阵乘
        Mmad(c1Local, a2Local, b2Local, bias2Local, mmadParams);
        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
        outQueueC2.FreeTensor(bias2Local);
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
    TQue<TPosition::C1, 1> inQueueC1;
    TQue<TPosition::C2, 1> outQueueC2;

    GlobalTensor<half> aGM;
    GlobalTensor<half> bGM;
    GlobalTensor<float> cGM;
    GlobalTensor<float> biasGM;
    uint16_t m = 32, k = 32, n = 32;
    uint16_t aSize, bSize, cSize;
```

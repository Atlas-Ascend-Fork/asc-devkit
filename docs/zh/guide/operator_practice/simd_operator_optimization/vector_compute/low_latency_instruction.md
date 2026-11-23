# 选择低延迟指令，优化归约操作性能

【优先级】高

> [!NOTE] 说明
> 本文介绍的优化建议不依赖具体API。为便于理解，“优化原理”小节沿用Basic API的接口及术语进行说明；不同API下的具体实现请参见[不同API下的实现](#different_api_implementations)。

## 优化原理

指令执行延迟（Instruction Execution Latency）指的是一条指令从开始执行到完全完成（即所有操作结束，结果可用）所消耗的时间，它直接影响程序的响应速度和实时性。在延迟敏感的场景中，降低指令执行延迟是提升性能的关键。下文以归约操作为例，介绍了几种归约方案的性能对比，便于开发者在使用归约指令时，能够根据具体的数据规模和场景，选择性能更高的方案。

### 二分累加方案和归约类指令方案的对比

根据单指令性能测试数据（开发者可以自行测试）分析，ReduceRepeat等归约指令的延迟时间约为Add指令的2-5倍。因此，对于连续数据的归约操作，可以采用Add指令和ReduceRepeat指令的组合，以优化整体性能。该方案简称为**二分累加方案**，具体方案说明如下：

-   二分累加：将数据一分为二，使用Add指令将两部分数据相加；将相加后的结果再次一分为二，继续使用Add指令进行累加，重复此过程。
-   当二分累加后的数据量小于等于256Byte（ReduceRepeat一条指令一个Repeat的数据操作量），使用ReduceRepeat指令，一次执行得到归约结果。

假设输入数据的数据类型为float，shape为\(5, 256\)，下图展示了一行数据的执行过程：

**图1**  二分累加方案示意图    
![](../../../figures/reduce_best_practice.png "二分累加方案示意图")

将以上过程，针对每一行，各自执行，得到最终归约结果，即shape为\(m, k\)的数据，归约完成后，shape为\(m, 1\)。

由于ReduceSum接口可能由多条指令组合实现，通常来说，数据量较大，循环次数较多的场景，二分累加方案性能 \> ReduceRepeat单指令操作性能 \> ReduceSum接口性能。而小数据量或者特殊shape下的场景，需要拆分开来，依据指令执行时间和指令执行数目等条件，具体问题具体分析。

下文给出了二分累加方案和归约类指令方案的核心代码片段和性能数据对比。完整样例请参考[ReduceCustom](../../../../../../examples/01_simd_cpp_api/03_basic_api/01_memory_vector_compute/reduce)。

【性能数据】

输入shape为30000，数据类型为float时，如下示例的性能数据对比如下，数据单位为cycle，通过GetSystemCycle接口获取。

| 二分累加方案 | ReduceRepeat单指令操作 |
| :--- | :--- |
| 172 | 242 |

[二分累加方案代码示例](#basic_reduce_example_0)。

[ReduceRepeat单指令操作代码示例](#basic_reduce_example_1)。

### ReduceDataBlock和ReduceRepeat归约方案对比

进一步测试分析可知，ReduceDataBlock单指令的执行效率优于ReduceRepeat，因此，根据不同的shape，通过不同的指令组合，可以达到更佳的执行性能。

例如数据类型为float，shape大小为256的数据，可以通过如下三种方式得到归约结果：

-   使用两次ReduceRepeat；
-   使用三次ReduceDataBlock；
-   一次ReduceDataBlock操作加一次ReduceRepeat操作。

通过分析单指令性能数据（开发者可以自行测试）可知：一次ReduceDataBlock操作加一次ReduceRepeat操作性能优于两次ReduceRepeat，同时也优于三次ReduceDataBlock的方案。

下文给出了上面三种方式的核心代码片段和性能数据对比。完整样例请参考[ReduceCustom](../../../../../../examples/01_simd_cpp_api/03_basic_api/01_memory_vector_compute/reduce)。

【性能数据】

输入shape为256，数据类型为float。如下示例的性能数据如下：

**表1**  两次ReduceRepeat、三次ReduceDataBlock、一次ReduceDataBlock加一次ReduceRepeat，三种归约操作的性能数据（循环100次的时间总和）

| 两次ReduceRepeat | 三次ReduceDataBlock | 一次ReduceDataBlock加一次ReduceRepeat |
| :--- | :--- | :--- |
| 13us | 13.94us | 8.44us |

[两次ReduceRepeat操作代码示例](#basic_reduce_example_2)。

[三次ReduceDataBlock操作代码示例](#basic_reduce_example_3)。

[ReduceDataBlock + ReduceRepeat操作代码示例](#basic_reduce_example_4)。

## 不同API下的实现<a id="different_api_implementations"></a>

<!-- npu="950" id50 -->

> [!NOTE] 说明
> C API实现部分仅适用于Ascend 950PR&950DT系列产品。


### C API

#### 30000个float的二分累加与分层归约

下面保留原30000个float的对照规模。输入UB按32768个float分配，函数先将末尾2768个元素清零，再比较二分累加与逐级整寄存器归约；两种写法均覆盖全部有效输入。src是可复用的工作区，执行过程中会被覆盖，正反例须分别重新搬入相同输入。dst至少预留8个float，src和dst互不重叠且起始地址32字节对齐。调用前等待GM到UB搬入完成，搬出结果前等待VF完成。原性能数据对应Basic API实现，C API耗时需另测。

```cpp
#include "c_api/asc_simd.h"
template <bool Binary>
__simd_vf__ inline void ReduceLongC(__ubuf__ float* src, __ubuf__ float* dst)
{
    vector_float zero, result;
    asc_duplicate_scalar(zero, 0.0f);
    vector_store_unalign padding;
    for (uint16_t i = 0; i < 44; ++i) {
        uint32_t offset = uint32_t(i) * 64;
        uint32_t count = 2768 - offset < 64 ? 2768 - offset : 64;
        asc_storeunalign(src + 30000 + offset, padding, zero, count);
    }
    asc_storeunalign_post(src + 30000, padding, 2768);
    asc_mem_bar(VST_VLD);
    auto all = asc_create_mask_b32(PAT_ALL);
    if constexpr (Binary) {
        uint32_t halfLength = 16384;
        // 32768个元素经9次二分相加后剩64个元素。
        for (uint16_t stage = 0; stage < 9; ++stage) {
            for (uint16_t block = 0; block < halfLength / 64; ++block) {
                uint32_t offset = uint32_t(block) * 64;
                vector_float left, right, sum;
                asc_loadalign(left, src + offset);
                asc_loadalign(right, src + halfLength + offset);
                asc_add(sum, left, right, all);
                asc_storealign(src + offset, sum, all);
            }
            asc_mem_bar(VST_VLD);
            halfLength /= 2;
        }
        vector_float value;
        asc_loadalign(value, src);
        asc_reduce_sum(result, value, all);
    } else {
        uint32_t length = 32768;
        // 逐级归约：32768 -> 512 -> 8 -> 1，部分和紧凑写回工作区。
        for (uint16_t stage = 0; stage < 3; ++stage) {
            uint32_t blocks = (length + 63) / 64;
            uint32_t remaining = length;
            vector_store_unalign partial;
            for (uint16_t block = 0; block < blocks; ++block) {
                vector_float value;
                asc_loadalign(value, src + uint32_t(block) * 64);
                asc_reduce_sum(result, value, asc_update_mask_b32(remaining));
                asc_storeunalign(src + block, partial, result, 1);
            }
            asc_storeunalign_post(src, partial, blocks);
            asc_mem_bar(VST_VLD);
            length = blocks;
        }
    }
    asc_storealign(dst, result, asc_create_mask_b32(PAT_VL1));
}
// 二分累加方案：asc_vf_call<ReduceLongC<true>>(src, dst)。
// 分层归约方案：asc_vf_call<ReduceLongC<false>>(src, dst)。
```

#### 256个float的归约组合

以下对比先二分累加再整寄存器归约、分层整寄存器归约，以及先按32B块归约再合并的写法。四个64元素寄存器覆盖一行，归约结果写入32B对齐输出的首元素。浮点求和顺序不同可能产生舍入差异。

```cpp
#include "c_api/asc_simd.h"
template <int Mode>
__simd_vf__ inline void ReduceRowC(__ubuf__ float* src, __ubuf__ float* dst)
{
    auto mask = asc_create_mask_b32(PAT_ALL);
    vector_float a, b, c, d, result;
    asc_loadalign(a, src); asc_loadalign(b, src + 64);
    asc_loadalign(c, src + 128); asc_loadalign(d, src + 192);
    if constexpr (Mode == 0) {
        asc_add(a, a, c, mask); asc_add(b, b, d, mask);
        asc_add(a, a, b, mask); asc_reduce_sum(result, a, mask);
    } else {
        if constexpr (Mode == 1) {
            asc_reduce_sum(a, a, mask); asc_reduce_sum(b, b, mask);
            asc_reduce_sum(c, c, mask); asc_reduce_sum(d, d, mask);
        } else {
            asc_reduce_sum_datablock(a, a, mask); asc_reduce_sum_datablock(b, b, mask);
            asc_reduce_sum_datablock(c, c, mask); asc_reduce_sum_datablock(d, d, mask);
        }
        asc_add(a, a, b, mask); asc_add(c, c, d, mask);
        asc_add(result, a, c, mask);
        if constexpr (Mode == 2) {
            asc_reduce_sum(result, result, asc_create_mask_b32(PAT_VL8));
        }
    }
    asc_storealign(dst, result, asc_create_mask_b32(PAT_VL1));
}
```


三次块内归约与“块内归约后整寄存器归约”的对照如下，分别选择 `ThreeStages=true` 和 `ThreeStages=false`。tmp为独立、32B对齐且至少64个float的UB临时区，前32个元素存放第一层部分和，其余空间保障完整寄存器读取合法。

```cpp
template <bool ThreeStages>
__simd_vf__ inline void BlockStagesC(__ubuf__ float* src, __ubuf__ float* dst, __ubuf__ float* tmp)
{
    auto mask = asc_create_mask_b32(PAT_ALL);
    for (uint16_t i = 0; i < 4; ++i) {
        vector_float v, part;
        asc_loadalign(v, src + i * 64);
        asc_reduce_sum_datablock(part, v, mask);
        asc_storealign(tmp + i * 8, part, asc_create_mask_b32(PAT_VL8));
    }
    asc_mem_bar(VST_VLD);
    vector_float partial, result;
    asc_loadalign(partial, tmp);
    if constexpr (ThreeStages) {
        asc_reduce_sum_datablock(result, partial, asc_create_mask_b32(PAT_VL32));
        asc_reduce_sum_datablock(result, result, asc_create_mask_b32(PAT_VL4));
    } else {
        asc_reduce_sum(result, partial, asc_create_mask_b32(PAT_VL32));
    }
    asc_storealign(dst, result, asc_create_mask_b32(PAT_VL1));
}
```

<!-- end id50 -->

### Basic API

【二分累加方案】<a id="basic_reduce_example_0"></a>

```cpp
__aicore__ inline void BinaryReduceSumImpl(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src, const uint32_t bsLength, const uint32_t hLength)
{
// src为二维数据，shape为(bsLength, hLength)，dst的shape为(bsLength,1)
AscendC::BinaryRepeatParams binaryParams;
AscendC::UnaryRepeatParams unaryParams;
AscendC::SetMaskCount();
for (uint32_t i = 0; i < bsLength; i++) {
    AscendC::LocalTensor<float> srcTmp = src[i * hLength];
    AscendC::LocalTensor<float> dstTmp = dst[i * hLength];
    uint32_t totalNum = hLength / 16 * 16;
    uint32_t remaining = hLength - totalNum;
    AscendC::LocalTensor<float> remainingTensor = srcTmp[totalNum];
    while (totalNum > ONE_REPEAT_FLOAT_SIZE) {
        uint32_t halfNum = AscendC::DivCeil(totalNum, 16) * DEFAULT_REP_STRIDE;
        AscendC::SetVectorMask<uint8_t, AscendC::MaskMode::COUNTER>(0, totalNum - halfNum);
        AscendC::Add<float, false>(dstTmp, srcTmp, srcTmp[halfNum], AscendC::MASK_PLACEHOLDER, 1, binaryParams);
        totalNum = halfNum;
        srcTmp = dstTmp;
    }
    if (remaining != 0 && hLength > ONE_REPEAT_FLOAT_SIZE) {
        AscendC::SetVectorMask<uint8_t, AscendC::MaskMode::COUNTER>(0, remaining);
        AscendC::Add<float, false>(dstTmp, dstTmp, remainingTensor, AscendC::MASK_PLACEHOLDER, 1, binaryParams);
    }
    AscendC::SetVectorMask<uint8_t, AscendC::MaskMode::COUNTER>(0, totalNum);
    AscendC::ReduceRepeat<AscendC::ReduceType::SUM, float, float, false>(dstTmp, srcTmp, AscendC::MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE,
        DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
}
AscendC::ResetMask();
AscendC::SetMaskNorm();
}
```

【ReduceRepeat单指令操作】<a id="basic_reduce_example_1"></a>

```cpp
__aicore__ inline void ReduceRepeatSumImpl(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src, const uint32_t bsLength, const uint32_t hLength)
{
// src为二维数据，shape为(bsLength, hLength)，dst的shape为(bsLength,1)
AscendC::SetMaskCount();
for (uint32_t i = 0; i < bsLength; i++) {
    uint32_t totalNum = hLength;
    AscendC::LocalTensor<float> srcTmp = src[i * hLength];
    AscendC::LocalTensor<float> dstTmp = dst[i * hLength];
    while (totalNum > 1) {
        AscendC::SetVectorMask<uint8_t, AscendC::MaskMode::COUNTER>(0, totalNum);
        AscendC::ReduceRepeat<AscendC::ReduceType::SUM, float, float, false>(dstTmp, srcTmp, AscendC::MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE,
            DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        totalNum = AscendC::DivCeil(totalNum, ONE_REPEAT_FLOAT_SIZE);
        srcTmp = dstTmp;
    }
}
AscendC::ResetMask();
AscendC::SetMaskNorm();
}
```

【两次ReduceRepeat操作】<a id="basic_reduce_example_2"></a>

```cpp
...
pipe.InitBuffer(calcBuf, totalLength * sizeof(DTYPE));
AscendC::LocalTensor<DTYPE> tempTensor1 = calcBuf.Get<DTYPE>();
const uint32_t repeatNum = (totalLength * sizeof(DTYPE) + REP_LEN - 1) / REP_LEN;
AscendC::SetMaskCount();
AscendC::SetVectorMask<DTYPE>(0, totalLength);
AscendC::ReduceRepeat<AscendC::ReduceType::SUM, DTYPE, DTYPE, false>(tempTensor1, xLocal, AscendC::MASK_PLACEHOLDER, 1,
DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<DTYPE>(0, repeatNum);
AscendC::ReduceRepeat<AscendC::ReduceType::SUM, DTYPE, DTYPE, false>(zLocal, tempTensor1, AscendC::MASK_PLACEHOLDER, 1,
DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskNorm();
...
```

【三次ReduceDataBlock操作】<a id="basic_reduce_example_3"></a>

```cpp
...
static constexpr uint32_t BLK_LEN = 32;
TBuf<TPosition::VECCALC> calcBuf;
constexpr uint32_t c0Count = BLK_LEN / sizeof(DTYPE_X);
const uint32_t blockNum0 = (totalLength + c0Count - 1) / c0Count;
const uint32_t blockNum1 = (blockNum0 + c0Count - 1) / c0Count;
AscendC::SetMaskCount();
AscendC::SetVectorMask<DTYPE_X>(0, totalLength);
AscendC::ReduceDataBlock<AscendC::ReduceType::SUM, DTYPE_X, DTYPE_X, false>(tempTensor1, xLocal, AscendC::MASK_PLACEHOLDER, 1,
DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<DTYPE_X>(0, blockNum0);
AscendC::ReduceDataBlock<AscendC::ReduceType::SUM, DTYPE_X, DTYPE_X, false>(tempTensor1, tempTensor1, AscendC::MASK_PLACEHOLDER, 1,
DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<DTYPE_X>(0, blockNum1);
AscendC::ReduceDataBlock<AscendC::ReduceType::SUM, DTYPE_X, DTYPE_X, false>(zLocal, tempTensor1, AscendC::MASK_PLACEHOLDER, 1,
DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskNorm();
...
```

【ReduceDataBlock + ReduceRepeat操作】<a id="basic_reduce_example_4"></a>

```cpp
...
pipe.InitBuffer(calcBuf, totalLength * sizeof(DTYPE));
AscendC::LocalTensor<DTYPE> tempTensor1 = calcBuf.Get<DTYPE>();
constexpr uint32_t c0Count = BLK_LEN / sizeof(DTYPE);
const uint32_t blockNum0 = (totalLength + c0Count - 1) / c0Count;
AscendC::SetMaskCount();
AscendC::SetVectorMask<DTYPE>(0, totalLength);
AscendC::ReduceDataBlock<AscendC::ReduceType::SUM, DTYPE, DTYPE, false>(tempTensor1, xLocal, AscendC::MASK_PLACEHOLDER, 1,
DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetVectorMask<DTYPE>(0, blockNum0);
AscendC::ReduceRepeat<AscendC::ReduceType::SUM, DTYPE, DTYPE, false>(zLocal, tempTensor1, AscendC::MASK_PLACEHOLDER, 1,
DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
AscendC::PipeBarrier<PIPE_V>();
AscendC::SetMaskNorm();
...
```

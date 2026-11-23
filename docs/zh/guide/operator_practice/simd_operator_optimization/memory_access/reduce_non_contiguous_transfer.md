# 非连续搬运场景减少搬运次数

【优先级】中

> [!NOTE] 说明
> 该性能优化建议适用于如下产品型号：
> <!-- npu="950" id1 -->
> - Ascend 950PR&950DT系列产品
> <!-- end id1 -->

> [!NOTE] 说明
> 本文介绍的优化建议不依赖具体API。为便于理解，“优化原理”小节沿用Basic API的接口及术语进行说明；不同API下的具体实现请参见[不同API下的实现](#different_api_implementations)。

## 优化原理

在非连续搬运场景可以使用DataCopyPad接口的Loop模式和DataCopy的多维数据搬运接口来减少搬运次数，优化搬运性能。

### 使用Loop模式减少非连续搬运的次数

【描述】DataCopyPad接口的Loop模式可以设置二级Loop，假设我们希望以下图的方式搬运8个48B大小的数据块：

<img src="../../../figures/copy_opt.png" alt="" width="920px" height="130px">

【反例】分别发起四次DataCopyPad搬运，每次处理两个数据块

**图1**  使用多次DataCopyPad接口进行搬运    
<img src="../../../figures/dcpad_api.png" alt="使用多次DataCopyPad接口进行搬运" width="920px" height="220px">

【正例】使用Loop模式进行搬运

**图2**  使用Loop模式进行搬运    
<img src="../../../figures/loop_copy.png" alt="使用Loop模式进行搬运" width="920px" height="291px">

【总结】当数据块之间需要插入不同大小Padding时，使用Loop模式搬运代替逐组搬运能够减少搬运指令的使用，提升性能。

### 使用多维数据搬运减少非连续搬运次数

【描述】假设我们希望以下图的方式搬运2个8B大小的数据块：

**图3**  搬运前后数据    
<img src="../../../figures/copy_data.png" alt="搬运前后数据" width="920px" height="175px">

【反例】使用多次DataCopyPad进行搬运

**图4**  使用多次DataCopyPad进行搬运    
<img src="../../../figures/dcpad_copy.png" alt="使用多次搬运进行搬运" width="920px" height="260px">

【正例】使用多维数据搬运

一次搬运描述多维步长和填充。

**图5**  搬运前后数据    
<img src="../../../figures/copy_data_56.png" alt="搬运前后数据-56" width="600px" height="207px">

【总结】使用多维数据搬运在部分场景下能够减少搬运指令的条数，从而提升性能。

## 不同API下的实现<a id="different_api_implementations"></a>

<!-- npu="950" id50 -->

> [!NOTE] 说明
> C API实现部分仅适用于Ascend 950PR&950DT系列产品。


### C API

两层循环例保留原96/128B和192/288B的源／目的步长；调用前将UB空隙预置为零并同步。多维填充例将两个8B数据块搬入4×4 float布局，中间两行各保留两个有效元素。两例均以 `Batch=false` 为反例、`Batch=true` 为正例。

```cpp
#include "c_api/asc_simd.h"
template <bool Batch>
__aicore__ inline void CopyLoopC(__ubuf__ float* dst, __gm__ float* src)
{
    constexpr auto cache = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM;
    if constexpr (Batch) {
        asc_set_gm2ub_loop_size(2, 2);
        asc_set_gm2ub_loop1_stride(96, 128);
        asc_set_gm2ub_loop2_stride(192, 288);
        asc_copy_gm2ub_align(dst, src, 2, 48, 0, 0, false, cache, 48, 48);
        asc_set_gm2ub_loop_size(1, 1);
    } else {
        constexpr uint32_t offset[] = {0, 32, 72, 104};
        for (uint16_t i = 0; i < 4; ++i) {
            asc_copy_gm2ub_align(dst + offset[i], src + i * 24,
                2, 48, 0, 0, false, cache, 48, 48);
        }
    }
}

template <bool Batch>
__aicore__ inline void CopyPaddedC(__ubuf__ float* dst, __gm__ float* src)
{
    constexpr auto cache = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM;
    if constexpr (Batch) {
        asc_set_ndim_loop0_stride(1, 1);
        asc_set_ndim_loop1_stride(4, 2);
        asc_set_ndim_loop2_stride(0, 0);
        asc_set_ndim_loop3_stride(0, 0);
        asc_set_ndim_loop4_stride(0, 0);
        asc_ndim_pad_count_config config{};
        config.loop1_lp_count = 1;
        config.loop1_rp_count = 1;
        asc_set_ndim_pad_count(config);
        asc_set_ndim_pad_value(0.0f);
        asc_ndim_copy_dci();
        asc_ndim_copy_gm2ub(dst, src, 2, 2, 1, 1, 1, 1, 1, true, cache);
    } else {
        asc_set_copy_pad_val(0.0f);
        asc_copy_gm2ub_align(dst, src, 1, 8, 5, 1, true, cache, 0, 0);
        asc_copy_gm2ub_align(dst + 8, src + 2, 1, 8, 1, 5, true, cache, 0, 0);
    }
}
```

【反例】调用多次搬运接口进行两层循环搬运

```cpp
CopyLoopC<false>(dst, src);
```

【正例】使用Loop模式进行搬运

```cpp
CopyLoopC<true>(dst, src);
```

【反例】调用多次搬运接口处理多维填充

```cpp
CopyPaddedC<false>(dst, src);
```

【正例】使用多维数据搬运

```cpp
CopyPaddedC<true>(dst, src);
```

读取UB结果前，应等待搬入流水完成。

<!-- end id50 -->

### Basic API

第一组代码的DataCopyExtParams中，blockLen、srcStride以字节计，dstStride以32B的DataBlock计；LoopModeParams的各循环步长以字节计，张量下标和左右填充数以元素计。Compact模式不使用dstStride，置0，目的数据块按有效长度紧凑排列。T须为常规整字节类型（如half、float、int8_t）；4bit类型的打包、下标和空间计算需另行处理，不能直接使用这里的sizeof(T)换算。第二组正例使用[多维数据搬运](../../../../api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopy_GMToUB_NDDMA.md)。

【反例】调用多次搬运接口进行搬运（以DataCopyPad为例）

```cpp
__aicore__ inline void CopyIn3(){
constexpr uint16_t blockCount = 2;
constexpr uint32_t blockLenBytes = 48;
constexpr uint32_t copyBytes = blockCount * blockLenBytes;
constexpr uint32_t dstGap1Bytes = 32;
constexpr uint32_t dstGap2Bytes = 64;
constexpr uint32_t dstGap3Bytes = 32;
constexpr uint32_t srcOffset1 = copyBytes / sizeof(T);
constexpr uint32_t srcOffset2 = 2 * copyBytes / sizeof(T);
constexpr uint32_t srcOffset3 = 3 * copyBytes / sizeof(T);
constexpr uint32_t dstOffset1 = (copyBytes + dstGap1Bytes) / sizeof(T);
constexpr uint32_t dstOffset2 = (2 * copyBytes + dstGap1Bytes + dstGap2Bytes) / sizeof(T);
constexpr uint32_t dstOffset3 = (3 * copyBytes + dstGap1Bytes + dstGap2Bytes + dstGap3Bytes) / sizeof(T);
AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
AscendC::Duplicate<T>(xLocal, 0, count);
AscendC::DataCopyExtParams dataCopyParams;
dataCopyParams.blockCount = blockCount;
dataCopyParams.blockLen = blockLenBytes;
dataCopyParams.srcStride = 0;
dataCopyParams.dstStride = 0;
dataCopyParams.rsv = 0;
AscendC::DataCopyPadExtParams<T> dataCopyPadParams;
dataCopyPadParams.isPad = 0;
dataCopyPadParams.leftPadding = 0;
dataCopyPadParams.rightPadding = 0;
dataCopyPadParams.paddingValue = 0;
AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(xLocal, xGm, dataCopyParams, dataCopyPadParams);
AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(
    xLocal[dstOffset1], xGm[srcOffset1], dataCopyParams, dataCopyPadParams);
AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(
    xLocal[dstOffset2], xGm[srcOffset2], dataCopyParams, dataCopyPadParams);
AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(
    xLocal[dstOffset3], xGm[srcOffset3], dataCopyParams, dataCopyPadParams);
inQueueX.EnQue<T>(xLocal);
}
```

【正例】使用Loop模式进行搬运

```cpp
__aicore__ inline void CopyIn3(){
AscendC::LoopModeParams loopModeParams;
loopModeParams.loop1Size = 2;
loopModeParams.loop2Size = 2;
loopModeParams.loop1SrcStride = 96;
loopModeParams.loop1DstStride = 128;
loopModeParams.loop2SrcStride = 192;
loopModeParams.loop2DstStride = 288;
AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
AscendC::Duplicate<T>(xLocal, 0, count);
AscendC::DataCopyExtParams dataCopyParams;
dataCopyParams.blockCount = 2;
dataCopyParams.blockLen = 48;
dataCopyParams.srcStride = 0;
dataCopyParams.dstStride = 0;
dataCopyParams.rsv = 0;
AscendC::DataCopyPadExtParams<T> dataCopyPadParams;
dataCopyPadParams.isPad = 0;
dataCopyPadParams.leftPadding = 0;
dataCopyPadParams.rightPadding = 0;
dataCopyPadParams.paddingValue = 0;
AscendC::SetLoopModePara(loopModeParams, AscendC::DataCopyMVType::OUT_TO_UB);
AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(xLocal, xGm, dataCopyParams, dataCopyPadParams);
AscendC::ResetLoopModePara(AscendC::DataCopyMVType::OUT_TO_UB);
inQueueX.EnQue<T>(xLocal);
}
```

【反例】使用多次DataCopyPad进行搬运

```cpp
__aicore__ inline void CopyIn5(){
constexpr uint32_t blockLenBytes = 8;
constexpr uint8_t leftPadding = 5;
constexpr uint8_t rightPadding = 1;
constexpr uint32_t srcOffset = blockLenBytes / sizeof(T);
constexpr uint32_t dstOffset =
    ((blockLenBytes + (leftPadding + rightPadding) * sizeof(T) + 31) / 32 * 32) / sizeof(T);
AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
AscendC::Duplicate<T>(xLocal, 0, count);
AscendC::DataCopyExtParams dataCopyParams;
dataCopyParams.blockCount = 1;
dataCopyParams.blockLen = blockLenBytes;
dataCopyParams.srcStride = 0;
dataCopyParams.dstStride = 0;
dataCopyParams.rsv = 0;
AscendC::DataCopyPadExtParams<T> dataCopyPadParams;
dataCopyPadParams.isPad = 1;
dataCopyPadParams.leftPadding = leftPadding;
dataCopyPadParams.rightPadding = rightPadding;
dataCopyPadParams.paddingValue = 0;
// 第一次搬运
AscendC::DataCopyPad<T, AscendC::PaddingMode::Normal>(xLocal, xGm, dataCopyParams, dataCopyPadParams);
dataCopyPadParams.isPad = 1;
dataCopyPadParams.leftPadding = rightPadding;
dataCopyPadParams.rightPadding = leftPadding;
dataCopyPadParams.paddingValue = 0;
// 第二次搬运
AscendC::DataCopyPad<T, AscendC::PaddingMode::Normal>(
    xLocal[dstOffset], xGm[srcOffset], dataCopyParams, dataCopyPadParams);
inQueueX.EnQue<T>(xLocal);
}
```

【正例】使用多维数据搬运

```cpp
__aicore__ inline void CopyIn6(){
AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
AscendC::Duplicate<T>(xLocal, 0, count);
AscendC::NdDmaLoopInfo<2> loopInfo{{1, 2}, {1, 4}, {2, 2}, {1, 1}, {1, 1}};
AscendC::NdDmaParams<T, 2> params = {loopInfo, 0};
AscendC::NdDmaDci();
static constexpr AscendC::NdDmaConfig config = {false};
AscendC::DataCopy<T, 2, config>(xLocal, xGm, params);
inQueueX.EnQue<T>(xLocal);
}
```

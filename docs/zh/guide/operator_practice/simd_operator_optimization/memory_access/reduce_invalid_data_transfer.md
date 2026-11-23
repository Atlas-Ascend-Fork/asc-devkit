# 非对齐场景减少无效数据的搬运

【优先级】中

> [!NOTE] 说明
> 该性能优化建议适用于如下型号：
> <!-- npu="950" id1 -->
> - Ascend 950PR&950DT系列产品
> <!-- end id1 -->

> [!NOTE] 说明
> 本文介绍的优化建议不依赖具体API。为便于理解，“优化原理”小节沿用Basic API的接口及术语进行说明；不同API下的具体实现请参见[不同API下的实现](#different_api_implementations)。

## 优化原理

【描述】在非对齐数据搬运场景中，Ascend 950PR&950DT系列产品在Basic API层面提供了DataCopyPad接口，该接口支持Normal、Compact（紧凑）两种搬运模式。搬运多块非32B对齐数据块的场景下，使用Compact模式可以减少搬运的无效数据量，节省带宽。

假设需要搬运三个数据块，每块数据块大小为48B，数据类型为float类型。除了这三个48字节的数据块之外，其他所有数据均为无效数据。

【反例】使用DataCopyPad接口进行Normal模式搬运数据。

搬运后UB内数据如下：

```text
[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.....]
```

**图1**  各数据块分别补齐    
<img src="../../../figures/norm_copy.png" alt="Normal模式搬运" width="920px" height="342px">

如图所示，由于每块数据块为48B，非32B对齐，因此搬运每块数据块时需要插入16B大小的padding数据使得数据32B对齐，最终搬运192B大小的数据到UB，其中包含48B的无效数据。

【正例】改用Compact模式搬运进行优化。

搬运后UB内数据如下：

```text
[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 0., 0., 0., 0....]
```

**图2**  数据块连续排列、末尾补齐    
<img src="../../../figures/compact_copy.png" alt="Compact模式搬运" width="920px" height="274px">

根据Compact模式搬运的示意图，最终搬运了160B大小的数据，其中包含16B的无效数据。

【总结】通过比较可以发现，搬运多个非32B对齐的数据块时，连续排列并只在末尾补齐，可以减少无效数据量，节省带宽。

## 不同API下的实现<a id="different_api_implementations"></a>

<!-- npu="950" id20 -->

> [!NOTE] 说明
> C API实现部分仅适用于Ascend 950PR&950DT系列产品。


### C API

使用[asc_copy_gm2ub_align](../../../../api/SIMD-API/c_api/vector_datamove/asc_copy_gm2ub_align/asc_copy_gm2ub_align_arch_3510.md)，源步长固定为48B，目的步长分别为64B和48B。步长为相邻块首地址的间隔。

Normal模式的目的UB至少预留192B，Compact模式至少预留160B；Compact模式的左右Padding参数必须为0，目的步长为48B，仅在三块有效数据整体末尾保留对齐空间。后续计算的地址步长必须与所选模式一致。若搬运后的数据交由Vector流水消费，应在消费前建立MTE2到Vector同步；复用目的UB前应等待Vector流水完成。

```cpp
#include "c_api/asc_simd.h"
template <bool Compact>
__aicore__ inline void CopyThreeBlocksC(__ubuf__ float* dst, __gm__ float* src)
{
    asc_set_copy_pad_val(0.0f);
    asc_copy_gm2ub_align(dst, src, 3, 48, 0, 0, true,
        asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, 48, Compact ? 48 : 64);
}
```

【反例】使用Normal模式搬运数据

```cpp
CopyThreeBlocksC<false>(dst, src);
```

【正例】改用Compact模式搬运进行优化

```cpp
CopyThreeBlocksC<true>(dst, src);
```

<!-- end id20 -->

### Basic API

以下T固定为float，Normal模式按每块64B预留UB，Compact模式按总长度向上补齐到32B预留；count须覆盖对应UB空间。

【反例】使用DataCopyPad接口进行Normal模式搬运数据

```cpp
__aicore__ inline void CopyIn(){
    AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    AscendC::Duplicate<T>(xLocal, 0, count);
    AscendC::DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 3;
    dataCopyParams.blockLen = 48; // 单位为字节
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    AscendC::DataCopyPadExtParams<T> dataCopyPadParams;
    dataCopyPadParams.isPad = 1;
    dataCopyPadParams.leftPadding = 0;
    dataCopyPadParams.rightPadding = 0; // 48B数据由硬件补齐；Compact不设置左右Padding
    dataCopyPadParams.paddingValue = 0;
    AscendC::DataCopyPad<T, AscendC::PaddingMode::Normal>(xLocal, xGm, dataCopyParams, dataCopyPadParams);
    inQueueX.EnQue<T>(xLocal);
}
```

【正例】改用Compact模式搬运进行优化

```cpp
__aicore__ inline void CopyIn(){
    AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    AscendC::Duplicate<T>(xLocal, 0, count);
    AscendC::DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 3;
    dataCopyParams.blockLen = 48; // 单位为字节
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    AscendC::DataCopyPadExtParams<T> dataCopyPadParams;
    dataCopyPadParams.isPad = 1;
    dataCopyPadParams.leftPadding = 0;
    dataCopyPadParams.rightPadding = 0; // 48B数据由硬件补齐；Compact不设置左右Padding
    dataCopyPadParams.paddingValue = 0;
    AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(xLocal, xGm, dataCopyParams, dataCopyPadParams);
    inQueueX.EnQue<T>(xLocal);
}
```

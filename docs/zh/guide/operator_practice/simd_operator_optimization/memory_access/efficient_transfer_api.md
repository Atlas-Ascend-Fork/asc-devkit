# 高效的使用搬运API

【优先级】高

> [!NOTE] 说明
> 本文介绍的优化建议不依赖具体API。为便于理解，“优化原理”小节沿用Basic API的接口及术语进行说明；不同API下的具体实现请参见[不同API下的实现](#different_api_implementations)。

## 优化原理

【描述】在使用搬运API时，应该尽可能地通过配置搬运控制参数实现连续搬运或者固定间隔搬运，避免使用for循环，二者效率差距极大。如下图示例，图片的每一行为16KB，需要从每一行中搬运前2KB，针对这种场景，使用for循环遍历每行，每次仅能搬运**2**KB。若直接配置DataCopyParams的blockCount、blockLen、srcStride和dstStride，则可以达到一次搬完的效果，每次搬运**32**KB；参考[尽量一次搬运较大的数据块](transfer_larger_data_blocks.md)章节介绍的搬运数据量和实际带宽的关系，建议一次搬完。

**图1**  待搬运数据排布    
![](../../../figures/pending_data.png "待搬运数据排布")

【反例】

从GM上每行16KB中搬运前2KB数据，共16行。使用for循环逐行提交搬运，每次搬运2KB，共提交16次。

【正例】

将16行的块数、每行2KB的有效长度以及行间步长交给搬运接口，一次提交搬运，在UB中连续存放32KB数据。

## 不同API下的实现<a id="different_api_implementations"></a>

<!-- npu="950" id1 -->

> [!NOTE] 说明
> C API实现部分仅适用于Ascend 950PR&950DT系列产品。

### C API

通过[asc_copy_gm2ub_align](../../../../api/SIMD-API/c_api/vector_datamove/asc_copy_gm2ub_align/asc_copy_gm2ub_align_arch_3510.md)搬运数据，显式配置块数、块长度和源、目的步长。`tensorGM`至少指向16行、每行4096个float的GM输入，`tensorIn`至少容纳8192个float，且UB起始地址32字节对齐。

【反例】

每次搬运一行的有效数据，循环提交16次搬运。

```cpp
#include "c_api/asc_simd.h"

__aicore__ inline void CopyRowsCBad(__ubuf__ float* tensorIn, __gm__ float* tensorGM)
{
    constexpr uint32_t copyWidth = 2 * 1024 / sizeof(float);
    constexpr uint32_t imgWidth = 16 * 1024 / sizeof(float);
    constexpr uint32_t imgHeight = 16;
    for (uint32_t i = 0; i < imgHeight; ++i) {
        asc_copy_gm2ub_align(tensorIn + i * copyWidth, tensorGM + i * imgWidth,
            copyWidth * sizeof(float));
    }
    asc_sync_pipe(PIPE_ALL);
}
```

【正例】

一次提交16个数据块，每块2048字节；源行首间距16384字节，目的行首间距2048字节。

```cpp
__aicore__ inline void CopyRowsCGood(__ubuf__ float* tensorIn, __gm__ float* tensorGM)
{
    constexpr uint32_t copyWidth = 2 * 1024 / sizeof(float);
    constexpr uint32_t imgWidth = 16 * 1024 / sizeof(float);
    constexpr uint32_t imgHeight = 16;
    asc_copy_gm2ub_align(tensorIn, tensorGM, imgHeight, copyWidth * sizeof(float),
        0, 0, false, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
        imgWidth * sizeof(float), copyWidth * sizeof(float));
    asc_sync_pipe(PIPE_ALL);
}
```

采用该方式时，需要注意以下事项：

- 本节950接口的长度和步长均以字节为单位，步长表示相邻块首地址间距，不是块间空隙。
- UB至少容纳32KB有效数据，块数、块长度及步长必须满足接口范围。
- 使用搬运结果前需保证搬入完成；片段末尾等待流水完成。

<!-- end id1 -->

### Basic API

通过DataCopy搬运数据，以DataCopyParams配置块数、块长度和块间间隔。`tensorGM`已关联GM输入，`tensorIn`已分配至少8192个float，起始地址32字节对齐；以下片段使用`kernel_operator.h`和`AscendC`命名空间。

【反例】

每次搬运一行的有效数据，循环提交16次DataCopy。

```cpp
constexpr int32_t copyWidth = 2 * 1024 / sizeof(float);
constexpr int32_t imgWidth = 16 * 1024 / sizeof(float);
constexpr int32_t imgHeight = 16;
for (int32_t i = 0; i < imgHeight; ++i) {
    DataCopy(tensorIn[i * copyWidth], tensorGM[i * imgWidth], copyWidth);
}
```

【正例】

通过DataCopyParams一次提交16行的搬运。

```cpp
constexpr int32_t copyWidth = 2 * 1024 / sizeof(float);
constexpr int32_t imgWidth = 16 * 1024 / sizeof(float);
constexpr int32_t imgHeight = 16;
DataCopyParams copyParams;
copyParams.blockCount = imgHeight;
copyParams.blockLen = copyWidth / 8; // 单位为32字节，每块有8个float。
copyParams.srcStride = (imgWidth - copyWidth) / 8; // 源块间空隙，单位为32字节。
copyParams.dstStride = 0; // 目的数据块连续排列。
DataCopy(tensorIn, tensorGM, copyParams);
```

采用该方式时，需要注意以下事项：

- 此处DataCopyParams的块长度与块间间隔以32字节为单位，不能照搬950 C API的字节步长。
- 源、目的参数顺序为目的LocalTensor在前、源GlobalTensor在后。
- 使用搬运结果前需通过队列或事件保证搬入完成。

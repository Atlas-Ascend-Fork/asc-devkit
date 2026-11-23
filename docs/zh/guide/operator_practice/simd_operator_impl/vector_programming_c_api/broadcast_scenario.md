# Broadcast场景

在某些场景下，可能会存在两个输入shape不相同的情况。由于[asc_add](../../../../api/SIMD-API/c_api/reg_compute/reg_arith/asc_add.md)接口只支持对shape相同的输入进行计算，因此需要先对输入进行shape变换，再进行Add计算。本节将对满足Broadcast条件的输入在算子实现中的Broadcast处理进行介绍，其他场景可以参考本章节中提供的思路。

> [!NOTE]注意
> Broadcast机制通过扩展较小维度的数据，使得不同shape的输入能够进行运算，从而避免了显式的复制操作，提高了计算效率。数据进行Broadcast需满足：两个输入的维度个数相同，并且仅在某一个维度上的长度不同，某一个输入在此维度的长度为1。比如：shape为\(32, 8\)和 \(32, 1\)的两个输入可以进行Broadcast，因为它们都是二维，且第一个维度大小相等，而不相等的维度中第二个输入的维度为1，满足条件。

本节中将组合使用[asc_loadalign_brc_elem](../../../../api/SIMD-API/c_api/reg_compute/load/asc_loadalign_brc_elem.md)和非对齐寄存器存储接口完成Broadcast，因此输入需满足相关C API约束。其中AddVF使用的对齐加载、存储要求UB起始地址32字节对齐；BroadcastVF使用的标量广播加载和非对齐存储按各自接口约束访问。本节以输入维度为2、第二个轴（axis = 1）需要Broadcast为例进行说明。原基础API版本的完整样例代码请参见[输入Broadcast的Add算子样例](../../../../../../examples/01_simd_cpp_api/04_advanced_api/08_transpose/add_broadcast/README.md)。

## Tiling实现

与输入shape相同的场景相比，在Tiling结构体中增加相应的成员变量，表示是否需要对输入进行Broadcast、需要对哪个维度进行Broadcast、Broadcast的轴需要扩充的倍数。因此新增四个Tiling结构体成员：

-   xLen和yLen：表示两个输入的数据长度。
-   axis：表示对输入的哪个维度进行Broadcast。
-   coef：表示Broadcast的输入需要扩维的倍数。例如，x shape为\(m, 1\)，y shape为\(m, n\)，则coef = n。如下图所示，图中相同颜色部分为单次计算的数据块。

**图1**  axis=1时coef示意图    

![](../../../figures/axis_coef.png "axis-1时coef示意图")

Tiling结构体定义代码如下所示：

```cpp
struct AddCustomTilingData {
    uint32_t xLen;
    uint32_t yLen;
    uint32_t coef;
    uint32_t axis;
    ...
};
```

设需要进行Broadcast的输入长度为shorterAxisLen；不需要进行Broadcast的输入长度为totalLength。

```cpp
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 1; // 本例串行复用UB
...   // 读入数据
uint32_t totalLength = (xLen > yLen)? xLen : yLen;
uint32_t shorterAxisLen = (xLen < yLen)? xLen : yLen;
```

使用shorterAxisLen进行分核计算，并使用分核后的长度与coef相乘作为totalLength的分核长度。

```cpp
constexpr uint32_t BLOCK_SIZE = 32;
uint32_t alignCoef = (tiling->axis == 0U) ? shorterAxisLen : totalLength / shorterAxisLen;
uint32_t divDimCoef = (tiling->axis == 0U) ? totalLength / shorterAxisLen : shorterAxisLen;
if (divDimCoef % blockDim == 0U) {
    uint32_t blockLength = divDimCoef / blockDim * alignCoef;    
    ...
} else {
    uint32_t formerNum = (divDimCoef / BUFFER_NUM) % blockDim;
    uint32_t tailNum = blockDim - formerNum;

    uint32_t formerLength = ((divDimCoef / BUFFER_NUM) / blockDim + 1U) * BUFFER_NUM * alignCoef;
    uint32_t tailLength = ((divDimCoef / BUFFER_NUM) / blockDim) * BUFFER_NUM * alignCoef;
    ....
}
```

进行核内数据切分时，需要计算Unified Buffer（UB）数据块的数量向coef和BUFFER\_NUM对齐之后的数量ubBlockAligned。

```cpp
uint32_t ubBlockAligned =
        (MAX_AVAILABLE_UB_BLOCK_NUM * alignNum / (alignCoef * BUFFER_NUM) * (alignCoef * BUFFER_NUM) == 0U) ?
            0U :
            MAX_AVAILABLE_UB_BLOCK_NUM * alignNum / (alignCoef * BUFFER_NUM) * (alignCoef * BUFFER_NUM);
// Host保证length > 0，且一整行能放入UB；ubBlockAligned为0时需重新切分。
...
tileNum = length / ubBlockAligned;
if (length % ubBlockAligned == 0U || tileNum == 0U) {
    if (tileNum == 0U) {
        tileNum = 1U;
    }
    if (length < ubBlockAligned) {
        tileLength = length;
        lastTileLength = tileLength;
    } else {
        tileLength = ubBlockAligned;
        lastTileLength = tileLength;
    }
} else {
    tileNum++;
    tileLength = ubBlockAligned;
    lastTileLength = (uint32_t)(length - (tileNum - 1) * tileLength);
}
```

## 算子类实现

在核函数（Kernel）初始化阶段，根据Tiling结构体传入的参数确定对哪个输入进行Broadcast。由于针对输入的第二个轴（axis = 1）进行Broadcast，可以计算出，对于需要进行Broadcast的输入，每个核搬入数据长度为blockLength / coef。

初始化函数代码如下：

```cpp
__aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y,
    __gm__ uint8_t* z, AddCustomTilingData tiling,
    __ubuf__ T* xBuffer, __ubuf__ T* yBuffer, __ubuf__ T* zBuffer, __ubuf__ T* broadcastBuffer)
{
    xLocal = xBuffer;
    yLocal = yBuffer;
    zLocal = zBuffer;
    broadcastTmpTensor = broadcastBuffer;
    __gm__ T* longerInputPtr = (__gm__ T*)(tiling.xLen > tiling.yLen ? x : y);
    __gm__ T* shorterInputPtr = (__gm__ T*)(tiling.xLen > tiling.yLen ? y : x);
    this->coef = tiling.coef;
    uint32_t offset;
    if (tiling.isEvenCore) {
        this->tileNum = tiling.tileNum;
        this->tileLength = tiling.tileLength / BUFFER_NUM;
        this->lastTileLength = tiling.lastTileLength;
        offset = tiling.blockLength * block_idx;
    } else if (block_idx < tiling.formerNum) {
        this->tileNum = tiling.formerTileNum;
        this->tileLength = tiling.formerTileLength / BUFFER_NUM;
        this->lastTileLength = tiling.formerLastTileLength;
        offset = tiling.formerLength * block_idx;
    } else {
        this->tileNum = tiling.tailTileNum;
        this->tileLength = tiling.tailTileLength / BUFFER_NUM;
        this->lastTileLength = tiling.tailLastTileLength;
        offset = tiling.formerLength * tiling.formerNum +
            tiling.tailLength * (block_idx - tiling.formerNum);
    }
    xGm = longerInputPtr + offset;
    yGm = shorterInputPtr + offset / this->coef;
    zGm = (__gm__ T*)z + offset;
}
```

由于数据是向coef对齐的，在数据拷贝的过程中可能会出现地址不满足32字节对齐的场景，因此CopyIn函数中使用[asc_copy_gm2ub_align](../../../../api/SIMD-API/c_api/vector_datamove/asc_copy_gm2ub_align/asc_copy_gm2ub_align_arch_3510.md)、CopyOut函数中使用[asc_copy_ub2gm_align](../../../../api/SIMD-API/c_api/vector_datamove/asc_copy_ub2gm_align/asc_copy_ub2gm_align_arch_3510.md)进行数据拷贝。

CopyIn函数实现代码如下：

```cpp
__aicore__ inline void CopyIn(int32_t progress)
{
    uint32_t start = progress * this->tileLength;
    uint32_t rows = currentLength / this->coef;
    asc_copy_gm2ub_align(xLocal, xGm + start, currentLength * sizeof(T));
    asc_copy_gm2ub_align(yLocal, yGm + start / this->coef, rows * sizeof(T));
    asc_sync(); // 同时保证后续VF读取xLocal和yLocal的依赖
}
```

CopyOut函数实现代码如下：

```cpp
__aicore__ inline void CopyOut(int32_t progress)
{
    asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_copy_ub2gm_align(zGm + progress * this->tileLength,
        zLocal, currentLength * sizeof(T));
    asc_sync(); // 下一块复用同一组UB前等待搬出完成
}
```

在Compute函数中，调用Add接口前需要先对输入进行Broadcast。这里需要计算Broadcast前后的shape。基于前文提到的数据关系，可以计算得出Broadcast前后的shape分别为\{tileLength / broadcastCoef, 1\}和\{tileLength / broadcastCoef, broadcastCoef\}。在此基础上对输入进行Broadcast，并将计算结果存入临时空间中，然后进行Add计算。实现代码示例如下所示：

```cpp
__aicore__ inline void Compute(int32_t progress)
{
    asc_vf_call<BroadcastVF<T>>(broadcastTmpTensor, yLocal, currentLength / this->coef, this->coef);
    asc_vf_call<AddVF<T>>(zLocal, xLocal, broadcastTmpTensor, currentLength);
}
```

以上代码为同一个算子类中的片段，适用于NPU架构版本3510，包含头文件`"c_api/asc_simd.h"`。本例T仅取half或float，与AddVF的b16/b32掩码分支保持一致。调用者在核函数中声明32字节对齐的xLocal、yLocal、zLocal、broadcastTmpTensor数组，通过Init的xBuffer、yBuffer、zBuffer、broadcastBuffer参数传入并保存为UB指针成员；xLocal、zLocal和broadcastTmpTensor按最大单次处理长度向上取整到完整VL（256字节）预留，yLocal按较短输入长度向上取整到32字节预留。Host须保证分核和分块边界不切断一行，即offset、tileLength和尾块长度均为coef的整数倍。

每次先设置currentLength，再依次调用CopyIn、Compute、CopyOut；本例BUFFER_NUM固定为1，tileNum包含最后一个块；主块循环次数为tileNum - 1，最后一个块的currentLength为lastTileLength，且只处理一次。较短输入的核间偏移为offset / coef，搬入长度为currentLength / coef，不能沿用较长输入的长度。这里串行复用UB以说明广播逻辑；DoubleBuffer的两组缓冲区及事件调度参见[DoubleBuffer场景](double_buffer_scenario.md)。本节只实现原文的axis=1示例，不将其宣称为任意轴的广播接口。

每个非空核按以下顺序调用上述函数，避免最后一个块被重复处理：

```cpp
__aicore__ inline void Process()
{
    if (tileNum == 0) { return; }
    for (uint32_t progress = 0; progress + 1 < tileNum; ++progress) {
        currentLength = tileLength;
        CopyIn(progress);
        Compute(progress);
        CopyOut(progress);
    }
    currentLength = lastTileLength;
    CopyIn(tileNum - 1);
    Compute(tileNum - 1);
    CopyOut(tileNum - 1);
}
```

## Reg矢量函数实现

上述计算调用的矢量函数定义如下，实际源文件中须放在调用位置之前。广播结果通过非对齐存储连续写入，允许coef不满足32字节对齐；AddVF的输入数组按完整VL向上对齐预留。

```cpp
#include "c_api/asc_simd.h"

template <typename T>
__simd_vf__ inline void BroadcastVF(__ubuf__ T* dst, __ubuf__ T* src, uint32_t rows, uint32_t coef)
{
    vector_store_unalign writer;
    for (uint32_t row = 0; row < rows; ++row) {
        auto value = asc_loadalign_brc_elem(src + row);
        for (uint32_t i = 0; i < coef; i += 256 / sizeof(T)) {
            uint32_t n = coef - i < 256 / sizeof(T) ? coef - i : 256 / sizeof(T);
            asc_storeunalign(dst + row * coef + i, writer, value, n);
        }
    }
    asc_storeunalign_post(dst, writer, rows * coef);
}

template <typename T>
__simd_vf__ inline void AddVF(__ubuf__ T* dst, __ubuf__ T* x, __ubuf__ T* y, uint32_t count)
{
    uint32_t remain = count;
    for (uint32_t i = 0; i < count; i += 256 / sizeof(T)) {
        vector_bool mask;
        if constexpr (sizeof(T) == 2) {
            mask = asc_update_mask_b16(remain);
        } else {
            mask = asc_update_mask_b32(remain);
        }
        auto a = asc_loadalign(x + i);
        auto b = asc_loadalign(y + i);
        auto result = asc_add(a, b, mask);
        asc_storealign(dst + i, result, mask);
    }
}
```

# DoubleBuffer场景

因存在算子中多次搬入搬出数据的场景，为充分利用硬件资源，实现多流水并行，引入[DoubleBuffer](../../../technical_appendix/concepts_and_terms/performance_optimization/double_buffer.md)机制。[DoubleBuffer](../../../technical_appendix/concepts_and_terms/performance_optimization/double_buffer.md)是通过将输入数据分成大小相等的两块，充分利用AI Core的硬件资源，实现数据搬入、计算、数据搬出的并行执行方式。下面以“核间不均分，核内不均分”的样例为例，介绍算子中DoubleBuffer的实现。

**图1**  DoubleBuffer数据切分示意图    

![](../../../figures/dbuf_split.png "DoubleBuffer数据切分示意图")

## Tiling实现

开启DoubleBuffer后，每一个数据块会分成大小相等的两块，因此，若要开启DoubleBuffer，要求数据总量应该能够均分。为了简化处理，将可用的Unified Buffer（UB）空间以32字节为粒度，分成n块dataBlock，如果n不是偶数，则减1，这样就可以保证一套代码兼容开启或不开启DoubleBuffer功能。对应步骤如下：

1.  判断数据总长度totalLength是否满足32字节对齐，如不满足，则计算totalLength向上32字节对齐后的长度totalLengthAligned。

    ```cpp
    constexpr uint32_t BLOCK_SIZE = 32;
    // 为方便计算，这里根据数据类型定义变量alignNum作为对齐数
    uint32_t alignNum = BLOCK_SIZE / dataTypeSize;
    // totalLength为数据总量
    uint32_t totalLengthAligned = (totalLength % alignNum == 0)?
            totalLength : ((totalLength + alignNum - 1) / alignNum) * alignNum;
    ```

2.  根据totalLengthAligned，计算每个核的计算数据长度blockLength，分核策略可参照[尾核切分](multi_core_tiling/tail_core_split.md)。
3.  计算其余Tiling参数。

    对当前UB可用空间以32字节为粒度，进行切分，计算出数据块个数UB\_BLOCK\_NUM。根据是否开启DoubleBuffer计算出当前可用的最大数据块个数，记作MAX\_AVAILABLE\_UB\_BLOCK\_NUM。最后，以MAX\_AVAILABLE\_UB\_BLOCK\_NUM为粒度，对blockLength进行切分。为方便演示，如下代码直接给出UB\_BLOCK\_NUM，作为当前UB可用空间包含的block（32字节）数。

    ```cpp
    constexpr uint32_t BUFFER_NUM = 2;
    constexpr uint32_t UB_BLOCK_NUM = 21;  // UB最大可以使用的block数量
    constexpr uint32_t MAX_AVAILABLE_UB_BLOCK_NUM = UB_BLOCK_NUM / BUFFER_NUM * BUFFER_NUM;
    
    tileNum = blockLength / (alignNum * MAX_AVAILABLE_UB_BLOCK_NUM);
    if (tileNum == 0) {
        // 单核需要计算的长度小于UB可用空间，按照仅有尾块处理
        tileLength = 0;
        lastTileLength = (blockLength + alignNum - 1) / alignNum * alignNum;
    } else if ((blockLength / alignNum) % MAX_AVAILABLE_UB_BLOCK_NUM == 0) {
        // 单核的计算量能被当前可用UB空间均分，仅有主块，无尾块
        tileLength = MAX_AVAILABLE_UB_BLOCK_NUM * alignNum;
        lastTileLength = 0;
    } else {
        // 同时有主块和尾块
        tileLength = MAX_AVAILABLE_UB_BLOCK_NUM * alignNum;
        lastTileLength = blockLength - tileNum * tileLength;
    }
    ```

## 算子类实现

不开启DoubleBuffer时，只需要对每个核上最后一个分块的起始地址做处理；开启DoubleBuffer后，需要处理的数据块长度变成原来的一半，所以需要分别计算两组缓冲区的数据块起始地址。

开启DoubleBuffer，为每组输入、输出分别预留两套显式UB缓冲区，即BUFFER\_NUM=2。

```cpp
// 本例dataType=half，UB_BLOCK_NUM=21，可用20个DataBlock。
// 主块总长320个half，拆为两块各160个half；尾块最大不超过320个half。
// 为兼容不拆分的尾块，每个槽位按MAX_TILE_LENGTH预留。
static constexpr uint32_t BUFFER_NUM = 2;
static constexpr uint32_t MAX_TILE_LENGTH = 384;
__ubuf__ dataType xLocal[BUFFER_NUM * MAX_TILE_LENGTH];
__ubuf__ dataType yLocal[BUFFER_NUM * MAX_TILE_LENGTH];
__ubuf__ dataType zLocal[BUFFER_NUM * MAX_TILE_LENGTH];
```

同时在计算核内每个数据块的长度时，考虑DoubleBuffer场景，需要将Buffer数量，即BUFFER\_NUM=2带入计算。

```cpp
this->tileLength = tiling.tileLength / BUFFER_NUM;
```

由于无法保证尾块满足DoubleBuffer的条件，因此不对尾块进行切分。

```cpp
this->lastTileLength = tiling.lastTileLength;
```

Init函数实现代码如下：

```cpp
__aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* z, AddCustomTilingData tiling,
    __ubuf__ dataType* xBuffer, __ubuf__ dataType* yBuffer, __ubuf__ dataType* zBuffer)
{
    xLocal = xBuffer;
    yLocal = yBuffer;
    zLocal = zBuffer;
    if (tiling.isEvenCore) {
        this->blockLength = tiling.blockLength;
        this->tileNum = tiling.tileNum;
        this->tileLength = tiling.tileLength / BUFFER_NUM;
        this->lastTileLength = tiling.lastTileLength;

        xGm = (__gm__ dataType *)x + this->blockLength * block_idx;
        yGm = (__gm__ dataType *)y + this->blockLength * block_idx;
        zGm = (__gm__ dataType *)z + this->blockLength * block_idx;
    } else {
        if (block_idx < tiling.formerNum) {
            this->tileNum = tiling.formerTileNum;
            this->tileLength = tiling.formerTileLength / BUFFER_NUM;
            this->lastTileLength = tiling.formerLastTileLength;

            xGm = (__gm__ dataType *)x + tiling.formerLength * block_idx;
            yGm = (__gm__ dataType *)y + tiling.formerLength * block_idx;
            zGm = (__gm__ dataType *)z + tiling.formerLength * block_idx;
        } else {
            this->tileNum = tiling.tailTileNum;
            this->tileLength = tiling.tailTileLength / BUFFER_NUM;
            this->lastTileLength = tiling.tailLastTileLength;

            xGm = (__gm__ dataType *)x + tiling.formerLength * tiling.formerNum +
                tiling.tailLength * (block_idx - tiling.formerNum);
            yGm = (__gm__ dataType *)y + tiling.formerLength * tiling.formerNum +
                tiling.tailLength * (block_idx - tiling.formerNum);
            zGm = (__gm__ dataType *)z + tiling.formerLength * tiling.formerNum +
                tiling.tailLength * (block_idx - tiling.formerNum);
        }
    }

}
```

由于开启DoubleBuffer后，切分后的主块数据块个数翻倍，在Process函数中，需要将BUFFER\_NUM带入计算循环次数；尾块独立计算，不开启DoubleBuffer。后续主尾块在CopyIn、Compute、CopyOut阶段中的处理，与[尾块均分](multi_core_tiling/tail_block_even_split.md)相同。

```cpp
__aicore__ inline void Process()
{
    uint32_t loops = this->tileNum * BUFFER_NUM;
    for (uint32_t i = 0; i < loops; ++i) {
        uint32_t slot = i & 1;
        event_t event = slot == 0 ? EVENT_ID0 : EVENT_ID1;
        uint32_t localOffset = slot * MAX_TILE_LENGTH;
        uint32_t start = i * this->tileLength;
        uint32_t count = this->tileLength;
        if (i >= 2) {
            asc_sync_wait(PIPE_V, PIPE_MTE2, event);
        }
        asc_copy_gm2ub_align(xLocal + localOffset, xGm + start, count * sizeof(dataType));
        asc_copy_gm2ub_align(yLocal + localOffset, yGm + start, count * sizeof(dataType));
        asc_sync_notify(PIPE_MTE2, PIPE_V, event);
        asc_sync_wait(PIPE_MTE2, PIPE_V, event);
        if (i >= 2) {
            asc_sync_wait(PIPE_MTE3, PIPE_V, event);
        }
        asc_vf_call<AddVF<dataType>>(zLocal + localOffset, xLocal + localOffset, yLocal + localOffset, count);
        if (i + 2 < loops) {
            asc_sync_notify(PIPE_V, PIPE_MTE2, event);
        }
        asc_sync_notify(PIPE_V, PIPE_MTE3, event);
        asc_sync_wait(PIPE_V, PIPE_MTE3, event);
        asc_copy_ub2gm_align(zGm + start, zLocal + localOffset, count * sizeof(dataType));
        if (i + 2 < loops) {
            asc_sync_notify(PIPE_MTE3, PIPE_V, event);
        }
    }
    asc_sync();
    // 尾块进行计算，不做DoubleBuffer操作。
    if (this->lastTileLength > 0U) {
        uint32_t start = loops * this->tileLength;
        uint32_t count = this->lastTileLength;
        asc_copy_gm2ub_align(xLocal, xGm + start, count * sizeof(dataType));
        asc_copy_gm2ub_align(yLocal, yGm + start, count * sizeof(dataType));
        asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
        asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
        asc_vf_call<AddVF<dataType>>(zLocal, xLocal, yLocal, count);
        asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
        asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
        asc_copy_ub2gm_align(zGm + start, zLocal, count * sizeof(dataType));
        asc_sync();
    }
}
```

代码片段适用于NPU架构版本3510，包含头文件`"c_api/asc_simd.h"`，并在核函数开始调用`asc_init()`。Init和Process属于同一个算子类；xGm、yGm、zGm为GM指针，tileNum、tileLength、lastTileLength为成员变量。UB数组在调用者核函数中声明，通过Init新增的xBuffer、yBuffer、zBuffer参数传入，并保存为UB指针成员。每个槽位向上对齐预留384个half，输入、输出三组双缓冲共4608字节，Host侧UB预算须包含全部数组。事件仅在有后续同槽位任务时通知并等待；主块完成后排空流水，再独立处理尾块。实际性能收益需单独测量。

## Reg矢量函数实现

上述计算通过asc_vf_call调用以下矢量函数。实际源文件中将该定义放在调用位置之前；输入UB数组须按完整VL（256字节）向上对齐预留，搬运仍只处理实际数据长度。

```cpp
#include "c_api/asc_simd.h"

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

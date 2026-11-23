# 主块均分

长度为`totalLength`的输入数据可以按参与计算的核数和主块长度`mainTileLength`均匀切分，主块均分后无剩余数据。各核处理的数据量相同，每个核处理`tileNum`个主块，核内只包含主块。

**图1**  数据切分示意图    

![数据切分示意图](../../../../figures/maintile_even.png "数据切分示意图")

本场景原基础API样例的完整代码请参考[场景0：主块均分](../../../../../../../examples/01_simd_cpp_api/02_features/02_tiling_selector/vector_tiling_strategy/README.md#scenario0-main-tile-only)。样例中固定使用8个核，主块长度`mainTileLength`为3200个`half`元素，数据整体长度为256000个`half`元素。数据平均分配到8个核上处理，每个核计算32000个元素；每个核内共切分成10个主块，不涉及尾块和尾核处理。

## Tiling实现

基于本节的切分策略，Tiling需要定义如下参数：

- blockLength：每个核处理的数据长度，单位为元素。
- tileNum：每个核需要计算的主块个数。

据此定义`MainTileOnlyTiling`结构体，代码如下：

```cpp
struct MainTileOnlyTiling {
    uint32_t blockLength; // 每个核处理的元素数量
    uint32_t tileNum;     // 主块数量
};
```

接下来完成Tiling参数的计算。涉及的主要变量包括：逻辑核数`numBlocks`、一个DataBlock的字节数`dataBlockSize`、一个`half`元素的字节数`halfSize`、一个DataBlock包含的`half`元素数量`alignNum`、一个主块包含的`half`元素数量`mainTileLength`。步骤如下：

1. 根据`dataBlockSize`和`halfSize`计算`alignNum`，本样例中`alignNum`为16。`totalLength`为输入数据的元素总数，将其按`alignNum`向上对齐后得到对齐后的元素总数`totalLengthAligned`。

    ```cpp
    constexpr uint32_t dataBlockSize = 32;
    constexpr uint32_t halfSize = 2;
    constexpr uint32_t alignNum = dataBlockSize / halfSize;
    
    uint32_t totalLengthAligned = AlignUp(totalLength, alignNum);
    ```

2. 用`totalLengthAligned`除以`numBlocks * mainTileLength`得到每个核均分到的主块数量`mainTileNum`，取余得到主块均分后的剩余元素数量`mainTileRemainder`；再将`mainTileNum`乘以`mainTileLength`得到每个核已分配的元素数量`baseLength`，`baseTileNum`直接取`mainTileNum`。

    ```cpp
    constexpr uint32_t numBlocks = 8;
    constexpr uint32_t mainTileLength = 3200;
    
    uint32_t mainTileNum = totalLengthAligned / (numBlocks * mainTileLength);
    uint32_t mainTileRemainder = totalLengthAligned % (numBlocks * mainTileLength);
    uint32_t baseLength = mainTileNum * mainTileLength;
    uint32_t baseTileNum = mainTileNum;
    ```

3. 当`mainTileRemainder`为0时，表示所有数据都能按主块均匀分配到各核，此时使用`MainTileOnlyTiling`结构体传递Tiling切分信息。

    ```cpp
    if (mainTileRemainder == 0U) {
        MainTileOnlyTiling tiling{baseLength, baseTileNum};
        add_custom<MainTileOnlyTiling><<<numBlocks, 0, stream>>>(xDevice, yDevice, zDevice, tiling);
        return;
    }
    ```

## 算子类实现

核函数（Kernel）侧算子使用C API，通过GM指针和显式UB数组管理数据。内存管理和同步管理通过显式地址空间与C API同步接口完成，本节重点说明各核GM地址的计算以及主块数据的循环处理流程。

- 核函数（Kernel）入口接收Host侧传入的Tiling结构体，并依次调用算子类的`Init`与`Process`函数完成初始化与计算。主块均分场景中，传入的结构体类型为`MainTileOnlyTiling`。

    ```cpp
    template <typename TilingData>
    __vector__ __global__ void add_custom(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* z, TilingData tiling)
    {
        asc_init();
        KernelAdd op;
        op.Init(x, y, z, tiling);
        op.Process();
        asc_sync();
    }
    ```

- 由于数据被分配到多个核上处理，每个核访问的Global Memory地址范围各不相同。`Init`函数通过`tiling.blockLength`获取每个核处理的元素数量，并结合`block_idx`计算当前核所处理数据在Global Memory上的起始偏移`offset`，随后将偏移与长度传入`InitGm`完成GM地址设置；同时将`tiling.tileNum`保存到成员变量，供`Process`函数使用。

    ```cpp
    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* z,
                                MainTileOnlyTiling tiling)
    {
        uint32_t offset = block_idx * tiling.blockLength;
        InitGm(x, y, z, offset, tiling.blockLength);
        this->tileNum = tiling.tileNum;
        this->lastTileLength = 0U;
    }
    ```

- `InitGm`函数通过GM指针赋值为输入`xGm`、`yGm`和输出`zGm`设置Global Memory地址。以输入x为例，`(__gm__ half*)x + offset`即为当前核处理数据在Global Memory上的起始地址，`length`为该核处理的元素数量。

    ```cpp
    __aicore__ inline void InitGm(
        __gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* z, uint32_t offset, uint32_t length)
    {
        xGm = (__gm__ half*)x + offset;
        yGm = (__gm__ half*)y + offset;
        zGm = (__gm__ half*)z + offset;
    }
    ```

- `Process`函数基于成员变量`tileNum`循环处理主块数据。`xLocal`、`yLocal`、`zLocal`为显式分配的UB数组。循环次数采用`tileNum + (lastTileLength > 0U ? 1U : 0U)`的通用形式计算，主块均分场景下`lastTileLength`为0，循环次数即等于`tileNum`。每轮循环以`mainTileLength`作为当前主块的元素数量，通过`loopIdx * mainTileLength`计算核内偏移`startElement`，依次完成输入搬入、Add计算和输出搬出。

    ```cpp
    __aicore__ inline void Process()
    {
        __ubuf__ half xLocal[mainTileLength];
        __ubuf__ half yLocal[mainTileLength];
        __ubuf__ half zLocal[mainTileLength];
    
        uint32_t totalLoop = this->tileNum + (this->lastTileLength > 0U ? 1U : 0U);
        for (uint32_t loopIdx = 0; loopIdx < totalLoop; loopIdx++) {
            uint32_t curLen = (loopIdx < this->tileNum) ? mainTileLength : this->lastTileLength;
            uint32_t startElement = loopIdx * mainTileLength;
            asc_copy_gm2ub_align(xLocal, xGm + startElement, curLen * sizeof(half));
            asc_copy_gm2ub_align(yLocal, yGm + startElement, curLen * sizeof(half));
            asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
            asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
            asc_vf_call<AddVF<half>>(zLocal, xLocal, yLocal, curLen);
            asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
            asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
            asc_copy_ub2gm_align(zGm + startElement, zLocal, curLen * sizeof(half));
            // 下一轮复用同一组UB前，等待本轮读写全部结束。
            asc_sync();
        }
    }
    ```

上述代码片段属于同一个`KernelAdd`类，类中需要声明`__gm__ half *xGm, *yGm, *zGm;`、`uint32_t tileNum, lastTileLength;`和`static constexpr uint32_t mainTileLength = 3200;`。包含头文件`"c_api/asc_simd.h"`，核函数入口前先定义算子类。示例串行复用UB，跨块流水重叠参见[DoubleBuffer场景](../double_buffer_scenario.md)。Host按16个half向上对齐后，须为对齐部分预留GM空间。

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

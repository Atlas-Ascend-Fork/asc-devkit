# 较小矩阵长驻L1 Buffer，仅分次搬运较大矩阵

【优先级】高

## 优化原理

在进行cube计算时，当L1 Buffer无法全载左右矩阵时，可以让较小的矩阵长驻于L1 Buffer上，只分次搬运较大的矩阵，减少搬运次数。

【反例】

假设L1 Buffer的大小为512K，左矩阵和右矩阵的大小分别为992K、16K，数据类型为half，单次无法将左右矩阵全部载入L1 Buffer中。开发者规划的切分策略为：不切K轴，将左矩阵平均分成两块A1、A2，shape大小均为\[992, 256\]；将右矩阵平均分成两块，shape大小均为\[256, 16\]。计算时的加载顺序如下：先加载A1矩阵至L1 Buffer，将B1、B2依次加载并计算；然后再加载A2至L1 Buffer，将B1、B2依次加载并计算。

**图1**  反例切分策略图示    
<img src="../../../figures/bad_split.png" alt="反例切分策略图示" width="600px" height="531px">

【正例】

将较小的右矩阵一次性搬入L1 Buffer并长存于L1 Buffer上，循环内不断搬运A矩阵，当循环次数为2时，共需要3次GM到L1搬运。对应代码见下文[Basic API正例](#basic_l1_resident_example)。

## 不同API下的实现

<!-- npu="950" id50 -->

> [!NOTE] 说明
> C API实现部分仅适用于Ascend 950PR&950DT系列产品。


### C API

左矩阵为1984×256，右矩阵为256×32，输入均为half，输出为float。正例将完整右矩阵一次搬入L1 Buffer，反例按16列在内层循环重复搬入。每次向L1 Buffer加载992行左矩阵，再按最多128行搬入L0A Buffer计算，最后一个子块为96行。

调用者分别预留496 KiB和16 KiB的L1 Buffer，64 KiB的L0A Buffer、8 KiB的L0B Buffer和8 KiB的L0C Buffer；GM输出为1984×32个float。所有缓冲区满足对应接口的地址对齐要求。每个子块搬出后等待全部流水完成，保证下一子块复用L0及下一轮复用L1时，前序消费者已结束。

主要接口为[asc_copy_gm2l1_nd2nz](../../../../api/SIMD-API/c_api/cube_datamove/cube_compute_load/asc_copy_gm2l1_nd2nz_arch_3510.md)、[asc_copy_l12l0a](../../../../api/SIMD-API/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0a_2d_arch_3510.md)、[asc_copy_l12l0b_transpose](../../../../api/SIMD-API/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0b_2d_arch_3510.md)、[asc_mmad](../../../../api/SIMD-API/c_api/cube_compute/mmad_compute/asc_mmad.md)和[asc_copy_l0c2gm](../../../../api/SIMD-API/c_api/cube_datamove/cube_compute_store/asc_copy_l0c2gm_arch_3510.md)。

```cpp
#include "c_api/asc_simd.h"
template <bool Keep>
__aicore__ inline void StageMatrixC(__gm__ half* a, __gm__ half* b,
    __cbuf__ half* al1, __cbuf__ half* bl1, __ca__ half* al0,
    __cb__ half* bl0, __cc__ float* cl0, __gm__ float* output)
{
    constexpr auto cache = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM;
    if constexpr (Keep) {
        asc_set_gm2l1_nz_para(1, 1, 256, 0);
        asc_copy_gm2l1_nd2nz(bl1, b, 64, cache, 256, 32, 0, false);
    }
    for (uint32_t i = 0; i < 2; ++i) {
        asc_set_gm2l1_nz_para(1, 1, 992, 0);
        asc_copy_gm2l1_nd2nz(al1, a + i * 992 * 256, 512, cache, 992, 256, 0, false);
        for (uint32_t j = 0; j < 2; ++j) {
            auto tile = bl1 + (Keep ? j * 256 * 16 : 0);
            if constexpr (!Keep) {
                asc_set_gm2l1_nz_para(1, 1, 256, 0);
                asc_copy_gm2l1_nd2nz(tile, b + j * 16, 64, cache, 256, 16, 0, false);
            }
            asc_sync();
            for (uint32_t row = 0; row < 992; row += 128) {
                uint16_t rows = (992 - row < 128) ? 992 - row : 128;
                asc_copy_l12l0a(al0, al1, row / 16, 0, rows / 16, 16, 62, rows / 16);
                asc_copy_l12l0b_transpose(bl0, tile, 0, 0, 16, 1, 16, 1);
                asc_sync();
                asc_mmad(cl0, al0, bl0, rows, 256, 16,
                    asc_unit_flag_mode::DISABLE, true, false, true);
                asc_sync();
                asc_set_l0c_copy_nz_para(1, 0, 0);
                asc_copy_l0c2gm(output + (i * 992 + row) * 32 + j * 16,
                    cl0, 16, rows, 32, rows,
                    asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                    asc_unit_flag_mode::DISABLE, asc_quant_mode::NoQuant,
                    asc_relu_pre_mode::NONE, false, true, false, false);
                asc_sync();
            }
        }
    }
}
```

【反例】

```cpp
StageMatrixC<false>(a, b, al1, bl1, al0, bl0, cl0, output);
```

【正例】

```cpp
StageMatrixC<true>(a, b, al1, bl1, al0, bl0, cl0, output);
```

<!-- end id50 -->

### Basic API

【反例】

```cpp
...
public:
__aicore__ inline KernelSample()
{
    aSize = baseM * baseK;
    bSize = baseK * baseN;
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
    for (uint32_t i = 0; i < 2; i++) {
        CopyInA1(i);
        SplitA();
        for (uint32_t j = 0; j < 2; j++) {
            CopyInB1(j);
            SplitB();
            Compute(i, j);
        }
    }
    CopyOut();
}
private:
__aicore__ inline void CopyInA1(uint32_t i)
{
    LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
    // 左矩阵a1/a2分块载入L1 Buffer（A1）。
    Nd2NzParams dataCopyA1Params;
    dataCopyA1Params.ndNum = 1;
    dataCopyA1Params.nValue = baseM;
    dataCopyA1Params.dValue = baseK;
    dataCopyA1Params.srcNdMatrixStride = 0;
    dataCopyA1Params.srcDValue = baseK;
    dataCopyA1Params.dstNzC0Stride = baseM;
    dataCopyA1Params.dstNzNStride = 1;
    dataCopyA1Params.dstNzMatrixStride = 0;
    DataCopy(a1Local, aGM[i * baseM * baseK], dataCopyA1Params);
    inQueueA1.EnQue(a1Local);
}
__aicore__ inline void SplitA()
{
    LocalTensor<half> a1Local = inQueueA1.DeQue<half>();
    LocalTensor<half> a2Local = inQueueA2.AllocTensor<half>();
    // 左矩阵a1/a2分块从L1 Buffer（A1）搬运到L0A Buffer（A2）。
    LoadData2dParams loadL0AParams;
    loadL0AParams.repeatTimes = baseM * baseK * sizeof(half) / 512;
    loadL0AParams.srcStride = 1;
    loadL0AParams.dstGap = 0;
    LoadData(a2Local, a1Local, loadL0AParams);
    inQueueA2.EnQue(a2Local);
    inQueueA1.FreeTensor(a1Local);
}
__aicore__ inline void CopyInB1(uint32_t j)
{
    LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();
    // 右矩阵分块b1/b2载入L1 Buffer（B1）。
    Nd2NzParams dataCopyB1Params;
    dataCopyB1Params.ndNum = 1;
    dataCopyB1Params.nValue = baseK;
    dataCopyB1Params.dValue = baseN;
    dataCopyB1Params.srcNdMatrixStride = 0;
    dataCopyB1Params.srcDValue = n;
    dataCopyB1Params.dstNzC0Stride = baseK;
    dataCopyB1Params.dstNzNStride = 1;
    dataCopyB1Params.dstNzMatrixStride = 0;
    DataCopy(b1Local, bGM[j * baseN], dataCopyB1Params);
    inQueueB1.EnQue(b1Local);
}
__aicore__ inline void SplitB()
{
    LocalTensor<half> b1Local = inQueueB1.DeQue<half>();
    LocalTensor<half> b2Local = inQueueB2.AllocTensor<half>();
    // 右矩阵分块b1/b2从L1 Buffer（B1）搬运到L0B Buffer（B2）。
    LoadData2dTransposeParams loadL0BParams;
    loadL0BParams.startIndex = 0;
    loadL0BParams.repeatTimes = baseK / nBlockSize;
    loadL0BParams.srcStride = 1;
    loadL0BParams.dstGap = 1;
    LoadDataWithTranspose(b2Local, b1Local, loadL0BParams);
    inQueueB2.EnQue(b2Local);
    inQueueB1.FreeTensor(b1Local);
}
__aicore__ inline void Compute(uint32_t i, uint32_t j)
{
    LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
    LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
    LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
    // 矩阵乘。
    mmadParams.m = baseM;
    mmadParams.n = baseN;
    mmadParams.k = baseK;
    Mmad(c1Local[i * baseM * baseN + j * m * baseN], a2Local, b2Local, mmadParams);
    outQueueCO1.EnQue<float>(c1Local);
    inQueueA2.FreeTensor(a2Local);
    inQueueB2.FreeTensor(b2Local);
}
__aicore__ inline void CopyOut()
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

GlobalTensor<half> aGM;
GlobalTensor<half> bGM;
GlobalTensor<float> cGM;
uint16_t m = 1984, k = 256, n = 32;
uint16_t baseM = 992, baseK = 256, baseN = 16;
uint16_t aSize, bSize, cSize;
uint16_t nBlockSize = 16;
...
```

【正例】<a id="basic_l1_resident_example"></a>

```cpp
...
public:
__aicore__ inline KernelSample()
{
    aSize = baseM * baseK;
    bSize = baseK * n;
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
    CopyInB1();
    SplitB();
    for (uint32_t i = 0; i < 2; i++) {
        CopyInA1(i);
        SplitA();
        for (uint32_t j = 0; j < 2; j++) {
            Compute(i, j);
        }
    }
    CopyOut();
}
private:
__aicore__ inline void CopyInB1()
{
    LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();
    // 右矩阵全载入B1。
    Nd2NzParams dataCopyB1Params;
    dataCopyB1Params.ndNum = 1;
    dataCopyB1Params.nValue = baseK;
    dataCopyB1Params.dValue = n;
    dataCopyB1Params.srcNdMatrixStride = 0;
    dataCopyB1Params.srcDValue = n;
    dataCopyB1Params.dstNzC0Stride = baseK;
    dataCopyB1Params.dstNzNStride = 1;
    dataCopyB1Params.dstNzMatrixStride = 0;
    DataCopy(b1Local, bGM, dataCopyB1Params);
    inQueueB1.EnQue(b1Local);
}
__aicore__ inline void SplitB()
{
    LocalTensor<half> b1Local = inQueueB1.DeQue<half>();
    LocalTensor<half> b2Local = inQueueB2.AllocTensor<half>();
    // 右矩阵全部从L1 Buffer（B1）搬运到L0B Buffer（B2）。
    LoadData2dTransposeParams loadL0BParams;
    loadL0BParams.startIndex = 0;
    loadL0BParams.repeatTimes = baseK / nBlockSize;
    loadL0BParams.srcStride = 1;
    loadL0BParams.dstGap = 1;
    for (int blockNum = 0; blockNum < (n / nBlockSize); blockNum++) {
        LoadDataWithTranspose(b2Local[blockNum * 16 * nBlockSize], b1Local[blockNum * baseK * nBlockSize], loadL0BParams);
    }
    inQueueB2.EnQue(b2Local);
    inQueueB1.FreeTensor(b1Local);
}
__aicore__ inline void CopyInA1(uint32_t i)
{
    LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
    // 左矩阵a1/a2分块载入L1 Buffer（A1）。
    Nd2NzParams dataCopyA1Params;
    dataCopyA1Params.ndNum = 1;
    dataCopyA1Params.nValue = baseM;
    dataCopyA1Params.dValue = baseK;
    dataCopyA1Params.srcNdMatrixStride = 0;
    dataCopyA1Params.srcDValue = baseK;
    dataCopyA1Params.dstNzC0Stride = baseM;
    dataCopyA1Params.dstNzNStride = 1;
    dataCopyA1Params.dstNzMatrixStride = 0;
    DataCopy(a1Local, aGM[i * baseM * baseK], dataCopyA1Params);
    inQueueA1.EnQue(a1Local);
}
__aicore__ inline void SplitA()
{
    LocalTensor<half> a1Local = inQueueA1.DeQue<half>();
    LocalTensor<half> a2Local = inQueueA2.AllocTensor<half>();
    // 左矩阵a1/a2分块从L1 Buffer（A1）搬运到L0A Buffer（A2）。
    LoadData2dParams loadL0AParams;
    loadL0AParams.repeatTimes = baseM * baseK * sizeof(half) / 512;
    loadL0AParams.srcStride = 1;
    loadL0AParams.dstGap = 0;
    LoadData(a2Local, a1Local, loadL0AParams);
    inQueueA2.EnQue(a2Local);
    inQueueA1.FreeTensor(a1Local);
}
__aicore__ inline void Compute(uint32_t i, uint32_t j)
{
    LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
    LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
    LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
    // 矩阵乘。
    mmadParams.m = baseM;
    mmadParams.n = baseN;
    mmadParams.k = baseK;
    Mmad(c1Local[i * baseM * baseN + j * m * baseN], a2Local, b2Local, mmadParams);
    outQueueCO1.EnQue<float>(c1Local);
    inQueueA2.FreeTensor(a2Local);
    inQueueB2.FreeTensor(b2Local);
}
__aicore__ inline void CopyOut()
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

GlobalTensor<half> aGM;
GlobalTensor<half> bGM;
GlobalTensor<float> cGM;
uint16_t m = 1984, k = 256, n = 32;
uint16_t baseM = 992, baseK = 256, baseN = 16;
uint16_t aSize, bSize, cSize;
uint16_t nBlockSize = 16;
...
```

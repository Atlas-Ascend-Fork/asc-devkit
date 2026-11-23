# VF循环优化

## 优化原理

Ascend 950PR&950DT系列产品对应的架构中，Vector Function（VF）是实现高性能向量计算的核心载体。VF函数中可以包含最多四层嵌套循环，每层循环中还可以包含多个串行循环，同时支持非循环的向量操作和标量操作。VF对控制结构的支持有限，仅支持for循环和条件判断，不支持switch、do-while和while-do等其他控制结构。VF循环通过尽可能优化为硬件级向量循环（Hardware Loop），从而实现性能优化。

当VF函数中的循环满足[Hardware Loop编码规范](#hardware_loop_rules)时，会被编译器优化为Hardware Loop，提升整体执行性能；否则循环逻辑会由迭代变量和条件判断语句构成Software Loop，无法开启VF循环优化。

在遵循Hardware Loop编码规范、确保循环可被优化为Hardware Loop的基础上，可以通过成员变量访问、指令分布优化和地址管理优化等方面进一步提升性能。

### Hardware Loop编码规范<a id="hardware_loop_rules"></a>

为了能让编译器识别并生成Hardware Loop，对应的Loop代码必须符合硬件设计的要求。具体规范如下：

-   迭代变量类型

    VF内所有Hardware Loop的迭代变量必须是uint16\_t类型，非uint16_t类型不会有功能错误，但性能会变差。

-   起始值与步长

    循环起始值从0开始。

    每次迭代的步长必须是递增1。

-   循环嵌套层数

    VF内最多支持四层级的嵌套循环，否则最外层会被编译为Software Loop。

-   循环内不允许C++标准跳转控制流，比如if/else、三元运算符?:、switch、break、goto、continue等条件判断跳转。

    VF内if/else在Loop内会阻碍Hardware Loop的生成，编译器虽然会尽可能做if/else消除优化，但是不做完全性保证。如果循环内某些跳转指令无法被优化，此循环将会被编译为Aux Scalar构成的Software Loop，无功能性错误。

    if控制流可使用if constexpr或者for(1)替换。if constexpr在编译时已经完成，无运行时开销，但是需要传入的参数是编译时常量，不能依赖运行时变量；for(1)可以触发编译器的循环优化，在当前硬件条件下，Vector侧执行Loop的性能是远远优于条件分支跳转的。

    [尾块处理示例](#basic_hardware_loop_rules)展示了尾块处理场景，当尾块大小不为0时，hasTail为1，通过for(1)来代替if(hasTail)的判断，提高了Loop的性能。

-   一旦执行，循环计数/边界不允许被更改。
-   若要利用外层循环的计数作为循环边界，将外层循环计数器移动到另一个寄存器中，然后将其设置为循环边界。

-   避免不可终止的控制流

    如果存在程序无法结束的控制流（如无限循环、无终止条件的递归等），编译器会认为该行为未定义，编译期间尽可能识别，抛出编译错误。

硬件循环与软件循环的代码对照见[Basic API的Hardware Loop编码规范](#basic_hardware_loop_rules)。

### 循环内成员变量访问优化

在一个VF内，不推荐直接访问结构体或类对象的成员变量。直接访问结构体或类对象的成员变量相当于从栈上把内容搬运到Tensor寄存器中，之后通过地址访问Tensor上的内容，该操作将导致VF融合失效，建议在进入VF之前把所需成员复制到局部变量，再作为参数传入。

### 循环内指令分布优化

减少Loop循环内非索引相关语句。for循环中存在与索引无关的语句可以提出for循环外来减少指令数。

### 循环内地址管理优化

在VF循环中，当使用搬运指令时，需要计算搬入和搬出的地址偏移量，由此会引入较多的标量计算开销。在Ascend 950PR&950DT系列产品中，通过引入地址寄存器，可有效优化地址偏移量的计算。当满足如下的地址寄存器生成模式时，编译器有机会生成地址寄存器，从而消除相关的Scalar计算消耗，提升整体性能。

其中，地址寄存器最多支持4层循环寻址，如下图所示。

<img src="../../../../figures/addressing.png" alt="" width="920px" height="415px">

编译器通过识别对齐加载和存储指令进行模式匹配，当代码结构满足特定优化模式时，编译器将有机会进行高效优化，从而获得最佳的性能收益。相反，若直接使用地址寄存器存储偏移量或其它底层接口，可能会限制编译器的全局优化能力。

特别地，当四层循环构成连续访问场景时，地址管理可被简化为一维模式，从而实现更高效的搬运优化，进一步提升数据访问的局部性和执行效率。

## 不同API下的实现

<!-- npu="950" id50 -->

> [!NOTE] 说明
> C API实现部分仅适用于Ascend 950PR&950DT系列产品。


### C API

保留Hardware Loop编码约束、局部变量传参、不变量外提及地址管理四种手段。以下分别展示循环内不变量外提、尾块判断改为0次或1次循环，以及连续地址访问。NestedLoopC的bound示例取2；输入、输出各预留bound⁴×64个float且起始地址32字节对齐。runtimeStep必须大于0且循环计数不能溢出。CopyTailC的tail范围为0至64，输入仍预留完整64个float；AddressLoopC的stride以float元素为单位，为8的倍数，每行预留至少64个float。结构体字段应在进入VF前复制到局部变量，再作为下面的rows、stride和tail参数传入；不要在VF内反复读取对象成员。

```cpp
#include "c_api/asc_simd.h"
template <bool HardwareLoop>
__simd_vf__ inline void NestedLoopC(__ubuf__ float* src, __ubuf__ float* dst,
    uint16_t bound, uint16_t runtimeStart, uint16_t runtimeStep)
{
    auto mask = asc_create_mask_b32(PAT_ALL);
    for (uint16_t i = HardwareLoop ? 0 : runtimeStart;
        i < bound; i += HardwareLoop ? 1 : runtimeStep) {
        for (uint16_t j = 0; j < bound; ++j) {
            for (uint16_t k = 0; k < bound; ++k) {
                for (uint16_t m = 0; m < bound; ++m) {
                    uint32_t offset = (((uint32_t(i) * bound + j) * bound + k) * bound + m) * 64;
                    vector_float value;
                    asc_loadalign(value, src + offset);
                    asc_storealign(dst + offset, value, mask);
                }
            }
        }
    }
}

// 归约有效宽度1～64；srcK至少64且为8的倍数。
// 源UB覆盖srcM行的完整VL，目的UB至少srcM×8个float。
struct LastAxisShapeC { uint16_t k; };
struct SoftMaxTilingC { uint16_t srcM; uint16_t srcK; };
__simd_vf__ inline void ReduceRowsObjectC(__ubuf__ float* src, __ubuf__ float* dst,
    LastAxisShapeC shape, SoftMaxTilingC tiling)
{
    for (uint16_t i = 0; i < tiling.srcM; ++i) {
        uint32_t remaining = shape.k;
        auto mask = asc_update_mask_b32(remaining);
        vector_float value, result;
        asc_loadalign(value, src + i * tiling.srcK);
        asc_reduce_max(result, value, mask);
        asc_storealign(dst + i * 8, result, asc_create_mask_b32(PAT_VL1));
    }
}

__simd_vf__ inline void ReduceRowsScalarC(__ubuf__ float* src, __ubuf__ float* dst,
    uint16_t rows, uint16_t stride, uint16_t count)
{
    for (uint16_t i = 0; i < rows; ++i) {
        uint32_t remaining = count;
        auto mask = asc_update_mask_b32(remaining);
        vector_float value, result;
        asc_loadalign(value, src + uint32_t(i) * stride);
        asc_reduce_max(result, value, mask);
        asc_storealign(dst + uint32_t(i) * 8, result, asc_create_mask_b32(PAT_VL1));
    }
}

template <bool Localize>
__aicore__ inline void ReduceRowsC(__ubuf__ float* src, __ubuf__ float* dst,
    LastAxisShapeC shape, SoftMaxTilingC tiling)
{
    if constexpr (Localize) {
        uint16_t rows = tiling.srcM;
        uint16_t stride = tiling.srcK;
        uint16_t count = shape.k;
        asc_vf_call<ReduceRowsScalarC>(src, dst, rows, stride, count);
    } else {
        asc_vf_call<ReduceRowsObjectC>(src, dst, shape, tiling);
    }
}

template <bool Hoist>
__simd_vf__ inline void FillRowsC(__ubuf__ float* dst, float value, uint16_t rows, uint32_t stride)
{
    vector_float v;
    auto mask = asc_create_mask_b32(PAT_ALL);
    if constexpr (Hoist) { asc_duplicate_scalar(v, value); }
    for (uint16_t i = 0; i < rows; ++i) {
        if constexpr (!Hoist) { asc_duplicate_scalar(v, value); }
        asc_storealign(dst + i * stride, v, mask);
    }
}
template <bool UseLoop>
__simd_vf__ inline void CopyTailC(__ubuf__ float* src, __ubuf__ float* dst, uint32_t tail)
{
    uint16_t hasTail = !!tail;
    if constexpr (UseLoop) {
        for (uint16_t i = 0; i < hasTail; ++i) {
            vector_float v; auto mask = asc_update_mask_b32(tail);
            asc_loadalign(v, src); asc_storealign(dst, v, mask);
        }
    } else {
        if (hasTail) {
            vector_float v; auto mask = asc_update_mask_b32(tail);
            asc_loadalign(v, src); asc_storealign(dst, v, mask);
        }
    }
}
__simd_vf__ inline void AddressLoopC(__ubuf__ float* src, __ubuf__ float* dst, uint16_t rows, uint32_t stride)
{
    auto mask = asc_create_mask_b32(PAT_ALL);
    for (uint16_t i = 0; i < rows; ++i) {
        vector_float v; asc_loadalign(v, src + i * stride);
        asc_storealign(dst + i * stride, v, mask);
    }
}
// 【反例】调用NestedLoopC<false>(src, dst, bound, 0, 1)、ReduceRowsC<false>、
//          FillRowsC<false>、CopyTailC<false>。
// 【正例】调用NestedLoopC<true>(src, dst, bound, 0, 1)、ReduceRowsC<true>、
//          FillRowsC<true>、CopyTailC<true>、AddressLoopC。
```

<!-- end id50 -->

### Basic API

#### Hardware Loop编码规范<a id="basic_hardware_loop_rules"></a>

```cpp
//【反例】使用if语句
uint16_t tailK = srcK % floatRepSize;
uint16_t hasTail = 0;
// 通过!!tailK ，对是否产生尾块进行判断，如果srcK % floatRepSize余数为0，则hasTail对应bool值为0（false）,否则为1（true）
hasTail = !!tailK;
if(hasTail){
  // 尾块处理内容
}
```

```cpp
//【正例】使用for(1)来替代if判断语句
uint16_t tailK = srcK % floatRepSize;
uint16_t hasTail = 0;
hasTail = !!tailK;
for (uint16_t i = 0; i < hasTail; i++) {
  // 尾块处理内容
}
```

```cpp
//【正例】被编译器优化为Hardware Loop
// 嵌套循环
for (uint16_t i = 0; i < LoopBound; i++) {
for (uint16_t j = 0; j < LoopBound; j++) {
    for (uint16_t k = 0; k < LoopBound; k++) {
        for (uint16_t m = 0; m < LoopBound; m++) {
          // ...
        }
     }
 }
}
```

```cpp
//【反例】无法被编译器优化，构成Software Loop
for (uint16_t i = 0; i < LoopBound; i++) {     // Software Loop，循环内包含了if判断
if(cond){
    ...
}
}

for (uint16_t i = 2; i < LoopBound*3; i+=2){  // Software Loop，循环起始值不为0，循环步长不为1
...
}
```

#### 循环内成员变量访问优化

```cpp
//【反例】直接读取结构体或类对象的成员变量
__aicore__ inline void SoftMaxGenericNDImpVF(__ubuf__ float* dstAddr, __ubuf__ float* sumAddr, __ubuf__ float* maxAddr,
__ubuf__ float* srcAddr, __ubuf__ float* workAddr, const LastAxisShapeND originalSrcShape, const SoftMaxTiling tiling)
{
for (uint16_t i = 0; i < (uint16_t)tiling.srcM; i++) {
    AscendC::ReduceMax(maxAddr + i * FLOAT_NUM_PER_BLK, srcAddr + i * tiling.srcK, workAddr, (uint16_t)originalSrcShape.k);
}
}
```

```cpp
//【正例】通过局部变量传递VF内访问的结构体或类对象成员变量
__aicore__ inline void SoftMaxGenericNDImpVF(__ubuf__ float* dstAddr, __ubuf__ float* sumAddr, __ubuf__ float* maxAddr,
__ubuf__ float* srcAddr, __ubuf__ float* workAddr, const LastAxisShapeND originalSrcShape, const SoftMaxTiling tiling)
{
uint16_t srcK = tiling.srcK;
uint16_t srcM = tiling.srcM;
uint16_t reduceK = FLOAT_NUM_PER_BLK;
uint16_t originK = (uint16_t)originalSrcShape.k;
for (uint16_t i = 0; i < srcM; i++) {
    AscendC::ReduceMax(maxAddr + i * reduceK, srcAddr + i * srcK, workAddr, originK);
}
}
```

#### 循环内指令分布优化

```cpp
//【反例】Duplicate语句放在for循环中，每次循环都会执行一次
template<typename T>
__simd_vf__ inline void DuplicateVF(__ubuf__ T* dstAddr, T scalarValue, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
AscendC::Reg::RegTensor<T> dstReg;
AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<T>();
for (uint16_t i = 0; i < repeatTimes; i++) {
    AscendC::Reg::Duplicate(dstReg, scalarValue);
    AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
}
}
```

```cpp
//【正例】Duplicate放在for循环外，仅执行一次，有效减少指令数
template<typename T>
__simd_vf__ inline void DuplicateVF(__ubuf__ T* dstAddr, T scalarValue, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
AscendC::Reg::RegTensor<T> dstReg;
AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<T>();
AscendC::Reg::Duplicate(dstReg, scalarValue);
for (uint16_t i = 0; i < repeatTimes; i++) {
    AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
}
}
```

#### 循环内地址管理优化

```cpp
for(uint16_t i = 0;i < extent1; i++){
for(uint16_t j = 0;j < extent2; j++){
    for(uint16_t k = 0;k < extent3; k++){
        for(uint16_t m = 0;m < extent4; m++){
           AscendC::Reg::LoadAlign(srcReg, srcAddr + i * const1 + j * const2 + k * const3 + m * const4);
        }
    }
}
}
```

```cpp
AscendC::Reg::AddrReg aReg;
for(uint16_t i = 0;i < extent1; i++){
for(uint16_t j = 0;j < extent2; j++){
    for(uint16_t k = 0;k < extent3; k++){
        for(uint16_t m = 0;m < extent4; m++){
            aReg = AscendC::Reg::CreateAddrReg<T>(i, const1, j, const2, k, const3, m, const4);
            AscendC::Reg::LoadAlign(srcReg, srcAddr, aReg);
        }
    }
}
}
```

```cpp
// 开启向量地址生成指令优化
__simd_vf__ inline void ComputeModeVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
AscendC::Reg::RegTensor<T> dstReg;
AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<T>();
for (uint16_t i = 0; i < repeatTimes; ++i) {
    AscendC::Reg::LoadAlign(dstReg, srcAddr + i * oneRepeatSize);
    AscendC::Reg::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
}
}
```

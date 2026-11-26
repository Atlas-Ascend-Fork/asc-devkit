---
okftype: reference.api
description: "按 pat_mode 指定的元素位置模式生成 b8、b16 或 b32 位宽的掩码寄存器，用于筛选后续 Reg 矢量计算中参与计算的元素。"
tag:
  - roles: [configuration]
dependencies:
  - path: docs/zh/api/SIMD-API/c_api/reg_compute/overview.md
    kind: constrained_by
    condition: "Reg 矢量计算 C API 通用约束与掩码寄存器数量上限"
version: ">=9.0.0"
version_changes:
  - since: "9.2.0"
    kind: header_reorganized
    scope: "page"
    scope_basis: kind_default
    note: "9.2.0 起 include/c_api/reg_compute/compute/reg_mask.h 以对象式宏别名提供该接口（#define asc_create_mask_b8 pset_b8、asc_create_mask_b16 pset_b16、asc_create_mask_b32 pset_b32），9.0.0/9.1.0 的 include 中只有注释形式的调用示例、没有可注解的声明；因此本页三个标识符在 doc_overrides 中显示为未写入头，本页不使用 overload scope。"
---
# asc_create_mask

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- Atlas 200I/500 A2推理产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas推理系列产品AI Core：不支持
<!-- end id5 -->
<!-- npu="310p" id6 -->
- Atlas推理系列产品Vector Core：不支持
<!-- end id6 -->
<!-- npu="910" id7 -->
- Atlas训练系列产品：不支持
<!-- end id7 -->

## 功能说明

头文件路径为：`"c_api/reg_compute/compute/reg_mask.h"`。

根据传入的pat_mode生成对应的掩码寄存器，支持b8、b16、b32三种位宽模式。pat_mode参数指定mask的模式，即指定哪些位置的元素参与计算。

位宽模式说明：
- b8模式：每个bit对应一个8bit元素（共256元素），用于8bit数据类型的矢量计算。
- b16模式：每2个bit为一组对应一个16bit元素（共128元素），用于16bit数据类型的矢量计算。
- b32模式：每4个bit为一组对应一个32bit元素（共64元素），用于32bit数据类型的矢量计算。

**图1** asc_create_mask原理

![asc_create_mask原理](../../../../figures/capi_create_mask.png)

本接口为Reg数据类型定义接口，仅在AIV上生效。

本接口为Reg矢量计算接口，仅在AIV上生效。

## 函数原型

```c
asc_create_mask_b8(pat_mode)
asc_create_mask_b16(pat_mode)
asc_create_mask_b32(pat_mode)
```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| pat_mode | 输入 | mask模式，取值如下：<br>&bull; PAT_ALL：所有元素设置为有效数据，全部参与计算<br>&bull; PAT_VL1：最低1个元素设置为有效数据<br>&bull; PAT_VL2：最低2个元素设置为有效数据<br>&bull; PAT_VL3：最低3个元素设置为有效数据<br>&bull; PAT_VL4：最低4个元素设置为有效数据<br>&bull; PAT_VL8：最低8个元素设置为有效数据<br>&bull; PAT_VL16：最低16个元素设置为有效数据<br>&bull; PAT_VL32：最低32个元素设置为有效数据<br>&bull; PAT_VL64：最低64个元素设置为有效数据<br>&bull; PAT_VL128：最低128个元素设置为有效数据<br>&bull; PAT_M3：下标为3的倍数的元素设置为有效数据<br>&bull; PAT_M4：下标为4的倍数的元素设置为有效数据<br>&bull; PAT_H：低一半的元素设置为有效数据<br>&bull; PAT_Q：低四分之一的元素设置为有效数据<br>&bull; PAT_ALLF：所有元素设置为无效元素，均不参与计算 |

## 返回值说明

vector_bool，掩码寄存器。

## 约束说明

- Reg矢量计算C API通用约束请参见[通用约束](../overview.md#通用约束)。
- 掩码寄存器的数量上限为8，超过上限的掩码寄存器会写入预留的8K Unified Buffer（UB）内存中，可能引起性能劣化。编译器会自动复用生命周期结束的寄存器和预留内存，若两者均可用，优先复用寄存器。

## 调用示例

将代码保存为`example.asc`后，可通过`bisheng`命令编译运行，其中`--npu-arch`参数需根据实际产品型号指定对应的NPU架构，具体产品与NPU架构的映射关系请参考[\_\_NPU\_ARCH\_\_](../../../../../guide/programming_guide/language_extension/simd_builtin_keywords.md#npu-arch)。

<!-- npu="950" id10 -->
以Ascend 950PR&950DT系列产品（对应NPU架构为`dav-3510`）为例，编译运行命令如下：

```bash
bisheng example.asc -o main --npu-arch=dav-3510 && ./main
```
<!-- end id10 -->
```cpp
/*
 * 以下示例对259个half元素执行加法。
 * 每轮手动维护剩余元素数，前两轮使用`PAT_ALL`，最后一轮使用`PAT_VL3`处理3个尾部元素；
 * 尾块通过掩码限制有效处理范围。
 * pat_mode仅支持表1列出的模式，其他尾块长度需要选择适合的模式或采用其他处理方式。
*/
#include <cstdint>
#include <iostream>
#include <vector>
#include "c_api/asc_simd.h"
#include "utils/std/cmath.h"
#include "acl/acl.h"

namespace {
constexpr uint8_t MUTEX_ID = 0;
constexpr uint32_t ELEMENTS_PER_REGISTER = 128;
constexpr uint32_t TAIL_ELEMENTS = 3;
constexpr uint32_t TOTAL_LENGTH = 2 * ELEMENTS_PER_REGISTER + TAIL_ELEMENTS;
constexpr uint32_t REPEAT_COUNT = (TOTAL_LENGTH + ELEMENTS_PER_REGISTER - 1) / ELEMENTS_PER_REGISTER;
constexpr uint32_t BUFFER_ELEMENTS = REPEAT_COUNT * ELEMENTS_PER_REGISTER;
constexpr uint32_t BUFFER_BYTES = BUFFER_ELEMENTS * sizeof(half);

__simd_vf__ inline void add_vf(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1,
                                uint32_t one_repeat_size, uint16_t repeat_time)
{
    vector_half src0_reg, src1_reg, dst_reg;
    uint32_t remaining = TOTAL_LENGTH;
    for (uint16_t i = 0; i < repeat_time; ++i) {
        const uint32_t offset = i * one_repeat_size;
        // half为16位数据，使用b16；其他数据类型应根据元素位宽选择b8或b32。
        // 本例尾块固定为3个元素，因此选择PAT_VL3；接口不会自动更新remaining。
        vector_bool mask = remaining >= one_repeat_size ? asc_create_mask_b16(PAT_ALL)
                                                        : asc_create_mask_b16(PAT_VL3);
        asc_loadalign(src0_reg, src0 + offset);
        asc_loadalign(src1_reg, src1 + offset);
        asc_add(dst_reg, src0_reg, src1_reg, mask);
        asc_storealign(dst + offset, dst_reg, mask);
        remaining = remaining > one_repeat_size ? remaining - one_repeat_size : 0;
    }
}

__global__ __vector__ void asc_create_mask_kernel(__gm__ half* dst, __gm__ half* src0, __gm__ half* src1)
{
    asc_init();
    __ubuf__ half dst_local[BUFFER_ELEMENTS];
    __ubuf__ half src0_local[BUFFER_ELEMENTS];
    __ubuf__ half src1_local[BUFFER_ELEMENTS];
    asc_lock(PIPE_MTE2, MUTEX_ID);
    asc_copy_gm2ub_align(src0_local, src0, BUFFER_BYTES);
    asc_copy_gm2ub_align(src1_local, src1, BUFFER_BYTES);
    asc_unlock(PIPE_MTE2, MUTEX_ID);

    uint32_t one_repeat_size = static_cast<uint32_t>(asc_get_vf_len() / sizeof(half));
    uint16_t repeat_time = static_cast<uint16_t>(AscendC::Std::ceil_div(TOTAL_LENGTH, one_repeat_size));
    asc_lock(PIPE_V, MUTEX_ID);
    asc_vf_call<add_vf>(dst_local, src0_local, src1_local, one_repeat_size, repeat_time);
    asc_unlock(PIPE_V, MUTEX_ID);

    asc_lock(PIPE_MTE3, MUTEX_ID);
    asc_copy_ub2gm_align(dst, dst_local, BUFFER_BYTES);
    asc_unlock(PIPE_MTE3, MUTEX_ID);
    asc_sync();
}
} // namespace

int main()
{
    std::vector<half> src0(BUFFER_ELEMENTS, half(0));
    std::vector<half> src1(BUFFER_ELEMENTS, half(0));
    std::vector<half> output(BUFFER_ELEMENTS, half(0));
    std::vector<half> expected(BUFFER_ELEMENTS, half(0));
    for (uint32_t i = 0; i < TOTAL_LENGTH; ++i) {
        const int32_t lhs = static_cast<int32_t>(i % 31) - 15;
        const int32_t rhs = static_cast<int32_t>(i % 13) - 6;
        src0[i] = half(lhs);
        src1[i] = half(rhs);
        expected[i] = half(lhs + rhs);
    }

    if (aclInit(nullptr) != ACL_SUCCESS) return 1;
    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        aclFinalize();
        return 1;
    }
    half* src0_device = nullptr;
    half* src1_device = nullptr;
    half* dst_device = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&src0_device), BUFFER_BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&src1_device), BUFFER_BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&dst_device), BUFFER_BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(src0_device, BUFFER_BYTES, src0.data(), BUFFER_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1_device, BUFFER_BYTES, src1.data(), BUFFER_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dst_device, BUFFER_BYTES, output.data(), BUFFER_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);
    asc_create_mask_kernel<<<1, 0>>>(dst_device, src0_device, src1_device);
    aclrtSynchronizeDevice();
    aclrtMemcpy(output.data(), BUFFER_BYTES, dst_device, BUFFER_BYTES, ACL_MEMCPY_DEVICE_TO_HOST);

    bool passed = true;
    for (uint32_t i = 0; i < TOTAL_LENGTH && passed; ++i) {
        if (static_cast<float>(output[i]) != static_cast<float>(expected[i])) {
            std::cout << "Mismatch at " << i << ": actual=" << static_cast<float>(output[i])
                      << ", expected=" << static_cast<float>(expected[i]) << std::endl;
            passed = false;
        }
    }
    std::cout << (passed ? "[Success] asc_create_mask_b16 passed." : "[Failed] asc_create_mask_b16 failed.") << std::endl;
    if (src0_device != nullptr) aclrtFree(src0_device);
    if (src1_device != nullptr) aclrtFree(src1_device);
    if (dst_device != nullptr) aclrtFree(dst_device);
    aclrtResetDevice(0);
    aclFinalize();
    return passed ? 0 : 1;
}
```

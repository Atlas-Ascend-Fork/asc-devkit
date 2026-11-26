---
okftype: reference.api
description: "从特殊用途寄存器读取掩码值并按数据类型格式化，b16 模式把每个 bit 扩展为 2bit、b32 模式扩展为 4bit，拼成 256bit 掩码寄存器返回。"
tag:
  - roles: [query]
dependencies:
  - path: docs/zh/api/SIMD-API/c_api/reg_compute/overview.md
    kind: constrained_by
    condition: "Reg 矢量计算 C API 通用约束"
  - path: docs/zh/api/SIMD-API/c_api/vector_compute/vector_mask_config/asc_set_vector_mask.md
    kind: related
    condition: "在 VF 场景外由该接口设置掩码寄存器，再配合本接口在 Reg 场景获取"
version: ">=9.0.0"
version_changes:
  - since: "9.2.0"
    kind: header_reorganized
    scope: "page"
    scope_basis: kind_default
    note: "声明位置由 include/c_api/reg_compute/reg_load.h（9.1.0 及更早）迁移到 include/c_api/reg_compute/compute/reg_mask.h（9.2.0 起），与正文头文件路径一致；签名始终保持无参返回 vector_bool。"
---
# asc_get_mask_spr

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

从SPR（Special Purpose Register）中读取mask值，并按数据类型格式化后返回。

- 对于b16数据类型：从SPR读取mask共128bit数据，将每个bit扩展为2bit，拼成256bit掩码寄存器返回。
- 对于b32数据类型：从SPR读取mask共64bit数据，将每个bit扩展为4bit，拼成256bit掩码寄存器返回。

本接口为Reg矢量计算接口，仅在AIV上生效。

## 函数原型

```c
__simd_callee__ inline vector_bool asc_get_mask_spr_b16()
__simd_callee__ inline vector_bool asc_get_mask_spr_b32()
```

## 参数说明

无

## 返回值说明

vector_bool，掩码寄存器。

## 约束说明

- Reg矢量计算C API通用约束请参见[通用约束](../overview.md#通用约束)。
- 本接口需在Vector Function（`__simd_vf__`标记的函数）内调用。
- [asc_set_vector_mask](../../vector_compute/vector_mask_config/asc_set_vector_mask.md)用于在VF场景外设置掩码寄存器，在Reg矢量计算配置时，通过`asc_get_mask_spr_b16`或`asc_get_mask_spr_b32`配合获取掩码寄存器的值。更推荐用户使用VF场景内的接口（[asc_create_mask](./asc_create_mask.md)和[asc_update_mask.md](./asc_update_mask.md)）完成Reg矢量计算掩码配置。

## 调用示例

以下示例对259个`half`元素执行加法。在VF外通过`asc_set_vector_mask`设置SPR掩码，VF内使用`asc_get_mask_spr_b16`读取掩码。前两个完整块全部有效，尾块仅低3个元素有效；尾块通过掩码限制有效处理范围。

将代码保存为`example.asc`后，可通过`bisheng`命令编译运行，其中`--npu-arch`参数需根据实际产品型号指定对应的NPU架构，具体产品与NPU架构的映射关系请参考[\_\_NPU\_ARCH\_\_](../../../../../guide/programming_guide/language_extension/simd_builtin_keywords.md#npu-arch)。

<!-- npu="950" id8 -->
以Ascend 950PR&950DT系列产品（对应NPU架构为`dav-3510`）为例，编译运行命令如下：

```bash
bisheng example.asc -o main --npu-arch=dav-3510 && ./main
```
<!-- end id8 -->

```cpp
/*
 * 对259个half元素执行加法。在VF外设置SPR掩码，在VF内读取并使用该掩码。
 * 两个完整寄存器块使用全有效掩码，尾块使用仅低3位有效的掩码。
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
    // 读取VF外配置的SPR掩码；half数据使用b16模式。
    vector_bool mask = asc_get_mask_spr_b16();
    for (uint32_t i = 0; i < repeat_time; ++i) {
        const uint32_t offset = i * one_repeat_size;
        asc_loadalign(src0_reg, src0 + offset);
        asc_loadalign(src1_reg, src1 + offset);
        asc_add(dst_reg, src0_reg, src1_reg, mask);
        asc_storealign(dst + offset, dst_reg, mask);
    }
}

__global__ __vector__ void asc_get_mask_spr_kernel(__gm__ half* dst, __gm__ half* src0, __gm__ half* src1)
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
    uint16_t full_repeat_time = repeat_time - 1;
    // 两个完整块全部有效。每个SPR位对应一个half元素。
    asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
    asc_lock(PIPE_V, MUTEX_ID);
    asc_vf_call<add_vf>(dst_local, src0_local, src1_local, one_repeat_size, full_repeat_time);
    asc_unlock(PIPE_V, MUTEX_ID);
    // 等待前一个VF使用完SPR掩码，再设置尾块的低3位。
    asc_sync();
    asc_set_vector_mask(0, (1ULL << TAIL_ELEMENTS) - 1);
    asc_lock(PIPE_V, MUTEX_ID);
    asc_vf_call<add_vf>(dst_local + full_repeat_time * one_repeat_size,
                        src0_local + full_repeat_time * one_repeat_size,
                        src1_local + full_repeat_time * one_repeat_size, one_repeat_size, 1);
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
    asc_get_mask_spr_kernel<<<1, 0>>>(dst_device, src0_device, src1_device);
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
    std::cout << (passed ? "[Success] asc_get_mask_spr_b16 passed." : "[Failed] asc_get_mask_spr_b16 failed.") << std::endl;
    if (src0_device != nullptr) aclrtFree(src0_device);
    if (src1_device != nullptr) aclrtFree(src1_device);
    if (dst_device != nullptr) aclrtFree(dst_device);
    aclrtResetDevice(0);
    aclFinalize();
    return passed ? 0 : 1;
}
```

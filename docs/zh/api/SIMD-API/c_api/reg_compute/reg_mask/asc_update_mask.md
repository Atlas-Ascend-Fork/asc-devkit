---
okftype: reference.api
description: "按剩余元素个数生成掩码并通过引用参数更新剩余个数。支持b8、b16、b32模式，单次分别对应256、128、64个元素。"
tag:
  - roles: [configuration]
dependencies:
  - path: docs/zh/api/SIMD-API/c_api/reg_compute/overview.md
    kind: constrained_by
    condition: "遵循Reg矢量计算C API的通用约束"
version: ">=9.0.0"
version_changes:
  - since: "9.1.0"
    kind: parameter_changed
    scope: "param:value"
    scope_basis: resolved
    note: "三个掩码更新接口统一将计数参数由scalar更名为value。"
---
# asc_update_mask

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

根据元素个数value生成mask，并自动将value减去当前向量处理单元的元素个数。支持b8、b16、b32三种位宽模式，由于VL=256B，各模式的向量处理单元元素个数为：

- b8模式：每次处理256个元素，用于8 bit数据类型的矢量计算。
- b16模式：每次处理128个元素，用于16 bit数据类型的矢量计算。
- b32模式：每次处理64个元素，用于32 bit数据类型的矢量计算。

**图1** asc_update_mask更新流程

![asc_update_mask更新流程](../../../../figures/capi_update_mask.png)

本接口为Reg矢量计算接口，仅在AIV上生效。

## 函数原型

```c
__simd_callee__ inline vector_bool asc_update_mask_b8(uint32_t& value)
__simd_callee__ inline vector_bool asc_update_mask_b16(uint32_t& value)
__simd_callee__ inline vector_bool asc_update_mask_b32(uint32_t& value)
```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| value | 输入/输出 | 元素个数。调用后自动减去当前向量处理单元的元素个数（b8模式、b16模式、b32模式分别为256、128、64）。<br>执行完一次该接口后，value = (value < VL_T) ? 0 : (value - VL_T)，VL_T表示位宽为VL的矢量数据寄存器中，可存放数据类型T的元素个数。<br>例如，有320个b16数据类型的元素，每次处理128个元素：<br>第一次调用接口，生成的mask对应的元素全为有效数据，value = 320-128 = 192。<br>第二次调用接口，生成的mask对应的元素全为有效数据，value = 192-128 = 64。<br>第三次调用接口，生成的mask对应的元素只有低半部分为有效数据，value = 0。|

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义](../../defs/type/data_type_definition.md)。

## 返回值说明

vector_bool，掩码寄存器。

## 约束说明

- Reg矢量计算C API通用约束请参见[通用约束](../overview.md#通用约束)。
- 掩码寄存器的数量上限为8，超过上限的掩码寄存器会写入预留的8K Unified Buffer（UB）内存中，可能引起性能劣化。编译器会自动复用生命周期结束的寄存器和预留内存，若两者均可用，优先复用寄存器。
- 本接口需在Vector Function（`__simd_vf__`标记的函数）内调用。

## 调用示例

以下示例对385个`float`元素求绝对值。`asc_update_mask_b32`在每轮生成掩码并自动更新剩余元素数，最后一轮仅处理1个有效元素；结果校验仅覆盖有效元素范围。

将代码保存为`example.asc`后，可通过`bisheng`命令编译运行，其中`--npu-arch`参数需根据实际产品型号指定对应的NPU架构，具体产品与NPU架构的映射关系请参考[\_\_NPU\_ARCH\_\_](../../../../../guide/programming_guide/language_extension/simd_builtin_keywords.md#npu-arch)。

<!-- npu="950" id8 -->
以Ascend 950PR&950DT系列产品（对应NPU架构为`dav-3510`）为例，编译运行命令如下：

```bash
bisheng example.asc -o main --npu-arch=dav-3510 && ./main
```
<!-- end id8 -->

```cpp
#include <cstdint>
#include <cmath>
#include <iostream>
#include <vector>
#include "c_api/asc_simd.h"
#include "utils/std/cmath.h"
#include "acl/acl.h"

namespace {
constexpr uint8_t MUTEX_ID = 0;


template <typename T>
void print_data(const char* label, const std::vector<T>& values)
{
    std::cout << label << ":";
    const size_t count = values.size() < 8 ? values.size() : 8;
    for (size_t i = 0; i < count; ++i) std::cout << ' ' << +values[i];
    if (values.size() > count) std::cout << " ...";
    std::cout << std::endl;
}

template <typename T>
bool compare_data(const std::vector<T>& actual, const std::vector<T>& expected, size_t count, double tolerance = 0.0)
{
    if (count > actual.size() || count > expected.size()) return false;
    for (size_t i = 0; i < count; ++i) {
        if (actual[i] == expected[i]) continue;
        const double diff = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
        if (!(diff <= tolerance && diff >= -tolerance)) {
            std::cout << "Mismatch at " << i << ": actual=" << actual[i]
                      << ", expected=" << expected[i] << std::endl;
            return false;
        }
    }
    return true;
}

constexpr uint32_t ELEMENTS_PER_REGISTER = 64;
constexpr uint32_t TOTAL_LENGTH = 385;
constexpr uint32_t REPEAT_COUNT = (TOTAL_LENGTH + ELEMENTS_PER_REGISTER - 1) / ELEMENTS_PER_REGISTER;
constexpr uint32_t BUFFER_ELEMENTS = REPEAT_COUNT * ELEMENTS_PER_REGISTER;
constexpr uint32_t BUFFER_BYTES = BUFFER_ELEMENTS * sizeof(float);

__simd_vf__ inline void exercise(__ubuf__ float* output, __ubuf__ float* input, uint32_t one_repeat_size, uint16_t repeat_time)
{
    vector_float src_reg;
    vector_float dst_reg;

    uint32_t remaining_length = TOTAL_LENGTH;
    for (uint16_t i = 0; i < repeat_time; ++i) {
        const uint32_t offset = i * one_repeat_size;
        // 接口自动更新剩余元素数，并为尾块生成掩码。
        vector_bool mask = asc_update_mask_b32(remaining_length);
        asc_loadalign(src_reg, input + offset);
        asc_abs(dst_reg, src_reg, mask);
        asc_storealign(output + offset, dst_reg, mask);
    }
}

__global__ __vector__ void asc_update_mask_kernel(__gm__ float* output, __gm__ float* input)
{
    asc_init();
    __ubuf__ float output_local[BUFFER_ELEMENTS];
    __ubuf__ float input_local[BUFFER_ELEMENTS];
    asc_lock(PIPE_MTE2, MUTEX_ID);
    asc_copy_gm2ub_align(input_local, input, BUFFER_BYTES);
    asc_unlock(PIPE_MTE2, MUTEX_ID);

    uint16_t one_repeat_size = asc_get_vf_len() / sizeof(float);
    uint16_t repeat_time = static_cast<uint16_t>(AscendC::Std::ceil_div(TOTAL_LENGTH, one_repeat_size));
    asc_lock(PIPE_V, MUTEX_ID);
    asc_vf_call<exercise>(output_local, input_local, one_repeat_size, repeat_time);
    asc_unlock(PIPE_V, MUTEX_ID);

    asc_lock(PIPE_MTE3, MUTEX_ID);
    asc_copy_ub2gm_align(output, output_local, BUFFER_BYTES);
    asc_unlock(PIPE_MTE3, MUTEX_ID);
    asc_sync();
}
} // namespace

int main()
{
    std::vector<float> input(BUFFER_ELEMENTS, 0.0f);
    std::vector<float> output(BUFFER_ELEMENTS, 0.0f);
    std::vector<float> golden(BUFFER_ELEMENTS, 0.0f);
    for (uint32_t i = 0; i < TOTAL_LENGTH; ++i) {
        input[i] = static_cast<float>(static_cast<int32_t>(i % 31) - 15);
        golden[i] = std::fabs(input[i]);
    }
    aclInit(nullptr);
    aclrtSetDevice(0);
    float* input_device = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&input_device), BUFFER_BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
    float* output_device = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&output_device), BUFFER_BYTES, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(input_device, BUFFER_BYTES, input.data(), BUFFER_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(output_device, BUFFER_BYTES, output.data(), BUFFER_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);
    asc_update_mask_kernel<<<1, 0>>>(output_device, input_device);
    aclrtSynchronizeDevice();
    aclrtMemcpy(output.data(), BUFFER_BYTES, output_device, BUFFER_BYTES, ACL_MEMCPY_DEVICE_TO_HOST);
    print_data("Input", input);
    print_data("Output", output);
    print_data("Expected", golden);
    const bool passed = compare_data(output, golden, TOTAL_LENGTH);
    std::cout << (passed ? "[Success] asc_update_mask_b32 passed." : "[Failed] asc_update_mask_b32 failed.") << std::endl;
    aclrtFree(input_device);
    aclrtFree(output_device);
    aclrtResetDevice(0);
    aclFinalize();
    return passed ? 0 : 1;
}
```

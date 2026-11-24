---
okftype: reference.api
description: "将float矢量元素与标量相乘后转换为half并按dst_pos写入偶数或奇数索引位置的Reg融合接口，未写入位置置0，支持返回值与dst引用两种形式，仅在AIV上生效。"
tag:
  - roles: [compute]
  - operations: [conversion, multiply]
  - traits: [rounding_mode]
dependencies:
  - path: docs/zh/api/SIMD-API/c_api/reg_compute/reg_fused/asc_mul_scalar_float2half_rna.md
    kind: replaced_by
  - path: docs/zh/api/SIMD-API/c_api/defs/type/data_type_definition.md
    kind: related
    condition: "矢量数据寄存器与掩码寄存器的类型定义"
  - path: docs/zh/api/SIMD-API/c_api/reg_compute/overview.md
    kind: constrained_by
    condition: "Reg矢量计算C API通用约束见该页通用约束节"
  - path: docs/zh/guide/programming_guide/language_extension/simd_builtin_keywords.md
    kind: related
    condition: "产品型号与NPU架构映射说明"
version: ">=9.2.0"
version_changes:
  - since: "9.3.0"
    kind: deprecated
    scope: "page"
    scope_basis: kind_default
    note: "全部四个重载自9.3.0起废弃，计划在2027/11/02后移除，请迁移到asc_mul_scalar_float2half_rna；旧接口保留兼容，计算行为不变。"
---
# asc_mul_scalar_float2half_rn（废弃）

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

头文件路径为：`"c_api/reg_compute/compute/reg_fused.h"`。

**该接口的全部四个重载已废弃，请使用[asc_mul_scalar_float2half_rna](../reg_compute/reg_fused/asc_mul_scalar_float2half_rna.md)替代。自CANN 9.3.0起产生废弃告警，计划在2027/11/02后移除。**

旧接口仍可编译使用，参数、返回值、掩码和奇偶位置语义保持兼容。旧名称中的`_rn`不代表RINT模式；本接口实际使用ROUND模式，替代接口以`_rna`明确这一行为。

根据`mask`将源操作数`src`（float类型）按元素与标量`value`相乘后，按照ROUND舍入模式转换为half类型，得到计算结果。计算公式如下：

$$
dst_i = round\_to\_half(src_i \times value)
$$

由于源操作数（float，32位）与目的操作数（half，16位）的位宽比为2:1，一个`vector_float`的数据转换后会填满一个`vector_half`。通过`dst_pos`参数指定结果写入`dst`中索引为偶数或奇数的位置，未写入的位置置0。

本接口为Reg矢量计算接口，仅在AIV上生效。

## 函数原型

```cpp
// 通过函数返回值返回结果，数据写入索引为偶数的位置
__simd_callee__ inline vector_half asc_mul_scalar_float2half_rn(vector_float src,
                                                                float value,
                                                                vector_bool mask,
                                                                std::integral_constant<asc_position_mode, asc_position_mode::EVEN> dst_pos)

// 通过函数返回值返回结果，数据写入索引为奇数的位置
__simd_callee__ inline vector_half asc_mul_scalar_float2half_rn(vector_float src,
                                                                float value,
                                                                vector_bool mask,
                                                                std::integral_constant<asc_position_mode, asc_position_mode::ODD> dst_pos)

// 通过引用参数输出结果，数据写入索引为偶数的位置
__simd_callee__ inline void asc_mul_scalar_float2half_rn(vector_half& dst,
                                                         vector_float src,
                                                         float value,
                                                         vector_bool mask,
                                                         std::integral_constant<asc_position_mode, asc_position_mode::EVEN> dst_pos)

// 通过引用参数输出结果，数据写入索引为奇数的位置
__simd_callee__ inline void asc_mul_scalar_float2half_rn(vector_half& dst,
                                                         vector_float src,
                                                         float value,
                                                         vector_bool mask,
                                                         std::integral_constant<asc_position_mode, asc_position_mode::ODD> dst_pos)
```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| -------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| dst | 输出 | 目的操作数（矢量数据寄存器，half类型）。 |
| src | 输入 | 源操作数（矢量数据寄存器，float类型）。 |
| value | 输入 | 源操作数（标量，float类型）。 |
| mask | 输入 | 掩码寄存器，用于控制各元素是否参与计算。`mask`中与元素对应的比特位为1时，该元素参与计算；为0时，该元素不参与计算。 |
| dst_pos | 输入 | 数据写入位置模式，取值为`ASC_POSITION_EVEN`（偶数位置）或`ASC_POSITION_ODD`（奇数位置）。选择偶数位置时，奇数位置置0；选择奇数位置时，偶数位置置0。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义](../defs/type/data_type_definition.md)。

## 返回值说明

- 通过引用参数输出结果：计算结果写入`dst`。
- 通过函数返回值返回结果：返回计算结果，类型为`vector_half`。

## 约束说明

- Reg矢量计算C API通用约束请参见[通用约束](../reg_compute/overview.md#通用约束)。
- `mask`掩码位为0时，结果对应元素置0。
- 当`dst_pos`为`ASC_POSITION_EVEN`时，结果写入`dst`的偶数索引位置，奇数索引位置置0；当`dst_pos`为`ASC_POSITION_ODD`时，结果写入`dst`的奇数索引位置，偶数索引位置置0。

## 调用示例

迁移后的完整示例请参见[asc_mul_scalar_float2half_rna](../reg_compute/reg_fused/asc_mul_scalar_float2half_rna.md)。将函数名替换为`asc_mul_scalar_float2half_rna`即可，参数及调用顺序不变。

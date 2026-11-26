---
okftype: reference.api
description: "不经过DCache向Global Memory写入整型标量数据，避免多核按Cache Line读写导致的随机覆盖，写入后需按缓存一致性要求处理。"
tag:
  - roles: [data_movement]
  - operations: [store]
  - traits: [aligned_transfer, scalar_operand]
dependencies:
  - path: docs/zh/api/SIMD-API/c_api/cache_ctrl/asc_dcci.md
    kind: requires
    condition: "其他通路经缓存访问同一GM地址时需清理/失效Cache Line"
  - path: docs/zh/api/SIMD-API/c_api/sync/intra_core_sync/asc_sync_data_barrier.md
    kind: requires
    condition: "保证相关访存操作的执行顺序与数据可见性"
version: ">=9.0.0"
version_changes: []
---
# asc_store_dev

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

头文件路径为：`"c_api/scalar_compute/scalar_store.h"`。

不经过DCache向GM地址上写数据。

当多核操作GM地址时，如果数据无法对齐到Cache Line，经过DCache的方式下，由于按照Cache Line大小进行读写，会导致多核数据随机覆盖的问题。此时，可以采用不经过DCache直接读写GM地址的方式，从而避免上述随机覆盖的问题。

![多核数据随机覆盖问题示意图](../../../../figures/write_gm_bypass_dcache.png)

## 函数原型

```c
__aicore__ inline void asc_store_dev(__gm__ <dtype>* addr,
                                     <dtype> value)
```

### dtype支持数据类型

`dtype`取值为：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`int64_t`、`uint64_t`。

### 函数原型典型示例

```c
// 示例：写入32bit无符号整数
__aicore__ inline void asc_store_dev(__gm__ uint32_t* addr,
                                     uint32_t value)
```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| ------------ | ------------ | ----------- |
| addr | 输出 | 目标GM地址。 |
| value | 输入 | 待写入目标的数据。 |

## 返回值说明

无

## 约束说明

- `addr`起始地址须按写入`dtype`字节数对齐。
- `addr`须落在GM可访问地址空间内。
- 本接口运行在标量流水上，与后续依赖该写入结果的指令之间存在标量数据依赖；如后续有读取同一GM地址的指令，须通过同步指令建立依赖顺序，标量流水本身的顺序执行不保证跨指令访存可见性。
- 本接口访问GM时绕过DCache，不维护缓存一致性。若其他核或其他通路通过缓存访问同一GM地址，调用方需使用[asc_dcci](../../cache_ctrl/asc_dcci.md)清理或失效对应Cache Line，并使用[asc_sync_data_barrier](../../sync/intra_core_sync/asc_sync_data_barrier.md)保证相关访存操作的执行顺序和数据可见性。详情可参考[Scalar原子操作与DCache一致性](../../../../../guide/programming_guide/advanced_programming/memory_model/cache_coherence.md#scalar原子操作与dcache一致性)。

## 调用示例

具体调用示例请参考[asc_sync_data_barrier调用示例](../../sync/intra_core_sync/asc_sync_data_barrier.md)。

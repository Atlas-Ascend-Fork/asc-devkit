# SetReduceType

## 功能说明

设置Reduce操作类型，仅对有归约操作的通信任务生效。

## 函数原型

```
uint32_t SetReduceType(uint32_t reduceType, uint8_t dstDataType = 0, uint8_t srcDataType = 0)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| reduceType | 输入 | 归约操作类型，仅对有归约操作的通信任务生效。uint32_t类型，取值详见[表2](../HCCL_Kernel/HCCL_usage.md#table2469980529)参数说明。支持的操作类型及约束请参见[AllReduce的op说明](../HCCL_Kernel/AllReduce.md#op说明)和[ReduceScatter的op说明](../HCCL_Kernel/ReduceScatter.md#op说明)。 |
| dstDataType | 输入 | 通信任务中输出数据的数据类型。uint8_t类型，该参数的取值范围请参考[表1](../HCCL_Kernel/HCCL_usage.md#table116710585514)。<!-- npu="950" id1 --><br><br>Ascend 950PR&950DT系列产品，不同通信任务支持的输出数据类型不同，具体为：<br>对于AllReduce、AllGather、AllToAll、AllToAllV、AllToAllVWrite通信任务：输出的数据类型必须与输入的数据类型一致。各通信任务支持的输入数据类型请参考srcDataType。<br>对于ReduceScatter通信任务，AI CPU作为服务端时，输出的数据类型须与输入一致；CCU作为服务端时，输入的数据类型为int16_t、half、bfloat16_t、int32_t、float时，输出的数据类型须与其一致。int8_t的输入输出类型组合及算法约束请参见[ReduceScatter的op说明](../HCCL_Kernel/ReduceScatter.md#op说明)。<!-- end id1 --> |
| srcDataType | 输入 | 通信任务中输入数据的数据类型。uint8_t类型，该参数的取值范围请参考[表1](../HCCL_Kernel/HCCL_usage.md#table116710585514)。<!-- npu="950" id2 --><br><br>Ascend 950PR&950DT系列产品，CCU作为服务端时，不同通信任务支持的输入数据类型如下：<br>AllReduce通信任务：支持的数据类型及约束请参见[AllReduce数据类型](../HCCL_Kernel/AllReduce.md#数据类型)。<br>AllGather、AllToAll、AllToAllV、AllToAllVWrite通信任务：支持的输入类型为int8_t、uint8_t、hifloat8_t、fp8_e5m2_t、fp8_e4m3fn_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、double。<br>ReduceScatter通信任务：支持的数据类型及约束请参见[ReduceScatter数据类型](../HCCL_Kernel/ReduceScatter.md#数据类型)。<!-- end id2 --> |

## 返回值说明

-   0表示设置成功。
-   非0表示设置失败。

## 约束说明

<!-- npu="950" id3 -->
对于Ascend 950PR&950DT系列产品，该接口仅设置Tiling配置，不校验通信服务端及通信算法是否支持对应的数据类型与归约操作组合。配置时须满足[AllReduce](../HCCL_Kernel/AllReduce.md#op说明)或[ReduceScatter](../HCCL_Kernel/ReduceScatter.md#op说明)的约束，并与Hccl对象选择的服务端类型保持一致。
<!-- end id3 -->
<!-- npu="A3" id4 -->
对于Atlas A3系列产品，该接口暂不支持使用。
<!-- end id4 -->
<!-- npu="910b" id5 -->
对于Atlas A2系列产品，该接口暂不支持使用。
<!-- end id5 -->

## 调用示例

本接口的调用示例请见[调用示例](SetOpType.md#调用示例)。

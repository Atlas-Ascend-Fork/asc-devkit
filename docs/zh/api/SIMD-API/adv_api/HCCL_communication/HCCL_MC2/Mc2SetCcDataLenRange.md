# Mc2SetCcDataLenRange

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
- Atlas推理系列产品Vector Core：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- Atlas训练系列产品：不支持
<!-- end id6 -->
<!-- @ref: asc-devkit/res/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_MC2/Mc2SetCcDataLenRange_res.md#id1 -->

## 功能说明

设置MC2通信参数对象的数据长度范围，供[Mc2GetSupportAlgList](Mc2GetSupportAlgList.md)查询可用算法名单时过滤算法：只列出长度条件与该范围有重叠的算法。minLen与maxLen均为单卡数据长度（单卡数据个数×单个数据占用的字节数），不是所有卡的总数据量。名单表示该范围内存在可用的数据长度，不表示范围内所有长度都可用。

查询范围为闭区间[minLen, maxLen]，含端点。不调用本接口或设置为(0, 0)时，查询不按长度过滤。本接口只影响查询结果，不影响通信资源申请和通信服务端启动流程。

## 函数原型

```cpp
extern Mc2Result __attribute__((visibility("default"))) Mc2SetCcDataLenRange(void* ccArgs, uint64_t minLen, uint64_t maxLen)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ccArgs | 输入 | [Mc2GetCcArgs](Mc2GetCcArgs.md)返回的MC2通信参数对象指针。 |
| minLen | 输入 | 数据长度下界，单卡数据个数×单个数据占用的字节数。 |
| maxLen | 输入 | 数据长度上界，须大于等于minLen。 |

## 返回值说明

返回HcclResult类型的值，具体取值参考下表。

**表2**  返回值说明

| 返回值 | 描述 |
| --- | --- |
| HCCL_SUCCESS | 设置成功。 |
| HCCL_E_PTR | ccArgs为空指针。 |
| HCCL_E_PARA | minLen大于maxLen。 |

## 约束说明

无

## 调用示例

```cpp
// 每卡发送sendCount个FP32数据，实际数据量最多向上浮动16个数据：
Mc2Result ret = Mc2SetCcDataLenRange(ccArgs, sendCount * sizeof(float), (sendCount + 16) * sizeof(float));
```

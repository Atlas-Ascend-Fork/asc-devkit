# Mc2GetSupportAlgList

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
<!-- @ref: asc-devkit/res/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_MC2/Mc2GetSupportAlgList_res.md#id1 -->

## 功能说明

查询当前通信域上，指定通信任务类型和通信引擎下可用的算法名单。查询结果以JSON字符串写入调用方提供的缓冲区。

使用本接口前，必须先调用[Mc2GetCcArgs](Mc2GetCcArgs.md)创建通信参数对象，并调用[Mc2SetCcCommEngine](Mc2SetCcCommEngine.md)设置通信引擎；如需按数据长度过滤名单，可先调用[Mc2SetCcDataLenRange](Mc2SetCcDataLenRange.md)设置数据长度范围。

查询结果中的名字可直接作为[Mc2SetCcAlgConfig](Mc2SetCcAlgConfig.md)的算法名参数，设置后按正常流程申请通信资源并执行通信。若已通过[Mc2SetCcDataLenRange](Mc2SetCcDataLenRange.md)设置数据长度范围，名单按该范围过滤：算法的长度条件与该范围有重叠即列入。

本接口只做查询，不申请通信资源，也不启动通信服务端。

查询结果格式固定为三字段JSON，成功与失败通过error_msg字段是否为空区分：

-   查询成功：`{"count":N,"alg_list":["算法名",...],"error_msg":[]}`，N为算法个数。示例：

    ```text
    {"count":2,"alg_list":["sole[mesh]","sole[nhr]"],"error_msg":[]}
    ```

-   查询失败：`{"count":0,"alg_list":[],"error_msg":["失败原因"]}`。示例：

    ```text
    {"count":0,"alg_list":[],"error_msg":["alg list buffer insufficient: need[2110] len[2048], recommended 16K"]}
    ```

参数合法但当前拓扑下没有可用算法时，属于查询成功，alg_list为空数组：`{"count":0,"alg_list":[],"error_msg":[]}`。

## 函数原型

```cpp
extern Mc2Result __attribute__((visibility("default"))) Mc2GetSupportAlgList(
    HcclComm comm, uint8_t ccType, void* ccArgs, char* algs, uint32_t len)
```

本接口为C链接接口（声明于`extern "C"`块内，符号无C++名修饰），可在C与C++宿主程序中直接调用。

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| comm | 输入 | 已初始化的通信域句柄。 |
| ccType | 输入 | 通信任务类型，取值为HcclCMDType枚举值，当前支持AllGather、AllReduce、ReduceScatter、AlltoAll和AlltoAllV。 |
| ccArgs | 输入 | [Mc2GetCcArgs](Mc2GetCcArgs.md)创建并通过Mc2SetCc*接口填充的MC2通信参数对象。 |
| algs | 输出 | 查询结果缓冲区，由调用方分配，接口写入完整JSON并以'\\0'结尾。建议不小于16KB。 |
| len | 输入 | algs缓冲区大小。 |

## 返回值说明

返回HcclResult类型的值，具体取值参考下表。

**表2**  返回值说明

| 返回值 | 描述 |
| --- | --- |
| HCCL_SUCCESS | 查询成功。 |
| HCCL_E_PTR | comm或ccArgs为空指针，或algs为空指针且len大于0。 |
| HCCL_E_NOT_SUPPORT | 通信任务类型不支持、通信引擎不支持，或结果JSON超出缓冲区大小。 |

## 约束说明

-   查询成功时写入完整的结果JSON；结果超出缓冲区容量时写入完整的失败JSON（含所需的缓冲区大小），不写入截断的JSON。
-   缓冲区不足以容纳失败JSON时仅写入一个结束符'\0'；len为0时不写入任何内容。
-   通信引擎仅支持`CCU_SCHED`（对应CCU路径）和`AICPU_TS`（对应AI CPU路径），其他取值（含`CCU_MS`）不支持本接口查询。

## 调用示例

```cpp
constexpr uint8_t kCcuSchedEngine = 6U; // CCU_SCHED
const uint8_t ccType = static_cast<uint8_t>(AscendC::HcclCMDType::HCCL_CMD_ALLGATHER);
char algs[16384]; // 建议不小于16KB
// ccArgs已由Mc2GetCcArgs创建，并经Mc2SetCcCommEngine(ccArgs, kCcuSchedEngine)、
// Mc2SetCcSrcDataType设置；可选经Mc2SetCcDataLenRange设置数据长度范围

Mc2Result ret = Mc2GetSupportAlgList(comm, ccType, ccArgs, algs, sizeof(algs));
if (ret == HCCL_SUCCESS && strstr(algs, "\"error_msg\":[]") != nullptr) {
    // 解析alg_list，取一个算法名经Mc2SetCcAlgConfig设置后执行
    ret = Mc2SetCcAlgConfig(ccArgs, "sole[mesh]");
    if (ret != HCCL_SUCCESS) {
        // 错误处理
    }
}
```

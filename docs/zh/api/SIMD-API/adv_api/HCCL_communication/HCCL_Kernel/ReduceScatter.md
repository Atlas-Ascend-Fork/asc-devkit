# ReduceScatter

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
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
<!-- @ref: asc-devkit/res/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel/ReduceScatter_res.md#id1 -->

## 功能说明

集合通信算子ReduceScatter的任务下发接口，返回该任务的标识handleId给用户。ReduceScatter的功能为：将所有rank的输入相加（或其他归约操作）后，再把结果按照rank编号均匀分散到各个rank的输出buffer，每个进程拿到其他进程1/ranksize份的数据进行归约操作。

![](../../../../figures/ReduceScatter.png)

## 函数原型

```
template <bool commit = false>
__aicore__ inline HcclHandle ReduceScatter(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, uint64_t strideCount, uint8_t repeat = 1)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| commit | 输入 | bool类型。参数取值如下：<br>true：在调用Prepare接口时，Commit同步通知服务端可以执行该通信任务。<br>false：在调用Prepare接口时，不通知服务端执行该通信任务。 |

**表2**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| sendBuf | 输入 | 源数据buffer地址。 |
| recvBuf | 输出 | 目的数据buffer地址，集合通信结果输出到此buffer中。 |
| recvCount | 输入 | 参与ReduceScatter操作的recvBuf的数据个数；sendBuf的数据个数等于recvCount * rank size。 |
| dataType | 输入 | ReduceScatter操作的数据类型。不同产品和通信服务端支持的数据类型请参见[数据类型](#数据类型)。HcclDataType数据类型的介绍请参考[表1](HCCL_usage.md#table116710585514)。 |
| op | 输入 | ReduceScatter的操作类型。支持的操作类型及约束请参见[op说明](#op说明)。HcclReduceOp数据类型的介绍请参考[表2](HCCL_usage.md#table2469980529)。 |
| strideCount | 输入 | 当将一张卡上sendBuf中的数据scatter到多张卡的recvBuf时，需要用strideCount参数表示sendBuf上相邻数据块间的起始地址的偏移量。<br>strideCount=0，表示从当前卡发送数据给其它卡时，相邻数据块保持地址连续。本卡发送数据到卡rank[i]，且本卡数据块在sendBuf中的偏移为i\*recvCount。非多轮切分场景下，推荐用户设置该参数为0。<br>strideCount>0，表示从当前卡发送数据给其它卡时，相邻数据块在sendBuf中起始地址的偏移数据量为strideCount。本卡发送数据到卡rank[i]，且本卡数据块在SendBuf中的偏移为i\*strideCount。<br><br>注意：上述的偏移数据量为数据个数，单位为sizeof(dataType)。 |
| repeat | 输入 | 一次下发的ReduceScatter通信任务个数。repeat取值≥1，默认值为1。当repeat>1时，每个ReduceScatter任务的sendBuf和recvBuf地址由服务端自动算出，计算公式如下：<br><br>sendBuf[i] = sendBuf + recvCount * sizeof(datatype) * i, i∈[0, repeat)<br><br>recvBuf[i] = recvBuf + recvCount * sizeof(datatype) * i, i∈[0, repeat)<br><br>注意：当设置repeat>1时，须与strideCount参数配合使用，规划通信数据地址。 |

### 数据类型

不同产品和通信服务端支持的数据类型如下。

<!-- npu="950" id10 -->
- 对于Ascend 950PR&950DT系列产品：

    - 通信服务端为AI CPU时，支持int8_t、int16_t、half、bfloat16_t、int32_t、float、int64_t、uint64_t、double。
    - 通信服务端为CCU时，支持int16_t、half、bfloat16_t、int32_t、float；int8_t的使用须满足[op说明](#op说明)中的输入输出类型和算法约束。
<!-- end id10 -->

<!-- npu="A3" id11 -->
- 对于Atlas A3系列产品，支持int8_t、int16_t、half、bfloat16_t、int32_t、float、int64_t。
<!-- end id11 -->

<!-- npu="910b" id12 -->
- 对于Atlas A2系列产品，支持int8_t、int16_t、half、bfloat16_t、int32_t、float、int64_t。
<!-- end id12 -->

### op说明

<!-- npu="950" id13 -->
- 对于Ascend 950PR&950DT系列产品：

    - AI CPU服务端：支持sum、prod、max、min操作，其中prod操作不支持int16_t、bfloat16_t数据类型。
    - CCU服务端：仅支持sum、max、min操作。使用int8_t输入时，须选择支持int8_t的通信算法，并满足以下输入输出类型约束；当前CCU自动选路的部分算法不支持int8_t。
        - sum操作：仅适用于所选通信算法支持int8_t输入、浮点输出归约的场景。须通过[SetReduceType](../HCCL_Tiling/SetReduceType.md)将输出类型设置为half、bfloat16_t、float三者之一。
        - max、min操作：输出类型须与输入类型一致。
<!-- end id13 -->

<!-- npu="A3" id14 -->
- 对于Atlas A3系列产品，通信服务端为AI CPU，支持sum、prod、max、min操作，其中prod操作不支持int16_t、bfloat16_t数据类型。
<!-- end id14 -->

<!-- npu="910b" id15 -->
- 对于Atlas A2系列产品，通信服务端为AI CPU，支持sum、prod、max、min操作，其中prod操作不支持int16_t、bfloat16_t数据类型。
<!-- end id15 -->

**图1**  ReduceScatter通信示例

![ReduceScatter通信示例-58](../../../../figures/reducescatter_example_58.png)

以上图为例，假设4张卡的场景，每份数据被切分为3块（TileCnt为3），每张卡上的0-0、0-1、0-2数据最终reduce+scatter到卡rank0的recvBuf上，其余的每块1-y、2-y、3-y数据类似，最终分别reduce+scatter到卡rank1、rank2和rank3的recvBuf上。因此，对一张卡上的数据需要调用3次ReduceScatter接口，完成每份数据的3块切分数据的通信。对于每一份数据，本接口中参数recvCount为TileLen，strideCount为TileLen\*TileCnt（即数据块0-0和1-0间隔的数据个数）。由于本例为内存连续场景，因此也可以只调用1次ReduceScatter接口，并将repeat参数设置为3。

## 返回值说明

返回该任务的标识handleId，handleId大于等于0。调用失败时，返回 -1。

## 约束说明

-   调用本接口前确保已调用过[InitV2](InitV2.md)和[SetCcTilingV2](SetCcTilingV2.md)接口。
-   若HCCL对象的[config模板参数](HCCL_template_params.md#hccl-template-params)未指定下发通信任务的核，该接口只能在AIC核或者AIV核两者之一上调用。若HCCL对象的[config模板参数](HCCL_template_params.md#hccl-template-params)中指定了下发通信任务的核，则该接口可以在AIC核和AIV核上同时调用，接口内部会根据指定的核的类型，只在AIC核、AIV核二者之一下发该通信任务。
<!-- npu="910b" id7 -->
-   对于Atlas A2系列产品，一个通信域内，所有Prepare接口的总调用次数不能超过63。
<!-- end id7 -->
<!-- npu="A3" id8 -->
-   对于Atlas A3系列产品，一个通信域内，所有Prepare接口和InterHcclGroupSync接口的总调用次数不能超过63。
<!-- end id8 -->
<!-- npu="950" id9 -->
-   对于Ascend 950PR&950DT系列产品，一个通信域内，所有Prepare接口的总调用次数不能超过63。
-   对于Ascend 950PR&950DT系列产品，通信服务端为CCU时，最大支持8卡的全链接，默认仅支持FullMesh算法，并且单次最大通信数据量不能超过256M。
<!-- end id9 -->

## 调用示例

-   非多轮切分场景

    如下图所示，4张卡上均有300 \* 4=1200个float16数据，每张卡从xGM内存中获取到本卡数据，对各卡数据完成reduce sum计算后的结果数据，进行scatter处理，最终每张卡都得到300个reduce sum后的float16数据。

    **图2**  非多轮切分场景下4卡ReduceScatter通信

    ![](../../../../figures/ReduceScatter_gai.png)

    ```
    extern "C" __global__ __aicore__ void reduce_scatter_custom(
        GM_ADDR xGM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
    {
        auto sendBuf = xGM;       // xGM为ReduceScatter的输入GM地址
        auto recvBuf = yGM;       // yGM为ReduceScatter的输出GM地址
        uint64_t recvCount = 300; // 每张卡的通信结果数据个数
        uint64_t strideCount = 0; // 非切分场景strideCount可设置为0
        HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
        REGISTER_TILING_DEFAULT(ReduceScatterCustomTilingData); // ReduceScatterCustomTilingData为对应算子头文件定义的结构体
        GET_TILING_DATA_WITH_STRUCT(ReduceScatterCustomTilingData, tilingData, tilingGM);

        Hccl hccl;
        GM_ADDR contextGM = AscendC::GetHcclContext<0>(); // AscendC自定义算子核函数（Kernel）中，通过此方式获取HCCL context
        if (AscendC::g_coreType == AIV) {                 // 指定AIV核通信
            hccl.InitV2(contextGM, &tilingData);
            auto ret = hccl.SetCcTilingV2(offsetof(ReduceScatterCustomTilingData, reduceScatterCcTiling));
            if (ret != HCCL_SUCCESS) {
                return;
            }
            HcclHandle handleId1 = hccl.ReduceScatter<true>(
                sendBuf, recvBuf, recvCount, HcclDataType::HCCL_DATA_TYPE_FP16, reduceOp, strideCount);
            hccl.Wait(handleId1);
            AscendC::SyncAll<true>(); // 全AIV核同步，防止0核执行过快，提前调用hccl.Finalize()接口，导致其他核Wait卡死
            hccl.Finalize();
        }
    }
    ```

-   多轮切分场景

    开启多轮切分，等效处理上述非多轮切分示例的通信。如下图所示，每张卡的每份300个float16数据，被切分为2个首块，1个尾块。每个首块的数据量tileLen为128个float16数据，尾块的数据量tailLen为44个float16数据。在算子内部实现时，需要对切分后的数据分3轮进行ReduceScatter通信任务，将等效上述非多轮切分的通信结果。

    **图3**  各卡数据切分示意图  
    ![各卡数据切分示意图-59](../../../../figures/per_rank_data_split_59.png)

    具体实现为，第1轮通信，每个rank上的0-0\\1-0\\2-0\\3-0数据块进行ReduceScatter处理。第2轮通信，每个rank上0-1\\1-1\\2-1\\3-1数据块进行ReduceScatter处理。第3轮通信，每个rank上0-2\\1-2\\2-2\\3-2数据块进行ReduceScatter处理。每一轮通信的输入数据中，各卡上相邻数据块的起始地址间隔的数据个数为strideCount，以第一轮通信结果为例，rank0的0-0数据块和1-0数据块，或者1-0数据块和2-0数据块，两个相邻数据块起始地址间隔的数据量strideCount = 2\*tileLen+1\*tailLen=300。

    **图4**  第一轮4卡ReduceScatter示意图  
    ![第一轮4卡ReduceScatter示意图](../../../../figures/first_round_4rank_reducescatter.png)

    ```
    extern "C" __global__ __aicore__ void reduce_scatter_custom(
        GM_ADDR xGM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
    {
        constexpr uint32_t tileNum = 2U;   // 首块数量
        constexpr uint64_t tileLen = 128U; // 首块数据个数
        constexpr uint32_t tailNum = 1U;   // 尾块数量
        constexpr uint64_t tailLen = 44U;  // 尾块数据个数
        auto sendBuf = xGM;                // xGM为ReduceScatter的输入GM地址
        auto recvBuf = yGM;                // yGM为ReduceScatter的输出GM地址
        HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
        uint64_t strideCount = tileLen * tileNum + tailLen * tailNum;
        REGISTER_TILING_DEFAULT(ReduceScatterCustomTilingData); // ReduceScatterCustomTilingData为对应算子头文件定义的结构体
        GET_TILING_DATA_WITH_STRUCT(ReduceScatterCustomTilingData, tilingData, tilingGM);

        Hccl hccl;
        GM_ADDR contextGM = AscendC::GetHcclContext<0>(); // AscendC自定义算子核函数（Kernel）中，通过此方式获取HCCL context
        if (AscendC::g_coreType == AIV) {                 // 指定AIV核通信
            hccl.InitV2(contextGM, &tilingData);
            auto ret = hccl.SetCcTilingV2(offsetof(ReduceScatterCustomTilingData, reduceScatterCcTiling));
            if (ret != HCCL_SUCCESS) {
                return;
            }
            // 2个首块处理
            constexpr uint32_t tileRepeat = tileNum;
            // 除了sendBuf和recvBuf入参不同，处理2个首块的其余参数相同。故使用repeat=2，第2个首块ReduceScatter任务的sendBuf、recvBuf将由API内部自行更新
            HcclHandle handleId1 = hccl.ReduceScatter<true>(
                sendBuf, recvBuf, tileLen, HcclDataType::HCCL_DATA_TYPE_FP16, reduceOp, strideCount, tileRepeat);
            // 1个尾块处理
            constexpr uint32_t kSizeOfFloat16 = 2U;
            sendBuf += tileLen * tileNum * kSizeOfFloat16;
            recvBuf += tileLen * tileNum * kSizeOfFloat16;
            constexpr uint32_t tailRepeat = tailNum;
            HcclHandle handleId2 = hccl.ReduceScatter<true>(
                sendBuf, recvBuf, tailLen, HcclDataType::HCCL_DATA_TYPE_FP16, reduceOp, strideCount, tailRepeat);

            for (uint8_t i = 0; i < tileRepeat; i++) {
                hccl.Wait(handleId1);
            }
            hccl.Wait(handleId2);
            AscendC::SyncAll<true>(); // 全AIV核同步，防止0核执行过快，提前调用hccl.Finalize()接口，导致其他核Wait卡死
            hccl.Finalize();
        }
    }
    ```

# GroupBarrier

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
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
<!-- @ref: asc-devkit/res/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/GroupBarrier_res.md#id1 -->

## 功能说明

头文件路径为`"basic_api/core_mng/roc/kernel_operator_group_barrier_intf.h"`。

当同一个[CubeResGroupHandle](../../inter_core_communication/CubeResGroupHandle/CubeResGroupHandle.md)中的两个AIV任务之间存在依赖关系时，可以使用GroupBarrier控制同步。假设一组AIV A做完任务x以后，另外一组AIV B才可以开始后续业务，称AIV A组为Arrive组，AIV B组为Wait组。

GroupBarrier提供以下接口：

- <a name="constructor"></a>**GroupBarrier**：创建GroupBarrier对象，通过设置构造函数参数，确定Arrive组的Block个数和Wait组的Block个数。
- <a name="arrive"></a>**Arrive**：通知其他等待的AIV，本AIV已经完成其依赖的任务。
- <a name="wait"></a>**Wait**：阻塞当前AIV，直到Arrive组的AIV均完成依赖任务后，再开始执行后续任务。
- <a name="get-workspace-len"></a>**GetWorkspaceLen**：返回当前GroupBarrier所占用的Global Memory消息空间大小。

## 函数原型

```cpp
template <PipeMode pipeMode>
class GroupBarrier {
public:
    __aicore__ inline GroupBarrier(GM_ADDR groupWorkspace, uint32_t arriveSizeIn, uint32_t waitSizeIn);
    __aicore__ inline void Arrive(uint32_t arriveIndex);
    __aicore__ inline void Wait(uint32_t waitIndex);
    __aicore__ inline uint64_t GetWorkspaceLen();
};
```

## 参数说明

**表1** 类模板参数说明

| 参数 | 说明 |
| --- | --- |
| pipeMode | GroupBarrier发送组同步消息时使用的执行单元，仅支持MTE3_MODE。 |

```cpp
enum class PipeMode : uint8_t {
    SCALAR_MODE = 0,
    MTE3_MODE = 1,
    MAX
};
```

**表2** 构造函数和成员函数参数说明

| 接口 | 参数 | 输入/输出 | 说明 |
| --- | --- | --- | --- |
| GroupBarrier | groupWorkspace | 输入 | 该GroupBarrier在GM上的起始地址，用于存储组同步消息，必须512B对齐。用户自行管理这部分GlobalMemory，包括地址对齐和清零。 |
| GroupBarrier | arriveSizeIn | 输入 | Arrive组AIV个数。 |
| GroupBarrier | waitSizeIn | 输入 | Wait组AIV个数。 |
| Arrive | arriveIndex | 输入 | 该AIV在Arrive组的序号。范围为[0, arriveSize - 1]。 |
| Wait | waitIndex | 输入 | 该AIV在Wait组的序号。范围为[0, waitSize - 1]。 |

GetWorkspaceLen无参数。

## 返回值说明

| 接口 | 返回值 |
| --- | --- |
| GroupBarrier | GroupBarrier对象实例。 |
| Arrive | 无。 |
| Wait | 无。 |
| GetWorkspaceLen | 当前GroupBarrier所占用的Global Memory消息空间大小。 |

## 约束说明

- GroupBarrier构造函数的约束如下：
    - 使用该构造函数时，Unified Buffer（UB）空间末尾的1600B被占用。
    - 不能和[REGIST\_MATMUL\_OBJ](../../../adv_api/cube_compute/Matmul_Kernel/REGIST_MATMUL_OBJ.md)接口同时使用。使用资源管理API时，用户自主管理AIC和AIV的核间通信，REGIST\_MATMUL\_OBJ内部是由框架管理AIC和AIV的核间通信，同时使用可能会导致通信消息错误等异常。

- Arrive、Wait：支持在循环中使用，但是受限于多核间通信效率要求，循环最大次数不超过1,048,575次。

## 调用示例

<a name="usage"></a>

基于GroupBarrier的组同步使用步骤如下：

1. 创建GroupBarrier，并为其分配满足对齐要求且已清零的GlobalMemory空间。
2. 被等待的AIV调用Arrive，需要等待的AIV调用Wait。
3. 调用GetWorkspaceLen查询GroupBarrier所需的workspace大小。

以下kernel示例中，Block 0和Block 1属于Arrive组，Block 2至Block 7属于Wait组。workspace由调用方分配512B对齐且已清零的3072B空间。

```cpp
constexpr uint32_t ARRIVE_NUM = 2;
constexpr uint32_t WAIT_NUM = 6;

extern "C" __global__ __vector__ void GroupBarrierKernel(__gm__ uint8_t* workspace)
{
    AscendC::GroupBarrier<AscendC::PipeMode::MTE3_MODE> barrier(workspace, ARRIVE_NUM, WAIT_NUM);
    uint64_t workspaceLen = barrier.GetWorkspaceLen();
    (void)workspaceLen;

    uint32_t blockIdx = AscendC::GetBlockIdx();
    if (blockIdx < ARRIVE_NUM) {
        // Arrive-group vector operations go here.
        barrier.Arrive(blockIdx);
    } else if (blockIdx < ARRIVE_NUM + WAIT_NUM) {
        barrier.Wait(blockIdx - ARRIVE_NUM);
        // Wait-group vector operations go here.
    }
}
```
调用该kernel时，blockDim需设置为8。GroupBarrier需要的workspace大小为3072B，可通过GetWorkspaceLen查询。workspaceLen为设备侧查询结果，实际分配workspace应由调用方在host侧完成。
```text
workspaceLen = 3072
```
结果示例如下：

```text
占用的GlobalMemory空间(offset)的大小为: 3072
```

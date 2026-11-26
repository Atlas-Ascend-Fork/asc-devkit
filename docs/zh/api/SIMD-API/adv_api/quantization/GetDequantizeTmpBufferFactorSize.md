# GetDequantizeTmpBufferFactorSize

## 功能说明

该接口用于获取maxLiveNodeCnt和extraBuf，在固定空间大小的情况下，通过maxLiveNodeCnt和extraBuf可以推算算子单次最大计算元素数量。maxLiveNodeCnt表示临时空间是单次计算数据量所占空间的多少倍；extraBuf表示使用的额外临时空间大小。

推算示例如下：

-   算子实现需要调用Dequantize接口，开发者为其预留currBuff大小的空间，利用GetDequantizeTmpBufferFactorSize接口得到maxLiveNodeCnt、extraBuf输出值，可推导算子单次最大计算元素数量为：

    **currentShapeSize = \(currBuff - extraBuf\) / maxLiveNodeCnt / typeSize**

-   算子实现需要调用两个核函数（Kernel）侧API KernelIntf1、KernelIntf2，利用两个GetXxxTmpBufferFactorSize（其中Xxx为需要调用的两个高阶API）接口的两组输出值\(maxLiveNodeCnt、extraBuf\)以及当前现有的临时空间，推导单次最大计算元素数量currentShapeSize为：

    **currentShapeSize1 = \(currBuff - extraBuf1\) / maxLiveNodeCnt1 / typeSize**

    **currentShapeSize2 = \(currBuff - extraBuf2\) / maxLiveNodeCnt2 / typeSize**

    **currentShapeSize = min\(currentShapeSize1, currentShapeSize2\)**

注意上文中的currBuff表示接口计算可用的空间，需要去除用户输入输出等空间；另外，接口获取的maxLiveNodeCnt值可能为0，计算时需要判断该值非0，避免除零错误。

## 函数原型

```
void GetDequantizeTmpBufferFactorSize(const AscendC::TensorShape& srcShape, uint32_t& maxLiveNodeCnt, uint32_t& extraBuf)
```

## 参数说明

**表1**  参数列表

| 参数名 | 输入/输出 | 功能 |
| --- | --- | --- |
| srcShape | 输入 | 输入srcTensor的shape信息，参数类型为[AscendC::TensorShape](../data_structures/TensorShape.md)。 |
| maxLiveNodeCnt | 输出 | 最大存活节点数，表示临时空间是单次计算数据量所占空间的多少倍。 |
| extraBuf | 输出 | 使用的额外临时空间大小，单位为字节。 |

## 返回值说明

无

## 约束说明

当利用maxLiveNodeCnt，extraBuf反推出的currentShapeSize  \* typeSize < 256B时，currentShapeSize按照256B/typeSize的值向上取整。

## 调用示例

```
std::vector<int64_t> shapeVec = {2, 1024};
AscendC::TensorShape srcShape(shapeVec);
uint32_t maxLiveNodeCnt = 0;
uint32_t extraBuf = 0;
AscendC::GetDequantizeTmpBufferFactorSize(srcShape, maxLiveNodeCnt, extraBuf);
```

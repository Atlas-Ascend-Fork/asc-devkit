# 设置合理的L2 CacheMode

【优先级】高

> [!NOTE] 说明
> 该性能优化指导适用于如下产品型号：
> <!-- npu="950" id1 -->
> - Ascend 950PR&950DT系列产品
> <!-- end id1 -->
> <!-- npu="A3" id2 -->
> - Atlas A3系列产品
> <!-- end id2 -->
> <!-- npu="910b" id3 -->
> - Atlas A2系列产品
> <!-- end id3 -->

## 优化原理

【描述】L2 Cache常用于缓存频繁访问的数据，其物理位置如下图所示：

<img src="../../../figures/reduce_61.png" alt="" width="600px" height="469px">

L2 Cache的带宽相比GM的带宽有数倍的提升，因此当数据命中L2 Cache时，数据的搬运耗时会优化数倍。通常情况下，L2 Cache命中率越高，算子的性能越好，在实际访问中需要通过设置合理的L2 CacheMode来保证重复读取的数据尽量缓存在L2 Cache上。

### L2 Cache访问的原理及CacheMode介绍

数据通过MTE2搬运单元搬入时，L2 Cache访问的典型流程如下：

<img src="../../../figures/reduce_62.png" alt="" width="600px" height="330px">

数据通过MTE3或者Fixpipe搬运单元搬出时，L2 Cache访问的典型流程如下：

<img src="../../../figures/reduce_opt_63.png" alt="" width="600px" height="398px">

从上面的流程可以看出，当数据访问总量超出L2 Cache容量时，AI Core会对L2 Cache进行数据替换。由于Cache一致性的要求，替换过程中旧数据需要先写回GM（此过程中会占用GM带宽），旧数据写回后，新的数据才能进入L2 Cache。

开发者可以针对访问的数据设置其CacheMode，对于只访问一次的Global Memory数据设置其访问状态为不进入L2 Cache，这样可以更加高效的利用L2 Cache缓存需要重复读取的数据，避免一次性访问的数据替换有效数据。

## 不同API下的实现

<!-- npu="950" id20 -->

> [!NOTE] 说明
> C API实现部分仅适用于Ascend 950PR&950DT系列产品。


### C API

以下保留输入x、y和输出z的Shape均为(8192, 8192)、48核按行切6份和列切8份的场景；代码展示当前核的搬运块配置，计算及核间切分沿用原流程。输入只访问一次时禁用输入的L2 Cache，输出保持Normal。

通过[asc_copy_gm2ub_align](../../../../api/SIMD-API/c_api/vector_datamove/asc_copy_gm2ub_align/asc_copy_gm2ub_align_arch_3510.md)的l2_cache_mode参数设置每条搬运指令的策略。

```cpp
#include "c_api/asc_simd.h"
__aicore__ inline void CopyInputC(__ubuf__ half* local, __gm__ half* input, bool disable)
{
    auto mode = disable ? asc_load_l2_cache_mode::NOTALLOC_KEEP :
        asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM;
    asc_copy_gm2ub_align(local, input, 1, 2048, 0, 0, false, mode, 2048, 2048);
}
```

【反例】

```cpp
CopyInputC(xUb, xGm, false);
CopyInputC(yUb, yGm, false);
```

【正例】

```cpp
CopyInputC(xUb, xGm, true);
CopyInputC(yUb, yGm, true);
```

<!-- end id20 -->

### Basic API

#### 设置L2 CacheMode的方法

Ascend C基于GlobalTensor提供了SetL2CacheHint接口，用户可以根据需要指定CacheMode。

考虑如下场景，构造两个Tensor的相加计算，输入x、y和输出z的Shape均为\(8192, 8192\)。下文完整样例请参考[设置合理L2 CacheMode样例](../../../../../../examples/01_simd_cpp_api/05_best_practices/00_vector_compute/add_high_performance)。

【反例】

总数据量384MB，其中：x：128MB；y：128MB；z：128MB。

使用48个核参与计算，按行方向切分6份，列方向切分8份。

x、y、z对应GlobalTensor的CacheMode均设置为CACHE_MODE_NORMAL，需要经过L2 Cache，需要进入L2 Cache的总数据量为384MB。

```cpp
xGm.SetGlobalBuffer((__gm__ half *)x + startM * totalN + startN, bufferSize);
yGm.SetGlobalBuffer((__gm__ half *)y + startM * totalN + startN, bufferSize);
zGm.SetGlobalBuffer((__gm__ half *)z + startM * totalN + startN, bufferSize);
```

【正例】

总数据量384MB，其中：x：128MB；y：128MB；z：128MB。

使用48个核参与计算，按行方向切分6份，列方向切分8份。

z对应的GlobalTensor的CacheMode设置为CACHE_MODE_NORMAL；x和y对应的GlobalTensor的CacheMode设置为CACHE_MODE_DISABLE。只将z设置为需要经过L2 Cache。需要进入L2 Cache的总数据量为128MB。

```cpp
xGm.SetGlobalBuffer((__gm__ half *)x + startM * totalN + startN, bufferSize);
yGm.SetGlobalBuffer((__gm__ half *)y + startM * totalN + startN, bufferSize);
zGm.SetGlobalBuffer((__gm__ half *)z + startM * totalN + startN, bufferSize);
// disable the L2 Cache mode of x and y
xGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
```

> [!NOTE] 说明
> 你可以通过执行如下命令行，通过msOpProf工具获取上述示例的性能数据并进行对比。
> ```
> msopprof --launch-count=2 --output=./prof ./execute_add_op
> ```
> 重点关注Memory.csv中的aiv\_gm\_to\_ub\_bw\(GB/s\)和aiv\_main\_mem\_write\_bw\(GB/s\)写带宽的速率。

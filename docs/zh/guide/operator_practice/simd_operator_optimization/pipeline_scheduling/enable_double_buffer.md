# 开启Double Buffer

【优先级】中

## 优化原理

【描述】执行于AI Core上的指令队列主要包括如下几类：[Vector（矢量）](../../../technical_appendix/concepts_and_terms/glossary.md#p203923211517)指令队列V、[Cube（矩阵）](../../../technical_appendix/concepts_and_terms/glossary.md#p1038918214519)指令队列M、[Scalar（标量）](../../../technical_appendix/concepts_and_terms/glossary.md#p1739112219516)指令队列S和搬运指令队列[Memory Transfer Engine 1（MTE1）](../../../technical_appendix/concepts_and_terms/glossary.md#p339019220518)/[Memory Transfer Engine 2（MTE2）](../../../technical_appendix/concepts_and_terms/glossary.md#p53901325511)/[Memory Transfer Engine 3（MTE3）](../../../technical_appendix/concepts_and_terms/glossary.md#p183901322511)。不同指令队列间的相互独立性和可并行执行特性，是Double Buffer优化机制的基石。

以纯Vector计算为例，矢量计算前后的CopyIn、CopyOut过程使用搬运指令队列（MTE2/MTE3），Compute过程使用Vector指令队列（V）。不同指令队列可并行执行，意味着CopyIn、CopyOut过程和Compute过程可以并行。如下图所示，考虑一个完整的数据搬运和计算过程，CopyIn过程将数据从[Global Memory（GM）](../../../technical_appendix/concepts_and_terms/glossary.md#p1439018225112)搬运到[Local Memory（LM）](../../../technical_appendix/concepts_and_terms/glossary.md#p193901420518)，Vector计算单元完成Compute计算后，经过CopyOut过程将计算结果搬回GM。

**图1** 数据搬运与Vector计算过程    
![](../../../figures/data_vec_54.png "数据搬运与Vector计算过程")

**图2** 未开启Double Buffer的流水图    
![](../../../figures/no_doublebuf_pipe.png "未开启Double Buffer的流水图")

在此过程中，数据搬运与Vector计算串行执行，Vector计算单元不可避免地存在资源闲置问题。假设CopyIn、Compute、CopyOut三个阶段的耗时均为$t$，则Vector的利用率仅为三分之一，等待时间过长，Vector利用率严重不足。

为减少Vector等待时间，Double Buffer为待处理数据准备Ping和Pong两组缓冲区。如下表所示，t0时，MTE2将数据块0的输入搬入Ping；t1时，MTE2将数据块1的输入搬入Pong，Vector同时使用Ping计算数据块0；t2时，MTE2将数据块2的输入搬入Ping，Vector使用Pong计算数据块1，MTE3同时从Ping搬出数据块0的结果。由此，数据搬入、Vector计算和数据搬出可以跨数据块并行执行，缓解Vector计算单元闲置问题。

**表1** Ping/Pong Double Buffer的三个时间点

| 执行阶段 | t0 | t1 | t2 |
| --- | --- | --- | --- |
| MTE2搬入 | 数据块0输入到Ping输入区 | 数据块1输入到Pong输入区 | 数据块2输入到Ping输入区 |
| Vector计算 | 无 | 计算数据块0（Ping） | 计算数据块1（Pong） |
| MTE3搬出 | 无 | 无 | 数据块0结果从Ping输出区搬出 |

t2时，Ping中的输入区和输出区相互独立，因此数据块2搬入和数据块0搬出可以同时执行。

总体来说，Double Buffer是基于MTE指令队列与Vector指令队列的独立性和可并行性，通过将数据搬运与Vector计算并行执行以隐藏大部分的数据搬运时间，并降低Vector指令的等待时间，最终提高Vector单元的利用效率。开启Double Buffer后的整体流水如下图所示。

**图3** 开启Double Buffer的流水图    
![](../../../figures/dbuf_pipe.png "开启Double Buffer的流水图")

### 使用说明

多数情况下，采用Double Buffer能有效提升Vector的利用率，缩减算子执行时间。然而，Double Buffer机制缓解Vector闲置问题，并不代表它总能带来明显的整体性能提升。例如：

- 当数据搬运时间较短，而Vector计算时间较长时，由于数据搬运在整个计算过程中的时间占比较低，Double Buffer机制带来的性能收益会偏小。
- 当原始数据较小且Vector可以一次性完成所有数据量的计算时，无法形成跨数据块的流水重叠，额外的缓冲区和同步处理可能使效果适得其反。

使用Double Buffer还需要满足以下条件：

- 单核数据能够切分为至少两个数据块，使相邻数据块可以交替使用两组缓冲区。
- [Unified Buffer（UB）](../../../technical_appendix/concepts_and_terms/glossary.md#p17392122195118)能够容纳两组输入和输出缓冲区以及算子所需的其他临时数据。
- 同一缓冲区的写入、读取和再次复用顺序能够通过相应编程方式的同步机制正确管理。
- 缓冲区地址和数据搬运长度满足目标产品及搬运接口的对齐要求，尾块按照实际数据长度进行处理。

因此，Double Buffer的使用需综合考虑Vector算力、数据量、UB容量、数据块数量以及搬运与计算时间占比等因素。

【反例】

只使用一组缓冲区，在同一组空间上串行完成数据搬入、计算和搬出后，再处理下一块数据，搬运和计算不能跨数据块重叠。

【正例】

使用Ping和Pong两组缓冲区交替处理数据块，通过同步保证各缓冲区的读写和复用顺序，使不同数据块的搬入、计算和搬出能够重叠。

## 不同API下的实现

下面将上述缓冲区安排和读写依赖落实到不同API，各节分别说明产品范围和资源管理方式。

<!-- npu="950" id1 -->

> [!NOTE] 说明
> C API实现部分仅适用于Ascend 950PR&950DT系列产品。

### C API：基于指针的编程方式

采用基于指针的C语言编程方式时，Double Buffer不是单一开关，需要在核函数（Kernel）中手动申请Ping和Pong两组Unified Buffer（UB）缓冲区，根据数据块序号交替选择缓冲区，并使用[asc_lock](../../../../api/SIMD-API/c_api/sync/intra_core_sync/asc_lock.md)和[asc_unlock](../../../../api/SIMD-API/c_api/sync/intra_core_sync/asc_unlock.md)接口约束同一缓冲区的读取、写入和再次复用。

下面直接摘录[融合向量计算Double Buffer样例源码](../../../../../../examples/02_simd_c_api/02_features/01_reg_vector_compute/00_add_double_buffer/add_double_buffer.asc)中的`fused_compute_vf`、`add_baseline`和`add_double_buffer`函数，函数名、参数、计算和流水调度均与样例一致。本节适用于Ascend 950PR&950DT系列产品。头文件、编译期常量和Host启动代码见[样例说明](../../../../../../examples/02_simd_c_api/02_features/01_reg_vector_compute/00_add_double_buffer/README.md)，此处代码需结合完整样例使用。

样例先计算`t0 = x + y`，再执行`ti+1 = ti * x + y`融合递推；单缓冲区基线与双缓冲区版本使用相同的`fused_compute_vf`，仅改变缓冲区安排和同步调度。`MAX_DATA_COPY_LENGTH`确定单个缓冲区容量，`DATA_COPY_LENGTH`确定每轮搬运量，`current_length`处理尾块。常量定义与计算规格以样例源码为准。

```cpp
__simd_vf__ inline void fused_compute_vf(
    __ubuf__ half* z_local, __ubuf__ half* x_local, __ubuf__ half* y_local, uint32_t element_count)
{
    constexpr uint32_t elements_per_vf = VF_LENGTH / sizeof(half);
    const uint16_t repeat_times = static_cast<uint16_t>((element_count + elements_per_vf - 1) / elements_per_vf);
    vector_half x_reg;
    vector_half y_reg;
    vector_half z_reg;
    vector_half temp_reg;
    vector_bool mask;
    // 每次迭代处理一个向量宽度，最后一个不足向量宽度的数据块通过Mask处理。
    for (uint16_t i = 0; i < repeat_times; ++i) {
        mask = asc_update_mask_b16(element_count);
        asc_loadalign(x_reg, x_local + i * elements_per_vf);
        asc_loadalign(y_reg, y_local + i * elements_per_vf);
        // t0 = x + y。
        asc_add(z_reg, x_reg, y_reg, mask);
        // ti+1 = ti * x + y；中间结果保留在寄存器中，不增加GM和UB搬运量。
        for (uint32_t fused_idx = 0; fused_idx < FUSED_ITERATIONS; ++fused_idx) {
            asc_mul(temp_reg, z_reg, x_reg, mask);
            asc_add(z_reg, temp_reg, y_reg, mask);
        }
        asc_storealign(z_local + i * elements_per_vf, z_reg, mask);
    }
}

```

【反例】

复用一组输入和输出空间，使用同一个Mutex串联搬入、计算和搬出。

```cpp
__vector__ __global__ void add_baseline(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* z)
{
    asc_init();

    // 为每个核分配连续的矩阵行。
    int64_t core_idx = block_idx;
    constexpr int64_t base_core_m = TOTAL_M / SPLIT_M;
    constexpr int64_t remainder_m = TOTAL_M % SPLIT_M;
    int64_t single_core_m;
    int64_t start_m;
    if (core_idx < remainder_m) {
        single_core_m = base_core_m + 1;
        start_m = core_idx * single_core_m;
    } else {
        single_core_m = base_core_m;
        start_m = remainder_m * (base_core_m + 1) + (core_idx - remainder_m) * base_core_m;
    }
    int64_t total_elements_per_core = single_core_m * TOTAL_N;

    // 将GM指针偏移到当前核负责的首行。
    __gm__ half* x_gm = reinterpret_cast<__gm__ half*>(x) + start_m * TOTAL_N;
    __gm__ half* y_gm = reinterpret_cast<__gm__ half*>(y) + start_m * TOTAL_N;
    __gm__ half* z_gm = reinterpret_cast<__gm__ half*>(z) + start_m * TOTAL_N;

    // 基线场景在UB中静态申请并复用一组连续排列的x/y/z Buffer。
    __ubuf__ half ub_buffer[MAX_DATA_COPY_LENGTH * BASELINE_BUFFER_NUM];
    __ubuf__ half* x_local = ub_buffer;
    __ubuf__ half* y_local = x_local + MAX_DATA_COPY_LENGTH;
    __ubuf__ half* z_local = y_local + MAX_DATA_COPY_LENGTH;

    // 向上取整计算数据块数量，由最后一次迭代处理尾块。
    uint32_t total_blocks = static_cast<uint32_t>((total_elements_per_core + DATA_COPY_LENGTH - 1) / DATA_COPY_LENGTH);
    // 所有数据块复用同一组UB Buffer，因此使用一个Mutex串联MTE2、Vector和MTE3。
    constexpr uint8_t mutex_id = 0;

    for (uint32_t loop_idx = 0; loop_idx < total_blocks; ++loop_idx) {
        uint32_t start_element = loop_idx * DATA_COPY_LENGTH;
        int64_t remain_elements = total_elements_per_core - start_element;
        uint32_t current_length =
            remain_elements > DATA_COPY_LENGTH ? DATA_COPY_LENGTH : static_cast<uint32_t>(remain_elements);
        // 使用一次连续Burst搬运，不设置Padding和块间Stride，非对齐尾块由硬件自动填充。
        uint32_t burst_length = current_length * sizeof(half);
        uint64_t gm_stride = 0;
        uint32_t ub_stride = 0;

        // 通过MTE2流水将当前x/y数据块从GM搬入UB。
        asc_lock(PIPE_MTE2, mutex_id);
        asc_copy_gm2ub_align(
            x_local, x_gm + start_element, BURST_NUM, burst_length, 0, 0, false, LOAD_L2_CACHE_MODE, gm_stride,
            ub_stride);
        asc_copy_gm2ub_align(
            y_local, y_gm + start_element, BURST_NUM, burst_length, 0, 0, false, LOAD_L2_CACHE_MODE, gm_stride,
            ub_stride);
        asc_unlock(PIPE_MTE2, mutex_id);

        // 等待输入数据就绪，再通过Vector流水执行融合计算。
        asc_lock(PIPE_V, mutex_id);
        asc_vf_call<fused_compute_vf>(z_local, x_local, y_local, current_length);
        asc_unlock(PIPE_V, mutex_id);

        // 等待计算结果就绪，再通过MTE3流水将z数据块从UB搬回GM。
        asc_lock(PIPE_MTE3, mutex_id);
        asc_copy_ub2gm_align(
            z_gm + start_element, z_local, BURST_NUM, burst_length, STORE_L2_CACHE_MODE, gm_stride, ub_stride);
        asc_unlock(PIPE_MTE3, mutex_id);
    }
    // 等待最后一轮异步搬入、计算和搬出全部完成。
    asc_sync_pipe(PIPE_ALL);
}
```

【正例】

分配Ping和Pong两组空间，分别使用输入和输出Mutex管理复用。

```cpp
__vector__ __global__ void add_double_buffer(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* z)
{
    asc_init();

    // 为每个核分配连续的矩阵行。
    int64_t core_idx = block_idx;
    constexpr int64_t base_core_m = TOTAL_M / SPLIT_M;
    constexpr int64_t remainder_m = TOTAL_M % SPLIT_M;
    int64_t single_core_m;
    int64_t start_m;
    if (core_idx < remainder_m) {
        single_core_m = base_core_m + 1;
        start_m = core_idx * single_core_m;
    } else {
        single_core_m = base_core_m;
        start_m = remainder_m * (base_core_m + 1) + (core_idx - remainder_m) * base_core_m;
    }
    int64_t total_elements_per_core = single_core_m * TOTAL_N;

    // 将GM指针偏移到当前核负责的首行。
    __gm__ half* x_gm = reinterpret_cast<__gm__ half*>(x) + start_m * TOTAL_N;
    __gm__ half* y_gm = reinterpret_cast<__gm__ half*>(y) + start_m * TOTAL_N;
    __gm__ half* z_gm = reinterpret_cast<__gm__ half*>(z) + start_m * TOTAL_N;

    // 在UB中静态申请两组独立的x/y/z Buffer，用于Ping/Pong交替复用。
    __ubuf__ half ub_buffer[MAX_DATA_COPY_LENGTH * DOUBLE_BUFFER_NUM];
    __ubuf__ half* x_ping = ub_buffer;
    __ubuf__ half* y_ping = x_ping + MAX_DATA_COPY_LENGTH;
    __ubuf__ half* z_ping = y_ping + MAX_DATA_COPY_LENGTH;
    __ubuf__ half* x_pong = z_ping + MAX_DATA_COPY_LENGTH;
    __ubuf__ half* y_pong = x_pong + MAX_DATA_COPY_LENGTH;
    __ubuf__ half* z_pong = y_pong + MAX_DATA_COPY_LENGTH;

    // 向上取整计算数据块数量，由最后一次迭代处理尾块。
    uint32_t total_blocks = static_cast<uint32_t>((total_elements_per_core + DATA_COPY_LENGTH - 1) / DATA_COPY_LENGTH);

    for (uint32_t loop_idx = 0; loop_idx < total_blocks; ++loop_idx) {
        uint32_t start_element = loop_idx * DATA_COPY_LENGTH;
        int64_t remain_elements = total_elements_per_core - start_element;
        uint32_t current_length =
            remain_elements > DATA_COPY_LENGTH ? DATA_COPY_LENGTH : static_cast<uint32_t>(remain_elements);

        // ID 0/1分别保护Ping/Pong的x/y，输出Mutex ID位于输入Mutex ID之后。
        uint8_t input_mutex_id = static_cast<uint8_t>(loop_idx & 1);
        uint8_t output_mutex_id = static_cast<uint8_t>(input_mutex_id + DOUBLE_BUFFER_GROUP_NUM);
        // 偶数轮使用Ping Buffer，奇数轮使用Pong Buffer。
        __ubuf__ half* x_local = (loop_idx & 1) == 0 ? x_ping : x_pong;
        __ubuf__ half* y_local = (loop_idx & 1) == 0 ? y_ping : y_pong;
        __ubuf__ half* z_local = (loop_idx & 1) == 0 ? z_ping : z_pong;
        // 使用一次连续Burst搬运，不设置Padding和块间Stride，非对齐尾块由硬件自动填充。
        uint32_t burst_length = current_length * sizeof(half);
        uint64_t gm_stride = 0;
        uint32_t ub_stride = 0;

        // 搬入x/y，输入Mutex保证上一轮Vector读取完成后才能复用对应Buffer。
        asc_lock(PIPE_MTE2, input_mutex_id);
        asc_copy_gm2ub_align(
            x_local, x_gm + start_element, BURST_NUM, burst_length, 0, 0, false, LOAD_L2_CACHE_MODE, gm_stride,
            ub_stride);
        asc_copy_gm2ub_align(
            y_local, y_gm + start_element, BURST_NUM, burst_length, 0, 0, false, LOAD_L2_CACHE_MODE, gm_stride,
            ub_stride);
        asc_unlock(PIPE_MTE2, input_mutex_id);

        // 等待x/y数据就绪且z Buffer可写，再计算当前数据块。
        asc_lock(PIPE_V, input_mutex_id);
        asc_lock(PIPE_V, output_mutex_id);
        asc_vf_call<fused_compute_vf>(z_local, x_local, y_local, current_length);
        asc_unlock(PIPE_V, input_mutex_id);
        asc_unlock(PIPE_V, output_mutex_id);

        // 独立搬出z，此时MTE2可同时向同组Buffer重新搬入x/y。
        asc_lock(PIPE_MTE3, output_mutex_id);
        asc_copy_ub2gm_align(
            z_gm + start_element, z_local, BURST_NUM, burst_length, STORE_L2_CACHE_MODE, gm_stride, ub_stride);
        asc_unlock(PIPE_MTE3, output_mutex_id);
    }

    // Kernel退出前等待所有未完成的流水操作结束。
    asc_sync_pipe(PIPE_ALL);
}
```

采用该方式时，需要注意以下事项：

- 输入Mutex ID 0/1分别保护Ping/Pong的`x/y`，输出Mutex ID 2/3分别保护Ping/Pong的`z`。
- 再次搬入同组输入前，MTE2等待上次Vector读取完成；再次计算同组输出前，Vector等待上次MTE3搬出完成。
- 输入和输出使用独立子缓冲区，同组输入搬入和输出搬出可重叠；缓冲区容量、搬运长度和尾块处理使用样例中的常量与配置。
- 核函数（Kernel）退出前通过`asc_sync_pipe(PIPE_ALL)`等待所有流水操作完成。

<!-- end id1 -->

### Basic API：基于TPipe/TQue的编程方式

采用`TPipe/TQue`管理资源时，通过在算子类的`Init`函数中调用`InitBuffer`初始化队列，并设置内存块数量，控制是否开启Double Buffer。`num`参数设置为1时，每个队列分配一个内存块，不开启Double Buffer；设置为2时，每个队列分配两个内存块，开启Double Buffer。

【反例】

`InitBuffer`的`num`参数设置为1，输入和输出队列均只分配一个内存块，数据搬入、计算和数据搬出不能跨数据块重叠。

```cpp
__aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
{
    src0Global.SetGlobalBuffer((__gm__ half*)src0Gm);
    src1Global.SetGlobalBuffer((__gm__ half*)src1Gm);
    dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
    // 不开启Double Buffer，占用的物理空间是1 * sizeSrc0 * sizeof(half)。
    // 三个InitBuffer执行后总空间为：
    // 1 * (sizeSrc0 * sizeof(half) + sizeSrc1 * sizeof(half) + sizeDst0 * sizeof(half))。
    pipe.InitBuffer(inQueueSrc0, 1, sizeSrc0 * sizeof(half));
    pipe.InitBuffer(inQueueSrc1, 1, sizeSrc1 * sizeof(half));
    pipe.InitBuffer(outQueueDst, 1, sizeDst0 * sizeof(half));
}

__aicore__ inline void Process()
{
    // 需要round * 2次循环才能处理完数据。
    for (uint32_t index = 0; index < round * 2; ++index) {
        CopyIn(index);
        Compute();
        CopyOut(index);
    }
}
```

【正例】

在`Init`函数初始化输入和输出队列时，将`InitBuffer`的`num`参数设置为2。`Process`函数按照数据块顺序调用`CopyIn`、`Compute`和`CopyOut`，队列根据内存块的入队、出队和释放顺序管理缓冲区复用及流水同步。

```cpp
__aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
{
    src0Global.SetGlobalBuffer((__gm__ half*)src0Gm);
    src1Global.SetGlobalBuffer((__gm__ half*)src1Gm);
    dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
    // InitBuffer中使用2表示开启Double Buffer，占用的物理空间是2 * sizeSrc0 * sizeof(half)。
    // 三个InitBuffer执行后总空间为：
    // 2 * (sizeSrc0 * sizeof(half) + sizeSrc1 * sizeof(half) + sizeDst0 * sizeof(half))。
    pipe.InitBuffer(inQueueSrc0, 2, sizeSrc0 * sizeof(half));
    pipe.InitBuffer(inQueueSrc1, 2, sizeSrc1 * sizeof(half));
    pipe.InitBuffer(outQueueDst, 2, sizeDst0 * sizeof(half));
}

__aicore__ inline void Process()
{
    // 开启Double Buffer的前提是循环次数大于等于2。
    for (uint32_t index = 0; index < round; ++index) {
        CopyIn(index);
        Compute();
        CopyOut(index);
    }
}
```

采用该方式时，需要注意以下事项：

- `InitBuffer`应在算子类的`Init`函数中调用，三个参数依次表示队列、内存块数量和单个内存块的字节数。
- 开启Double Buffer后，输入和输出队列占用的UB空间增加为原来的两倍，需要重新计算单个数据块的长度和循环次数。
- `Process`中的循环次数取决于Tiling切分结果。上述`round`仅用于展示开启前后的代码位置，不能作为所有算子的固定计算方式。
- 如果尾块不能满足主块的Double Buffer切分条件，应单独处理尾块。

完整的Tiling、队列初始化和主尾块处理方法请参见[Double Buffer场景](../../simd_operator_impl/vector_programming_basic_api/double_buffer_scenario.md)。

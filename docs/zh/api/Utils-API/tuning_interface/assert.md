# assert/ascendc_assert

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
- Atlas推理系列产品AI Core：支持
<!-- end id5 -->
<!-- npu="310p" id6 -->
- Atlas推理系列产品Vector Core：支持
<!-- end id6 -->
<!-- npu="910" id7 -->
- Atlas训练系列产品：不支持
<!-- end id7 -->
<!-- npu="x90" id8 -->
- KirinX90处理器系列产品：不支持
<!-- end id8 -->
<!-- npu="9030" id9 -->
- Kirin9030处理器系列产品：不支持
<!-- end id9 -->
<!-- @ref: asc-devkit/res/docs/zh/api/Utils-API/tuning_interface/assert_res.md#id1 -->

不同产品对各编程场景的支持情况如下表所示。

**表1**  支持能力

<table>
    <tr>
        <th>产品</th>
        <th>SIMD</th>
        <th>SIMT VF</th>
        <th>SIMD VF</th>
    </tr>
    <!-- npu="950" id10 -->
    <tr>
        <td>Ascend 950PR&950DT系列产品</td>
        <td>支持</td>
        <td>支持</td>
        <td>支持</td>
    </tr>
    <!-- end id10 -->
    <!-- npu="A3" id11 -->
    <tr>
        <td>Atlas A3系列产品</td>
        <td>支持</td>
        <td>不支持</td>
        <td>不支持</td>
    </tr>
    <!-- end id11 -->
    <!-- npu="910b" id12 -->
    <tr>
        <td>Atlas A2系列产品</td>
        <td>支持</td>
        <td>不支持</td>
        <td>不支持</td>
    </tr>
    <!-- end id12 -->
    <!-- npu="310p" id13 -->
    <tr>
        <td>Atlas推理系列产品AI Core</td>
        <td>支持</td>
        <td>不支持</td>
        <td>不支持</td>
    </tr>
    <!-- end id13 -->
    <!-- npu="310p" id14 -->
    <tr>
        <td>Atlas推理系列产品Vector Core</td>
        <td>支持</td>
        <td>不支持</td>
        <td>不支持</td>
    </tr>
    <!-- end id14 -->
</table>

## 功能说明

头文件路径为：`"utils/debug/asc_assert.h"`。

`assert`和`ascendc_assert`用于在Device侧代码中进行断言检查。断言条件成立时，后续代码继续执行；断言条件不成立时，接口打印断言失败信息并触发异常，算子执行失败。

若需要在断言失败时输出自定义错误信息，可以在条件后传入格式字符串和对应参数。

- SIMD编程场景和SIMD VF编程场景下`assert`和`ascendc_assert`提供相同的断言能力，可以任选其一使用。
- SIMT编程场景和SIMT VF编程场景下`assert`只支持不带消息的形式，不输出fmt和args指定的自定义错误信息；`ascendc_assert`支持输出`fmt`和`args`指定的自定义错误信息。

> [!CAUTION]注意
> 该接口用于调测，使用时会影响算子性能，生产环境建议通过设置[ASCENDC_DUMP=0](../../SIMD-API/basic_api/debug_interface/disable_ascendc_dump.md)关闭。

## 函数原型

```cpp
assert(expr)
assert(expr, fmt, args...)

ascendc_assert(expr)
ascendc_assert(expr, fmt, args...)
```

## 参数说明

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| expr | 输入 | 断言条件。为`true`时，核函数（Kernel）继续运行；为`false`时，接口打印断言失败信息并触发异常。 |
| fmt | 输入 | 可选。自定义错误信息的格式控制字符串，包含普通字符和转换说明。普通字符原样输出；转换说明以百分号（%）开始，用于控制`args`中对应参数的转换和输出。支持的转换类型和数据类型参见[printf](printf.md)的参数说明。 |
| args | 输入 | 可选。与`fmt`中的转换说明对应的附加参数。每个参数替换`fmt`中对应的转换说明，参数数量和类型应与转换说明匹配。 |

## 返回值说明

无

## 约束说明

### 通用约束

- 在CMake工程的Release模式下，默认使能`NDEBUG`宏定义，本接口不生效。
- 使用`assert`时，如需包含标准库头文件`<cassert>`，请先包含`<cassert>`，再包含`"kernel_operator.h"`或`"utils/debug/asc_assert.h"`，避免标准库覆盖Ascend C的`assert`定义。

### SIMD编程场景

- CPU域调试时，多参数调用仅检查断言条件，不打印`fmt`和`args`指定的自定义错误信息，行为与单参数调用相同。
- 单次调用本接口打印的数据总量不可超过打印大小限制，默认为32KB。超出限制时，断言失败信息不会打印，但接口仍会触发异常。可以通过[aclInit接口](https://gitcode.com/cann/runtime/blob/master/docs/zh/api_ref/02_initialization_and_deinitialization.md#aclinit)中的`simd_printf_fifo_size_per_core`字段配置，配置范围为`[1KB,64MB]`。PyTorch调用和算子入图场景暂不支持该配置。

### SIMD VF编程场景

- CPU域调试时，多参数调用仅检查断言条件，不打印`fmt`和`args`指定的自定义错误信息，行为与单参数调用相同。
- `fmt`可以直接使用字符串字面量。使用字符串指针时，该指针必须指向Unified Buffer（UB）中的字符串，示例如下：

    ```cpp
    __ubuf__ const char* fmt = "value is %d.\n";
    assert(value == 6, fmt, value);
    ```

- 每个AIV核在单次`asc_vf_call`执行期间使用2KB预留UB空间临时保存调测数据。同一次`asc_vf_call`中的`assert`、`ascendc_assert`、`printf`和`asc_dump`共享该空间。该空间中的数据传输完成后会被复用，因此上述接口产生的累计调测数据可以超过2KB。单条断言失败信息必须能完整保存在该空间中，否则该条信息不会打印，但接口仍会触发异常。
- SIMD VF场景下，本接口需要使用2KB预留UB空间传递断言信息。开启`--cce-disable-asc-reserved-ubuf`选项后，本接口不可用。
- SIMD VF场景下，`simd_printf_fifo_size_per_core`建议配置为3KB以上。配置过小且打印数据量较大时，断言失败信息不会被打印，但接口仍会触发异常。

<!-- npu="950" id15 -->
### SIMT编程场景/SIMT VF编程场景

- 不支持CPU域调试。
- 断言失败信息通过Global Memory中的缓存空间输出，与[printf](printf.md)共享该缓存空间，缓存空间大小默认为2MB。
- 本接口会引入一定的性能开销，开销大小受断言调用次数、线程规模和Kernel构成等因素影响。SIMT编程场景、SIMT VF编程场景下，使用单条assert的Kernel相较于不使用assert的Kernel，性能劣化小于10%。各线程规模与断言数量下的实测开销参考值见[附录](#simt编程场景assert性能)，附录数据为特定测试配置下的观测值，可在评估Kernel性能时酌情参考。
<!-- end id15 -->

## 调用示例

### SIMD编程场景

条件不满足时，接口打印AIC/AIV类型、Block号、源码位置、函数名和条件表达式。传入格式字符串和对应参数时，还会打印自定义错误信息。随后，接口触发异常。

以下示例中，`value`为7，不满足`value == 6`，因此触发断言。

```cpp
extern "C" __global__ __vector__ void AssertSimdKernel()
{
    int32_t value = 7;
    AscendC::printf("before assert, value is %d.\n", value);
    assert(value == 6, "value is %d.\n", value);
}
```

程序运行时会触发断言，实际打印如下：

```plain
[AIV Block 0/1] before assert, value is 7.
[AIV Block 0/1] [ASSERT] .../assert_simd.asc:10: void AssertSimdKernel(): Assertion 'value == 6' failed. value is 7.
```

### SIMD VF编程场景

条件不成立时，接口打印源码位置、函数名和条件表达式；如果传入了格式字符串和对应参数，自定义错误信息会追加在该断言失败信息之后。随后，接口触发异常。

```cpp
__simd_vf__ inline void AssertSimdVf()
{
    int32_t value = 7;
    assert(value == 6, "value is %d.\n", value);
}
```

程序运行时会触发断言，打印格式如下：

```plain
[ASSERT] .../assert_simd_vf.asc:12: void AssertSimdVf(): Assertion 'value == 6' failed. value is 7.
```

### SIMT VF编程场景

-   调用`assert`，传入`expr`，当条件不成立时，接口打印源码位置、函数名和条件表达式，随后接口触发异常：

    ```cpp
    __simt_vf__ __launch_bounds__(1024) inline void simt_kernel(__gm__ float* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        assert(!isnan(x[idx]));
    }
    ```

    程序运行时会触发断言，打印效果如下：

    ```plain
    [ASSERT] /home/.../assert_simt_vf.asc:44: void simt_kernel(__gm__ float *): Assertion `!isnan(x[idx])' failed.
    ```

-   调用`ascendc_assert`，传入`expr`、`fmt`和`args`，当条件不成立时，接口在打印源码位置、函数名和条件表达式之后，追加打印自定义错误信息，随后接口触发异常：

    ```cpp
    __simt_vf__ __launch_bounds__(THREAD_COUNT) inline void simt_check(__gm__ int32_t* data, __ubuf__ float* local)
    {
        if (threadIdx.x >= ELEM_COUNT) {
            return;
        }
        if (threadIdx.x == 0) {
            int32_t v = data[0];
            ascendc_assert(v >= 0, "[simt] value %d must be non-negative (tid=%u)\n", v, threadIdx.x);
        }
        local[threadIdx.x] = static_cast<float>(threadIdx.x);
    }
    ```

    程序运行时会触发断言，打印效果如下：

    ```plain
    [ASSERT] /home/.../assert_simt_vf.asc:43: void simt_check(__gm__ int32_t *, __ubuf__ float *): Assertion `v >= 0' failed. [simt] value -5 must be non-negative (tid=0).
    ```

### SIMT编程场景

-   调用`assert`，传入`expr`，当条件不成立时，接口打印源码位置、函数名和条件表达式，随后接口触发异常：

    ```cpp
    __global__ __launch_bounds__(1024) inline void simt_kernel(float* x)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        assert(!isnan(x[idx]));
    }
    ```

    程序运行时会触发断言，打印效果如下：

    ```plain
    [ASSERT] /home/.../assert_simt.asc:44: void simt_kernel(float *): Assertion `!isnan(x[idx])' failed.
    ```

-   调用`ascendc_assert`，传入`expr`、`fmt`和`args`，当条件不成立时，接口在打印源码位置、函数名和条件表达式之后，追加打印自定义错误信息，随后接口触发异常：

    ```cpp
    constexpr uint32_t MIN_SHAPE = 64;

    __global__ void simt_assert(float* input, uint32_t in_shape)
    {
        ascendc_assert(in_shape >= MIN_SHAPE, "in_shape %u must be >= %u.\n", in_shape, MIN_SHAPE);
    }
    ```

    程序运行时会触发断言，打印效果如下：

    ```plain
    [ASSERT] /home/.../assert_simt.asc:52: void simt_assert(float *, uint32_t): Assertion `in_shape >= MIN_SHAPE' failed. in_shape 32 must be >= 64.
    ```

详细示例请参考[断言打印样例](../../../../../examples/03_simt_api/01_utilities/01_assert/README.md)。

<!-- npu="950" id16 -->
## 附录

### SIMT编程场景assert性能

以下为SIMT编程场景下的assert接口的实测性能数据，均测量断言通过时的开销。作为参照，同时测量了语义等价的手写`if (!cond) __trap()`的结果，即[\_\_trap接口](__trap.md)，下表简称__trap；对照组为不含断言的Kernel。每组数据取多次采样的平均值，实际开销与Kernel的计算、访存构成有关，以下数据供性能评估参考。

在Ascend 950PR产品上执行如下核函数代码获取如下性能数据。Kernel以单Block启动，线程数即下表中各列的线程规模。核函数如下：

```cpp
// 编译期宏三选一定义被测变体：
//   定义GATHER_ASSERT -> assert组；定义GATHER_TRAP -> __trap组；均不定义 -> 对照组
#ifdef GATHER_ASSERT
#define CHECK(cond) assert(cond)
#elif defined(GATHER_TRAP)
#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            __trap(); \
        } \
    } while (0)
#else
#define CHECK(cond) ((void)(cond))
#endif

template <int THREADS, int WIDTH>
__global__ __launch_bounds__(THREADS) void task_gather(float* input, uint32_t* index, float* output,
    uint32_t in_rows, uint32_t index_len)
{
    uint32_t out_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_row >= index_len) {
        return;
    }
    uint32_t in_row = index[out_row];
    uint32_t in_total = in_rows * WIDTH;
    uint32_t out_total = index_len * WIDTH;
    for (uint32_t col = 0; col < WIDTH; col++) {
        uint32_t input_idx = in_row * WIDTH + col;
        uint32_t output_idx = out_row * WIDTH + col;
        bool flag = (input_idx < in_total) && (output_idx < out_total) && __isfinite(input[input_idx]);
        CHECK(flag);
        output[output_idx] = input[input_idx];
    }
}
```

断言通过时各线程规模下的平均耗时如下，单位为us。单元格内三个数值依次为assert、__trap、对照组的耗时：

| 断言次数 | 32线程 | 64线程 | 128线程 | 256线程 | 512线程 | 1024线程 | 2048线程 |
|---|---|---|---|---|---|---|---|
| 1 | 1.808/1.715/1.675 | 1.747/1.735/1.715 | 1.815/1.797/1.745 | 1.993/2.013/1.947 | 2.117/2.082/2.028 | 2.205/2.180/2.065 | 2.317/2.312/2.202 |
| 10 | 2.987/3.035/2.437 | 2.967/3.088/2.562 | 3.097/3.062/2.903 | 3.665/3.797/3.357 | 5.188/5.250/4.573 | 8.613/8.283/7.393 | 24.255/18.158/14.133 |
| 50 | 11.052/10.607/8.420 | 12.183/11.527/9.170 | 15.565/14.920/12.830 | 25.863/25.173/21.565 | 58.193/57.348/55.635 | 192.797/197.992/143.702 | 446.747/400.637/283.445 |
| 100 | 20.862/19.405/14.228 | 23.012/21.650/16.763 | 29.327/28.427/23.260 | 56.492/55.872/47.762 | 134.682/134.305/117.195 | 380.883/386.450/270.468 | 894.965/799.477/554.070 |
| 200 | 36.387/33.810/23.058 | 43.478/40.062/30.568 | 60.358/59.007/47.900 | 118.100/117.657/97.740 | 284.813/285.177/233.223 | 767.997/780.993/555.575 | 1786.717/1602.532/1098.692 |

相比于对照组增加的净开销百分比如下。单元格内两个数值依次为assert、__trap相比于对照组增加的净开销：

| 断言次数 | 32线程 | 64线程 | 128线程 | 256线程 | 512线程 | 1024线程 | 2048线程 |
|---|---|---|---|---|---|---|---|
| 1 | +7.9/+2.4 | +1.8/+1.2 | +4.0/+3.0 | +2.4/+3.4 | +4.4/+2.6 | +6.8/+5.6 | +5.2/+5.0 |
| 10 | +22.6/+24.6 | +15.8/+20.6 | +6.7/+5.5 | +9.2/+13.1 | +13.4/+14.8 | +16.5/+12.0 | +71.6/+28.5 |
| 50 | +31.3/+26.0 | +32.9/+25.7 | +21.3/+16.3 | +19.9/+16.7 | +4.6/+3.1 | +34.2/+37.8 | +57.6/+41.3 |
| 100 | +46.6/+36.4 | +37.3/+29.2 | +26.1/+22.2 | +18.3/+17.0 | +14.9/+14.6 | +40.8/+42.9 | +61.5/+44.3 |
| 200 | +57.8/+46.6 | +42.2/+31.1 | +26.0/+23.2 | +20.8/+20.4 | +22.1/+22.3 | +38.2/+40.6 | +62.6/+45.9 |

<!-- end id16 -->

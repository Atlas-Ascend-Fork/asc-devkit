# 矩阵编程（C API）

> [!NOTE]说明
> 本文针对NPU架构版本3510的分离模式，使用C API进行矩阵乘法编程；示例包含头文件`"c_api/asc_simd.h"`。

针对[分离模式](../../programming_guide/advanced_programming/hardware_implementation/basic_architecture.md#section1574769433)，由于硬件架构差异，在使用C API进行矩阵乘法算子的实现上存在差异：
- 分离模式中，Cube计算单元与Vector计算单元分离部署，每个核有自己的Scalar单元，能独立加载自己的代码段。在核函数（Kernel）编写时，可通过函数类型限定符\_\_cube\_\_标识该核函数（Kernel）在设备端aicore上的Cube计算单元执行。
- 分离模式中支持Fixpipe硬件加速，支持从L0C Buffer直接搬出数据到L1 Buffer/Global Memory/Unified Buffer（UB）的同时，完成量化、反量化、数据排布格式转换等功能。

遵循算子分析、核函数（Kernel）定义、算子类实现的开发流程，下面以Matmul算子为例，给出分离模式下Matmul算子的代码框架。

## 算子分析

Matmul算子完成形状为\[m, k\]的矩阵a和形状为\[k, n\]的矩阵b的矩阵乘，得到形状为\[m, n\]的矩阵c。为了方便这里取m=k=n=32。Matmul算子的输入输出、计算逻辑以及需要调用的Ascend C接口如下表所示。

**表1** Ascend C Matmul算子设计规格

| 项目 | 内容 |
|------|------|
| 算子类型（OpType） | Matmul |
| 算子输入a | shape：(m, k) = (32, 32)；data type：half；format：[ND](../../technical_appendix/concepts_and_terms/neural_networks_and_operators/data_layout.md) |
| 算子输入b | shape：(k, n) = (32, 32)；data type：half；format：[ND](../../technical_appendix/concepts_and_terms/neural_networks_and_operators/data_layout.md) |
| 算子输出c | shape：(m, n) = (32, 32)；data type：half；format：[ND](../../technical_appendix/concepts_and_terms/neural_networks_and_operators/data_layout.md) |
| 核函数（Kernel）名称 | mmad_custom |
| 使用的主要接口 | [asc_copy_gm2l1_nd2nz](../../../api/SIMD-API/c_api/cube_datamove/cube_compute_load/asc_copy_gm2l1_nd2nz_arch_3510.md)：Global Memory到L1 Buffer数据搬运 + ND转NZ格式转换接口<br>[asc_copy_l12l0a](../../../api/SIMD-API/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0a_2d_arch_3510.md)、[asc_copy_l12l0b_transpose](../../../api/SIMD-API/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0b_2d_arch_3510.md)：L1 Buffer到L0 Buffer数据搬运 + NZ转NZ/ZN格式转换接口<br>[asc_mmad](../../../api/SIMD-API/c_api/cube_compute/mmad_compute/asc_mmad.md)：矩阵乘计算接口<br>[asc_copy_l0c2gm](../../../api/SIMD-API/c_api/cube_datamove/cube_compute_store/asc_copy_l0c2gm_arch_3510.md)：L0C Buffer到Global Memory数据搬出 + NZ转ND格式转换 + 精度转换接口 |

GM到L1 Buffer搬入前，使用[asc_set_gm2l1_nz_para](../../../api/SIMD-API/c_api/cube_datamove/cube_load_aux_config/asc_set_gm2l1_nz_para.md)配置ND到NZ的目标步长；L0C Buffer到GM搬出前，使用[asc_set_l0c_copy_nz_para](../../../api/SIMD-API/c_api/cube_datamove/cube_store_aux_config/asc_set_l0c_copy_nz_para.md)配置NZ转换参数。

## 核函数（Kernel）定义

根据[核函数（Kernel）](../../programming_guide/programming_model/ai_core_simd_programming/kernel_function.md)中介绍的规则进行核函数（Kernel）的定义。核函数（Kernel）名为mmad\_custom，有3个参数a，b，c，其中a，b都为输入内存，c为输出内存。使用函数类型限定符\_\_global\_\_来标识它是一个核函数（Kernel），可以被<<<\>\>\>调用；使用函数类型限定符\_\_cube\_\_来标识该核函数（Kernel）在设备端aicore上的Cube核执行。

核函数（Kernel）中，算子类的Init函数，完成内存初始化相关工作，Process函数完成算子实现的核心逻辑。核函数（Kernel）在开始时调用`asc_init()`初始化硬件状态，结束时调用`asc_sync()`等待所有指令完成。

```cpp
template <uint32_t M, uint32_t K, uint32_t N>
__global__ __cube__ void mmad_custom(__gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c)
{
    asc_init();
    KernelMatmul<M, K, N> op;
    op.Init(a, b, c);
    op.Process();
    asc_sync();
}
```

> [!NOTE]说明
> 核函数（Kernel）使用模板参数传入矩阵的维度信息，这样可以在编译期确定循环次数，有利于编译器优化。其中M/K/N为总矩阵维度。

## 算子类实现

在分离模式中，一个Matmul矩阵乘算子的实现分为以下四步流水。

1.  CopyIn任务（Global Memory --> L1 Buffer）  
    使用[asc_copy_gm2l1_nd2nz](../../../api/SIMD-API/c_api/cube_datamove/cube_compute_load/asc_copy_gm2l1_nd2nz_arch_3510.md)接口将Global Memory中的数据搬运到L1 Buffer，将ND格式转换为NZ格式。

2.  Split任务（L1 Buffer--> L0 Buffer）    
    使用[asc_copy_l12l0a](../../../api/SIMD-API/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0a_2d_arch_3510.md)、[asc_copy_l12l0b_transpose](../../../api/SIMD-API/c_api/cube_datamove/cube_compute_load/asc_copy_l12l0b_2d_arch_3510.md)接口将数据从L1 Buffer中搬运到L0A Buffer/L0B Buffer，同时完成NZ到NZ/ZN的格式转换。

3.  Compute任务（矩阵乘计算）    
    使用[asc_mmad](../../../api/SIMD-API/c_api/cube_compute/mmad_compute/asc_mmad.md)接口在完成矩阵乘计算，结果存放在L0C Buffer。

4.  CopyOut任务（L0C Buffer--> Global Memory）    
    使用[asc_copy_l0c2gm](../../../api/SIMD-API/c_api/cube_datamove/cube_compute_store/asc_copy_l0c2gm_arch_3510.md)接口将结果从L0C Buffer直接搬运到Global Memory，同时利用Fixpipe的随路格式转换功能将NZ格式转回ND格式。

一个典型的Matmul算子类的代码框架如下所示。

```cpp
template <uint32_t M, uint32_t K, uint32_t N>
class KernelMatmul {
public:
    __aicore__ inline KernelMatmul() {}
    __aicore__ inline void Init(__gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c)
    {
        aGM = (__gm__ half*)a;
        bGM = (__gm__ half*)b;
        cGM = (__gm__ half*)c;
    }

    __aicore__ inline void Process()
    {
        __cbuf__ half a1Local[M * K];
        __ca__ half a2Local[M * K];
        __cbuf__ half b1Local[K * N];
        __cb__ half b2Local[K * N];
        __cc__ float cLocal[M * N];

        CopyInA(a1Local);
        CopyInB(b1Local);

        asc_sync();

        DataLoadA(a1Local, a2Local);
        DataLoadB(b1Local, b2Local);

        asc_sync();

        Compute(cLocal, a2Local, b2Local);

        asc_sync();
        CopyOut(cLocal);
    }

private:
    // 搬入函数，完成矩阵A从GM到L1 Buffer的CopyIn任务。
    __aicore__ inline void CopyInA(__cbuf__ half* a1Local)
    {
        asc_set_gm2l1_nz_para(1, 1, M, 0);
        asc_copy_gm2l1_nd2nz(a1Local, aGM, K * sizeof(half),
            asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, M, K, 0, false);
    }
    // 搬入函数，完成矩阵B从GM到L1 Buffer的CopyIn任务。
    __aicore__ inline void CopyInB(__cbuf__ half* b1Local)
    {
        asc_set_gm2l1_nz_para(1, 1, K, 0);
        asc_copy_gm2l1_nd2nz(b1Local, bGM, N * sizeof(half),
            asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, K, N, 0, false);
    }
    // 搬入函数，完成矩阵A从L1 Buffer到L0A Buffer的Split任务。
    __aicore__ inline void DataLoadA(__cbuf__ half* a1Local, __ca__ half* a2Local)
    {
        asc_copy_l12l0a(a2Local, a1Local, 0, 0, M / 16, K / 16, M / 16, M / 16);
    }
    // 搬入函数，完成矩阵B从L1 Buffer到L0B Buffer的Split任务。
    __aicore__ inline void DataLoadB(__cbuf__ half* b1Local, __cb__ half* b2Local)
    {
        asc_copy_l12l0b_transpose(b2Local, b1Local, 0, 0, K / 16, N / 16, K / 16, N / 16);
    }
    // 计算函数，完成L0上的计算任务
    __aicore__ inline void Compute(__cc__ float* cLocal, __ca__ half* a2Local, __cb__ half* b2Local)
    {
        asc_mmad(cLocal, a2Local, b2Local, M, K, N, asc_unit_flag_mode::DISABLE, true, false, true);
    }
    // 搬出函数，完成矩阵C从L0C Buffer到GM的CopyOut任务。
    __aicore__ inline void CopyOut(__cc__ float* cLocal)
    {
        asc_set_l0c_copy_nz_para(1, 0, 0);
        asc_copy_l0c2gm(cGM, cLocal, N, M, N, M,
            asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, asc_unit_flag_mode::DISABLE,
            QuantMode_t::F322F16, asc_relu_pre_mode::NONE, false, true, false, false);
    }

private:
    // 私有成员变量
    __gm__ half *aGM, *bGM, *cGM;
};
```

> [!NOTE]说明
> CopyIn、Split、Compute和CopyOut四个阶段的操作分别在不同的硬件流水线上执行。为保证数据依赖正确——即每个阶段读取的数据必须由前一阶段写入完毕，需要在各阶段之间设置同步屏障。这里使用全局流水同步[asc_sync](../../../api/SIMD-API/c_api/sync/intra_core_sync/asc_sync.md)。

本例使用NPU架构版本3510的接口，包含头文件`"c_api/asc_simd.h"`；实际源文件中须在核函数入口前定义KernelMatmul，并以M=K=N=32实例化。四个阶段均保留显式数据通路：GM->L1 Buffer、L1 Buffer->L0A Buffer/L0B Buffer、矩阵乘计算及L0C Buffer->GM。L0A Buffer采用NZ，L0B Buffer采用ZN，外部输入输出仍为half、ND格式；搬出使用F322F16完成float到half转换。

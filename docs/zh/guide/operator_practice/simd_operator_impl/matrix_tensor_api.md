# 矩阵编程（Tensor API）

> [!NOTE]说明
> 本节内容为针对分离模式，使用Tensor API进行矩阵乘法的编程指导。

针对[分离模式](../../programming_guide/advanced_programming/hardware_implementation/basic_architecture.md#section1574769433)，由于硬件架构差异，在使用Tensor API进行矩阵乘法算子的实现上存在差异：
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
| 使用的主要接口 | [copy（GM到L1）](../../../api/SIMD-API/tensor_api/matrix_compute/cube_compute_load/copy_gm_to_l1.md)：Global Memory到L1 Buffer数据搬运 + ND转NZ格式转换接口<br>[copy（L1到L0A）](../../../api/SIMD-API/tensor_api/matrix_compute/cube_compute_load/copy_l1_to_l0a.md)、[copy（L1到L0B）](../../../api/SIMD-API/tensor_api/matrix_compute/cube_compute_load/copy_l1_to_l0b.md)：L1 Buffer到L0 Buffer数据搬运 + NZ转NZ/ZN格式转换接口<br>[mmad](../../../api/SIMD-API/tensor_api/matrix_compute/mmad_compute/mmad.md)：矩阵乘计算接口<br>[copy（L0C到GM）](../../../api/SIMD-API/tensor_api/matrix_compute/cube_compute_store/copy_l0c_to_gm.md)：L0C Buffer到Global Memory数据搬出 + NZ转ND格式转换 + 精度转换接口 |

## 核函数（Kernel）定义

根据[核函数（Kernel）](../../programming_guide/programming_model/ai_core_simd_programming/kernel_function.md)中介绍的规则进行核函数（Kernel）的定义。核函数（Kernel）名为mmad\_custom，有3个参数a，b，c，其中a，b都为输入内存，c为输出内存。使用函数类型限定符\_\_global\_\_来标识它是一个核函数（Kernel），可以被<<<\>\>\>调用；使用函数类型限定符\_\_cube\_\_来标识该核函数（Kernel）在设备端aicore上的Cube核执行。

核函数（Kernel）中，算子类的Init函数，完成内存初始化相关工作，Process函数完成算子实现的核心逻辑。核函数（Kernel）在开始时调用`asc::te::init_soc_state()`初始化硬件状态，结束时调用`asc::te::sync_pipe(pipe_t::PIPE_ALL)`等待所有指令完成。

```cpp
template <uint32_t M, uint32_t K, uint32_t N>
__global__ __cube__ void mmad_custom(__gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c)
{
    asc::te::init_soc_state();
    KernelMatmul<M, K, N> op;
    op.Init(a, b, c);
    op.Process();
    asc::te::sync_pipe(pipe_t::PIPE_ALL);
}
```

> [!NOTE]说明
> 核函数（Kernel）使用模板参数传入矩阵的维度信息，这样可以在编译期确定循环次数，有利于编译器优化。其中M/K/N为总矩阵维度。

## 算子类实现

在分离模式中，一个Matmul矩阵乘算子的实现分为以下四步流水。

1.  CopyIn任务（Global Memory --> L1 Buffer）  
    使用[copy（GM到L1）](../../../api/SIMD-API/tensor_api/matrix_compute/cube_compute_load/copy_gm_to_l1.md)接口将Global Memory中的数据搬运到L1 Buffer，将ND格式转换为NZ格式。

2.  Split任务（L1 Buffer--> L0 Buffer）    
    使用[copy（L1到L0A）](../../../api/SIMD-API/tensor_api/matrix_compute/cube_compute_load/copy_l1_to_l0a.md)、[copy（L1到L0B）](../../../api/SIMD-API/tensor_api/matrix_compute/cube_compute_load/copy_l1_to_l0b.md)接口将数据从L1 Buffer中搬运到L0A Buffer/L0B Buffer，同时完成NZ到NZ/ZN的格式转换。

3.  Compute任务（矩阵乘计算）    
    使用[mmad](../../../api/SIMD-API/tensor_api/matrix_compute/mmad_compute/mmad.md)接口完成矩阵乘计算，结果存放在L0C Buffer。

4.  CopyOut任务（L0C Buffer--> Global Memory）    
    使用[copy（L0C到GM）](../../../api/SIMD-API/tensor_api/matrix_compute/cube_compute_store/copy_l0c_to_gm.md)接口将结果从L0C Buffer直接搬运到Global Memory，同时利用Fixpipe的随路格式转换功能将NZ格式转回ND格式。

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
        using namespace asc::te;
        __cbuf__ half a1Local[M * K], b1Local[K * N];
        __ca__ half a2Local[M * K];
        __cb__ half b2Local[K * N];
        __cc__ float cLocal[M * N];
        auto a1 = make_tensor(make_mem_ptr(a1Local), make_frame_layout<nz_layout_ptn, half>(M, K));
        auto b1 = make_tensor(make_mem_ptr(b1Local), make_frame_layout<nz_layout_ptn, half>(K, N));
        auto a2 = make_tensor(make_mem_ptr(a2Local), make_frame_layout<nz_layout_ptn, half>(M, K));
        auto b2 = make_tensor(make_mem_ptr(b2Local), make_frame_layout<zn_layout_ptn, half>(K, N));
        auto result = make_tensor(make_mem_ptr(cLocal), make_frame_layout<nz_layout_ptn, 16>(M, N));
        CopyInA(a1);
        CopyInB(b1);
        sync_pipe(pipe_t::PIPE_ALL);
        DataLoadA(a1, a2);
        DataLoadB(b1, b2);
        sync_pipe(pipe_t::PIPE_ALL);
        Compute(result, a2, b2);
        sync_pipe(pipe_t::PIPE_ALL);
        CopyOut(result);
    }
private:
    template <typename Tensor>
    __aicore__ inline void CopyInA(Tensor a1)
    {
        using namespace asc::te;
        auto source = make_tensor(make_mem_ptr<location::gm>(aGM), make_frame_layout<nd_layout_ptn>(M, K));
        copy(a1, source);
    }
    template <typename Tensor>
    __aicore__ inline void CopyInB(Tensor b1)
    {
        using namespace asc::te;
        auto source = make_tensor(make_mem_ptr<location::gm>(bGM), make_frame_layout<nd_layout_ptn>(K, N));
        copy(b1, source);
    }
    template <typename L1Tensor, typename L0Tensor>
    __aicore__ inline void DataLoadA(L1Tensor a1, L0Tensor a2)
    {
        asc::te::copy(a2, a1);
    }
    template <typename L1Tensor, typename L0Tensor>
    __aicore__ inline void DataLoadB(L1Tensor b1, L0Tensor b2)
    {
        asc::te::copy(b2, b1);
    }
    template <typename CTensor, typename ATensor, typename BTensor>
    __aicore__ inline void Compute(CTensor c, ATensor a, BTensor b)
    {
        using namespace asc::te;
        auto atom = make_mmad(mmad_operation{}).with(
            asc::te::mmad_params{M, N, K, unit_flag_mode::disable, true});
        asc::te::mmad(atom, c, a, b);
    }
    template <typename Tensor>
    __aicore__ inline void CopyOut(Tensor c)
    {
        using namespace asc::te;
        auto destination = make_tensor(make_mem_ptr<location::gm>(cGM),
            make_frame_layout<nd_layout_ptn>(M, N));
        copy(make_copy(copy_l0c_to_gm{}), destination, c);
    }
    __gm__ half *aGM, *bGM, *cGM;
};
```

> [!NOTE]说明
> CopyIn、DataLoad、Compute和CopyOut四个阶段的操作分别在不同的硬件流水线上执行。为保证数据依赖正确——即每个阶段读取的数据必须由前一阶段写入完毕，需要在各阶段之间设置同步屏障。这里使用全局流水同步[sync_pipe](../../../api/SIMD-API/tensor_api/sync/sync_pipe.md)。

本节示例使用NPU架构版本3510的Tensor API完成数据搬运、矩阵计算、初始化和同步控制。Tensor API未覆盖的能力（如部分系统变量查询）可配合C API使用，接口分层与能力范围参见[Tensor API总览](../../../api/SIMD-API/tensor_api/tensor_api.md)。

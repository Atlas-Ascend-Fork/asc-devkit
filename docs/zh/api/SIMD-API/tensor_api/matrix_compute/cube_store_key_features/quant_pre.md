# 随路量化

随路量化是矩阵计算结果从L0C Buffer搬出到Global Memory或Unified Buffer（UB）过程中的数据转换能力。L0C Buffer到Global Memory/UB搬运根据是否传入`quant`参数自动选择不量化、直接cast、scalar量化或tensor量化模式。

## 调用形式

不传入`quant`参数时，接口根据源类型和目的类型选择不量化输出或直接cast输出：

```cpp
auto atom = make_copy(copy_l0c_to_gm{}).with(fixpipe_params);
copy(atom, dst, src);
```

传入`uint64_t`类型的`quant`参数时，表示scalar量化。整个C矩阵共用同一个量化参数：

```cpp
uint64_t quant = 1;
auto atom = make_copy(copy_l0c_to_gm{}).with(fixpipe_params);
copy(atom, dst, src, quant);
```

传入Tensor API张量作为`quant`参数时，表示tensor量化。量化参数通常位于L1 Buffer，元素类型为`uint64_t`，每个channel维度对应一个量化参数：

```cpp
auto quant = make_tensor(make_mem_ptr(quant_addr),
    make_frame_layout<nd_ext_layout_ptn>(1, n));
auto atom = make_copy(copy_l0c_to_gm{}).with(fixpipe_params);
copy(atom, dst, src, quant);
```

## 量化参数

scalar量化参数和tensor量化参数的元素类型均为`uint64_t`。

<!-- npu="950" id1 -->
针对Ascend 950PR&950DT系列产品，量化参数的bit位含义如下：

| bit位 | 含义 |
| :--- | :--- |
| 0~12 | 无效比特位，用户无需关注。 |
| 13~31 | M1。数据类型视为float，硬件以(1, 8, 10)的格式，即1个符号位、8个指数位和10个尾数位作为量化计算所需要乘的值。其中31位为符号位，23~30位为指数位，13~22位为尾数位。其值不能为inf/nan。 |
| 32~36 | 无效比特位，用户无需关注。 |
| 37~45 | Offset。9bit整形数据，源数据乘以量化系数或者随路系数的计算结果，可以与Offset表示的整数值进行相加。若不使用offset，请置为0。 |
| 46 | Sign标志位。如果置为1，表明量化结果是signed(int8)；如果为置为0，表明量化结果是unsigned(uint8)。 |
| 47~63 | 无效比特位，用户无需关注。 |
<!-- end id1 -->

## 约束说明

- tensor量化参数张量应位于L1 Buffer，元素类型为`uint64_t`，地址要求32字节对齐。
- 量化参数不能为`inf`、`nan`或非规格化数。该约束属于硬件约束，接口不单独检查参数值。
- `round_mode::hybrid`仅在源类型为`float`、目的类型为`hifloat8_t`的量化输出场景支持。
- 不传入`quant`时，仅支持不量化输出或`float`到`half`/`bfloat16_t`的直接cast输出。

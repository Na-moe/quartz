#### 前言

鉴于最近一系列有影响力的工作都用 `Triton` 进行了高性能的实现，决定趁寒假的空闲时间学习一下。

官网的 [Tutorial](https://triton-lang.org/main/getting-started/tutorials/index.html) 对于初学者而言稍微有些含混。在知乎遍览众多入门材料后，感觉 [Sasha Rush 教授](https://rush-nlp.com/) 的 [Triton-Puzzles](https://github.com/srush/Triton-Puzzles/) 比较有趣。在此基础上，[@SiriusNEO](https://www.zhihu.com/people/chaofanlin) 做了些改进（[Triton-Puzzles-Lite](https://zhuanlan.zhihu.com/p/5964285807)），大幅减少依赖并且提高了题面的清晰度，加之更方便的调试，使其对初学者更加友好。

在这里记录下自己做这些 Puzzle 的解答和一些思路。

#### Puzzle 1: Constant Add

>  Add a constant to a vector. Uses one program id axis. Block size `B0` is always the same as vector `x` with length `N0`.
>  
>  $z_i=10+x_i,\text{ for } i=1,\dots,N_0$
>  
> ![[triton-puzzles-01.png]]


练手题，主要帮助掌握 `tl.load` 和 `tl.store`。鉴于 `B0=N0`，也就不需要 program id axis 了，同理在 `load` 和 `store` 时也不需要考虑访问越界。

直接创建一个 `0` 到 `B0` 的 offset 即可。

```python
@triton.jit
def add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    # We name the offsets of the pointers as "off_"
    off_x = tl.arange(0, B0)
    x = tl.load(x_ptr + off_x)
    # Finish me!
    z = x + 10.0
    tl.store(z_ptr + off_x, z)
    return
```



#### Puzzle 2: Constant Add Block

>Add a constant to a vector. Uses one program block axis (no `for` loops yet). 
>
>Block size `B0` is now smaller than the shape vector `x` which is `N0`.
>
  $z_i = 10 + x_i, \text{ for } i = 1\ldots N_0$
  >
>![[triton-puzzles-02.png]]

在 `N0` 维度启动若干 threads，每个 thread 处理大小为 `B0` 的加法操作。

```python
@triton.jit
def add_mask2_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    # Finish me!
    pid = tl.program_id(0)
    off_x = pid * B0 + tl.arange(0, B0)
    mask = off_x < N0
    x = tl.load(x_ptr + off_x, mask=mask)
    z = x + 10.0
    tl.store(z_ptr + off_x, z, mask=mask)
    return
```



#### Puzzle 3: Outer Vector Add

>Add two vectors.
>
>Uses one program block axis. Block size `B0` is always the same as vector `x` length `N0`.
>
>Block size `B1` is always the same as vector `y` length `N1`.
>
  $z_{j, i} = x_i + y_j,\text{ for } i = 1\ldots B_0,\ j = 1\ldots B_1$
  >
> ![[triton-puzzles-03.png]]

形式类似于向量外积，两个一维向量生成了一个二维向量。

`PyTorch` 的示例代码中使用了 `broadcast` 机制：`x[None, :] + y[:, None]`。首先将 `x` 的 `shape` 从 `(N0)` 变为 `(1, N0)`，`y` 的 `shape` 从 `(N1)` 变为 `(N1, 1)`，然后在相加时触发广播机制，两者的 `shape` 均广播到 `(N1, N0)`。

类似地，在 `Triton` 中也存在这样的机制，我们利用这点得到 `z` 及其 `offset` 。

由于 `B0=N0, B1=N1`，略去`pid` 和 `mask`。

```python
@triton.jit
def add_vec_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    # Finish me!
    off_x = tl.arange(0, B0)
    off_y = tl.arange(0, B1)
    off_z = off_y[:, None] * N0 + off_x[None, :]
    x = tl.load(x_ptr + off_x)
    y = tl.load(y_ptr + off_y)
    z = x[None, :] + y[:, None]
    tl.store(z_ptr + off_z, z)
    return
```



#### Puzzle 4: Outer Vector Add Block

> Add a row vector to a column vector.
>
> Uses two program block axes. Block size `B0` is always less than the vector `x` length `N0`.
>
> Block size `B1` is always less than vector `y` length `N1`.
> 
  $z_{j, i} = x_i + y_j,\text{ for } i = 1\ldots N_0,\ j = 1\ldots N_1$
>
> ![[triton-puzzles-04.png]]

加上 `pid` 和 `mask` 即可：

```python
@triton.jit
def add_vec_block_kernel(
    x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    block_id_x = tl.program_id(0)
    block_id_y = tl.program_id(1)
    # Finish me!
    off_x = block_id_x * B0 + tl.arange(0, B0)
    off_y = block_id_y * B1 + tl.arange(0, B1)
    off_z = off_y[:, None] * N0 + off_x[None, :]
    mask_x = off_x < N0
    mask_y = off_y < N1
    mask_z = mask_y[:, None] & mask_x[None, :]

    x = tl.load(x_ptr + off_x, mask=mask_x)
    y = tl.load(y_ptr + off_y, mask=mask_y)
    z = x[None, :] + y[:, None]

    tl.store(z_ptr + off_z, z, mask=mask_z)
    return
```



#### Puzzle 5: Fused Outer Multiplication

>Multiply a row vector to a column vector and take a relu.
>
>Uses two program block axes. Block size `B0` is always less than the vector `x` length `N0`.
>
>Block size `B1` is always less than vector `y` length `N1`.
>
 $z_{j, i} = \text{relu}(x_i \times y_j),\text{ for } i = 1\ldots N_0,\ j = 1\ldots N_1$
 >
> ![[triton-puzzles-05.png]]

相比 Puzzle 4 外和变外积，又多了个 `relu` （$\text{relu}(x)=\text{maximum}(x,0)$）。

```python
@triton.jit
def mul_relu_block_kernel(
    x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    block_id_x = tl.program_id(0)
    block_id_y = tl.program_id(1)
    # Finish me!
    off_x = block_id_x * B0 + tl.arange(0, B0)
    off_y = block_id_y * B1 + tl.arange(0, B1)
    off_z = off_y[:, None] * N0 + off_x[None, :]
    mask_x = off_x < N0
    mask_y = off_y < N1
    mask_z = mask_y[:, None] & mask_x[None, :]

    x = tl.load(x_ptr + off_x, mask=mask_x)
    y = tl.load(y_ptr + off_y, mask=mask_y)
    z = x[None, :] * y[:, None]
    z = tl.maximum(z, 0)
    
    tl.store(z_ptr + off_z, z, mask=mask_z)
    return
```



#### Puzzle 6: Fused Outer Multiplication - Backwards

>Backwards of a function that multiplies a matrix with a row vector and take a relu.
>
>Uses two program blocks. Block size `B0` is always less than the vector `x` length `N0`.
>
>Block size `B1` is always less than vector `y` length `N1`. Chain rule backward `dz` is of shape `N1` by `N0`
>
   $f(x, y) = \text{relu}(x_{j, i} \times y_j),\text{ for } i = 1\ldots N_0,\ j = 1\ldots N_1$
 >
 > $dx_{j, i} = f_x'(x, y)_{j, i} \times dz_{j, i}$
 >
>
> ![[triton-puzzles-06.png]]

这题是算反向传播，根据链式法则有 $\mathrm{d}x_{j,i}=\text{relu}'(x_{j,i}\times y_{j})\times y_{j}\times \mathrm{d}z_{j,i}$ （$\text{relu}'(x)=\text{where}(x>0,1,0)$）。

```python
@triton.jit
def mul_relu_block_back_kernel(
    x_ptr, y_ptr, dz_ptr, dx_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    block_id_i = tl.program_id(0)
    block_id_j = tl.program_id(1)
    # Finish me!
    off_i = block_id_i * B0 + tl.arange(0, B0)
    off_j = block_id_j * B1 + tl.arange(0, B1)
    off_ji = off_j[:, None] * N0 + off_i[None, :]
    mask_i = off_i < N0
    mask_j = off_j < N1
    mask_ji = mask_j[:, None] & mask_i[None, :]

    x = tl.load(x_ptr + off_ji, mask=mask_ji)
    y = tl.load(y_ptr + off_j, mask=mask_j)
    dz = tl.load(dz_ptr + off_ji, mask=mask_ji)

    df = tl.where(x * y[:, None] > 0, 1.0, 0)
    dxy_x = y[:, None]
    dx = df * dxy_x * dz
    tl.store(dx_ptr + off_ji, dx, mask=mask_ji)
    return
```



#### Puzzle 7: Long Sum

>Sum of a batch of numbers.
>
>Uses one program blocks. Block size `B0` represents a range of batches of  `x` of length `N0`.
>
>Each element is of length `T`. Process it `B1 < T` elements at a time.  
>
  $z_{i} = \sum^{T}_j x_{i,j},  \text{ for } i = 1\ldots N_0$
 >
>Hint: You will need a for loop for this problem. These work and look the same as in Python.
>
> ![[triton-puzzles-07.png]]

在`axis=0`做并行，在`axis=1`做求和。参数 `N1` 似乎没用。

注意在 `axis=1` 维度每次处理 `B1` 个元素，需要用 `for-loop` 逐次累加。

```python
@triton.jit
def sum_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    # Finish me!
    block_id_i = tl.program_id(0)
    off_i = block_id_i * B0 + tl.arange(0, B0)
    mask_i = off_i < N0

    z = tl.zeros((B0,), dtype=tl.float32)

    for j in range(0, T, B1):
        off_j = tl.arange(j, j + B1)
        off_ij = off_i[:, None] * T + off_j[None, :]
        mask_j = off_j < T
        mask_ij = mask_i[None, :] & mask_j[None, :]
        x = tl.load(x_ptr + off_ij, mask=mask_ij)
        z += tl.sum(x, axis=1)
        
    tl.store(z_ptr + off_i, z, mask=mask_i)
    return
```



#### Puzzle 8: Long Softmax

> Softmax of a batch of logits.
>
> Uses one program block axis. Block size `B0` represents the batch of `x` of length `N0`.
>
> Block logit length `T`.  Process it `B1 < T` elements at a time.  
> 
   $z_{i, j} = \text{softmax}(x_{i,1} \ldots x_{i, T}), \text{ for } i = 1\ldots N_0$
>
> Note softmax needs to be computed in numerically stable form as in Python. In addition in Triton 
>
> they recommend not using `exp` but instead using `exp2`. You need the identity
> 
   $\exp(x) = 2^{\log_2(e) x}$
  >
> Advanced: there one way to do this with 3 loops. You can also do it with 2 loops if you are clever. 
>
> Hint: you will find this identity useful:
> 
   $\exp(x_i - m) =  \exp(x_i - m/2 - m/2) = \exp(x_i - m/ 2) /  \exp(m/2)$
  >
> ![[triton-puzzles-08.png]]

我们先实现一个没有优化的三个循环版本：

首先有 $\text{softmax}(x_1,\dots,x_T)=\frac{\mathrm{e}^{x_1-x^\text{max}}}{\sum_1^T\mathrm{e}^{x_i-x^\text{max}}},\dots,\frac{\mathrm{e}^{x_T-x^\text{max}}}{\sum_1^T\mathrm{e}^{x_i-x^\text{max}}}$

注意：减 $x^\text{max}=\max(x_1,\dots,x_T)$ 的作用是防止指数运算数值溢出。

这里我们先用第一个循环得到 $x^\text{max}$，在 `axis=1` 维度上做 `max` 操作，每次处理 `B1` 个元素；

然后类似地，我们用第二个循环计算 $\sum_1^T\mathrm{e}^{x_i-x^\text{max}}$；

最后，我们得到 $\text{softmax}$ 每个位置的结果 $\frac{\mathrm{e}^{x_i-x^\text{max}}}{\sum_1^T\mathrm{e}^{x_i-x^\text{max}}}$。

```python
@triton.jit
def softmax_kernel_brute_force(
    x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr
):
    """3 loops ver."""
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504
    # Finish me!
    off_i = block_id_i * B0 + tl.arange(0, B0)
    mask_i = off_i < N0
    # x_max
    x_max = tl.full((B0,), -float('inf'), dtype=tl.float32)
    for j in range(0, T, B1):
        off_j = tl.arange(j, j + B1)
        off_ij = off_i[:, None] * T + off_j[None, :]
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]
        x = tl.load(x_ptr + off_ij, mask=mask_ij)
        x_max = tl.maximum(x_max, tl.max(x, axis=1))
    # sum_x_exp
    sum_x_exp = tl.zeros((B0,), dtype=tl.float32)
    for j in range(0, T, B1):
        off_j = tl.arange(j, j + B1)
        off_ij = off_i[:, None] * T + off_j[None, :]
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]
        x = tl.load(x_ptr + off_ij, mask=mask_ij)
        x_exp = tl.exp2(log2_e * (x - x_max[:, None]))
        sum_x_exp += tl.sum(x_exp, axis=1)
    # softmax
    for j in range(0, T, B1):
        off_j = tl.arange(j, j + B1)
        off_ij = off_i[:, None] * T + off_j[None, :]
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]
        x = tl.load(x_ptr + off_ij, mask=mask_ij)
        x_exp = tl.exp2(log2_e * (x - x_max[:, None]))
        z = x_exp / sum_x_exp[:, None]
        tl.store(z_ptr + off_ij, z, mask=mask_ij)
    return
```

然后我们实现一个 Online Softmax 版本，可以只使用两个循环得到结果。

此处参考 **[Zhao Ye](mailto:zhye@cs.washington.edu)** 的 [Flash Attention 讲义](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)。

关键在于我们可以同时迭代求解 $x^\text{max}$ 和 $\sum_1^T\mathrm{e}^{x_i-x^\text{max}}$：

我们定义 $m_i=\max(x_1,\dots,x_i)$ 和 $d_i=\sum_{j=1}^i\mathrm{e}^{x_j-m_i}$，则有 $x^\text{max}=m_T$，$\sum_1^T\mathrm{e}^{x_i-x^\text{max}}=d_T$；

并且有 $m_i=\max(m_{i-1}, x_i)$，$d_i=d_{i-1}\frac{m_{i-1}}{m_i}+\mathrm{e}^{x_i-m_i}$。

在 `axis=1` 上按块迭代求解即可：

```python
@triton.jit
def softmax_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    """2 loops ver."""
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504
    # Finish me!
    off_i = block_id_i * B0 + tl.arange(0, B0)
    mask_i = off_i < N0
    # x_max and sum_x_exp'
    x_max = tl.full((B0,), -float('inf'), dtype=tl.float32)
    sum_x_exp = tl.zeros((B0,), dtype=tl.float32)
    for j in range(0, T, B1):
        off_j = tl.arange(j, j + B1)
        off_ij = off_i[:, None] * T + off_j[None, :]
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]
        x = tl.load(x_ptr + off_ij, mask=mask_ij)
        x_max_now = tl.maximum(x_max, tl.max(x, axis=1))
        x_exp = tl.exp2(log2_e * (x - x_max_now[:, None]))
        factor = tl.exp2(log2_e * (x_max - x_max_now))
        x_max = x_max_now
        sum_x_exp = sum_x_exp * factor + tl.sum(x_exp, axis=1)
    # softmax
    for j in range(0, T, B1):
        off_j = tl.arange(j, j + B1)
        off_ij = off_i[:, None] * T + off_j[None, :]
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]
        x = tl.load(x_ptr + off_ij, mask=mask_ij)
        x_exp = tl.exp2(log2_e * (x - x_max[:, None]))
        z = x_exp / sum_x_exp[:, None]
        tl.store(z_ptr + off_ij, z, mask=mask_ij)
    return
```



#### Puzzle 9: Simple FlashAttention

> A scalar version of FlashAttention.
>
> Uses zero programs. Block size `B0` represent the batches of `q` to process out of `N0`. Sequence length is `T`. Process it `B1 < T` elements (`k`, `v`) at a time for some `B1`.
> 
   $z_{i} = \sum_{j=1}^{T} \text{softmax}(q_i k_1, \ldots, q_i k_T)_j v_{j}, \text{ for } i = 1\ldots N_0$
>
> This can be done in 1 loop using a similar trick from the last puzzle.
>
> Hint: Use `tl.where` to mask `q dot k` to -inf to avoid overflow (NaN).
>
> ![[triton-puzzles-09.png]]

令 $x_{i,j}=q_ik_j$，`softmax` 部分可以划归为上一道 puzzle。因此我们可以很容易地写出一个两个循环的版本（在上一道 puzzle 的第二个循环中，对 `z` 乘以 `v[:, None]` 即可）。

然而，我们还可以推导出一个更简单的版本，只用一个循环完成任务。

此处参考 **[Zhao Ye](mailto:zhye@cs.washington.edu)** 的 [Flash Attention 讲义](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)。

令 $o_i=\sum_{j=1}^i{\frac{\mathrm{e}^{x_j-m_i}}{d_i}}v[i,:]$, 则有 $o_i=o_{i-1}\frac{d_{i-1}\mathrm{e}^{x_i-m_i}}{d_i}+\frac{\mathrm{e}^{x_i-m_i}}{d_i}v[i,:]$。

可以看到，所有符号的下标只和 $i$ 以及 $i-1$ 有关，因此我们可以在一个循环内获得 $o_T$（注意这里所有的下标都是作用在 `axis=1` 上的）。

```python
@triton.jit
def flashatt_kernel(
    q_ptr, k_ptr, v_ptr, z_ptr, N0, T, B0: tl.constexpr, B1: tl.constexpr
):
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504
    myexp = lambda x: tl.exp2(log2_e * x)
    # Finish me!

    off_i = block_id_i * B0 + tl.arange(0, B0)
    mask_i = off_i < N0

    q = tl.load(q_ptr + off_i, mask=mask_i)

    x_max = tl.full((B0,), -float('inf'), dtype=tl.float32)
    sum_x_exp = tl.zeros((B0,), dtype=tl.float32)
    z = tl.zeros((B0,), dtype=tl.float32)

    for j in range(0, T, B1):
        off_j = tl.arange(j, j + B1)
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]

        k = tl.load(k_ptr + off_j, mask=mask_j)
        
        x = q[:, None] * k[None, :] + tl.where(mask_ij, 0, -1e6)
        
        x_max_now = tl.maximum(x_max, tl.max(x, axis=1))
        factor = myexp(x_max - x_max_now)
        x_exp = myexp(x - x_max_now[:, None])
        sum_x_exp_now = sum_x_exp * factor + tl.sum(x_exp, axis=1)

        v = tl.load(v_ptr + off_j, mask=mask_j)
        qkv = v[None, :] * x_exp / sum_x_exp_now[:, None]
        z = tl.fma(z, factor * sum_x_exp / sum_x_exp_now, tl.sum(qkv, axis=1))
        
        x_max = x_max_now
        sum_x_exp = sum_x_exp_now

    tl.store(z_ptr + off_i, z, mask=mask_i)
    return
```



#### Puzzle 10: Two Dimensional Convolution

>A batched 2D convolution.
>
>Uses one program id axis. Block size `B0` represent the batches to process out of `N0`.
>
>Image `x` is size is `H` by `W` with only 1 channel, and kernel `k` is size `KH` by `KW`.
>
  $z_{i, h, w} = \sum_{kh=0, kw=0}^{kh< KH, kw< KW} k_{kh,kw} \times x_{i,h + kh, w + kw}, \text{ for } i = 1\ldots N_0,\ h = 1\ldots H,\ w = 1\ldots W$
>
> ![[triton-puzzles-10.png]] 

按照 `PyTorch` 逻辑实现即可：

```python
@triton.jit
def conv2d_kernel(
    x_ptr, k_ptr, z_ptr, N0, H, W, KH: tl.constexpr, KW: tl.constexpr, B0: tl.constexpr
):
    block_id_i = tl.program_id(0)
    # Finish me!
    off_i = block_id_i * B0 + tl.arange(0, B0)
    mask_i = off_i < N0
    
    off_kh = tl.arange(0, KH)
    off_kw = tl.arange(0, KW)
    off_k = off_kh[:, None] * KW + off_kw[None, :]
    k = tl.load(k_ptr + off_k)

    for h in range(0, H):
        for w in range(0, W):
            off_h_kh = h + off_kh[None, :, None]
            off_w_kw = w + off_kw[None, None, :]
            off_x = off_i[:, None, None] * H * W + off_h_kh * W + off_w_kw
            mask_x = mask_i & (off_h_kh < H) & (off_w_kw < W)
            x = tl.load(x_ptr + off_x, mask=mask_x, other=0)
            
            z = (x * k[None, :, :]).sum(1).sum(1)
            off_z = off_i * H * W + h * W + w
            tl.store(z_ptr + off_z, z, mask=mask_i)

    return
```



#### Puzzle 11: Matrix Multiplication

> A blocked matrix multiplication.
>
> Uses three program id axes. Block size `B2` represent the batches to process out of `N2`.
>
> Block size `B0` represent the rows of `x` to process out of `N0`. Block size `B1` represent the cols 
>
> of `y` to process out of `N1`. The middle shape is `MID`.
> 
   $z_{i, j, k} = \sum_{l} x_{i,j, l} \times y_{i, l, k}, \text{ for } i = 1\ldots N_2, j = 1\ldots N_0, k = 1\ldots N_1$
 > 
> You are allowed to use `tl.dot` which computes a smaller mat mul.
>
> Hint: the main trick is that you can split a matmul into smaller parts.
> 
   $z_{i, j, k} = \sum_{l=1}^{L/2} x_{i,j, l} \times y_{i, l, k} +  \sum_{l=L/2}^{L} x_{i,j, l} \times y_{i, l, k}$
 > 
> ![[triton-puzzles-11.png]]


在 `MID` 上按块做矩阵乘，结果累加，如图示：

![image-20250127220014995](C:\Users\Namoe\AppData\Roaming\Typora\typora-user-images\image-20250127220014995.png)

```python
@triton.jit
def dot_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    N0,
    N1,
    N2,
    MID,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B_MID: tl.constexpr,
):
    block_id_j = tl.program_id(0)
    block_id_k = tl.program_id(1)
    block_id_i = tl.program_id(2)
    # Finish me!
    off_i = block_id_i * B2 + tl.arange(0, B2)[:, None, None]
    off_j = block_id_j * B0 + tl.arange(0, B0)[None, :, None]
    off_k = block_id_k * B1 + tl.arange(0, B1)[None, None, :]

    mask_i = off_i < N2
    mask_j = off_j < N0
    mask_k = off_k < N1

    z = tl.zeros((B2, B0, B1), dtype=tl.float32)

    for l in range(0, MID, B_MID):
        off_l = l + tl.arange(0, B_MID)
        mask_l = off_l < MID
        off_x = off_i * N0 * MID + off_j * MID + off_l[None, None, :]
        off_y = off_i * N1 * MID + off_l[None, :, None] * N1 + off_k
        mask_x = mask_i & mask_j & mask_l[None, None, :]
        mask_y = mask_i & mask_l[None, :, None] & mask_k

        x = tl.load(x_ptr + off_x, mask=mask_x)
        y = tl.load(y_ptr + off_y, mask=mask_y)
        z = tl.dot(x, y, acc=z)

    off_z = off_i * N0 * N1 + off_j * N1 + off_k
    mask_z = mask_i & mask_j & mask_k
    tl.store(z_ptr + off_z, z, mask=mask_z)

    return
```



#### Puzzle 12: Quantized Matrix Mult

> When doing matrix multiplication with quantized neural networks a common strategy is to store the weight matrix in lower precision, with a shift and scale term.
>
> For this problem our `weight` will be stored in 4 bits. We can store `FPINT` of these in a 32 bit integer. In addition for every `group` weights in order we will store 1 `scale` float value and 1 `shift` 4 bit value. We store these for the column of weight. The `activation`s are stored separately in standard floats.
>
> Mathematically it looks like.
> 
   $z_{j, k} = \sum_{l} sc_{j, \frac{l}{g}} (w_{j, l} - sh_{j, \frac{l}{g}}) \times y_{l, k}, \text{ for } j = 1\ldots N_0, k = 1\ldots N_1$
 > 
> Where `g` is the number of groups (`GROUP`).
>
> However, it is a bit more complex since we need to also extract the 4-bit values into floats to begin.
>
> Note:
>
> \- We don't consider batch size, i.e. `i`, in this puzzle.
>
> \- Remember to unpack the `FPINT` values into separate 4-bit values. This contains some shape manipulation.
> ![[triton-puzzles-12.png]]

注意仔细观察示意图：

我们最终要获取到 `true_weight` 去做矩阵乘，而 `true_weight=scale*(weight-shift)`。

有两点需要注意：
1. 每个 `shift` 和 `scale` 对应一组8个 `weight`
2. `weight` 和 `shift` 按 `4-bit` 存储

所以需要两步：
1. 将 `weight` 和 `shift` 从 `32-bit` 的存储中提取出来
2. 将 `shift` 和 `scale` 按组对应

步骤一可以仿照 `PyTorch` 代码实现，输入 `shape` 为 `[dim_1, dim_2, ..., dim_n]` 的32位张量，输出 `shape` 为 `[dim_1, dim_2, ..., dim_n, 8]` 的张量，且只有低4位不为0。

步骤二可以通过 `broadcast` 机制实现，注意到 `scale`, `shift(extracted)` 的维度是 `[N0, MID // GROUP]`, 而 `weight` 的维度是 `[N0, MID // FPINT] = [N0, MID // GROUP]`。因此只需将 `scale`, `shift` 进行 `expand_dim(-1)` 即可。之后 `scale * (weight- shift)` 可以在最后一维做 `broadcast`。

最后做矩阵乘即可。

注意：这个矩阵乘不是一个通用矩阵乘，由于 `offset` 只有维度0，导致其只支持 $\le 8$ 组 `weight`，也即 `weight.shape[1] <= 8`，也即 `true_weight.shape[1] <= 64`。

```python
FPINT = 32 // 4
GROUP = 8

@triton.jit
def quant_dot_kernel(
    scale_ptr,
    offset_ptr,
    weight_ptr,
    activation_ptr,
    z_ptr,
    N0,
    N1,
    MID,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B_MID: tl.constexpr,
):
    block_id_j = tl.program_id(0)
    block_id_k = tl.program_id(1)

    # Finish me!
    def extract(x):
        BITS = 32 // FPINT
        bit_shift = tl.arange(0, FPINT) * BITS
        bit_mask = (1 << BITS) - 1
        return (x.expand_dims(-1) >> bit_shift) & bit_mask

    off_j = block_id_j * B0 + tl.arange(0, B0)
    off_k = block_id_k * B1 + tl.arange(0, B1)
    mask_j = off_j < N0
    mask_k = off_k < N1

    z = tl.zeros((B0, B1), dtype=tl.float32)
    off_z = off_j[:, None] * N1 + off_k[None, :]
    mask_z = mask_j[:, None] & mask_k[None, :]

    for l in range(0, MID, B_MID):
        # scale
        off_l_div_g = (l // GROUP) + tl.arange(0, B_MID // GROUP)
        mask_l_div_g = off_l_div_g < (MID // GROUP)
        off_scale = off_j[:, None] * (MID // GROUP) + off_l_div_g[None, :]
        mask_scale = mask_j[:, None] & mask_l_div_g[None, :]
        scale = tl.load(scale_ptr + off_scale, mask=mask_scale)

        # offset
        offset = tl.load(offset_ptr + off_j, mask=mask_j)
        offset = extract(offset)

        # weight
        off_weight_l = (l // FPINT) + tl.arange(0, B_MID // FPINT)
        mask_weight_l = off_weight_l < (MID // FPINT)
        off_weight = off_j[:, None] * (MID // FPINT) + off_weight_l[None, :]
        mask_weight = mask_j[:, None] & mask_weight_l[None, :]
        weight = tl.load(weight_ptr + off_weight, mask=mask_weight)
        weight = extract(weight)

        # activation
        off_l = l + tl.arange(0, B_MID)
        mask_l = off_l < MID
        off_activation = off_l[:, None] * N1 + off_k[None, :]
        mask_activation = mask_l[:, None] & mask_k[None, :]
        activation = tl.load(activation_ptr + off_activation, mask=mask_activation)

        # true weight
        true_weight = scale[:, :, None] * (weight - offset[:, :, None])
        true_weight = true_weight.reshape(B0, B_MID)
        
        # quantized dot product
        z = tl.dot(true_weight, activation, acc=z)

    tl.store(z_ptr + off_z, z, mask=mask_z)

    return
```

---

答案部分参考了 `Triton-Puzzles-Lite` 的官解以及知乎文章：[Triton Puzzles - 寒假摸鱼 (1)](https://zhuanlan.zhihu.com/p/20269643126)。

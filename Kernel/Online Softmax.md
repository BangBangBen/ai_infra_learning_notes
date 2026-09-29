---
course: AI Infra
topic: online softmax
tags: [ai-infra, gpu-kernel, attention, softmax]
---

# Online Softmax（在线 Softmax）

> **Summary**
> Online softmax 用可合并的运行统计量处理分块到达的 logits，无需先存下完整向量或完整 attention score 矩阵。它维护当前最大值、归一化指数和（以及加权值累积），可稳定地跨 tile 合并计算，是 FlashAttention 等 IO-aware attention 算法的基础。实现时必须正确处理最大值变化、mask 和空块。

## 1. 普通 softmax 的中间存储问题

对 logits $x_i$，softmax 为 $p_i=\exp(x_i)/\sum_j\exp(x_j)$。直接计算时需要知道全局最大值以保证数值稳定，并需要分母归一化项。长序列 attention 若物化整个 $QK^T$，会产生大量 HBM 写入和读取。

## 2. 可合并的运行状态

处理一组 logits 时维护：最大值 $m$ 与缩放指数和 $\ell$：

$$
m = \max_i x_i, \qquad \ell = \sum_i \exp(x_i-m)
$$

两个块 A、B 的状态 $(m_A,\ell_A)$、$(m_B,\ell_B)$ 可合并为：

$$
m = \max(m_A,m_B), \qquad
\ell = \ell_A\exp(m_A-m)+\ell_B\exp(m_B-m)
$$

新的最大值改变了指数基准，因此要缩放旧的累积值。这个在线归约避免指数溢出/下溢的同时支持分块计算。

## 3. Attention 中同时累积输出

若每个 score $x_i$ 对应 value $v_i$，同时维护未归一化加权和 $o=\sum_i \exp(x_i-m)v_i$。合并两个块时：

$$
o = o_A\exp(m_A-m)+o_B\exp(m_B-m), \qquad y=o/\ell
$$

这样一个 query 行可逐块读取 K/V：每块算局部 score，再在线更新 $(m,\ell,o)$，不必把所有 score 或概率写回 HBM。

## 4. 伪代码

```text
m = -infinity; l = 0; o = 0
for each key/value tile:
    scores = Q @ K_tile.T + mask
    m_tile = row_max(scores)
    p = exp(scores - m_tile)
    l_tile = row_sum(p)
    o_tile = p @ V_tile
    m_new = max(m, m_tile)
    alpha = exp(m - m_new)
    beta = exp(m_tile - m_new)
    l = alpha * l + beta * l_tile
    o = alpha * o + beta * o_tile
    m = m_new
output = o / l
```

伪代码展示数学结构；实际 kernel 还需处理 causal mask、padding、低精度累积、tile 布局和全 mask 行。

## 5. 性能含义与注意事项

- 在线统计量让算法可以流式遍历 tile，减少中间张量的 HBM 读写；
- 它不减少 attention 的数学计算量，也不自动解决并行度、occupancy 或访存布局问题；
- 多线程分块归约需要正确组合局部状态；
- 对全被 mask 的行，必须定义安全处理，避免 `-inf - -inf` 导致 NaN；
- 在低精度下通常需要更高精度的 max、sum 或累加策略，并与参考实现比较误差。

## 6. 一句话总结

> Online softmax 把全局归一化改写为数值稳定、可合并的运行统计量，使 attention 能分块计算并省去完整 score/probability 矩阵的物化。

## References

- Tri Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)

*Last reviewed: 2026-09-29.*

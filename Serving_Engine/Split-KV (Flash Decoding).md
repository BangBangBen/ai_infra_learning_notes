---
course: AI Infra
topic: split-KV flash decoding
tags: [ai-infra, llm-serving, attention, decode]
---

# Split-KV（Flash Decoding）

> **Summary**
> Split-KV 将单个 decode query 所需的长 KV cache 切成多个区间，在多个 GPU thread blocks 中并行计算局部 attention 结果，再用 log-sum-exp 统计量合并。它为 batch 较小、上下文很长的 decode 增加序列维并行度；代价是 partial output、合并步骤和调度开销，短上下文或较大 batch 下未必更快。

## 1. Decode attention 的并行度瓶颈

Decode 时每条序列通常只有一个新 query token。若 batch 很小，query/head 维可提供的 block 数有限；与此同时每个 query 仍要扫描很长的 KV cache。只按 query 维并行可能无法充分利用 GPU。

## 2. 分割与合并

将 KV 序列划分为多个 split。每个 split 独立算局部 attention，并输出局部最大值 $m_s$、指数和 $\ell_s$ 及未归一化加权输出 $o_s$。之后设 $m=\max_s m_s$：

$$
\ell=\sum_s \ell_s\exp(m_s-m), \qquad
o=\sum_s o_s\exp(m_s-m), \qquad
y=o/\ell
$$

这个合并利用 online softmax 统计量，结果在数学上等价于对整个 KV 序列做 softmax attention（浮点舍入除外）。

## 3. 执行心智模型

```text
query + KV[0:S]   -> partial (m0, l0, o0) ┐
query + KV[S:2S]  -> partial (m1, l1, o1) ├─ combine -> output
query + KV[2S:3S] -> partial (m2, l2, o2) ┘
```

## 4. 收益与成本

| 条件 | 可能结果 |
|---|---|
| batch 小、上下文长 | 增加并行 block，隐藏 KV 读取延迟并改善利用率 |
| batch 已大、query/head 并行充分 | split 增益减小，partial buffer 和 combine 可能成为负担 |
| split 数过多 | 并行度提高，但局部工作太少、调度与合并成本上升 |
| KV cache 已在较快 cache 层命中 | 预期的内存并行收益可能较小 |

## 5. 与 Chunked Prefill 的区别

Split-KV 把单个 decode attention 对历史 KV 的读取分片，并合并部分 softmax 结果；chunked prefill 把 prompt prefill token 切成调度单元。前者针对 decode kernel 并行度，后者针对 serving scheduler 的阶段混排。

## 6. 调优和正确性检查

- 根据 batch、上下文长度、head 数和 GPU 规模选择 split 数与分块大小；
- 计入 partial buffer、combine kernel 和额外 launch 的成本；
- 校验 causal mask、GQA/MQA head 映射、变长序列和低精度结果；
- 同时测量 attention kernel 和端到端 inter-token latency；
- 对短序列和高并发负载保留不分割路径。

## 7. 一句话总结

> Split-KV 用序列维并行度加速长上下文、小 batch 的 decode attention，再以 online softmax 合并局部结果；split 粒度应按实际负载调优。

## References

- Dao et al., [Flash-Decoding for Long-Context Inference](https://princeton-nlp.github.io/flash-decoding/)
- Tri Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

*Last reviewed: 2026-09-29. Split-KV 是多种 attention kernel 的实现策略，具体参数和 dispatch 条件依实现而异。*

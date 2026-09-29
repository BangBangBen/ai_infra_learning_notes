---
course: AI Infra
topic: paged decode
tags: [ai-infra, llm-serving, inference, kv-cache]
---

# Paged Decode（分页式 KV Cache 解码）

> **Summary**
> 在 paged decode 路径中，请求的 KV cache 按固定 token 数分成逻辑块，映射到可不连续的物理显存块。attention kernel 通过 block table 找到各请求的 KV 页面并参与解码。分页缓解连续显存分配和外部碎片问题，支持动态增长与共享前缀；代价是索引、间接寻址和 kernel 管理复杂度。

## 1. 解码为何受 KV cache 管理影响

自回归解码每生成一个 token，都要读取历史 token 的 key/value。并发请求的上下文长度不断变化，若每条请求预留一段最大长度连续空间，会浪费显存；若按需连续扩展，可能因碎片或扩展失败而难以稳定接纳请求。

## 2. 分页心智模型

```text
逻辑序列块:  [0] [1] [2] [3] [4]
物理显存块:  [17][ 4][29][ 8][12]   (无需连续)
block table:  17  -> 4 -> 29 -> 8 -> 12
```

每个请求维护逻辑块到物理 KV block 的映射。解码时 kernel 按映射读取对应 K/V，写入新 token 的状态；最后一块通常可能只部分填充。

## 3. 关键操作

1. 新请求按需分配 KV blocks，并写入 prefill 产生的状态；
2. decode 继续增长序列，块满后申请新的物理块；
3. 请求结束、抢占或迁移时回收或转移块；
4. 多请求或共享前缀场景可通过引用计数等策略共享只读前缀块；
5. attention kernel 根据 block table 遍历 KV，而非假设单段连续内存。

## 4. 收益与成本

| 方面 | 收益 | 代价 |
|---|---|---|
| 显存分配 | 降低连续预留造成的浪费和外部碎片 | 最后一页内部可能有未使用空间 |
| 动态长度 | 逐块增长，便于在线调度 | allocator、block table 和回收逻辑更复杂 |
| 前缀复用 | 可让多个请求引用共同前缀状态 | 引用计数、写时复制和生命周期管理 |
| Kernel | 支持非连续 KV 存储 | 增加间接寻址，需设计高效的 block-aware attention |

## 5. 不要混淆的概念

“Paged decode”通常指 paged KV cache 支撑的 decode 执行路径；PagedAttention 是相关的 attention 计算与内存访问机制。它们不代表把模型权重分页到 CPU，也不等同于 chunked prefill。分页管理 KV 存储；chunked prefill 管理 prompt 工作在迭代中的切分。

## 6. 评估要点

- 同时统计 KV block 利用率、内部浪费、可接纳并发数和抢占次数；
- 观察长上下文下的 attention 延迟与 block table 访问成本；
- 验证共享前缀的正确生命周期及请求取消后的及时回收；
- 对照目标引擎和版本中的 block size、内存池和调度配置。

## 7. 一句话总结

> Paged decode 让每条序列的 KV 状态按块分散存放并通过映射读取，以动态分配换取更高的显存利用灵活性；实际表现还取决于 block 粒度和 attention kernel。

## References

- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- vLLM, [PagedAttention design](https://docs.vllm.ai/en/stable/design/paged_attention/)

*Last reviewed: 2026-09-29. 引擎中 paged decode 的具体实现和 block 参数可能变化。*

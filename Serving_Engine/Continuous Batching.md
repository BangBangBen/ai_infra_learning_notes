---
course: AI Infra
topic: continuous batching
tags: [ai-infra, llm-serving, inference, scheduling]
---

# Continuous Batching（连续批处理）

> **Summary**
> Continuous batching（也称 iteration-level scheduling）在生成迭代边界动态加入新请求、移除已完成请求，并为每个活跃请求推进一个 decode token。它减少静态 batch 因请求长度不同产生的空槽，提升 GPU 忙碌度；调度公平性、请求延迟、KV cache 容量和 batch 形状仍需要管理。

## 1. 静态 batching 的限制

静态 batching 把一组请求放入同一批次，并通常等整批都完成后再换入新请求。生成长度不一时，短请求结束后留下空闲槽位；长请求会拖住整批，降低有效 batch 利用率。

## 2. 执行心智模型

```text
迭代 0: A  B  C
迭代 1: A  B  C  + 接纳 D
迭代 2: A     C  D  (B 已结束，释放槽位)
迭代 3:    E  C  D  (接纳 E)
```

每次模型迭代后 scheduler 检查请求状态：完成的请求释放资源；仍在生成的请求继续 decode；新请求在满足资源与策略约束时进入 batch。Prompt prefill 与 decode 如何共批由引擎策略决定。

## 3. 收益和成本

| 维度 | 影响 |
|---|---|
| GPU 利用率 | 释放的 batch 槽可由新请求补上，减少等待整批结束的空闲 |
| 请求延迟 | 新请求可能更快启动；竞争激烈时，既有请求也可能排队或等待 |
| KV cache | 活跃请求集合动态变化，需要在线分配、回收和可能的抢占 |
| 调度开销 | 每轮要管理队列、预算、优先级、停止条件和错误恢复 |
| batch 形状 | 变动的序列长度和并发数要求 kernel 支持动态形状或高效打包 |

## 4. 调度策略与指标

- 定义请求准入、队列排序、优先级、公平性和最大并发策略；
- 结合 KV cache 预算判断请求能否接纳，不能只看 batch 中 token 数；
- 观察 TTFT、TBT/ITL、端到端延迟分位数、请求吞吐与抢占/重计算次数；
- 避免只用总 token/s 评价用户体验；
- 测试不同 prompt/output 长度混合和到达率，而非固定长度离线输入。

## 5. 与相关机制的关系

- **Chunked prefill**：限制每轮处理的 prompt token 数，使长 prefill 可以拆分参与迭代调度。
- **Paged KV cache**：让变长请求的 KV 状态以块为单位分配和回收。
- **Static batching**：整批协调、实现较简单；continuous batching 以迭代为边界更新活跃请求。
- **Prefill/decode disaggregation**：把两个阶段放在不同资源池，改变部署位置；continuous batching 是调度策略，可出现在任一资源池中。

## 6. 一句话总结

> Continuous batching 以每次生成迭代为调度边界，持续填补已完成请求留下的空位；收益取决于调度策略、KV cache 容量、请求分布和 kernel 对动态 batch 的支持。

## References

- Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- vLLM, [PagedAttention design](https://docs.vllm.ai/en/stable/design/paged_attention/)
- vLLM, [Optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)

*Last reviewed: 2026-09-29. Serving engine 的调度策略和默认值随版本变化。*

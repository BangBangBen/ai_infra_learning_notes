---
course: AI Infra
topic: chunked prefill
tags: [ai-infra, llm-serving, inference, scheduling]
---

# Chunked Prefill（分块预填充）

> **Summary**
> Chunked prefill 将长 prompt 的 prefill 阶段拆成多个较小的 token 块，允许 serving scheduler 在一次迭代中把 prefill 块和已有请求的 decode 一起执行。这样可以限制单次 prefill 对 GPU 的占用时间，减少长 prompt 阻塞 decode 的风险，并更灵活地使用每轮 token budget；代价是调度更复杂，而且 chunk 大小需要在吞吐、首 token 延迟和 decode 延迟之间权衡。

## 1. LLM 推理的两个阶段

一条请求通常经历以下阶段：

| 阶段                     | 输入与工作                                               | 典型硬件特征                                   | 常用延迟指标                                                    |
| ---------------------- | --------------------------------------------------- | ---------------------------------------- | --------------------------------------------------------- |
| **Prefill（预填充）**       | 并行处理 prompt 中已有的 token，计算并写入 KV cache，产生第一个输出 token | 大量 token 可并行计算，常较偏 compute-bound         | Time to First Token（TTFT）                                 |
| **Decode（逐 token 解码）** | 每轮为每条活跃请求生成下一个 token，并读取已有 KV cache                 | 每条序列每轮只新增一个 token，常受权重或 KV cache 的内存访问影响 | Time Between Tokens（TBT，也常以 inter-token latency / ITL 表示） |

普通的 iteration-level batching 可以在迭代边界调整 batch，但一个很长的 prefill 迭代仍可能占用较长时间，让已有 decode 请求等待。Chunked prefill 限制一次迭代处理的 prefill token 数，给 scheduler 更多插入 decode 的机会。

## 2. 心智模型

```text
长 prompt:  [ p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 ]

分块后:    [ p0 ... p3 ] → [ p4 ... p7 ] → [ p8 ... p11 ]
                 ↘ decode batch 可在迭代边界参与调度 ↙

一次调度迭代的工作量受 token budget 限制
```

```text
scheduler iteration
  ├─ 优先放入已有请求的 decode token
  └─ 用剩余预算安排一个或多个 prefill chunk
       └─ 未处理完的 prompt 留在队列中，后续迭代继续
```

图中每块大小仅作示意。实际 scheduler 可按当前批次剩余预算动态决定可安排多少 prefill token；并非所有系统都会把每个 prompt 切成固定且相等大小的块。

## 3. 执行过程

设一个 prompt 有 10,000 个 token，而某轮 scheduler 给 prefill 留下的预算是 2,000 个 token：

1. scheduler 选取最多 2,000 个尚未处理的 prompt token；
2. 模型执行该 prefill 部分，并把对应的 key/value 状态写入请求的 KV cache；
3. 同一轮或相邻迭代中，scheduler 可把已在生成的请求的 decode token 纳入 batch；
4. 剩下的 prompt token 保持待处理状态；后续轮次继续 prefill；
5. prompt 完成 prefill 后，该请求进入正常的自回归 decode。

chunk 的边界是调度和执行的切分点，不改变 prompt token 的逻辑顺序，也不会把请求变成彼此独立的多个 prompt。模型仍需保留足够的 KV cache 状态，以便后续 chunk 和 decode 使用。

## 4. 为什么要分块

| 问题                            | 不分块时                                         | 分块后的机会                                       |
| ----------------------------- | -------------------------------------------- | -------------------------------------------- |
| 长 prompt 占据一次迭代很久             | 已在生成的请求可能长时间得不到下一 token                      | 限制每轮 prefill 工作量，让 decode 更频繁地参与调度           |
| Prefill 与 decode 的计算形态不同      | 大 prefill 可能拉高单轮时长；decode batch 可能无法充分利用计算资源 | 同批组合两类工作，利用 prefill 的计算并行度补充 decode 的较低计算利用率 |
| 固定 batch 难以适配混合请求长度           | 长短请求和新到请求互相影响                                | scheduler 可在 token budget 内调整每轮工作组成          |
| Pipeline parallelism 中微批计算量不均 | prefill 与 decode 工作差异可能带来 pipeline bubbles   | 在特定设计中，限制 prefill 块大小可以让各轮计算量更均匀             |

Chunking 只是让 scheduler 有更多调度机会，不会自动保证公平，也不会消除所有等待。策略仍需决定 decode 优先级、prefill 的准入时机以及各类请求的 token budget。

## 5. 主要权衡

| 调整方向 | 可能收益 | 可能代价 |
|---|---|---|
| **减小每轮 prefill token 数** | 限制单轮 prefill 工作量，通常有利于控制 decode 的等待时间 | 长 prompt 需要更多迭代才能完成；调度与 kernel 执行次数可能增加，TTFT 或整体吞吐可能变差 |
| **增大每轮 prefill token 数** | 更快处理长 prompt，可能提高 prefill 吞吐并改善 TTFT | 单轮工作变重，decode 请求等待可能增加；也可能挤占本轮其他请求的预算 |
| **prefill 和 decode 混合 batching** | 让计算密集型 prefill 与较偏内存访问的 decode 在同一调度迭代协同执行 | batch 形态、attention kernel 和模型结构会影响收益；混合不一定在所有硬件或负载上更快 |
| **更积极地接纳长 prompt** | 提高新请求处理速度或总体 token 吞吐 | KV cache 占用上升，可能触发请求抢占、缓存抖动或可运行并发量下降 |

要明确区分两个限制：`max_num_batched_tokens` 一类预算约束单次迭代可调度的 token 数；KV cache 则存储活跃请求的注意力状态，受 GPU 内存容量约束。减小每轮 token 预算不会让长 prompt 所需的总 KV cache 自动变小。

## 6. 关键参数与调优

不同引擎和版本的参数语义不同。以 vLLM 为例，V1 文档说明 chunked prefill 在适用时默认启用；scheduler 会先安排待处理的 decode，再用剩余 `max_num_batched_tokens` 预算安排 prefill。若一个 prefill 放不进剩余预算，可以拆分后续处理。具体默认值和开关应以正在使用的 vLLM 版本为准。

调优时建议：

- 先记录真实请求分布：prompt 长度、输出长度、到达率和并发度；
- 明确优化目标是 TTFT、TBT / ITL、端到端延迟、吞吐，还是满足 SLO 下的 serving capacity；
- 改变 `max_num_batched_tokens` 或引擎提供的长 prefill 限制时，一次只调整一个变量；
- 同时观察 TTFT、P50/P99 TBT、请求吞吐、GPU 利用率、KV cache 占用和抢占/重计算次数；
- 使用代表性的混合长短请求做负载测试，不要只用固定长度的离线 prompt；
- 另外验证多模态请求。图像或视频输入可能含有不可任意拆开的多模态片段，支持部分调度的规则依实现而异。

单看 GPU 利用率或总 token/s 可能掩盖用户可见的长尾延迟；应在相同模型、硬件、并发和请求分布下比较完整延迟分布。

## 7. 与相关技术的边界

- **Continuous batching / iteration-level scheduling**：在迭代边界动态加入和移除请求。Chunked prefill 解决长 prefill 工作如何切成较小的调度单元，可与 continuous batching 配合。
- **PagedAttention / paged KV cache**：管理 KV cache 的分配与访问，减少内存碎片并支持灵活的请求长度。它处理的是 KV cache 管理；chunked prefill 处理的是调度迭代中的 prefill 工作量。
- **Disaggregated prefill and decode**：把 prefill 与 decode 放到不同实例或设备上运行，需传输或共享 KV 状态。它改变阶段的部署位置；chunked prefill 可在单实例的迭代调度中使用，两者不是同一机制。
- **Sarathi-Serve**：提出 chunked-prefill 与 stall-free scheduling 等技术，目标是在不中断已有 decode 的情况下安排新的 prefill 工作。分块本身与完整的 stall-free scheduler 不是同义词。

## 8. 一句话总结

> Chunked prefill 把长 prompt 的计算切成可由 scheduler 控制的较小工作单元，使 serving 系统更容易在 prefill 与 ongoing decode 之间分配每轮 GPU 时间；实际效果取决于 token budget、KV cache 容量、请求分布、模型与 kernel，必须用目标 SLO 下的端到端负载测试验证。

## References

- Amey Agrawal et al., [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)
- Amey Agrawal et al., [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310)
- vLLM, [Optimization and Tuning — Chunked Prefill](https://docs.vllm.ai/en/latest/configuration/optimization/)
- vLLM, [CLI Reference — `enable-chunked-prefill` and token budget options](https://docs.vllm.ai/en/latest/cli/serve/)

*Last reviewed: 2026-09-29. Serving engines change defaults and scheduling behavior across releases; check the documentation for the deployed version before applying configuration examples.*

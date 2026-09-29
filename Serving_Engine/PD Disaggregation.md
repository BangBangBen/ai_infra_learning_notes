---
course: AI Infra
topic: prefill-decode disaggregation
tags: [ai-infra, llm-serving, inference, disaggregation]
---

# Prefill-Decode Disaggregation（Prefill/Decode 解耦部署）

> **Summary**
> PD disaggregation 将 prompt prefill 与逐 token decode 放到不同 GPU 实例或资源池，分别按各阶段的计算与内存访问特征配置容量。Prefill 完成后，需要把请求 KV 状态传给 decode 侧。解耦有机会减少阶段间资源争抢、独立扩缩容并改善 SLO goodput；收益必须覆盖 KV 传输、调度与资源碎片成本。

## 1. 为什么考虑分离阶段

Prefill 一次处理多个 prompt token，通常具有较多并行计算；decode 每条活跃序列每轮新增一个 token，反复读取权重和增长中的 KV cache。二者的 batch 需求与瓶颈不同，混在同一资源池可能互相干扰。

## 2. 执行流程

```text
请求到达 -> Prefill pool 计算 prompt 并生成 KV
                         │ KV 状态/元数据传输
                         v
              Decode pool 持续逐 token 生成
```

解耦会移动 KV 状态或让两侧共享可访问存储。传输路径、布局转换、量化格式、模型并行拓扑和失败恢复都会影响切换时间。

## 3. 主要收益与成本

| 维度 | 潜在收益 | 代价或风险 |
|---|---|---|
| 资源隔离 | 避免重 prefill 对 decode token latency 的直接争抢 | 资源池之间负载不均时可能有闲置容量 |
| 独立扩缩容 | 按 prompt/output 长度与到达率调配两侧容量 | 需要预测流量并协调扩容与路由 |
| SLO | 可针对 TTFT 与 TPOT/TBT 分别配置 | 更复杂的排队和网络路径可能增加尾延迟 |
| KV 迁移 | decode 节点获得 prefill 结果后继续服务 | 大上下文传输占用互连带宽并增加启动延迟 |
| 部署 | 可尝试为阶段匹配不同硬件 | 模型权重、拓扑、版本和故障处理更复杂 |

## 4. 何时有价值

若 prefill 与 decode 的负载比例差异大、同池干扰明显、且节点间有足够快的互连，解耦更可能改善目标 SLO 下的 goodput。若 KV 传输很大、请求短、流量不稳定或资源池规模太小，聚合部署可能更简单且更高效。

评估应同时测量 TTFT、TPOT/TBT、端到端延迟分位数、每 SLO 内完成请求数、KV 传输字节/时间、GPU 利用率和空闲资源。

## 5. 与相邻技术的区别

- **Chunked prefill**：在同一调度系统中切小 prefill 工作单元；不要求阶段分置。
- **Continuous batching**：每轮动态维护活跃 batch，可用于 prefill 池或 decode 池。
- **KV cache paging**：改变 KV 状态的存储分配方式；与跨节点迁移可组合。
- **PD disaggregation**：改变阶段的资源放置和数据流，需要显式连接两侧生命周期。

## 6. 一句话总结

> PD disaggregation 用 KV 状态传输和系统复杂度换取 prefill、decode 的资源隔离与独立配置；是否提高 goodput，要在实际互连和 SLO 下衡量。

## References

- Zhong et al., [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](https://arxiv.org/abs/2401.09670)
- Patel et al., [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677)
- Agrawal et al., [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310)

*Last reviewed: 2026-09-29. 具体 KV 传输和调度协议依引擎、网络拓扑及版本而异。*

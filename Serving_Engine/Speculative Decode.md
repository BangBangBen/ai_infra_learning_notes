---
course: AI Infra
topic: speculative decoding
tags: [ai-infra, llm-serving, inference, decoding]
---

# Speculative Decoding（推测解码）

> **Summary**
> Speculative decoding 让较快的 draft model（或模型内部的 draft 机制）先提出若干候选 token，再由 target model 并行验证。采用正确的接受/拒绝采样规则时，可保持 target model 的输出分布，同时有机会一次目标模型前向接受多个 token。收益取决于 draft 成本、接受率、验证并行度和 serving 负载，拒绝过多时可能不加速。

## 1. 自回归解码的串行性

普通 decode 每轮根据此前 token 计算下一个 token；在 memory-bound 场景，每轮都要读取模型权重，但只生成一个 token。Speculation 尝试用便宜的草稿工作填满目标模型的一次并行验证。

## 2. 执行流程

```text
+ draft:  x -> y1 -> y2 -> y3
+ target: 对 (x, y1, y2, y3) 做一次因果验证
+          接受连续匹配前缀；遇到拒绝时采样校正 token
+          下轮从已确认的前缀继续
```

Greedy 设置可按 target 的贪心结果检查草稿；随机采样若要严格保持 target 分布，需要相应的 rejection sampling 和 residual distribution 校正。只做“draft token 与 target token 相同才接受”一般不足以保证采样分布不变。

## 3. 关键变量与权衡

| 变量 | 增大时可能的效果 |
|---|---|
| 草稿长度 | 一次验证覆盖更多候选；但长草稿更可能在较早位置失配 |
| Draft 质量/速度 | 高接受率有帮助；草稿模型太贵可能抵消收益 |
| 验证 batch | target 一次处理多 token，可摊薄权重读取成本；也增加计算和临时状态 |
| 请求并发 | 高并发时 target 可能已被充分利用，speculation 的边际收益可能降低 |
| KV 与调度开销 | 草稿和校验需要维护临时/回滚状态，可能影响显存与延迟 |

## 4. 常见实现路线

- **独立 draft model**：结构较小的模型生成草稿；要处理 tokenizer/词表兼容和额外权重成本。
- **Self-speculation / early exit**：复用目标模型的部分层或内部预测头，减少独立模型开销。
- **树状候选**：提出共享前缀的多个分支，以更高验证工作量换潜在接受机会。
- **多 token prediction heads**：训练时加入辅助预测结构，具体校验算法因模型设计而异。

## 5. 评估方式

- 记录平均接受 token 数、每步草稿成本、target 验证成本和 KV 开销；
- 用目标采样参数验证输出分布或任务质量约束；严格无损结论需对应具体算法和配置；
- 同时比较单请求延迟、并发吞吐和 SLO 下 goodput；
- 覆盖不同提示类型与生成长度，草稿接受率可能随任务显著变化。

## 6. 一句话总结

> Speculative decoding 用便宜的候选生成加目标模型并行校验来跨越逐 token 串行瓶颈；正确算法可保持 target 分布，但能否加速由接受率和额外成本决定。

## References

- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- Chen et al., [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- vLLM, [Speculative decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode/)

*Last reviewed: 2026-09-29. Serving engines support different draft methods and sampling guarantees; verify version-specific behavior.*

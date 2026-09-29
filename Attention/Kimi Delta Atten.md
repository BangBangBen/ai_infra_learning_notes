---
course: AI Infra
topic: Kimi Delta Attention
tags: [ai-infra, attention, linear-attention, kimi]
---

# Kimi Delta Attention（KDA）

> **Summary**
> Kimi Delta Attention（KDA）是 Kimi Linear 中使用的一种线性注意力模块。它基于 delta-rule recurrent memory，并以细粒度、按通道的门控控制记忆衰减。相较于标准 softmax attention，它可以用固定大小的状态逐 token 更新，降低长序列推理时随上下文增长的 KV 状态需求；代价是检索机制与标准注意力不同，表达能力、训练稳定性和硬件效率取决于具体模型与实现。

## 1. 背景：从 softmax attention 到 recurrent state

标准因果注意力在每个 token 处读取此前所有位置的 key/value。上下文越长，解码时需要访问的 KV cache 越大。线性注意力尝试把历史压缩到一个递归状态，使状态大小不随序列长度线性增长，但压缩可能损害精确检索能力。

Delta-rule 一类方法把状态看作可更新的记忆：先读取当前 key 对应的旧记忆，再依据新旧信息之间的误差修正记忆。KDA 在这一框架中细化了遗忘控制。

## 2. 心智模型

```text
历史 token ──递归写入──> 有限大小的状态 S_t
                                  │
当前 key 查询 S_t ──> 读出旧记忆
                                  │
当前 value 与旧读出之差 ──> delta 更新
                                  │
按通道门控控制保留/衰减 ──> S_(t+1)
```

可用下式表达核心直觉（省略论文中的具体投影、门控参数化和多头细节）：

$$
\text{read}_t = S_{t-1} k_t, \qquad e_t = v_t - \text{read}_t
$$

随后用 key 与误差更新状态，并通过门控调节旧状态的保留。该式用于建立直觉，不是完整的 KDA 实现公式。

## 3. KDA 的关键变化

Kimi Linear 技术报告将 KDA 描述为对 Gated DeltaNet 的细粒度门控改进：门控由较粗粒度的衰减控制扩展到按通道控制，使不同记忆维度可以采用不同的衰减行为。这样模型可更灵活地表达位置与记忆保留模式。

Kimi Linear 是混合架构：KDA 与 Multi-Head Latent Attention（MLA）按层组合。不能把 KDA 单独等同于整套模型，也不能仅凭“线性”就推断它在所有硬件和序列长度下更快。

## 4. 推理含义与权衡

| 方面 | 潜在收益 | 需要关注的代价 |
|---|---|---|
| 历史状态 | recurrent state 大小可不随上下文逐 token 增长 | 状态仍随层数、头数和状态维度占用显存；混合模型还可能保留传统 KV cache |
| 解码 | 每步更新固定大小状态，适合长序列递归执行 | 小 batch 或短上下文时，专用 kernel 的常数开销可能抵消理论优势 |
| 表达能力 | 通道级遗忘提供比单一衰减更细的控制 | 压缩记忆与全量 token 检索不同，任务效果需要实测 |
| 实现 | 可将序列训练实现为块并行算法、推理实现为递归更新 | 训练与解码 kernel 结构不同，数值、布局和并行化较复杂 |

## 5. 与相关机制的边界

- **标准 softmax attention**：直接对历史 token 的 key/value 做内容寻址；通常依赖随上下文增长的 KV cache。
- **Linear attention**：更广的类别，通常将历史压缩到固定大小状态；不同变体的更新规则和能力不同。
- **Gated DeltaNet**：KDA 的相关基础之一；KDA 的重点是更细粒度的记忆衰减门控。
- **MLA**：通过潜在表示压缩注意力 KV 表示的另一条路线；Kimi Linear 将其与 KDA 分层混合。

## 6. 阅读与评估要点

- 区分论文的渐近复杂度、实际 kernel 延迟和端到端 serving 指标。
- 比较相同模型规模、训练数据、上下文长度与硬件下的质量和延迟。
- 长上下文评估应覆盖精确检索、长文问答、困惑度和生成质量，而不只看单一 benchmark。
- 报告推理内存时，要区分 recurrent state、传统 KV cache、权重和临时 workspace。

## 7. 一句话总结

> KDA 用通道级门控增强 delta-rule 线性注意力的记忆控制，目标是在有限状态下更有效地处理长序列；它的优势取决于混合架构、模型质量和实现效率，不能只由复杂度符号判断。

## References

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Kimi Team, [Kimi Linear Technical Report](https://yzhang.site/assets/pubs/techreport/2025/kda.pdf)
- Yang et al., [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)

*Last reviewed: 2026-09-29. KDA 是较新的架构，后续模型报告与 kernel 实现可能更新具体细节。*

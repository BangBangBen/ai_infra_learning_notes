---
course: AI Infra
topic: Double-P
tags: [ai-infra, llm-serving, attention, sparse-attention, paper]
---

# Double-P: Hierarchical Top-P Sparse Attention for Long-Context LLMs

> **Summary**
> Double-P is a hierarchical sparse-attention method for long-context decoding. It first estimates attention mass at the cluster level, then selectively refines token-level attention under a top-p mass criterion. The paper targets the trade-off among attention fidelity, selection overhead, and sparse computation. Reported speedups are paper-specific and should not be treated as guarantees for other models or serving stacks.

## 1. Problem setting

During autoregressive decoding, a query may attend over a growing KV cache. Fixed top-k selection gives every head a fixed token budget, although attention distributions vary across queries, heads, and layers. Top-p selection instead aims to retain enough tokens to cover a chosen portion of attention mass, but identifying those tokens can itself be expensive.

## 2. Method overview

1. Partition historical tokens into clusters.
2. Use size-weighted cluster summaries to estimate coarse attention mass.
3. Select/refine clusters adaptively according to a top-p criterion.
4. Compute token-level attention only for selected regions.
5. Combine the selected results with the appropriate normalization.

The hierarchy avoids first scoring every cached token at full token granularity. Correct mass estimation and refinement are important: pruning too aggressively can omit relevant tokens, while selecting too much erodes the compute savings.

## 3. Why two levels

| Stage | Purpose | Trade-off |
|---|---|---|
| Cluster-level estimate | Reduce the amount of detailed scoring | Centroid estimates may not perfectly represent every token |
| Token-level refinement | Spend compute where estimated mass is concentrated | Refinement and bookkeeping have overhead |
| Sparse attention | Avoid processing unselected token blocks | Irregular access and kernel overhead can reduce hardware efficiency |

The method's benefit depends on whether saved attention work exceeds the cost of selection, metadata, and irregular memory access.

## 4. Evaluation and interpretation

The paper reports near-zero accuracy drop on its evaluated long-context benchmarks, reduced attention computation overhead, and end-to-end decode gains over selected fixed-budget sparse-attention baselines. These are measurements for the paper's evaluated settings, not universal guarantees.

When reproducing or applying the method, record:

- model architecture, checkpoint, context length, and KV layout;
- quality and retrieval metrics alongside speed;
- selection overhead, attention-kernel time, and end-to-end inter-token latency;
- cluster size, top-p thresholds, and sensitivity to workload and head distribution.

## 5. Relation to other methods

- **Dense attention** computes over all eligible cached tokens and is the fidelity reference.
- **Fixed top-k sparsity** bounds selected tokens but may allocate compute poorly when attention is concentrated or diffuse.
- **Top-p attention** adapts token selection to cumulative attention mass; selection overhead is a central challenge.
- **FlashAttention / Split-KV** optimize dense attention execution and IO/parallelism; they do not themselves decide which tokens to prune.

## 6. One-sentence takeaway

> Double-P uses cluster-level estimation followed by adaptive token-level refinement to make top-p sparse attention more practical for long-context decode; quality and speed depend on the selection policy, kernel, and workload.

## References

- Ni et al., [Double-P: Hierarchical Top-P Sparse Attention for Long-Context LLMs](https://arxiv.org/abs/2602.05191)

*Last reviewed: 2026-09-29. This note summarizes the paper abstract and reported method; consult the full paper for exact algorithm and benchmark setup.*

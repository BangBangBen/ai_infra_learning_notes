---
course: AI Infra
topic: split-K GEMM
tags: [ai-infra, gpu-kernel, gemm, parallelism]
---

# Split-K GEMM（沿 K 维切分矩阵乘法）

> **Summary**
> Split-K 将 GEMM 的归约维度 K 切成多个区间，由更多 thread blocks 并行计算部分结果，再通过 workspace reduction 或原子累加合并。它能为 M、N 较小而 K 较大的问题增加并行度；代价是额外写读、归约开销、临时空间和浮点累加顺序变化，因此并非通用加速开关。

## 1. GEMM 的并行度问题

矩阵乘法 $C=AB$ 中，输出 tile 通常映射给一个 thread block，K 维在 block 内循环累加。当 M、N 很小而 K 很大时，可分配的输出 tile 数可能远少于 GPU 的计算单元数量，导致设备利用不足。

## 2. 心智模型

```text
K = K0 | K1 | K2 | K3
block 0: C0 = A[:,K0] B[K0,:] ┐
block 1: C1 = A[:,K1] B[K1,:] ├─ reduction -> C
block 2: C2 = A[:,K2] B[K2,:] ┤
block 3: C3 = A[:,K3] B[K3,:] ┘
```

若切成 S 份，则每个 split 对每个输出 tile 产生部分和；数学上 $C=\sum_{s=0}^{S-1} C_s$，同时还需正确处理 $\alpha$、$\beta C$、bias 和 epilogue。

## 3. 两类常见归约方式

| 方法 | 做法 | 主要代价 |
|---|---|---|
| Workspace + reduction kernel | 各 split 写入独立部分结果，再由第二个 kernel 相加 | 临时显存、额外全局内存流量和 kernel launch |
| Atomic accumulation | 各 split 对输出执行原子加 | 写冲突、原子吞吐、初始化与结果顺序问题 |

CUTLASS 的 `GemmSplitKParallel` 使用 K 维分区并另行执行 reduction。具体实现和可用选项取决于库版本与 kernel 配置。

## 4. 什么时候考虑

- 输出 tile 数不足以填满 GPU，而 K 维提供较多独立计算；
- 单次 GEMM 延迟比总吞吐更重要，额外归约仍可获益；
- split 粒度足够大，使增加的并行度超过同步与归约成本。

通常当 M、N 已能产生大量输出 tile、K 不大、或输出矩阵很大时，额外的部分结果流量可能让 split-K 更慢。

## 5. 数值与调优

- split 数越多，部分结果和归约开销越大，但可用并行度可能增加；
- 浮点加法不满足结合律，切分会改变累加顺序，需设定可接受的误差标准；
- 对比相同 dtype、布局、alpha/beta、epilogue 和同步条件下的基准；
- 检查 workspace 大小、原子争用、占用率和 reduction kernel 时间；
- 同时测试代表性的 M/N/K，而不是只测一个形状。

## 6. 与 Stream-K 的边界

传统 split-K 按固定 K 区间拆分，再显式归约部分输出。Stream-K 以工作量为中心，把 K 迭代工作分配给 thread blocks，以改善不同 GEMM 形状下的负载均衡；二者都利用 K 维并行，但调度与部分和组织方式不同。

## 7. 一句话总结

> Split-K 用额外归约和临时结果换取更多 GPU 并行度，适合输出 tile 并行不足而 K 较长的 GEMM；收益需按问题形状实测。

## References

- NVIDIA CUTLASS, [GEMM API: GemmSplitKParallel](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html)
- NVIDIA CUTLASS, [Split-K example](https://github.com/NVIDIA/cutlass/blob/main/examples/06_splitK_gemm/splitk_gemm.cu)
- Osama et al., [Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU](https://arxiv.org/abs/2301.03598)

*Last reviewed: 2026-09-29. CUTLASS kernel API 会随版本变化。*

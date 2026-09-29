---
course: AI Infra
topic: Stream-K GEMM
tags: [ai-infra, gpu-kernel, gemm, parallelism]
---

# Stream-K GEMM

> **Summary**
> Stream-K 是一种以 GEMM 计算工作量为中心的 GPU 并行分解方法。它把逻辑上的 K 维乘加工作分配给受控数量的 thread blocks，使不同形状的 GEMM 更均匀地利用 GPU，并通过 partial tile/归约机制处理跨 block 的输出累积。它改善的是调度和负载均衡，不改变 GEMM 的数学结果；额外调度和归约可能使某些形状变慢。

## 1. 动机：tile 数与设备规模不匹配

传统 GEMM 将输出矩阵切成二维 tiles，每个 tile 由一个 thread block 完成完整 K 归约。当输出 tile 数恰好适配 GPU 时效果很好；但 tile 数很少时设备空闲，tile 数与 SM 数不能整除时则可能有尾部负载不均。

## 2. 心智模型

```text
传统方式：每个输出 tile 是完整工作单元
[完整 K] [完整 K] [完整 K] ...

Stream-K：将 K 方向的 tile 迭代视为可调度工作
工作流 -> block 0: 部分输出区间
       -> block 1: 部分输出区间 -> 必要时合并 partial sums
       -> block 2: 部分输出区间
```

Stream-K 以工作量而非只以完整输出 tile 为调度单位，分配任务时尽量使并行单元有相近工作量。论文和实现包含不同变体，具体 partition 与 reduction 细节以所用实现为准。

## 3. 可能收益与成本

| 场景 | 预期影响 |
|---|---|
| GEMM 形状产生的输出 tile 太少 | 可切分 K 方向工作，提升活跃 block 数 |
| tile 数无法整齐匹配设备规模 | 可能减少尾部空闲、改善负载均衡 |
| 输出 tile 已足够多且负载均匀 | 传统路径可能更简单、更快 |
| 需要合并 partial results | 增加同步、workspace/原子操作或归约开销 |

## 4. 与 Split-K 的区别

Split-K 通常按固定份数切 K，并对每个输出 tile 生成多份部分结果后做归约。Stream-K 更侧重把整体 K-tile 工作按需分配到 thread blocks，目标是让工作量与硬件并行资源相匹配。实际库可能支持多种调度变体，名称不能替代对具体 kernel 的检查。

## 5. 评估方法

- 按 M、N、K、dtype、layout 和 batch 维度分组测试；
- 报告 kernel latency 与吞吐，不将单个大矩阵的峰值结果外推到全部形状；
- 检查 partial reduction 时间、workspace、原子争用和数值误差；
- 与生产库（如 cuBLAS/CUTLASS）在相同硬件、时钟和计时方法下对比；
- 对端到端模型 workload，确认 GEMM 不是被其他算子或数据移动掩盖的非瓶颈。

## 6. 一句话总结

> Stream-K 重新组织 GEMM 的 K 维工作分配，以减少输出 tile 数不合适时的 GPU 空闲和尾部不均；是否更快取决于形状与 partial reduction 的成本。

## References

- Osama et al., [Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU](https://arxiv.org/abs/2301.03598)
- NVIDIA CUTLASS, [GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html)

*Last reviewed: 2026-09-29.*

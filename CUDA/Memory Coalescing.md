---
course: AI Infra
topic: GPU memory coalescing
tags: [ai-infra, cuda, gpu, memory]
---

# GPU Memory Coalescing（全局内存合并访问）

> **Summary**
> Memory coalescing 描述 warp 中线程的全局内存访问如何合并为尽可能少的内存事务。让相邻线程访问相邻且对齐的数据，通常能减少事务数量、提高有效带宽；实际事务粒度和合并规则由 GPU 架构与访问类型决定，应以目标 CUDA 版本文档和 profiler 为准。

## 1. 基本机制

GPU 以 warp 为单位执行。一个 warp 中多个 lane 发出内存请求时，硬件根据地址、数据宽度和对齐情况，将请求映射到一个或多个内存事务。若每个 lane 访问连续数组中的相邻元素，请求往往可以合并；若 lane 跨大步长访问，可能需要更多事务，传输大量未使用的数据。

## 2. 心智模型

```text
连续访问：lane 0  1  2  3 ...  -> 连续地址范围 -> 少量事务
跨步访问：lane 0      1      2 -> 分散地址     -> 更多事务
```

对于行优先矩阵 `A[row][col]`，通常让相邻 lane 沿 `col` 方向访问连续元素。矩阵转置中读和写的理想方向相反，常用 shared memory 作为中间 tile，分别优化读取与写回。

## 3. 为什么影响性能

| 访问模式 | 典型结果 |
|---|---|
| 相邻 lane 访问连续、对齐地址 | 较少内存事务，数据利用率高 |
| 相邻 lane 访问大步长地址 | 事务增多，带宽利用率可能下降 |
| 地址未对齐或边界掩码较多 | 可能需要额外事务或产生部分利用 |
| 数据已命中 cache | HBM 流量影响可能较小，仍需关注指令和 cache 行利用率 |

Coalescing 是访问模式属性，不等同于“内存带宽已饱和”。Kernel 还可能受计算、依赖链、占用率、缓存命中、同步或 launch 开销限制。

## 4. 常见改进方法

- 设计索引映射，让相邻 lane 处理连续元素；
- 检查元素类型、向量化宽度和首地址对齐；
- 将转置或复杂重排拆成共享内存 tile，并注意 tile 边界；
- 通过 profiler 检查 global load/store efficiency、sector/transaction 数量和实际吞吐；
- 对比正确性与性能，确保改动没有引入越界访问或不必要的数据搬运。

## 5. 与相关概念的边界

- **Shared-memory bank conflicts**：发生在片上 shared memory 的 bank 映射，与 global memory coalescing 是不同层次的问题。
- **Cache locality**：讨论数据重用和层级命中；合并访问可改善传输效率，但不保证重用。
- **Vectorized load/store**：可能减少指令数或改善对齐访问，不会自动修复差的 lane 地址布局。

## 6. 一句话总结

> 让同一 warp 的线程尽量读取或写入相邻、对齐的数据，通常可以用更少的全局内存事务完成同样的工作；最终收益要结合架构、缓存行为和 profiler 结果判断。

## References

- NVIDIA, [CUDA C++ Programming Guide: Writing CUDA Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
- NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)

*Last reviewed: 2026-09-29.*

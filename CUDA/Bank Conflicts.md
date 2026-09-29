---
course: AI Infra
topic: shared memory bank conflicts
tags: [ai-infra, cuda, gpu, shared-memory]
---

# Shared Memory Bank Conflicts（共享内存 Bank 冲突）

> **Summary**
> Shared memory 被划分为多个 bank，warp 内不同线程访问不同 bank 时可并行服务。多个线程访问同一 bank 的不同地址时，访问会被拆成多个冲突自由的请求，降低吞吐。广播同一地址等情形可有特殊处理；bank 宽度和映射细节应按目标 GPU 文档确认。

## 1. 基本机制

Shared memory 是按 bank 交错组织的片上存储。典型配置中，连续 word 映射到连续 bank，地址按 bank 数取模映射。warp 的一次 shared-memory 指令若需要同一 bank 服务多个不同地址，就产生 bank conflict；访问同一地址的读取通常可广播，不属于普通的多地址冲突。

## 2. 典型例子：转置 tile

若二维 tile 声明为 `tile[32][32]`，warp 按列访问时，各 lane 的地址可能以 32 个 word 为步长，反复映射到同一 bank。将行宽填充为 33（`tile[32][33]`）可改变步长的模数关系，常用于避免经典转置访问中的冲突。

```text
未填充：列访问 stride = 32 words -> 同一 bank 周期性重叠
填充后：列访问 stride = 33 words -> bank 映射错开
```

这是针对特定布局、元素宽度和架构的示例，不是所有 shared-memory 冲突的通用修复。

## 3. 影响与排查

| 现象 | 说明 |
|---|---|
| 不同 lane 同 bank、不同地址 | 请求被分多次执行，吞吐下降 |
| 多个 lane 读同一地址 | 可由广播满足，通常不增加串行冲突 |
| 同一 lane 的向量化访问跨多个 bank | 属于访问宽度/分段问题，需按架构规则分析 |

使用 Nsight Compute 等工具查看 shared-memory wavefront、replay 或 bank-conflict 指标，并把指标对应到具体 load/store 指令和布局。先画出 lane 到地址再到 bank 的映射，通常比盲目加 padding 更有效。

## 4. 常见优化思路

- 调整 tile 的行跨度或增加 padding；
- 改变线程到矩阵坐标的映射，让 warp 访问不同 bank；
- 使用 swizzle/layout 转换以适配矩阵乘法或异步搬运；
- 同时评估额外 padding 对 shared-memory 用量、occupancy 和边界处理的影响。

## 5. 与 Memory Coalescing 的区别

Global memory coalescing 关注 warp 请求如何形成全局内存事务；bank conflict 关注 shared memory 内部的 bank 并行度。优化一种访问不保证另一种也理想，例如矩阵转置常用 shared memory 修复全局读写方向，但中间 tile 自身也需要避免 bank conflict。

## 6. 一句话总结

> Bank conflict 是多个线程让同一 shared-memory bank 同时服务不同地址而导致的串行化；通过布局、padding 或线程映射改变 bank 分布，并用 profiler 验证。

## References

- NVIDIA, [CUDA C++ Programming Guide: Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- NVIDIA, [CUDA C++ Programming Guide: Writing CUDA Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)

*Last reviewed: 2026-09-29. Bank 映射和指标名称可能随 GPU 架构与工具版本变化。*

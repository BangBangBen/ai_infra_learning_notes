---
course: AI Infra
topic: graph dispatching
tags: [ai-infra, graph-compiler, runtime, dispatch]
---

# Graph Dispatching（计算图分发）

> **Summary**
> Graph dispatching 是编译器或运行时依据计算图中的算子、形状、属性和目标硬件，选择可执行实现、kernel 或子图分区的过程。好的 dispatch 将工作交给兼容且高效的实现；它还必须处理约束、fallback、缓存和可观测性。它与图模式匹配相关，但匹配识别候选结构，dispatch 决定候选如何执行。

## 1. 分发发生在哪一层

“Graph dispatching”不是一个所有框架都统一命名的单一算法。它可能指：

- 根据 op type 与属性选择算子实现或 kernel；
- 根据 shape、dtype、layout 选择 specialized kernel；
- 将子图分配给 CPU、GPU、NPU 或不同设备；
- 在 eager、compiled、captured 等执行路径间选择；
- 在多个已编译版本之间做 runtime dispatch。

需要先明确讨论的是算子级 dispatch、子图 partition 还是设备级 placement。

## 2. 决策流程

```text
graph node/subgraph
       |
读取 op、shape、dtype、layout、device、约束
       |
匹配可用实现与合法性条件
       |
cost model / priority / heuristic 选择
       |
编译缓存命中? --是--> 调用实现
       |否
   编译或选择 fallback
```

选择器必须保证语义兼容；成本模型只在合法候选中比较，不能为了性能违反数值、内存布局或设备约束。

## 3. 常见选择信号

| 信号 | 可能影响 |
|---|---|
| 算子类型与属性 | 决定实现是否语义匹配，例如 stride、padding、groups |
| Shape 与动态维度 | 决定 tile、并行策略及是否需要通用 kernel |
| dtype 与精度规则 | 决定指令支持、累加类型和误差约束 |
| Layout / strides | 决定是否直接执行或先做数据重排 |
| 设备与 capability | 限制可用 kernel、指令和内存路径 |
| 子图边界与数据传输 | 决定融合收益是否会被切分或跨设备开销抵消 |

## 4. 静态与动态 dispatch

- **Compile-time dispatch**：形状等信息已知时，在编译期选择或生成专用实现；启动快，但可能产生较多变体。
- **Runtime dispatch**：运行时按输入 shape、设备状态或策略选择；适配动态输入，但增加分支、查表或测量开销。
- **Autotuning**：对候选实现基准测试并缓存选择；可能获得较好性能，但需要调优时间、稳定的测量环境和缓存键设计。

## 5. 设计与排查要点

- 明确候选实现的正确性约束和 fallback；
- 缓存键应覆盖影响合法性或性能的关键信息，避免错误复用；
- 动态 shape 下限制编译变体数量，控制编译时间和缓存增长；
- 记录选中实现与拒绝原因，便于解释意外性能；
- 使用目标工作负载测成本模型，防止单算子峰值误导端到端选择；
- 对不同 kernel 做数值容差与边界 shape 检查。

## 6. 与 Graph Pattern Matching 的区别

Graph pattern matching 在图中识别满足结构与属性约束的子图；graph dispatching 在候选实现间选择执行路线，也可能利用 pattern matcher 的结果。一个匹配规则可以生成融合候选，dispatch 再决定是否应用以及调用哪种实现。

## 7. 一句话总结

> Graph dispatching 将图的合法算子或子图映射到具体执行实现，核心是在语义约束内结合 shape、硬件和成本作出可解释的选择。

## References

- NVIDIA, [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/)
- TVM, [TensorIR and runtime documentation](https://tvm.apache.org/docs/)
- MLIR, [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)

*Last reviewed: 2026-09-29. 各框架对 dispatch、partition、placement 的命名范围不同。*

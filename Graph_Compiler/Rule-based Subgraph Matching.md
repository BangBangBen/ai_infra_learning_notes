---
course: AI Infra
topic: rule-based subgraph matching
tags: [ai-infra, graph-compiler, compiler-optimization, subgraph]
---

# Rule-based Subgraph Matching（基于规则的子图匹配）

> **Summary**
> Rule-based subgraph matching applies declared structural and attribute constraints to find candidate regions in a computation graph, then invokes a rewrite or implementation choice. Traversal order helps search efficiently but does not itself prove that a rewrite is correct.

## 1. 为什么需要遍历计算图？

在图模式匹配中，**目标是在一个大的计算图（data graph）中找出所有与给定模式（pattern）结构匹配的子图**。为了做到这一点，系统必须**系统性地访问图中的节点和边**，尝试在每个可能的位置“尝试匹配”模式。

这就需要一种**图遍历策略**——决定以什么顺序访问节点，如何探索邻居，如何避免重复或遗漏。

---

## 2. 计算图的结构特点

神经网络的计算图通常是：

- **有向图**（Directed）：数据从输入流向输出
- **无环图**（Acyclic）：训练/推理图一般不允许循环（除非显式支持 RNN/while_loop）
- **DAG（有向无环图）**：这是关键性质，使得**拓扑排序**可行

节点 = 算子（如 Conv、Add、Relu）边 = 张量（数据依赖）：从生产者（producer）指向消费者（consumer）

例如：

```
Input → Conv → BatchNorm → Relu → Output
```

---

## 3. 三种常用遍历策略

### 3.1 深度优先搜索（DFS）

#### 原理：

从一个起始节点出发，沿着一条路径尽可能深入，直到无法继续，再回溯尝试其他分支。

#### 在模式匹配中的用途：

- 适合**递归定义的模式匹配器**（如 TVM 的 matcher 是递归实现的）
- 可用于**从输出节点反向搜索**（backward traversal），因为很多优化模式是从“sink”（如 loss）往回找

#### 伪代码（简化版）：

```
def dfs_match(node, pattern_node):
    if not compatible(node, pattern_node):
        return False
    # 递归匹配所有输入（反向边）
    for i, input_node in enumerate(node.inputs):
        if not dfs_match(input_node, pattern_node.inputs[i]):
            return False
    return True
```

该递归 matcher 沿 producer 边检查前驱；真实实现还需处理分支、共享节点、回溯与去重。

#### 优点：

- 实现简单（天然递归）
- 内存局部性好（栈式访问）

#### 缺点：

- 容易陷入深分支（虽然 DAG 无环，但宽图可能栈溢出）
- 不保证按计算顺序访问（可能先访问深层节点）

---

### 3.2 广度优先搜索（BFS）

#### 原理：

从起始节点开始，逐层访问其邻居，用队列管理待访问节点。

#### 在模式匹配中的用途：

- 较少直接用于完整模式匹配，但可用于**预处理**（如收集所有 Conv 节点作为候选起点）
- 适合**多起点并行匹配**

#### 伪代码（找所有潜在起点）：

```
from collections import deque
candidates = deque()
for node in graph.nodes:
    if node.op_type == "Relu":  # 假设我们想找以 Relu 结尾的模式
        candidates.append(node)

while candidates:
    node = candidates.popleft()
    if try_match_pattern_ending_at(node):
        apply_rewrite(node)
```

#### 优点：

- 可控制搜索深度（如只匹配长度 ≤3 的子图）
- 适合并行化

#### 缺点：

- 匹配复杂模式时需额外维护状态（不如 DFS 自然）

---

### 3.3 拓扑排序（Topological Sort）

#### 原理：

对 DAG 中的节点进行线性排序，使得**所有边都从前指向后**（即每个节点在其所有前驱之后、所有后继之前）。

这是计算图最**自然且常用**的遍历顺序，因为：

- 符合数据流执行顺序（前向传播顺序）
- 保证在处理一个节点时，其所有输入（前驱）已经处理完毕

#### 如何用于模式匹配？

- **正向遍历**（从输入到输出）：适用于前缀模式（如 `Input → Conv`）
- **反向遍历**（逆拓扑序，从输出到输入）：**更常用**，因为大多数优化模式是“以某个算子结尾”的（如 `... → Relu`）

> 例如：要匹配 `Conv → BN → Relu`，通常从 `Relu` 节点开始，**逆拓扑序**回溯其输入。

#### 伪代码（逆拓扑匹配）：

```python
# 获取逆拓扑序（从输出往输入）
nodes_in_reverse_topo = reversed(topological_sort(compute_graph))

for node in nodes_in_reverse_topo:
    if node.op_type == "Relu":
        # 尝试匹配前驱是否为 BN，再前驱是否为 Conv
        if match_pattern_ending_with_relu(node):
            rewrite_subgraph(node)
```

#### 如何获取拓扑序？

- **Kahn 算法**（基于入度）：
    
    ```python
    from collections import deque
    
    def topological_sort(graph):
        in_degree = {node: 0 for node in graph.nodes}
        for node in graph.nodes:
            for child in node.outputs:
                in_degree[child] += 1
        
        queue = deque([n for n in graph.nodes if in_degree[n] == 0])
        result = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for child in node.outputs:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return result
    ```
    

#### 优点：

- **顺序合理**：天然符合数据依赖
- **高效**：O(V + E) 时间
- **候选顺序稳定**：可按数据依赖顺序处理节点；是否去重取决于 matcher 的状态与规则设计

#### 缺点：

- 需要先构建完整图（不适用于流式图）
- 对带环图失效（但神经网络推理图通常无环）

---

## 4. 实际系统中的选择

> ✅ **经验法则**：

---

## 5. 一个完整小例子

假设计算图：

```
x → Conv → y → BatchNorm → z → Relu → out
```

我们要匹配模式：`Conv → BatchNorm → Relu`

### 步骤：

1. 对图做拓扑排序：`[x, Conv, y, BatchNorm, z, Relu, out]`
2. 取逆序：`[out, Relu, z, BatchNorm, y, Conv, x]`
3. 遍历逆序：
    - 遇到 `Relu` → 检查其输入是否为 `BatchNorm`
    - 是 → 检查 `BatchNorm` 的输入是否为 `Conv`
    - 是 → 匹配成功！将这三个节点替换为 `FusedConvBNRelu`

---

## 6. 与相关概念的边界

- **Graph pattern matching** 是更广义的匹配任务；rule-based matching 是按显式规则定义的一种方式。
- **Graph rewriting** 在匹配成功后执行变换，必须维护图的语义和类型信息。
- **Subgraph isomorphism** 可用于严格结构比较，但优化规则往往允许通配输入和属性约束。
- **Graph dispatching** 在实现或设备候选中选执行路径，可把匹配结果作为输入。

## 7. 一句话总结

在 DAG 上，拓扑序可提供符合数据依赖的候选处理顺序；它本身不保证匹配完备性或重写正确性，这些性质取决于 matcher 与规则定义。

如果你正在实现一个图优化 pass，建议：

1. 先对计算图做拓扑排序
2. 逆序遍历节点
3. 对每个“可能结尾”的节点启动递归 matcher；
4. 检查外部消费者、属性约束和重写语义；
5. 在代表性 shape 与 dtype 上验证性能及数值行为。

## References

- Apache TVM, [Relay pattern language](https://tvm.apache.org/docs/reference/langref/relay_pattern.html)
- PyTorch, [FX subgraph rewriter](https://docs.pytorch.org/docs/stable/fx.html)
- MLIR, [Pattern Rewriter](https://mlir.llvm.org/docs/PatternRewriter/)

*Last reviewed: 2026-09-29. Matcher and rewrite APIs vary across compiler frameworks and versions.*

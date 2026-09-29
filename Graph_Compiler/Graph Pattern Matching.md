---
course: AI Infra
topic: graph pattern matching
tags: [ai-infra, graph-compiler, compiler-optimization, pattern-matching]
---

# Graph Pattern Matching（计算图模式匹配）

> **Summary**
> 图模式匹配是在计算图中寻找符合目标拓扑、算子类型和属性约束的子图，常用于算子融合、图重写、硬件映射与模型分析。深度学习编译器更常做带语义约束的子图匹配，而非严格图同构；匹配成功后仍需验证重写的数值与形状语义。

## 1. 计算图与模式

在深度学习框架（如 PyTorch、ONNX、TVM）中，计算图通常是有向图；推理图常可视作 DAG。节点表示算子（如 Conv、MatMul、Relu），边表示张量依赖。

**“神经网络计算图的图模式匹配”** 通常指在计算图中识别特定的子图结构（模式），用于：

- **图优化**（如算子融合、常量折叠）
- **硬件加速**（如将 Conv+BN+Relu 映射为一个硬件指令）
- **模型压缩**（如识别可剪枝/量化的结构）
- **安全检测**（如检测后门触发器模式）
- **中间表示转换**（如 ONNX → TensorRT）

---

## 2. 核心特点

> ✅ 关键：匹配往往基于**算子类型序列+数据流拓扑+属性约束**，而非严格图同构。

---

## 3. 常用方法

### 3.1 基于规则的子图匹配

这是主流深度学习编译器（如 TVM Relay、TensorFlow Grappler、ONNX Optimizer）的核心方法。

#### 步骤：

1. **定义模式图**（Pattern）：用 DSL 或代码描述目标子图（如 `Conv → BatchNorm → Relu`）
2. **遍历计算图**：采用 DFS/BFS 或拓扑排序
3. **匹配与重写**：若匹配成功，替换为优化后的算子（如 fused ConvBNRelu）

#### 示例（TVM Relay）：

```python
# 定义模式
conv = is_op("nn.conv2d")(wildcard(), wildcard())
bn = is_op("nn.batch_norm")(conv, wildcard(), wildcard(), wildcard(), wildcard())
relu = is_op("nn.relu")(bn)

# 注册优化规则
@relay.ir_pass.function_pass(opt_level=3)
class ConvBnReluFuse:
    def transform_function(self, func, mod, ctx):
        return rewrite(relu, fused_op)(func)
```

#### 代表系统：

- **TVM Relay**：使用 `Pattern Matcher`
- **TensorFlow Grappler**：`Remapper` 优化器识别常见模式
- **ONNX Optimizer**：基于 `onnx.utils.extract_model` + 规则
- **TensorRT**：`Layer/Graph Fuser` 自动融合模式

---

### 3.2 基于图同构的匹配

仅在严格结构匹配场景使用（如模型克隆检测），但因计算图对属性敏感（stride、padding 等），通常需扩展为 **带属性的子图同构**（Attributed Subgraph Isomorphism）。

- 可用 VF2 算法，但需扩展节点兼容性函数（如：Conv 节点仅当 `kernel_size` 和 `groups` 相同时才匹配）

---

### 3.3 基于图神经网络的相似性检索

用于**模糊匹配**或**语义相似性检测**（如检测两个模型是否功能等价）：

- 将子图编码为向量（GNN Embedding）
- 通过向量相似度判断是否匹配
- 应用于模型水印、版权保护、后门检测

> 🔍 例如：NeuroX、Graph2Vec 用于计算图表示学习。

---

## 4. 关键技术挑战

1. **属性匹配**：不仅要看算子类型，还要检查 `kernel_size=3`, `stride=1`, `in_channels=64` 等属性。
2. **拓扑灵活性**：
    - 模式可能有分支（如 ResNet 的 skip connection）
    - 支持通配符（`wildcard()`）或可选节点（如 BN 可能被吸收到 Conv 中）
3. **高效性**：编译期需快速匹配，避免指数级搜索。
4. **跨框架兼容**：ONNX 作为中间表示，常用于统一模式定义。

---

## 5. 常用工具

✅ PyTorch FX 示例：

```python
from torch.fx import subgraph_rewriter

def pattern(x):
    return torch.relu(torch.nn.functional.batch_norm(conv(x), ...))

def replacement(x):
    return fused_conv_bn_relu(x)

subgraph_rewriter.replace_pattern(gm, pattern, replacement)
```

---

## 6. 与相关概念的边界

神经网络计算图的图模式匹配 ≠ 传统图同构，而是：

- **以优化为导向的语义匹配**
- **基于规则 + 属性约束 + 数据流拓扑**
- **广泛应用于编译器和推理引擎**

## 7. 一句话总结

> 图模式匹配用拓扑、算子类型和属性约束定位优化候选；匹配只是编译决策的一步，重写还必须保持语义正确并带来目标 workload 上的收益。

## References

- Apache TVM, [Relay pattern language](https://tvm.apache.org/docs/reference/langref/relay_pattern.html)
- PyTorch, [FX subgraph rewriter](https://docs.pytorch.org/docs/stable/fx.html)
- ONNX, [ONNX IR specification](https://onnx.ai/onnx/repo-docs/IR.html)

*Last reviewed: 2026-09-29. Framework APIs and pattern semantics vary by version.*

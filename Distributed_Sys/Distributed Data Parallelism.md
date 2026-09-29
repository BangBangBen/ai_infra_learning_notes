---
course: AI Infra
topic: distributed data parallelism
tags: [ai-infra, distributed-training, pytorch, ddp]
---

# Distributed Data Parallelism（数据并行训练）

> **Summary**
> Distributed Data Parallel（DDP）让多个进程各自持有一份模型副本，并用不同的数据分片计算梯度；反向传播期间，各副本通过 all-reduce 同步梯度，再执行一致的优化器更新。它提升训练吞吐与数据规模，但模型、优化器状态仍需适配每个副本的显存，通信、负载不均和同步等待会限制扩展效率。

## 1. 一步训练如何工作

1. 初始化时各 rank 建立通信组并确保模型初始状态一致；
2. 每个 rank 处理本地 mini-batch，执行 forward 与 backward；
3. 梯度 ready 后分桶并发起 all-reduce，可与尚未完成的反向计算重叠；
4. 同步后的梯度供每个 rank 的 optimizer 更新本地副本；
5. sampler 通常为不同 rank 分配不重复的数据子集。

在常见同步 SGD 设置中，all-reduce 后的平均梯度对应所有 rank 的联合 batch 梯度（具体缩放还受 loss reduction 与配置影响）。

## 2. 通信心智模型

```text
rank 0: 数据分片 0 -> 前/反向 -> 梯度 ┐
rank 1: 数据分片 1 -> 前/反向 -> 梯度 ├─ All-Reduce -> 同步更新
rank N: 数据分片 N -> 前/反向 -> 梯度 ┘
```

DDP 的数据并行不切分单个模型副本。若模型及优化器状态装不进单卡，需要 FSDP、ZeRO 或 tensor/pipeline parallel 等模型切分方法。

## 3. 性能与扩展性

| 因素 | 影响 |
|---|---|
| 每 rank 的 batch/计算量 | 计算越多，通信越容易被反向传播隐藏；过小 batch 时同步占比上升 |
| 梯度 bucket 大小 | 小 bucket 较早通信但通信次数多；大 bucket 启动次数少但可能较晚开始 |
| 网络带宽和拓扑 | 决定 all-reduce 代价，机间通信常比机内链路更敏感 |
| 数据与计算均衡 | 慢 rank 会让其他 rank 在同步点等待 |
| 参数参与情况 | 未使用参数检测、静态图假设等配置会改变开销与合法性 |

## 4. 常见检查项

- 用 `torchrun` 等 launcher 启动每个 GPU 对应的进程，并确保 rank、world size 与设备映射正确；
- 配置 DistributedSampler，并在每个 epoch 调用相应的 epoch 更新接口；
- 确认每个进程的数据和随机种子策略符合实验可复现要求；
- 确认所有 rank 以兼容顺序参与 collective，避免某个进程提前退出导致挂起；
- 观察 step time 中计算、通信和 dataloader 的占比，再调整 bucket、batch 与拓扑；
- 区分 global batch、每 rank batch 与梯度累积后的有效 batch。

## 5. 与其他并行策略的关系

- **Data parallelism**：每个 rank 计算不同样本、复制模型并同步梯度。
- **Tensor parallelism**：把层内张量计算切到多个设备，常有更频繁的激活通信。
- **Pipeline parallelism**：按层切模型，用微批在阶段间传递激活。
- **FSDP/ZeRO**：分片参数、梯度或优化器状态，降低副本显存占用并引入参数通信。

这些方法可以组合，组合后需要共同考虑通信拓扑和 batch 划分。

## 6. 一句话总结

> DDP 复制模型、切分数据、同步梯度，是模型能放入单卡时常见的多 GPU 训练方式；扩展效果取决于每步计算量能否覆盖梯度通信和慢 rank 等待。

## References

- PyTorch, [DistributedDataParallel API](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- PyTorch, [Distributed Data Parallel notes](https://docs.pytorch.org/docs/stable/notes/ddp.html)
- PyTorch, [Distributed package](https://docs.pytorch.org/docs/stable/distributed.html)

*Last reviewed: 2026-09-29. PyTorch API 参数与后端行为随版本变化，应核对部署版本文档。*

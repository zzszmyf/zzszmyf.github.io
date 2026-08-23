---
title: "深入解读 Primus：面向大规模大语言模型的高性能训练框架"
date: 2024-04-11T00:00:00+08:00
draft: false
categories: ["研究笔记"]
tags: ["研究笔记", "LLM", "AI Infra"]
---

# 深入解读 Primus：面向大规模大语言模型的高性能训练框架

> **原文链接**：https://zhuanlan.zhihu.com/p/2011808660167337197
> 
> **原文作者**：Vidushi Goyal, Wei Cai, Yao Fu, George Wang, Wen Xie, Xiaobo Chen
> 
> **发布机构**：AMD中国（AMD开发者中心）

---

## 简介

**Primus** [1] 是 AMD 推出的统一训练框架，面向大规模大语言模型（LLM）的高性能、可扩展训练场景，支持多个后端，包括 TorchTitan 和 Megatron-LM。Primus 提供统一的命令行（CLI）入口，同时为不同后端提供预先调优好的配置，覆盖主流开源模型。这些后端预设专门针对 AMD GPU 优化，开箱即可获得优异训练性能。

本文将围绕如何在 Primus 上训练「dense LLMs (稠密大模型)」时获得接近峰值的性能，做一次系统的深度解读和实战建议。

---

## 性能瓶颈分析

要搞清楚应该优先优化哪里，先以 Llama 3.1 70B 为例，分析 dense LLMs 在 Primus 上训练时的性能瓶颈。

### GPU 时间线统计（表 1）

在 Primus（TorchTitan 后端，未启用优化）上运行 Llama 3.1 70B 的数据如表 1 所示：

| 统计项 | 占比 |
|--------|------|
| 计算时间 | >99% |
| 通信开销 | 很小 |
| 空闲时间 | 很小 |

**结论**：超过 99% 的总训练时间都花在计算上；通信开销和空闲时间都很小，说明整体任务高度**计算受限（compute-bound）**。

### Kernel 算子统计（表 2）

进一步从算子/Kernel 维度看：

| 算子类型 | 占比 |
|----------|------|
| aten::mm（GEMM）| ~47% |
| FlashAttention | ~47% |
| 其他 | ~6% |

**结论**：`aten::mm`（GEMM）和 FlashAttention 两类算子加总占据了约 **94%** 的训练时间，这些核心算子几乎决定了 dense LLMs 的整体训练成本。

因此，Primus 生态中集成了一套优化 Kernel 库，**Primus-Turbo**，针对 GEMM 和 FlashAttention 做了**架构感知（architecture-aware）**调优，并提供基于 ROCm 的高性能实现，从而显著提升整体训练吞吐。

---

## FlashAttention 优化

既然 FlashAttention 是 dense LLMs 的主要计算热点之一，Primus 通过 Primus-Turbo 集成了来自 **AITER** [3] Kernel 库的优化实现。

当启用 Primus-Turbo 时，Primus 会自动切换到 AITER 中的：
- `aiter::fmha_v3_bwd`（后向算子）
- `aiter::fmha_v3_fwd`（前向算子）

### 性能对比（图 1）

| 算子 | 优化后延迟降低 |
|------|---------------|
| 后向算子（fmha_v3_bwd）| 约 **75%** |
| 前向算子（fmha_v3_fwd）| 约 **47%** |

这些 Kernel 将 FlashAttention 的时延显著拉低，对 Llama 3.1 70B 等模型的端到端训练性能提升非常明显。

---

## GEMM 调优

除了 FlashAttention 之外，GEMM（`aten::mm`）是 dense LLMs 训练时间的最大开销来源。ROCm 生态中提供了两种互补的 GEMM 调优方式：

### 方式一：在线 GEMM 调优（Online Tuning）

- **工具**：ROCm Transformer Engine
- **特点**：使用较小的搜索空间，在训练过程中轻量地进行 Kernel 选择
- **优势**：适合需要在训练时完成调优的场景，集成成本低
- **适用**：需要实时动态调优的训练任务

### 方式二：离线 GEMM 调优（Offline Tuning）

- **工具**：hipblaslt-bench
- **特点**：探索更大的搜索空间
- **优势**：一次离线搜完后，可缓存结果，在后续训练中重复使用，从而在不增加训练时间开销的前提下获得更优性能
- **适用**：追求极致性能，可预先准备调优结果

### 共同点

两种方式都基于 AMD 推荐的 **hipBLASLt** 后端，从多个实现中选出性能最佳的 GEMM Kernel。

### 性能收益（图 2）

对 Llama 3.1 70B 用到的 GEMM Kernel 进行调优后，性能最高可提升约 **5%**。

### 集成优势

基于 AITER FlashAttention 和 GEMM 调优带来的性能收益，Primus 将这些优化直接集成到了端到端训练工作流中，用户无需额外手动配置即可持续受益于最佳 Kernel 性能。

---

## 使用 Primus 进行稠密 LLM 端到端训练

本节将分别讨论基于 PyTorch 的两个后端，Megatron-LM 与 TorchTitan，在 Primus 中进行 dense LLMs 训练时推荐的分片与并行策略。

---

## A) Primus-Megatron 训练

Primus 支持多种 dense LLMs。下面以三个模型为例，给出在 Megatron-LM 后端下的端到端训练配置方案：

### 1、Qwen2.5 7B 训练配置方案

**模型特点**：
- 参数量：7B
- 层数：28 层
- 相对较小的 dense LLM

**关键特性**：
配合分布式优化器，它的全部参数可以轻松放入一块 AMD GPU 里。

**推荐策略：纯数据并行（DDP）**

优势：
- 最大化利用单卡显存容量
- 避免在单节点 8 卡之间通过相对较慢的 p2p 链路进行多余的集体通信
- 在小型 dense LLM 上，可以在 AMD GPU 上获得最高吞吐

### 2、Llama 3.1 70B 训练配置方案

**模型特点**：
- 参数量：70B
- 已超出单卡显存容量，需要进行模型分片

**推荐策略：FSDP2**

配置要点：
- **FSDP2** 同时对参数、梯度和优化器状态进行高效分片
- 开启 `overlap_grad_reduce = true`，在进行梯度规约的同时与计算重叠，隐藏通信时延
- 结合**全量激活重计算（full activation recompute）**，可以让 Llama 3.1 70B 完整模型容纳在单个 AMD GPU 节点中

**多节点扩展（8 个节点）**：
当训练规模扩展到 8 个节点后，由于参数、梯度和优化器状态进一步分散在更多 GPU 上，显存压力随之降低，这时可以：
- 适当放宽激活重算策略
- 只对部分层进行重算
- 在保持模型可放入显存的前提下减少重算开销，进一步提升吞吐

### 3、Llama 3.1 405B 训练配置方案

**模型特点**：
- 参数量：405B
- 远大于 70B，已经无法在单节点（8 卡）内容纳
- 必须采用多节点训练

**推荐策略：TP + PP + VPP 组合**

| 并行策略 | 说明 |
|----------|------|
| **Tensor Parallelism（TP）** | 张量并行 |
| **Pipeline Parallelism（PP）** | 流水线并行 |
| **Virtual Pipeline Parallelism（VPP）** | 虚拟流水线并行 |

**重要限制**：
与 70B 不同的是，在 Primus-Megatron 下**不推荐对 405B 使用 FSDP2**。原因：
- Megatron 的 FSDP2 实现并未对激活进行分片
- 导致激活占用显存过高
- 即便是 AMD GPU 多节点集群也容易 OOM

### Primus-Megatron 端到端性能测试结果（图 3）

对 Qwen2.5 7B、Llama 3.1 70B、Llama 3.1 405B 分别采用上述推荐分片与优化策略后，在 AMD GPU 上做了端到端训练吞吐测试。

测试配置：
- **单节点（1N）**：1 个节点测试
- **8 节点（8N）**：8 个节点测试
- **Qwen2.5 7B**：使用 DDP
- **Llama 3.1 70B**：使用 FSDP2
- **Llama 3.1 405B**：使用 TP/PP/VPP 组合

整体表现出良好的扩展性和跨模型规模的优化训练性能。

---

## B) Primus-TorchTitan 训练

本节介绍在 TorchTitan 后端下，使用 Primus 训练两类 dense LLMs 的端到端训练配置方案：

### 1、Llama 3.1 8B 训练配置方案

**TorchTitan DDP 与 Megatron DDP 的区别**：
TorchTitan 的 DDP 与 Megatron 的 DDP 不同：TorchTitan **不使用分布式优化器**，因此单卡显存占用更高。

**推荐策略：FSDP 分片**

为了在 AMD GPU 上提升可用 batch size、减小显存压力，即便是相对较小的 8B 模型，仍然推荐使用 FSDP 分片进行训练：
- FSDP 对参数、梯度、优化器状态三者一起分片
- 有助于在相同硬件上容纳更大 batch size，更好地利用算力

### 2、Llama 3.1 70B 训练配置方案

对于 Llama 3.1 70B，推荐策略与 Primus-Megatron 基本一致：
- 使用 **FSDP2** 进行分布式分片
- 配合**激活重计算**，显著降低激活显存占用
- 使 Llama 3.1 70B 可以放入单个 AMD GPU 节点中

这一组合在 TorchTitan 后端上兼顾吞吐和显存效率，非常适合大规模 dense LLMs 训练。

### Primus-TorchTitan 端到端性能测试结果（图 4）

在对 Llama 3.1 8B 和 Llama 3.1 70B 使用上述基于 FSDP2 的训练配置方案后，在单个 AMD GPU 节点上测试了端到端训练吞吐。

---

## 更多性能参考

关于更多模型及不同设备上的训练性能与推荐设置，可参考 AMD GPU 性能页面 [4]。

---

## 总结

要在大规模场景下高效训练 dense LLMs，需要在**计算效率、显存利用和并行策略**之间找到精细的平衡。随着模型规模持续增大，性能瓶颈会越来越集中在 GEMM 与 Attention 等核心 Kernel 上，因此对这些低层算子的优化尤为关键。

### Primus 优化手段总结

Primus 通过多层次的手段，在 AMD GPU 上系统性地优化了 dense LLMs 训练：

1. **聚焦核心计算热点**：GEMM 和 FlashAttention
2. **集成 Kernel 级加速**：通过 Primus-Turbo 集成 AITER 和 hipBLASLt 调优
3. **针对不同模型规模与训练后端，给出具体可落地的并行与分片策略**：
   - 小模型（7B/8B）：DDP 或 FSDP
   - 中模型（70B）：FSDP2 + 激活重计算
   - 大模型（405B）：TP + PP + VPP，避免 Megatron 的 FSDP2

本文展示了 Primus 如何把这些优化能力整合到统一的训练工作流中，帮助用户理解性能瓶颈的位置、如何有效缓解这些瓶颈，以及如何在 AMD GPU 上以最小调参成本，获得可扩展的高性能 dense LLMs 训练。

---

## 附录：上手路径与进一步阅读

### 使用 Primus 进行训练

| 资源 | 链接/说明 |
|------|----------|
| 使用 Primus + Megatron-LM 训练模型 [5] | 基于 Megatron 后端，在 Primus 中完成 dense LLMs 的端到端环境搭建与训练流程 |
| 使用 Primus + PyTorch（TorchTitan）训练模型 [6] | 适合希望使用 TorchTitan 后端进行 dense LLMs 训练的用户 |

### 性能调优与性能分析

| 资源 | 链接/说明 |
|------|----------|
| 离线 GEMM 调优（文档）[7] | 介绍如何使用 hipBLASLt 进行离线 GEMM 调优 |
| 离线 GEMM 调优（Primus 应用示例）[8] | 提供在 Primus 中进行离线调优的示例和具体用法 |
| TraceLens：性能分析工具 [2] | 帮助你从系统层与 Kernel 层深入分析性能瓶颈 |

### 更多 Primus 能力与场景

| 资源 | 链接/说明 |
|------|----------|
| Primus-SaFE：面向基础模型的可扩展训练平台 [9] | 一套面向大规模部署的全栈训练平台，关注多节点 AMD GPU 环境中的集群稳定性、可调试性与可观测性 |
| Primus for Large Models：面向大模型的训练方案 [10] | 详细介绍 Primus-Turbo 及 Primus 全栈在大模型训练上的能力，包括性能优化库和可扩展训练工作流 |

---

## 免责声明

第三方内容由各自的第三方权利人直接授权给你，并非由 AMD 授权。所有链接的第三方内容均按"现状"提供，不附带任何形式的担保。你对该等第三方内容的使用完全由你自行决定，因使用第三方内容造成的任何损失，AMD 在任何情况下均不承担责任。你需要自行承担使用第三方内容所带来的全部风险，并对由此产生的任何损害负责。

---

## 参考链接

| 编号 | 名称 | 链接 |
|------|------|------|
| [1] | Primus：在 AMD GPU 上用于大规模模型的轻量化统一训练框架 | - |
| [2] | TraceLens | https://github.com/AMD-AGI/TraceLens/tree/main |
| [3] | AITER | https://github.com/ROCm/aiter |
| [4] | AMD GPU 性能结果页 | https://www.amd.com/en/developer/resources/rocm-hub/dev-ai.html |
| [5] | 使用 Primus 和 Megatron-LM 训练模型 | https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/index.html |
| [6] | 使用 Primus 和 PyTorch（TorchTitan）训练模型 | Use ROCm for training |
| [7] | hipBLASLt 离线 GEMM 调优文档 | Using hipBLASLt offline tuning |
| [8] | Primus 离线 GEMM 调优示例 | https://github.com/AMD-AGI/Primus/tree/main/.github |
| [9] | Primus-SaFE：Scalable and Efficient Training for Foundation Models | 规模化稳定性：AMD 面向大模型训练的全栈平台 |
| [10] | Primus for Large Models | Primus-Turbo 简介：在 AMD GPU 上加速 Transformer 模型的高性能库 |

---
title: "深入解读 Primus：面向大规模大语言模型的高性能训练框架"
description: "Primus 是 AMD 推出的大规模 LLM 统一训练框架，支持 Megatron-LM 与 TorchTitan 后端。深度解读其性能瓶颈（GEMM 与 FlashAttention 合计约占 94% 训练时间）、Primus-Turbo 集成 AITER 与 hipBLASLt 调优的收益，以及 7B/70B/405B 模型分别对应 DDP、FSDP2、TP+PP+VPP 的并行策略选择。"
date: 2026-08-23T17:25:14+08:00
lastmod: 2026-10-18T12:24:51+08:00
draft: false
weight: 100
aliases: ["/posts/primus-llm-training-framework/"]
categories: ["研究笔记"]
tags: ["研究笔记", "LLM", "AI Infra"]
---

Llama 3.1 70B 在 Primus 的 TorchTitan 后端上跑一遍，超过 99% 的时间花在计算上，通信和空闲都只占很小一部分。再往下拆，`aten::mm` 和 FlashAttention 各占约 47%，两类算子合计约 94%。这组数字把优化顺序限制得很清楚：先看 attention 和 GEMM，别一上来就调通信。

## 一、先 profile，再谈优化

这组数据的上下文要说清楚：模型是 Llama 3.1 70B，后端是 TorchTitan，Primus 没有启用优化，测试跑在 AMD GPU 上。换一个模型、换一套并行策略，通信占比可能上来；但在 dense LLM 的常规训练里，GEMM 和 attention 通常是最大的两块计算。Primus 的判断是，把这两类 kernel 的峰值性能榨出来，比先动通信层更划算。

Primus 是一个统一训练框架，支持 TorchTitan 和 Megatron-LM 两个后端，提供统一的 CLI 入口和针对 AMD GPU 预先调好的配置。它的性能优化集中在 Primus-Turbo：一套面向 GEMM 和 FlashAttention 的架构感知 kernel 库，底层接 ROCm 的 AITER 和 hipBLASLt。这个分层很重要：上层并行策略负责切模型，下层 kernel 负责把切开的每一步算快；两层都做对，端到端吞吐才上得去。

## 二、FlashAttention：47% 和 75% 是算子级数字

启用 Primus-Turbo 后，Primus 会自动把 FlashAttention 切到 AITER 的 `aiter::fmha_v3_fwd` 和 `aiter::fmha_v3_bwd`。原文给出的延迟降幅是前向约 47%、后向约 75%。后向降得更多，和后向本身更重有关：它要多读一遍激活、多算一遍 QK^T 和 softmax 的反向，还要处理 dropout 掩码，数据搬运和访存的压力都比前向大；融合 kernel 把注意力的多个步骤压进一个 kernel，减少 HBM 往返，收益在后向更明显。

这两个数字是算子级延迟，不是端到端训练吞吐。原文没有给出启用 AITER 后整步训练快了多少，也没有说明测试用的序列长度、head dim 和 batch 形状 [待补：fmha_v3 前向/后向在目标 shape 下的端到端 tokens/s 提升]。attention kernel 的收益高度依赖 shape：序列长度太短，融合带来的收益会被固定开销吃掉；head dim 或 batch 不合适，kernel 可能选不到最优的 tile。把 47% 和 75% 直接乘到模型吞吐上，会高估实际收益。

## 三、GEMM：在线调优和离线调优的取舍

GEMM 是另一块主计算热点。ROCm 生态给了两种调优方式，二者都落在 hipBLASLt 后端上，从多个实现里选性能最好的 GEMM kernel。在线调优用 ROCm Transformer Engine，搜索空间较小，在训练过程中轻量地做 kernel 选择，集成成本低，适合形状会随训练配置变化的场景；代价是搜索范围有限，碰到极端 shape 不一定能找到最优解。

离线调优用 hipblaslt-bench，搜索空间大，一次搜完把结果缓存下来，后续训练直接复用，不增加训练时的搜索开销。它适合追求极致性能、并且能提前准备好调优结果的团队；前提是你要有一组有代表性的 shape，模型结构、batch、序列长度一变，缓存就可能失效，需要重新搜。在线和离线可以组合使用：先用离线搜出大致的候选，再用在线调优覆盖训练中实际出现的形状。

原文给出的 GEMM 调优收益是“最高可提升约 5%”。这句话没有写清是 kernel 级还是端到端 [待补：5% 的统计口径是单 kernel 延迟还是端到端训练吞吐]，规划时应该按上限理解，不要把它当成每个 GEMM 都能拿到的稳定收益。5% 和 attention 的 47%/75% 不在一个量级，原因在 GEMM 的 kernel 生态更成熟、可融合的空间更小；这也决定了优化的先后顺序：先把 attention 的融合收益吃掉，再回头抠 GEMM 的 5%。

## 四、按模型规模选并行策略

Primus-Megatron 对三个模型给了三套配置。Qwen2.5 7B 是 28 层，FP16 权重约 14GB，单卡放得下；优化器状态通过分布式优化器分片到 DDP 组，单卡显存压力进一步下降。Primus 推荐纯数据并行（DDP）：每张卡持有完整模型，单节点内梯度 All-Reduce 走 NVLink，扩到多节点时再走 InfiniBand，不需要张量并行或流水线并行的额外通信。这个选择的收益是最大化单卡显存、避免为 TP/PP 额外插入 p2p 集合通信，在小模型上能拿到最高吞吐。

Llama 3.1 70B 已经超出单卡显存，Primus-Megatron 推荐 FSDP2。FSDP2 对参数、梯度和优化器状态一起分片，前向之前按层 All-Gather 参数，反向之后 Reduce-Scatter 梯度，通信和计算可以重叠；配置里把 `overlap_grad_reduce` 设为 `true`，让梯度规约和反向计算同时进行，把通信时延藏起来。再加上全量激活重计算，70B 可以完整放进一个 8 卡 AMD GPU 节点。训练扩到 8 个节点后，参数、梯度和优化器状态分散到更多 GPU 上，显存压力下降，这时可以把全量重算放宽成只对部分层重算，减少重算带来的额外计算。

Llama 3.1 405B 必须多节点，Primus-Megatron 推荐 TP + PP + VPP 组合。张量并行在节点内切矩阵乘法，流水线并行跨节点切层，虚拟流水线并行（VPP）把每个 stage 再切成多个虚拟 stage，让一个 GPU 负责不连续的层，减少流水线气泡。405B 不用 FSDP2 的原因很具体：Megatron 的 FSDP2 实现没有对激活做分片，参数和优化器状态虽然切开了，但 405B 的激活占用仍然过高，多节点集群也容易 OOM。这个限制不是调参能绕过去的，选型时只能换并行方案。

## 五、TorchTitan 后端为什么要换一套推荐

同一个模型规模，在 TorchTitan 后端下的推荐和 Megatron 不完全一样。TorchTitan 的 DDP 不使用分布式优化器，单卡显存占用比 Megatron 的 DDP 高；Llama 3.1 8B 虽然不大，Primus 仍然推荐用 FSDP 分片，把参数、梯度和优化器状态切开，以便在相同硬件上容纳更大的 batch size。7B/8B 这个量级在 Megatron 后端可以纯 DDP，在 TorchTitan 后端却要 FSDP，差别来自 DDP 实现，和模型本身无关。

Llama 3.1 70B 在 TorchTitan 后端推荐 FSDP2 加激活重计算，和 Megatron 后端的思路基本一致：用分片解决参数和优化器状态的显存，用重计算解决激活的显存，让 70B 放进单个 AMD GPU 节点。原文在单节点上测了 8B 和 70B 的端到端训练吞吐，但没有在正文给出具体数字 [待补：图 4 中 8B/70B 在单节点上的 tokens/s 和扩展效率]。照搬 Megatron 的 DDP 配置到 TorchTitan，或者反过来，都会在显存上撞墙；选后端的时候要连配置一起选。

## 六、上手路径和边界

原文的性能测试覆盖了 Megatron 后端的 1 节点和 8 节点、7B/70B/405B 三组配置，以及 TorchTitan 后端的单节点 8B/70B，但正文只给了配置，没有给 tokens/s 和扩展效率 [待补：图 3 中 7B/70B/405B 在 1N/8N 下的吞吐与扩展效率]。这意味着你无法从文章直接判断 405B 在 8 节点上的 MFU，也无法判断 FSDP2 的通信重叠实际藏住了多少时间；真要选型，得在自己的集群上复现。

上手入口在 AMD 的 ROCm 文档和 Primus 仓库里：Megatron-LM 后端的训练流程、TorchTitan 后端的训练流程、hipBLASLt 离线调优的文档和 Primus 示例、TraceLens 性能分析工具，以及面向大规模部署的 Primus-SaFE 和 Primus for Large Models。TraceLens 值得先跑：它能从系统层和 kernel 层看时间花在哪，确认你的 top kernel 是不是也在 GEMM 和 attention 上。如果 profile 出来通信占了大头，这篇文章的优化顺序就不适用，应该先查并行策略和网络拓扑。

这篇解读的边界也在这里：94% 是 Llama 3.1 70B 在 TorchTitan 后端、未启用优化时的切分，47%/75% 是 fmha_v3 的算子级延迟降幅，5% 是 GEMM 调优的最高收益。三组数字的统计口径不同，不能互相乘。先在自己的模型和集群上复现 profile，再决定把工程时间投到 attention、GEMM 还是通信上；不同形状下的 kernel 收益差别很大，vendor 给的百分比只能当上限参考。

本文整理自 AMD 中国开发者中心发布的 Primus 性能文章，原文链接是 zhuanlan.zhihu.com/p/2011808660167337197。原文作者是 Vidushi Goyal、Wei Cai、Yao Fu、George Wang、Wen Xie、Xiaobo Chen，发布机构是 AMD 中国。更完整的配置和性能数据可以看 AMD GPU 性能页面与 Primus 仓库。

> 第三方内容由各自的第三方权利人直接授权给你，并非由 AMD 授权。所有链接的第三方内容均按"现状"提供，不附带任何形式的担保。你对该等第三方内容的使用完全由你自行决定，因使用第三方内容造成的任何损失，AMD 在任何情况下均不承担责任。你需要自行承担使用第三方内容所带来的全部风险，并对由此产生的任何损害负责。

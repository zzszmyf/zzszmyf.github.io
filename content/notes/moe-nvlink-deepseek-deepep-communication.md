---
title: "MoE数据特征、训练和推理的通信特点、基于NVLink/Scale Up那些Feature，DeepSeek的DeepEP和对MoE All-to-All数据的处理"
date: 2024-04-08T00:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/moe-nvlink-deepseek-deepep-communication/"]
categories: ["研究笔记"]
tags: ["研究笔记", "LLM", "AI Infra"]
---

# MoE数据特征、训练和推理的通信特点、基于NVLink/Scale Up那些Feature，DeepSeek的DeepEP和对MoE All-to-All数据的处理

> 原文链接：https://zhuanlan.zhihu.com/p/2011403053715174158
> 作者：Ethan（芯片互联 & 访存设计）
> 收录于：Scale-Up互联

## 目录

- [一、MoE数据特征](#一moe数据特征)
- [二、训练与推理中的数据传输流程](#二训练与推理中的数据传输流程)
- [三、具体用到的NVLink核心功能](#三具体用到的nvlink核心功能)
- [四、NVLink/Scale Up未来的需求](#四nvlinkscale-up未来的需求)
- [五、MoE 通信优化与 Scale-Up的论文](#五moe-通信优化与-scale-up的论文)
- [六、DeepSeek开源通信库DeepEP](#六deepseek开源通信库deepep)
- [七、MoE中的All-to-All通信](#七moe中的all-to-all通信)
- [八、具体如何实现的让数据分布更均匀](#八具体如何实现的让数据分布更均匀)

---

## 一、MoE数据特征

### 1、稀疏性与动态性 (Sparsity & Dynamism)

**特征**：对于每个输入token，门控网络（Gating Network/Router）动态选择Top-K个专家（通常K=1或2）。这意味着不同token需要访问不同的GPU（如果专家分布在不同的GPU上）。

**影响**：数据流向是动态且不规则的，无法像稠密模型那样进行静态的张量并行划分。

### 2、All-to-All 通信模式

**特征**：在专家并行（Expert Parallelism, EP）策略下，拥有Token的GPU需要将数据发送给拥有对应专家的GPU。由于每个token的目标专家可能不同，这本质上是一个All-to-All（全对全）通信问题。

**规模**：如果有 N 个GPU，每个GPU都可能向其他所有 N−1 个GPU发送数据，同时也接收来自其他GPU的数据。

### 3、小消息与大吞吐的矛盾

**特征**：单个token的数据量很小（例如FP8精度下，隐藏层维度为4096，仅几KB），但总token数量巨大（Batch Size × Sequence Length）。

**影响**：通信延迟（Latency）敏感，同时需要极高的聚合带宽（Throughput）。如果消息切分过细，延迟占主导；如果聚合过大，负载不均衡会导致等待。

### 4、负载不均衡 (Load Imbalance)

**特征**：某些专家可能被频繁选中（"热门专家"），导致对应GPU接收的数据量远超其他GPU。

**影响**：NVLink链路可能出现拥塞，部分GPU空闲等待，降低整体效率。

### 5、低精度数据 (Low Precision)

**特征**：现代MoE（如DeepSeek-V3）广泛使用FP8甚至FP4精度。

**影响**：数据体积减小，对带宽压力略有缓解，但对通信库处理非标准数据格式的能力提出了要求。

---

## 二、训练与推理中的数据传输流程

### 1. 训练阶段 (Training)

在训练过程中，MoE层的前向传播和反向传播都涉及复杂的专家并行通信。

#### 前向传播 (Forward Pass)

**路由计算**：每个GPU本地计算门控得分，确定每个Token的目标专家ID。

**Token分发 (All-to-All Dispatch)**：
- **操作**：GPU根据目标专家ID，将Token重新排序并打包。
- **NVLink传输**：利用NVLink Switch实现的全互联拓扑，执行高带宽的All-to-All通信。数据从源GPU显存直接通过NVLink链路传输到目标GPU显存。
- **关键点**：DeepSeek的DeepEP库在此阶段优化了数据打包（Packing）和路由表生成，最大化利用NVLink的单向和双向带宽。

**专家计算**：目标GPU接收完属于自己专家的所有Token后，进行FFN计算。

**结果汇聚 (All-to-All Combine)**：
- **操作**：计算完成后，结果需要按原始Token顺序发回给源GPU。
- **NVLink传输**：再次执行逆向的All-to-All通信，将结果写回源GPU显存。

#### 反向传播 (Backward Pass)

梯度通信流程与前向传播类似，但方向相反。梯度的All-to-All通信同样依赖NVLink的高带宽。此外，还需要进行参数梯度的All-Reduce（如果在专家内部使用了数据并行），这也高度依赖NVLink的低延迟特性。

#### 利用的NVLink功能

- **点对点直连 (P2P Direct Access)**：GPU显存直接映射，无需经过CPU内存，极大降低延迟。
- **原子操作 (Atomic Operations)**：在某些负载均衡或锁机制中可能用到。
- **多播/广播 (Multicast/Broadcast)**：虽然MoE主要是All-to-All，但在同步路由参数或全局状态时，NVLink的硬件广播功能非常高效。
- **NVSwitch的非阻塞交换**：在72卡（如GB200 NVL72）规模下，NVSwitch确保任意两卡之间的通信带宽不被其他通信对占用，实现真正的线速All-to-All。

### 2. 推理阶段 (Inference)

推理阶段对延迟极其敏感，尤其是解码（Decoding）阶段，每次只生成一个token。

#### 预填充阶段 (Prefill)

类似训练的前向传播，处理整个Prompt。数据量大，主要瓶颈是吞吐量。

**NVLink作用**：利用高带宽进行大规模的All-to-All Token分发，尽可能重叠计算与通信。

#### 解码阶段 (Decoding)

**特征**：每次迭代只处理一个（或少数几个）token。

**挑战**：通信延迟成为主导因素。传统的All-to-All开销过大。

**优化传输**：
- **细粒度通信**：DeepEP等库针对推理提供了基于RDMA或纯NVLink的低延迟内核。
- **异步传输**：利用NVLink的异步引擎，在GPU计算当前层时，预取下一层需要的Token数据。
- **专家缓存**：如果可能，将热门专家常驻在特定GPU，减少跨节点通信（但这在单节点内主要靠NVLink解决）。

#### 利用的NVLink功能

- **低延迟链路**：NVLink的物理层延迟远低于PCIe。
- **流控制 (Flow Control)**：防止快速发送方淹没接收方，特别是在负载不均衡时。
- **错误纠正 (ECC)**：保证长时推理的数据完整性。

---

## 三、具体用到的NVLink核心功能

### 1、高带宽双向通道 (High-Bandwidth Bidirectional Links)

NVLink 5.0/6.0提供每GPU 900GB/s - 3.6TB/s的带宽。MoE的All-to-All通信是双向的（发送+接收），NVLink的全双工特性使得发送和接收可以同时进行，带宽利用率翻倍。

### 2、NVSwitch 全互联拓扑 (Full All-to-All Topology)

在单机多卡（如8卡H100/H800）或机架级（如GB200 NVL72）系统中，NVSwitch消除了"跳数"限制。任何GPU到任何GPU的通信都是单跳（Single-hop）且带宽一致的。这对于MoE至关重要，因为热点专家可能导致特定链路拥塞，全互联避免了瓶颈链路。

### 3、内存一致性/统一寻址 (Unified Memory Addressing / P2P Access)

CUDA程序可以直接通过指针访问远程GPU显存（通过NVLink）。通信库（如DeepEP, NCCL）利用此特性实现零拷贝（Zero-Copy）或最小拷贝的数据传输，直接将数据从发送方的显存缓冲区DMA到接收方的显存缓冲区。

### 4、硬件多播 (Hardware Multicast)

虽然MoE主要是单播，但在路由表同步、专家参数更新（如果使用共享专家）时，硬件多播比软件模拟效率高得多。

### 5、协议卸载 (Protocol Offload)

NVLink控制器硬件处理数据包的分片、重组、重传和流量控制，释放GPU SM（流多处理器）资源用于计算，这对MoE这种计算通信比（Arithmetic Intensity）较低的架构尤为重要。

---

## 四、NVLink/Scale Up未来的需求

### 1、硬件级的动态负载均衡 (Hardware-assisted Dynamic Load Balancing)

**现状**：目前负载不均衡主要由软件（如DeepEP）通过预测和重排来解决，增加了计算开销。

**期望**：NVLink交换机能感知各端口的队列深度，硬件自动调整路由策略，或将数据智能分流到空闲链路，甚至支持"工作窃取"（Work Stealing）机制的硬件原语。

### 2、更细粒度的通信原语 (Fine-grained Communication Primitives)

**现状**：All-to-All通常需要显式的数据打包和解包。

**期望**：支持散列-聚集（Scatter-Gather）或键值对路由（Key-Value Routing）的硬件指令。GPU只需发出"将Token X发送给专家ID Y"的指令，NVLink控制器自动完成寻址和传输，进一步降低Kernel开发复杂度。

### 3、原生支持稀疏数据格式压缩 (Native Sparse Data Compression)

**现状**：传输的是稠密打包后的Token数据。

**期望**：在链路层直接支持稀疏张量格式的压缩传输，或者在传输过程中自动剔除Padding数据，进一步节省宝贵的带宽。

### 4、跨节点NVLink扩展 (Extended Cross-Node NVLink)

**现状**：NVLink主要在单机或机架内。跨机通信依赖InfiniBand/RoCE，延迟和带宽差距大。

**期望**：虽然物理距离受限，但希望能有类似NVLink over Optical的更长距离扩展方案，或者在协议层实现NVLink与高速以太网/光互联的无缝融合，让多机集群像单机一样进行All-to-All通信（即"超级节点"概念）。

### 5、增强的遥测与可观测性 (Enhanced Telemetry)

**期望**：提供更实时的链路拥塞、丢包、延迟分布的硬件计数器，并暴露给软件栈，以便MoE路由器能实时感知网络状态并动态调整路由策略（如避开拥塞的专家节点）。

### 6、低精度数据的原生路由 (Native Low-Precision Routing)

随着FP4/FP8的普及，希望NVLink控制器能原生识别这些格式，进行更高效的数据对齐和传输，减少GPU端的格式转换开销。

### 7、语义拓展

- **内存语义（Load/Store）**：小粒度、低延迟，适合控制
- **消息语义（Send/Recv）**：大块数据异步传输
- **张量语义（Push/Pull）**：针对1~100KB张量优化，支持批量/流式、显式/隐式确认

### 8、光互联的应用

MoE模型采用专家并行（EP），要求网络提供超大带宽和超低时延，且EP域越来越大（从几十卡向几百卡扩展）。

超节点（如NVL72、华为CM384），因为现有铜互连方案受限于距离（通常仅机柜内），高密机柜设计带来制造、散热、供电、可靠性等挑战。

而光互连是扩展规模的必然选择，但传统可插拔光模块（FRO）成本高、功耗大、时延高，且可靠性不如铜缆；如何低成本、高可靠地实现光互连是关键。

从LPO->NPO->CPO演进。

---

## 五、MoE 通信优化与 Scale-Up的论文

### 1、GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding (ICLR 2021)

**作者**: Google Brain / DeepMind

**内容**: MoE 领域的开山之作之一。详细描述了 Expert Parallelism (EP) 的实现，包括 All-to-All 通信在 TPU/NVIDIA GPU 集群上的挑战，以及负载均衡算法。

**关联性**: 解释了为什么 MoE 需要高带宽互联，以及通信瓶颈在哪里。

**下载链接**: https://arxiv.org/pdf/2006.16668.pdf

### 2、Tutel: Adaptive Mixture-of-Experts at Scale (MLSys 2023)

**作者**: Microsoft

**内容**: 专注于 MoE 的系统优化。详细讨论了 通信内核优化、稀疏性处理 以及在 NVIDIA GPU 上如何利用高速互联（NVLink/InfiniBand）进行高效的 Token 分发。

**关联性**: 提到了类似 DeepEP 的优化技术，如通信与计算重叠、自适应批处理。

**下载链接**: https://arxiv.org/pdf/2206.03382.pdf

### 3、Llama-MoE: Building Mixture-of-Experts from Llama with Progressive Pre-training

**作者**: Meta AI

**内容**: 虽然主要讲模型，但其系统部分会提及在大规模 GPU 集群上训练 MoE 时的通信策略。

**下载链接**: https://arxiv.org/pdf/2405.04434.pdf 

**注意**: Llama 3 主要是稠密模型，Llama 3.1 有 MoE 版本但未发详细系统论文。建议参考 Switch Transformers (Google)。

**替代推荐**: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (JMLR 2022)

**链接**: https://arxiv.org/pdf/2101.03961.pdf

### 4、DeepEP: Efficient Expert Parallelism Communication Library

**说明**: DeepSeek 目前尚未发布名为 "DeepEP" 的独立系统论文（它包含在 V2/V3 技术报告中）。但可以参考 Alpa 或 Megatron-LM 的相关系统论文，它们实现了类似的 EP 通信原语。

**推荐**: Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning (OSDI 2022)

**内容**: 自动寻找最佳的并行策略（包括 EP），并在 NVLink 集群上进行通信优化。

**链接**: https://arxiv.org/abs/2201.12023

#### Megatron-LM策略

在transformer层并行化self-attention模型：

对于大型的transformer模型，权重的数量非常大，因此不能使用数据并行。所以需要对权重进行分片。在这个模型里面，第一个权重矩阵已经被按列分片，第二个矩阵已经被按行分片，从而减少通信代价。

#### GShard Mixture-of-Expert

专家层（红色部分）以专家维分片，非专家层以批处理维度分片做数据并行。

为了应用这样的并行策略，模型开发者不得不去重写他们的模型定义，特定化分片策略，并且插入必要的通信原语，比如这里的all-to-all。这使得开发一个新的模型或者寻找异质模型变得困难。

#### Alpa 简介

Alpa 是一个自动并行化大模型（如千亿参数的 GPT-3 ）训练的编译器。它端到端的效果是，开发者只需在（Jax 框架上）需要并行的训练函数签名前加上一行 Python 装饰器 `@alpa.parallelize`（类似于现有装饰器 `@jax.jit`，只不过新的装饰器是为了编译在一个分布式集群上运行的 Jax 代码，而非一台机器），它就能在编译时自动地生成近似最优的并行化可执行代码，达到加速的效果。

乍看之下，Alpa 不需要显式地指明对于待训练模型和目标集群的描述，使得使用起来几乎没有工程负担。

工程已开源：https://github.com/alpa-projects/alpa

---

## 六、DeepSeek开源通信库DeepEP

DeepSeek-R1的推理系统即是依赖DeepEP+NVSHMEM+GDRCopy+IBGDA的方案替代了NCCL进行高效通信。

### NVSHMEM通信库

DeepEP利用了NVSHMEM的能力进行高效通信。

NVSHMEM是一个基于OpenSHMEM的专门用于NVIDIA GPU的通信库，其核心思想是将所有GPU节点上的显存视为一个大的显存池来进行管理即分区全局地址空间（PGAS）。

该库支持通过GPU共享内存直接进行数据访问，提供如shmem_put、shmem_get等可以进行细粒度数据传输的API接口。

除此之外，还集成了IBGDA（InfiniBand GPUDirect Async）进行高性能的GPUDirect RDMA通信。

结构图示意：
```
[GPU0] <--NVLink--> [GPU1] <--NVLink--> [GPU2] ... [GPUN]
   |                    |                    |          |
   └──────┬─────────────┴─────────────┬──────┘          |
          │                           │                 |
    [NVSHMEM PGAS]            [IBGDA RDMA]        [IBGDA RDMA]
          │                           │                 |
    [显存池统一管理]           [跨节点通信]        [跨节点通信]
```

### NCCL VS NVSHMEM

NVSHMEM通信库和经典的NCCL集合通信库的对比如下表所示：

| 特性 | NCCL | NVSHMEM |
|------|------|---------|
| 通信模型 | 集合通信（Collective） | 分区全局地址空间（PGAS） |
| 编程接口 | 高级集合操作（AllReduce等） | 细粒度Put/Get操作 |
| 适用场景 | 大规模数据并行训练 | 细粒度、动态数据访问 |
| 延迟特性 | 相对较高 | 更低延迟 |
| 数据访问 | 需显式同步 | 直接内存访问 |
| 灵活性 | 固定模式 | 动态路由支持 |

### GDRCopy低延时库

官方对GDRCopy的定义如下：基于GPUDirect RDMA技术的低延时GPU显存拷贝库，允许CPU直接访问GPU显存。

从图中我们可以看出，在使用了GDRCopy的能力后，H2D的链路缩短了，这优化了H2D的延时。NVIDIA官方给出的性能测试结果如下：

- 在小消息传输的场景下，和传统的cudaMemcpy相比，利用GDRCopy后的延时有了很大程度的降低。

### InfiniBand GPUDirect Async技术

InfiniBand GPUDirect Async简称IBGDA，是NVIDIA推出的基于InfiniBand GPUDirect RDMA（简称GDR）技术进一步优化的高效通信技术。

#### GDR的流程

1. 应用程序launch cuda kernel，在显存中生成数据
2. SM写一个work descriptor到在主机内存中的一个代理线程的proxy buffer中
3. proxy通知cpu进行相应的网络操作
4. CPU创建work descriptor到WQ队列中
5. CPU更新doorbell record（DBR）
6. CPU注册相关信息到NIC的DB中以通知NIC进行数据传输
7. NIC从WQ中读取work descriptor
8. NIC通过GDR从显存中读取数据
9. NIC发送数据到远端节点
10. NIC写完成event到CQ队列中
11. CPU从CQ中确认网络操作完成
12. CPU通知GPU操作完成，此步依赖GDRCopy

从上述流程可以看出，经典的GDR技术有比较多的非应用数据传输的步骤需要CPU的参与。由于GPU和Mellanox高性能网卡的数据处理能力都在快速增长，且远远超过CPU的处理能力，因此在对延时有极高要求的场景下，经典的GDR技术在CPU侧会成为瓶颈。

#### IBGDA流程

为进一步优化通信效率，NVIDIA在GDR的基础上推出了IBGDA：

1. 应用程序launch cuda kernel，在显存中生成数据
2. SM创建work descriptor到WQ中
3. SM更新DBR
4. SM通知NIC
5. NIC通过GDR从WQ中读取work descriptor
6. NIC通过GDR从显存中读取数据
7. NIC发送数据到远端节点
8. NIC通过GDR向CQ中写入完成事件

从上述流程可以看出，IBGDA将在CPU上进行的相关操作全部放到GPU中，整个过程完全不需要CPU的参与，进一步减少了通信链路，提高了通信效率。

NVIDIA官方基于IBGDA技术在All-to-All场景下的延时测试（为凸显IBGDA的效果，该测试禁用了A100节点内的NVLink）：

- 32 PEs可以理解为有32张A100
- IBRC表示未启用IBGDA
- 在小消息传输的场景下，启用IBGDA后延时有了大幅的下降

### 总结

DeepSeek基于上述相关技术，在DeepEP中实现了：
- 专门用于训练和推理Prefilling阶段的高吞吐Kernel
- 专门用于推理Decoding阶段的低延时Kernel

此外，DeepSeek内部还实现了一套P-D分离的推理系统来高效的部署DeepSeek相关模型以支撑庞大的用户请求量。

---

## 七、MoE中的All-to-All通信

在大规模MoE模型进行训练和推理的过程中，面临的最主要问题就是大规模专家并行的All-to-All通信瓶颈，All-to-All通信主要分为两个阶段：Dispatch阶段和Combine阶段。这两个阶段共同完成了"数据去找专家"和"专家结果回传"的过程。

### Dispatch阶段

在MoE模型中，dispatch的目的是将输入数据分发到不同的专家进行处理。由于每个输入token只需要激活top-k个专家，因此dispatch需要根据门控路由的结果将数据发送到对应的专家上。

**具体流程**：

1. **路由计算**：使用一个路由网络（FFN）计算每个token对所有专家的分数，根据分数选择top-k个专家，生成对应的索引和权重
2. **数据分发**：根据top-k索引，将输入数据分发到对应的专家，每个专家会接收到属于自己的输入数据
3. **通信机制**：使用All-to-All通信方式，确保每个节点上的数据可以被正确的分发到所有其它节点上的专家

### Combine阶段

在MoE模型中，combine的目的是将各个专家的输出结果合并回一个完整的输出张量。由于每个token的输出是由top-k个专家的输出加权求和得到的，因此需要重新将这些数据进行组合。

**具体流程**：

1. **结果聚合**：根据top-k专家的索引和权重，将每个专家的输出结果聚合到对应的token上，通常使用加权求和的方式进行聚合
2. **通信机制**：使用All-to-All通信方式，将各个节点上的专家输出数据回传到原节点，以便进行结果聚合

### 详细介绍

#### 1. Dispatch 阶段 (前向传播的数据分发)

**目标**：将分散在各个 GPU 上的 Token，根据路由算法（Router/Gating Network）的选择，发送给持有对应专家（Expert）的 GPU。

**数据流向**：
- **输入**：每个 GPU 持有一批本地 Token（形状通常为 [Local_Batch, Hidden_Dim]）。
- **路由决策**：每个 GPU 本地计算 Router，得到每个 Token 的目标专家 ID（Target Expert ID）。
- **重排与打包**：
  - 根据目标专家 ID，将 Token 重新排序。
  - 将发往同一个目标 GPU 的 Token 聚合在一起（Packing），形成连续的大块内存，以减少 NVLink 的小包传输开销。
- **通信动作 (All-to-All)**：
  - GPU i 将属于专家 Ej​ （位于 GPU j ）的所有 Token，通过 NVLink 直接发送给 GPU j 。
  - 这是一个典型的 Scatter 操作：源端分散，目的端按专家聚合。
- **输出**：每个 GPU 接收到了所有指派给其本地专家的 Token（形状变为 [Num_Experts_On_GPU * Tokens_Per_Expert, Hidden_Dim]）。

**NVLink 的关键作用**：
- **高带宽吞吐**：由于所有 Token 都要移动，数据量巨大（通常是激活参数量的数倍），需要 NVLink 的 TB/s 级带宽。
- **P2P 直连**：避免经过 CPU 内存，直接 GPU 显存对拷。
- **负载均衡挑战**：如果路由不均匀（某些专家过热），会导致目标 GPU 接收数据过多，造成 Recv 阻塞，而其他 GPU 空闲等待。DeepSeek-V3 的"无辅助损失负载均衡"就是为了解决这个问题，确保每个 GPU 接收的数据量几乎相等，从而跑满 NVLink 带宽。

#### 2. 专家计算阶段 (本地计算)

在 Dispatch 完成后，每个 GPU 上现在全是"属于自己专家"的 Token。

- GPU 本地执行 FFN（前馈神经网络）计算。
- 注意：此阶段没有通信，是纯计算。这也是 MoE 能用大参数量换取高计算密度的原因。

#### 3. Combine 阶段 (前向传播的结果汇聚 / 反向传播的梯度分发)

**目标**：将专家计算后的结果，按照原始 Token 的顺序，发回给拥有该 Token 的源 GPU。

**数据流向**：
- **输入**：每个 GPU 持有计算完成的专家输出（形状同 Dispatch 输出）。
- **逆向路由**：利用 Dispatch 阶段保存的路由元数据（Metadata）（即：哪个 Token 来自哪个 GPU，原始索引是多少）。
- **重排与打包**：
  - 根据源 GPU ID，将结果重新排序。
  - 将发往同一个源 GPU 的结果聚合。
- **通信动作 (All-to-All)**：
  - GPU j 将计算结果发回给原始的源 GPU i 。
  - 这是一个典型的 Gather 操作：源端按专家聚合，目的端分散还原。
- **输出**：每个 GPU 恢复了原始 Batch 的 Token 顺序和归属，得到完整的输出张量（形状恢复为 [Local_Batch, Hidden_Dim]）。

**NVLink 的关键作用**：
- **对称带宽**：Combine 阶段的数据量与 Dispatch 阶段完全一致（只是精度可能不同，如激活值是 FP8/BF16，梯度可能是 FP32 累加）。NVLink 的全双工特性允许在某些优化策略下，部分重叠 Dispatch 和 Combine（虽然在标准串行流程中是先后发生的）。
- **低延迟**：在推理的 Decoding 阶段，Batch Size 很小，Combine 阶段的延迟直接决定了生成速度（Token/sec）。

#### 4. 两个阶段的对比与特征总结

| 特征 | Dispatch (分发) | Combine (汇聚) |
|------|-----------------|----------------|
| 通信模式 | Scatter (多对多，按目标聚合) | Gather (多对多，按源聚合) |
| 数据内容 | 原始激活值 (Activations) | 专家输出值 (Outputs) 或 梯度 (Gradients) |
| 依赖关系 | 依赖 Router 计算结果 | 依赖 Dispatch 阶段生成的路由元数据 |
| 负载敏感性 | 极高。若路由不均，接收方显存可能溢出或计算负载倾斜。 | 高。若 Dispatch 不均，Combine 的发送方数据量也不均，导致发送阻塞。 |
| NVLink 压力 | 写压力为主 (目标 GPU 接收写入) | 读压力为主 (源 GPU 读取发送) |
| 优化关键 | 动态负载均衡、微批次聚合 | 元数据快速索引、零拷贝还原 |

#### 5. 高级优化技术 (针对这两个阶段)

为了在 NVLink 上极致优化这两个阶段，现代系统（如 DeepEP, Tutel, GShard）采用了以下技术：

##### 1、通信与计算重叠 (Overlap)

在 GPU 计算当前层的专家时，异步启动下一层（或上一层）的 Dispatch/Combine 通信。

利用 NVLink 的异步引擎（CUDA Stream），让数据传输在后台进行。

##### 2、精确的元数据管理

Dispatch 阶段生成的 indices (目标位置) 和 locations (源位置) 必须高效存储。Combine 阶段直接使用这些索引进行 memcpy 或 gather 操作，避免二次路由计算。

##### 3、混合精度通信

- Dispatch 传输激活值时，通常使用 FP8 或 BF16 以减少 NVLink 带宽占用。
- Combine 传输梯度时（反向传播），可能需要 FP32 累加，数据量翻倍，对 NVLink 带宽要求更高。

##### 4、稀疏性感知打包 (Sparse-Aware Packing)

不发送空的 Padding 数据。只打包有效的 Token，并在接收端根据元数据还原到正确位置（可能包含 Padding）。这能显著减少无效数据的 NVLink 传输量。

##### 5、双缓冲 (Double Buffering)

在显存中开辟两块缓冲区，一块用于当前计算，一块用于下一次通信，彻底隐藏通信延迟。

### 总结

**Dispatch** 和 **Combine** 是 MoE 架构的一体两面。

- **Dispatch** 是"把活分下去"，关键在于负载均衡，防止个别专家累死（GPU 拥塞）。
- **Combine** 是"把活收上来"，关键在于索引效率，确保结果准确归位。

在 NVIDIA NVLink 架构下，这两个阶段都依赖于 NVSwitch 的全互联 All-to-All 能力。DeepSeek 等厂商的核心竞争力，就在于通过算法（如均匀路由）让这两个阶段的数据流变得规则且均匀，从而让 NVLink 硬件发挥出理论峰值性能，避免因为负载不均导致的"木桶效应"。

---

## 八、具体如何实现的让数据分布更均匀

DeepSeek-V2 和 V3 的技术报告揭示了一个核心洞察：NVLink 的硬件性能（带宽、延迟）是固定的，但 MoE 的通信效率完全取决于数据分布的均匀性。

如果数据分布不均匀（长尾分布），NVLink 就会出现"木桶效应"：最忙的那条链路决定了整体速度，其他链路闲置。DeepSeek 通过算法强制将数据分布从"自然长尾"重塑为"人工均匀"，从而让 NVLink 能够以确定性、满带宽的状态运行。

### 1. 核心问题：自然路由导致的"灾难性"数据分布

在没有强约束的情况下，MoE 的路由器（Router/Gating Network）倾向于"马太效应"：

**现象**：少数几个"热门专家"吸引了 60%-80% 的 Token，而大量"冷门专家"几乎无人问津。

**对 NVLink 的打击**：
- **拥塞（Congestion）**：持有热门专家的 GPU 接收数据量巨大，NVLink 接收缓冲区溢出，触发流控（Flow Control），发送方被迫暂停。
- **空闲（Idle）**：持有冷门专家的 GPU 瞬间收完数据，然后空转等待全局同步（Barrier）。

**结果**：NVLink 的平均利用率极低，整体通信时间由最慢的那个 GPU 决定（Straggler Problem）。

### 2. DeepSeek-V2 的策略：细粒度专家 + 共享专家 + 辅助损失

DeepSeek-V2 首先通过架构设计缓解了分布不均，但未完全消除。

#### A. 细粒度专家 (Fine-Grained Experts)

**策略**：将传统的少量大专家（如 8 个）拆分为大量小专家（如 256 个）。

**数据分布改变**：
- **大数定律**：根据概率论，当专家数量 N 增大时，Token 落入每个专家的概率方差会自然减小。
- **效果**：将极端的长尾分布"磨平"了一些，减少了单个 GPU 负载过重的风险。

**NVLink 适配**：
- 数据包变得更小、更碎。虽然分布稍好，但仍需依赖软件层面的聚合（Packing）来适应 NVLink 的大包传输特性。

#### B. 共享专家 (Shared Experts)

**策略**：设置一部分所有 Token 都会访问的"共享专家"，只有剩余 Token 走路由选择"路由专家"。

**数据分布改变**：
- 分流了大部分通用知识相关的流量，减轻了路由专家的竞争压力。

**NVLink 适配**：
- 共享专家通常位于所有 GPU 上（数据并行），不需要 All-to-All 通信，直接减少了 NVLink 上的动态流量总量。

#### C. 辅助负载均衡损失 (Auxiliary Load Balancing Loss)

**策略**：在训练目标中加入一个惩罚项，如果某个专家被选中的频率过高，就增加 Loss。

**局限性**：V2 发现这种方法不够稳定，且需要调节超参数，有时会导致模型为了平衡而牺牲精度（强行把 Token 发给不合适的专家）。

### 3. DeepSeek-V3 的突破：无辅助损失负载均衡 (Auxiliary-Loss-Free)

这是 DeepSeek-V3 技术报告中最关键的系统创新。它彻底改变了数据分布的形态，使其完美适配 NVLink。

#### A. 核心机制：硬约束与动态容量 (Hard Constraint & Dynamic Capacity)

**策略**：
1. **取消辅助损失**：不再依赖 Loss 函数去"劝"路由器平衡。
2. **设定严格容量上限**：为每个专家设定严格的 Token 容量上限（Capacity Factor，通常 Factor 接近 1.0）。
3. **贪婪选择 + 滚动回退**：
   - Router 依然按得分高低选择 Top-K 专家。
   - **关键一步**：如果某个专家已满（达到容量上限），后续选到该专家的 Token 会被强制重新分配给当前批次中负载最低的专家（或者丢弃，但在 V3 中主要是重分配）。

**数据分布改变**：
- 从"长尾"变为"矩形"：无论原始得分如何，最终每个专家接收到的 Token 数量严格相等（或差异极小，仅在 Batch 边缘）。
- **确定性**：每个 GPU 在通信开始前，就已经确切知道自己要接收多少数据（Exact Count）。

#### B. 这种分布如何极致适配 NVLink？

##### 1. 消除同步气泡 (Eliminating Synchronization Bubbles)

**之前**：GPU A 收 1GB，GPU B 收 100MB。GPU B 等 GPU A，NVLink 闲置 90% 时间。

**现在**：所有 GPU 都收 500MB。

**NVLink 收益**：所有 NVLink 链路同时开始、同时结束。零等待时间。NVLink 的带宽利用率从"受限于最慢链路"提升到"所有链路满载"。

##### 2. 预分配与零拷贝 (Pre-allocation & Zero-Copy)

**之前**：因为不知道每个专家会来多少数据，必须预留很大的缓冲池（Over-provisioning），或者使用复杂的动态内存管理，导致显存碎片化，DMA 效率低。

**现在**：由于数据量是确定的（Deterministic），系统可以在 Kernel 启动前精确计算并分配显存地址。

**NVLink 收益**：
- 可以直接使用固定大小的连续内存块进行 DMA 传输。
- NVLink 控制器最喜欢连续的大块内存，这样可以最大化总线利用率，减少协议开销。
- 实现了真正的Zero-Copy，无需中间拷贝。

##### 3. 完美的微批次聚合 (Perfect Micro-batching)

**之前**：负载不均导致无法有效打包，小包多，延迟高。

**现在**：既然每个目标 GPU 的数据量已知且相等，通信库（如 DeepEP）可以将数据完美地切分成大小一致的 Micro-batches。

**NVLink 收益**：
- 可以流水线式地发送这些大小一致的数据包。
- 极大地降低了 NVLink 的启动延迟（Latency）占比，提升了吞吐（Throughput）。

##### 4. 避免流控停顿 (Avoiding Flow Control Stalls)

**之前**：热点专家导致接收端 Credit 耗尽，发送端暂停，整个 NVLink 网络出现反压（Backpressure）。

**现在**：流入每个节点的数据速率严格匹配其处理能力。

**NVLink 收益**：NVLink 的信用机制（Credit-based Flow Control）始终处于平滑流动状态，没有停顿和重试。

### 4. 直观比喻对比

想象 NVLink 是一个有 8 条车道的收费站（8 卡互联）：

**无负载均衡 (V2 之前/传统 MoE)**：
- 车道 1 排了 1000 辆车（拥堵，后面车过不去）。
- 车道 2-8 只有 10 辆车（空荡荡，收费员发呆）。
- **结果**：整个收费站必须等车道 1 处理完才能放行下一批。效率 = 1/8。

**DeepSeek-V3 负载均衡**：
- 算法在入口处强行指挥：每条车道必须正好进 125 辆车。
- 车道 1-8 同时开始处理，同时结束。
- **结果**：所有收费员都在工作，没有等待。效率 = 100%。

### 5. 总结：从"尽力而为"到"确定性工程"

DeepSeek-V2/V3 的负载均衡策略本质上是将 MoE 的通信问题从一个随机的、概率的网络拥塞问题，转化为了一个确定性的、规则的内存拷贝问题。

- **数据分布特征变化**：High Variance (高方差) → Zero Variance (零方差/均匀)。

**NVLink 适配效果**：
- **带宽跑满**：消除了 Straggler，所有链路并行满载。
- **延迟最低**：确定的数据量允许最优的 Kernel 调度和重叠（Overlap）。
- **显存高效**：无需过度预留缓冲，显存利用率更高。

---

*本文整理自知乎专栏文章，原文链接：https://zhuanlan.zhihu.com/p/2011403053715174158*

*作者：Ethan（芯片互联 & 访存设计），收录于 Scale-Up互联专栏*

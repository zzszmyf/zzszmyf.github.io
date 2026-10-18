---
title: "MoE数据特征、训练和推理的通信特点、基于NVLink/Scale Up那些Feature，DeepSeek的DeepEP和对MoE All-to-All数据的处理"
description: "MoE All-to-All通信与NVLink优化解析：拆解专家并行下Token分发的Dispatch和Combine两阶段数据流，梳理训练与推理对带宽、延迟的不同要求，以及DeepSeek开源通信库DeepEP的高吞吐/低延迟Kernel设计；重点讲DeepSeek-V3无辅助损失负载均衡如何把长尾路由重塑成均匀分布，让NVLink链路跑满带宽。"
date: 2024-04-08T00:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/moe-nvlink-deepseek-deepep-communication/"]
categories: ["研究笔记"]
tags: ["研究笔记", "LLM", "AI Infra"]
---

推理服务进入 decode 阶段后，通常每一步只生成一个 token。路由器把这一小批 token 按专家分到不同 GPU，任何一张卡上的热门专家排队，整批请求的 TPOT 就会被这一跳拖住。MoE 的通信优化，很多时候就是在处理这个看起来很不起眼、但每步都要发生一次的 All-to-All。

## 一、All-to-All 的难点在消息形态

MoE 的路由器对每个 token 算 Top-K 专家，K 由路由策略决定，早期 MoE 常用 1 或 2，DeepSeek-V2 选 top-6、V3 选 top-8。专家分布在不同 GPU 上，token 要去哪张卡由路由结果决定，所以数据流是动态的、不规则的；稠密模型那种按固定张量切分的静态并行，在这里不成立。

规模上，N 张卡里每张都可能向另外 N-1 张卡发数据，同时也从 N-1 张卡收数据。单 token 的数据量不大：FP8、hidden 维度 4096 时只有几 KB；但 batch 乘上 sequence，总量会很大，聚合带宽需求很高。消息切得太碎，延迟占主导；包得太大，负载不均又会让先到的卡空等。

更麻烦的是负载分布。原文给出的经验值是少数热门专家吸走 60%-80% 的 token，持有这些专家的 GPU 接收缓冲区先满，流控把发送方压住，其他 GPU 收完自己的份额后空转。All-to-All 的完成时间由最慢的那条链路决定，NVLink 的峰值带宽在这种长尾分布下跑不出来。

低精度缓解了一部分体积压力。DeepSeek-V3 这类模型用 FP8 甚至 FP4，数据量下来了，但通信库要处理非标准格式的对齐和转换，低精度并没有让问题消失。

训练和推理对通信的要求不一样。训练和前向 prefill 处理整批 token，瓶颈是吞吐，通信要尽量和计算重叠；decode 每步只处理一个或少数几个 token，瓶颈是延迟，传统 All-to-All 的一次同步就足以吃掉收益。

## 二、Dispatch 和 Combine 是一次往返

Dispatch 阶段，每张卡先算本地 token 的 router 分数，得到目标专家 ID；然后按目标 GPU 重排 token，把发往同一张卡的 token 打包成连续内存块，减少 NVLink 上的小包数量。发送时，GPU i 把属于 GPU j 上专家的 token 发过去，接收端按专家聚合，形状从 `[Local_Batch, Hidden_Dim]` 变成 `[Num_Experts_On_GPU * Tokens_Per_Expert, Hidden_Dim]`。专家在本地做 FFN，这一段没有通信。Combine 阶段按 Dispatch 保存的元数据反向走一遍，把专家输出按原始 token 顺序发回源 GPU，恢复 `[Local_Batch, Hidden_Dim]`。

元数据是这条链路的关键。Dispatch 生成的 indices 和 locations 记录了哪个 token 来自哪张卡、原始位置在哪，Combine 直接拿它做 gather，避免二次路由计算。元数据如果存得不好，Combine 阶段就要重新算一遍路由，省下的通信时间会被计算吃回去。

训练的反向传播把方向反过来：梯度从专家侧发回 token 侧，再在专家内部的数据并行组里做 All-Reduce。前向传的是激活值，反向可能是 FP32 累加的梯度，数据量翻倍，NVLink 的压力更大。

推理的 prefill 和训练前向类似，处理整批 token，瓶颈在吞吐；decode 的数据量小、延迟敏感，每步一次 All-to-All 的同步开销就会吃掉收益。两种负载对通信库的要求不同，后文再展开。

更早的 GShard 把专家并行的 All-to-All 挑战写清楚了，Tutel 做了通信内核和计算重叠，Switch Transformer 用简单的稀疏路由把 MoE 推到万亿参数，Alpa 把专家并行纳入自动并行搜索。DeepEP 的工程位置，是把这些系统思路落在一套面向训练和推理的通信库里。

## 三、NVLink 的哪几个特性真正被用到

NVLink 的带宽要按代际说清楚。H100 的 NVLink 4 是每 GPU 900GB/s，GB200 的 NVLink 5 是 1.8TB/s，Rubin 的 NVLink 6 到 3.6TB/s；这些数字是双向聚合带宽，单向要减半。原文把 900GB/s 到 3.6TB/s 笼统写成 5.0/6.0 的范围，容易让人把 H100 的带宽当成 Blackwell 的起点。

MoE 的 All-to-All 同时有发送和接收，全双工让两个方向可以并行，带宽利用率比单向传输高。NVSwitch 在单机 8 卡或 NVL72 机架内提供任意两卡单跳、带宽一致的连接；热点专家把某条链路打满时，其他通信对不会被这条链路挡住。P2P 直连让 GPU 直接访问远端显存，通信库可以把数据从发送方缓冲区 DMA 到接收方缓冲区，不需要经过 CPU 内存。

硬件多播在 MoE 里不是主路径。All-to-All 基本是单播，但路由表同步、共享专家参数更新这类一对多操作，硬件多播比软件模拟省事。协议卸载则直接影响 SM 的可用算力：NVLink 控制器负责分片、重组、重传和流控，GPU 的 SM 不用停下来处理协议，这对计算通信比不高的 MoE 很重要。

这几个特性分别对应不同瓶颈：全双工和 NVSwitch 解决带宽与拥塞，P2P 解决延迟和拷贝，协议卸载解决 CPU 和 SM 的占用。选机器时只看总带宽不够，还要看这些路径在目标拓扑上是不是都成立。

## 四、DeepEP 的软件栈：把 CPU 从数据路径上拿掉

DeepEP 不是只靠 NVLink。DeepSeek 公开的 R1 推理系统把 DeepEP、NVSHMEM、GDRCopy、IBGDA 组合在一起；NCCL 的集合通信模型擅长规整的 AllReduce 和 AllGather，面对按 token 动态寻址的 All-to-All，细粒度 put/get 更合适。

NVSHMEM 的核心是把所有 GPU 显存当成一个分区全局地址空间（PGAS），提供 `shmem_put`、`shmem_get` 这类细粒度接口。NCCL 把通信抽象成集合操作，调用方要显式同步；NVSHMEM 允许直接读写远端显存，适合动态路由。代价是编程模型更底层，同步和内存序要自己管。

```text
[GPU0] <--NVLink--> [GPU1] <--NVLink--> [GPU2] ... [GPUN]
   |                    |                    |          |
   └──────┬─────────────┴─────────────┬──────┘          |
          │                           │                 |
    [NVSHMEM PGAS]            [IBGDA RDMA]        [IBGDA RDMA]
          │                           │                 |
    [显存池统一管理]           [跨节点通信]        [跨节点通信]
```

GDRCopy 解决的是 CPU 访问 GPU 显存的延迟。传统 `cudaMemcpy` 的 H2D 路径长，GDRCopy 让 CPU 直接读写 GPU 显存，小消息的延迟明显下降。真正把 CPU 从数据路径上拿掉的是 IBGDA，但它的起点是 GDR 的代理模型。

经典的 GDR 流程要走 12 步：应用启动 kernel 在显存里生成数据，SM 把 work descriptor 写到主机内存的 proxy buffer，proxy 通知 CPU，CPU 创建 WQ 条目、更新 doorbell record、把信息注册到 NIC 的 DB，NIC 再从 WQ 读 descriptor、通过 GDR 读显存数据、发到远端、把完成事件写回 CQ，最后 CPU 确认完成并通知 GPU。这里面 CPU 参与了太多步；GPU 和网卡的处理能力早就超过 CPU，CPU 就成了小消息路径上的瓶颈。

IBGDA 把这套流程压到 8 步：SM 直接创建 work descriptor 写到 WQ，SM 更新 DBR 并通知 NIC，NIC 通过 GDR 读 WQ 和显存数据，发到远端，再把完成事件写回 CQ。CPU 从数据路径上消失，链路短了，延迟就下来了。

DeepEP 在 NVSHMEM 之上做了两套 kernel：训练和 prefill 用高吞吐 kernel，decode 用低延迟 kernel；低延迟路径走纯 RDMA，高吞吐路径同时用 NVLink 和 RDMA [待补：DeepEP 在 H800 或 NVL72 上的 decode 延迟与 All-to-All 带宽实测]。这个分工比“一套 All-to-All 打天下”更贴近实际，因为训练和 decode 的瓶颈根本不在一个维度上。DeepSeek 的推理系统还把预填充和解码拆成不同的部署池，P-D 分离之后，decode 池可以用更小的 batch 和更低的延迟跑，prefill 池负责把长 prompt 的吞吐吃满。

## 五、负载均衡：把长尾路由压短

负载均衡是 MoE 通信优化的另一半，前面提到的长尾分布要靠路由算法压短。DeepSeek-V2 先用细粒度专家和共享专家缓解这个问题。V2 是 160 个路由专家加 2 个共享专家，每个 token 选 top-6 路由专家；V3 进一步到 256 个路由专家加 1 个共享专家，top-8。专家数量变多，单个专家的负载方差会下降；共享专家吸收通用知识，不参与 All-to-All，直接减少动态流量。原文把 V2 写成“8 个专家拆成 256 个”，这个数字对应的是 V3，V2 的配置是 160 加 2。

V2 还用了辅助负载均衡损失，给被选中频率过高的专家加惩罚。这个办法能压住长尾，但需要调超参数，训练目标里多了一项，模型可能为了平衡把 token 塞给不合适的专家，精度和稳定性都要打折。

V3 的主负载均衡换成了无辅助损失的方案。它在路由分数上加一个按专家维护的偏置 b_i，这个偏置只影响 top-k 选择，不影响被选中专家的组合权重；每一步训练后按负载偏差的符号更新，负载高于均值的专家把偏置调低，低于均值的调高。V3 另外保留了一个序列级的平衡损失，防止单条序列内部极端倾斜。原文把 V3 写成“容量上限加强制重分配”，这个说法和技术报告不符：V3 没有用硬容量截断，也没有把 token 强制塞给负载最低的专家，它靠的是偏置的反馈控制。

这个机制对 NVLink 的帮助是让流量更可预测。路由偏置把每个专家的负载往均值拉，通信库可以按容量上界预留缓冲，不需要为极端长尾留一个巨大的过载池；每个目标 GPU 的数据量更接近，微批次可以切得更均匀，流控不容易被单条热点链路打停。它不能保证每个专家严格相等，原文说“矩形分布、零方差”是夸张；实际能做到的是把长尾压短，让最慢链路和平均值的差距变小。

收费站的比喻可以留着，但要把数字改对：不均衡时一条车道排 1000 辆车，其他车道只有 10 辆，整体效率由那条最堵的车道决定；偏置均衡把车流往空闲车道引，最堵的车道会降到接近均值，但不会做到每条车道恰好 125 辆。通信优化真正买到的，是最慢链路变快；完全均匀只是逼近它的手段。

## 六、还没解决的部分

负载均衡目前仍在软件层做，NVLink 交换机看不到端口的实时队列深度，做不到硬件级的动态路由；细粒度通信原语也还要显式打包和解包，GPU 发出“把 token X 发给专家 Y”之后，寻址和传输仍由软件库完成。链路拥塞、丢包、延迟分布的硬件计数器还不够细，路由器拿不到实时网络状态，只能按历史负载调偏置。

跨节点的问题更大。NVLink 的优势在单机或机架内，出了机架要靠 InfiniBand 或 RoCE，延迟和带宽都掉一档；NVL72 把 72 张卡放进一个机架，华为 CM384 也在做类似的事，但铜互连的距离限制摆在那里 [待补：跨机 EP 扩到几百卡时的 All-to-All 延迟曲线]。光互联是继续扩大的方向，LPO、NPO、CPO 一路演进，成本、功耗和可靠性都还没有完全解决。低精度格式的原生路由也还没有落到链路层，FP4/FP8 的格式转换仍然要在 GPU 侧做。

这也是 DeepEP 的边界：它把单机或机架内的 All-to-All 做到了接近硬件的水平，跨机的 All-to-All 仍然受物理距离和网络协议限制。MoE 的 EP 域从几十卡扩到几百卡时，最先撞墙的往往是跨机那一段，GPU 算力通常还轮不到。

下次看到 MoE 推理的 TPOT 抖动，先看路由分布和跨机链路的队列，再看 kernel；单机 NVLink 的带宽通常不是第一嫌疑人。DeepEP 能解掉的是机架内的搬运，跨机那一段还得靠网络和路由算法一起解。

参考材料：DeepSeek-V2 和 V3 的技术报告（github.com/deepseek-ai/DeepSeek-V2、github.com/deepseek-ai/DeepSeek-V3），DeepEP 的代码和 README（github.com/deepseek-ai/DeepEP），以及 GShard、Tutel、Switch Transformer、Alpa 的论文。原文是 Ethan 的知乎专栏文章，链接为 zhuanlan.zhihu.com/p/2011403053715174158。

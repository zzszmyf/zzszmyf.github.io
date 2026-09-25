---
title: "SGLang RadixAttention 原理是什么？前缀缓存怎么提升命中率"
date: 2026-08-17T00:00:00+08:00
draft: false
description: "前缀缓存把「相同前缀、KV 用一次就扔」的浪费消掉。本文推导命中率 h 到 prefill 计算节省的收益模型，讲清 RadixAttention 的 radix tree、LRU leaves-first 驱逐与引用计数，排列 GPU→CPU→分布式→磁盘的 KV 存储层级，并解释 cache-aware routing 与 PD 分离。"
weight: 58
tags: ["LLM推理优化", "注意力内核"]
---

> 系列导航：[注意力与计算内核精读笔记总览](/notes/llm注意力内核精读笔记-00-总览与学习地图/)（共 8 篇）｜上一篇：[注意力优化上线怎么验收](/notes/llm注意力内核精读笔记-07-系统集成与生产验收/)

> 对应：Baseten *Inference Engineering* Ch5（Techniques）的 Caching 与 Disaggregation 部分；Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*（arXiv:2312.07104）；衔接 05 章（PagedAttention/vLLM）与量化系列 08 章（KV 量化）。
> 前置：01 章（KV cache 账本、prefill/decode 形态）、05 章（分页与共享）。学完本章你应该能：① 说出前缀缓存的三个杀手场景与它消除的浪费；② 推导"命中率 $h$ → prefill 计算省 $h$ 比例"的收益模型；③ 复述 RadixAttention 的 radix tree、LRU leaves-first 驱逐与引用计数机制；④ 排列 KV 存储层级并给出各自适用场景；⑤ 解释 cache-aware routing 与 disaggregation 中 KV 的角色；⑥ 手算命中率对 TTFT、KV 传输对延迟的影响。

---

## 目录（本章）

1. 本章目标
2. 问题：共享前缀，但 KV 用一次就扔
3. 收益模型：命中率与 prefill 节省
4. RadixAttention：把 KV 缓存当一棵树来管
5. Cache-Aware 调度：让缓存命中最大化
6. KV 存储层级：GPU → CPU → 分布式 → 磁盘
7. Cache-Aware Routing：把请求路由到"有缓存"的副本
8. Disaggregation：prefill 与 decode 分离
9. 组合：分页 + 前缀缓存 + 量化 + 批处理
10. 数值算例
11. 实验数据
12. 本章小结
13. 习题与解答
14. 延伸阅读

---

## 2. 问题：共享前缀，但 KV 用一次就扔

01 章讲过：prefill 阶段把整个 prompt 从第一层算到最后一层，产出 **KV cache**；decode 阶段逐 token 读 KV、续写输出。大多数 serving 系统里，一个请求算完、输出结束，**KV cache 就被丢弃**。

SGLang 论文（arXiv:2312.07104）把这件事指认为 LLM serving 最大的系统性浪费之一：

> **KV cache 的计算只依赖前缀 token。** 因此，凡是共享同一前缀的请求，本可以复用同一份 KV，却各自重新 prefill 了一遍。

现实负载里共享前缀无处不在：

```
多轮对话：每轮的 prompt = 系统提示 + 历史轮次 + 新问题
          → 历史轮次的 KV 完全可复用，只有新问题需要 prefill
代码补全：文件上下文 + 游标前的代码是公共前缀，多次补全共享
Agent（ReAct 等）：系统提示 + 工具定义 + 已执行轨迹，每一步都要"重发"整个轨迹
少样本（few-shot）：MMLU 每道题都带同样的 5 个示例
          → 1000 道题把示例 prefill 了 1000 遍；HellaSwag 甚至是两级共享
          （few-shot 示例 + 公共问题前缀）
自洽采样 / Tree-of-Thought：同一前缀分叉出多个候选分支，分支共享前缀 KV
```

浪费有两笔账：

```
① 计算账：无缓存时每个请求 prefill FLOPs ≈ 2LN（L 为 prompt 长度，N 为参数量）
           共享前缀越长、请求越多，重复计算越离谱
② 显存账：每份 KV 都重新占一块显存；相同前缀的 KV 在显存里重复存放
```

前缀缓存（prefix caching）的答案一句话：**把请求结束后的 KV 留下来，下一个共享前缀的请求直接复用，跳过重复的 prefill。**

---

## 3. 收益模型：命中率与 prefill 节省

### 3.1 定义

设请求的 prompt 长度为 $L$，其中 $L_p$ 个 token 能在缓存中命中，新 token 数为 $L_s = L - L_p$。定义命中率（论文定义）：

$$
h = \frac{L_p}{L} = \frac{\text{缓存命中的 prompt token 数}}{\text{总 prompt token 数}}
$$

### 3.2 逐层账本推导：省下的比例恰好是 $h$

先写单层的前向成本（沿用 01 章的记法：注意力 $4L^2d$、MLP $8d^2L$，这里 $d$ 为隐藏维度）：

**无缓存**——$L$ 个 token 全部前向：

$$
F_{\text{no}} = 4L^2d + 8d^2L
$$

**有缓存**——前缀 token 的前向整体跳过（QKV 投影、注意力、MLP 都不用重算）；后缀的 $L_s$ 个 token 仍然要 attend **全部** $L$ 个位置（它们必须看到前缀），所以注意力项是 $4 L_s L d$：

$$
F_{\text{cached}} = 4L_s L d + 8d^2 L_s
$$

两式相减：

$$
F_{\text{no}} - F_{\text{cached}}
= 4Ld(L - L_s) + 8d^2(L - L_s)
= (L - L_s)\big(4Ld + 8d^2\big)
$$

$$
= \frac{L_p}{L}\left(4L^2d + 8d^2L\right)
= h\, F_{\text{no}}
$$

结论：**命中率 $h$ 的请求，prefill 计算正好省下 $h$ 的比例**——注意力与 MLP 两项同时按 $h$ 节省，因为两者都随序列长度线性扩展，前缀跳过后就只剩后缀部分。任何 per-token 项（嵌入、QKV 投影、MLP）都有同样的性质，所以这个结论对任意模型结构都成立（只要命中段的前向确实被整体跳过）。

### 3.3 对 TTFT 的含义

prefill 是计算受限的（06 章），所以 prefill 时间随 FLOPs 近似线性：

$$
t_{\text{prefill}}(h) \approx (1 - h)\, t_{\text{prefill}}(0)
$$

TTFT 不能直接按 $(1-h)$ 缩，因为还有不随命中率缩水的固定开销：

$$
t_{\text{TTFT}} \approx \underbrace{t_{\text{fixed}}}_{\text{路由/调度/缓存查找/kernel 启动}} + (1-h)\, t_{\text{prefill}}
$$

### 3.4 三个容易算错的地方

```
① 命中省的是 prefill 计算，decode 每步成本不变
   （但显存被释放 → batch 可更大 → 吞吐上升，这是间接收益）
② 后缀 token 仍要"读"前缀的 KV：省的是 FLOPs，不是带宽
③ 命中率按 token 算；若命中段很短而固定开销很大，TTFT 收益被摊薄
```

---

## 4. RadixAttention：把 KV 缓存当一棵树来管

SGLang 的 RadixAttention 是第一套**自动、系统化**的 KV 复用机制：不再"请求结束就丢 KV"，而是把 KV 放进一棵 **radix tree** 里长期缓存，用 LRU 策略驱逐，并配合缓存感知的调度器。

### 4.1 数据结构：radix tree（压缩前缀树）

radix tree 是 trie 的空间高效变体：**边可以标注一串 token，而不是单个字符**。节点对应一个前缀，节点上挂着这个前缀的 KV cache 张量（分页存储，SGLang 里一页一个 token——天然兼容 05 章的 PagedAttention）。

看一个多轮对话的例子：

```
root
 └─ "What is the capital of "        ← 公共前缀（节点分裂产生）
     ├─ "France?"                     ← 请求 1 的 prompt（缓存）
     │    └─ "Paris. And its population?"  ← 请求 2 追加（复用请求 1 全部 KV）
     └─ "Germany?"                    ← 请求 3 的 prompt（缓存）
```

插入"Germany?"这条路径时，原树只有 "What is the capital of France?"，两者最长公共前缀是 "What is the capital of "——于是**节点分裂**，公共前缀上提，两个国家各自成为叶子。这就是 radix tree 的**多级共享**：系统提示、问题模板、历史轮次，每一级都能被复用。

### 4.2 三个核心操作

```
① Match（最长前缀匹配）：新请求沿树查找最长匹配节点
   → 命中即复用该前缀的 KV，这些 token 的 prefill 直接跳过
② Insert（插入）：未匹配的后缀作为新节点挂上去；
   请求的输出 token 也插入树（下一轮对话就能复用"上一轮的答案"）
③ Evict（驱逐）：显存不足时按 LRU 驱逐"最久未使用的叶子"
```

驱逐策略的两个关键设计：

```
Leaves-first（先叶子后祖先）：
  叶子是"私有尾巴"，驱逐它不影响其他请求复用公共祖先；
  祖先只有等所有子节点都被驱逐、自己也变成叶子后，才可驱逐
  → 热的公共前缀活得最久

引用计数（reference count）：
  正在运行的请求会"用"某些节点，refcount > 0 的节点不可驱逐
  → 连续批处理下，运行中请求的 KV 不会被缓存策略误杀
```

还有一个容易被忽略但很重要的设计——**共享内存池**：

```
SGLang 不预分配一块"缓存专用"的固定显存（比如 20% 给缓存、80% 给请求），
而是让"缓存中的 KV"和"运行中请求的 KV"住在同一个池子里：
缓存 = 当前没人用的 KV。
等待请求多时，系统自动驱逐缓存 token、把显存让给更大的 batch。
```

这样做的好处：显存永远不会"留给缓存却用不上"，也不会"缓存把请求挤爆"。代价是命中率随负载动态浮动——这正是第 5 节调度器要优化的对象。

### 4.3 伪代码

```
HandleRequest(prompt, rid):
  node, matched = radix_tree.match_longest_prefix(prompt)   # 最长前缀匹配，O(树高)
  if matched < len(prompt):
      node = radix_tree.insert(prompt[matched:], parent=node)  # 新后缀入树
  prefill_and_decode(node)       # 只对未命中 token 算 KV
  # 请求结束后 KV 保留在树中（不再丢弃）

EvictIfNeeded(required):
  while free_memory < required:
      leaf = LRU_least_recently_used_leaf()
      if leaf.refcount == 0:
          evict(leaf)            # 只驱逐没有运行请求在用的叶子
```

多轮对话/分叉场景还有一个配合技巧：前端解释器先发送"前缀提示"（frontend hint），让运行时知道这个分支会被继续复用，避免误驱逐——这体现了**前端语言与运行时协同设计**的价值。

### 4.4 开销

在没有复用机会的负载上（ShareGPT 100 个请求），RadixAttention 的数据结构管理只花 0.2 秒、占总时间 74.3 秒的 **不到 0.3%**——树操作是线性的，因此论文结论是：**默认开启，无需配置**。

---

## 5. Cache-Aware 调度：让缓存命中最大化

有了缓存，**请求的执行顺序**就变成了第一等的性能杠杆。命中率定义为：

$$
\text{hit rate} = \frac{\text{缓存命中的 prompt token 数}}{\text{总 prompt token 数}}
$$

先到先服务（FCFS）的问题是**缓存抖动（thrashing）**：调度器在不同前缀的请求之间反复横跳，缓存里刚热起来的 KV 很快被驱逐，下个请求又 miss。

RadixAttention 的调度策略是 **longest-shared-prefix-first（最长共享前缀优先）**：把等待队列按"已匹配前缀长度"降序排列，优先执行共享前缀最长的请求。可以证明这个顺序等价于对请求的 radix tree 做**深度优先遍历（DFS）**：

```
Schedule(batch_size, queue):
  ordered = sort_by_matched_prefix_len_desc(queue)   # 最长共享前缀优先 ≈ DFS 序
  return contiguous_batch(ordered[:batch_size])      # 与连续批处理结合
```

论文给出了一个离线情形下的最优性定理：

> **定理 3.1**：对一批请求，若缓存容量 ≥ 最大请求长度，按 DFS 序访问请求的 radix tree 可达到**最优命中率**；最长共享前缀优先序即 DFS 序。

直觉：DFS 沿着一条前缀链一口气处理完所有共享它的请求，公共 KV 只计算一次、一直热着；BFS/轮询式调度会让每个前缀刚建立就被换走。

权衡与边界：

```
贪婪提升吞吐，但可能饿死共享前缀很短的请求（论文留作未来工作：
与公平调度结合）
延迟敏感场景：可以容忍"有限重排"来换缓存命中
实测：命中率 50%–99%，调度器平均达到最优命中率的 ~96%
```

---

## 6. KV 存储层级：GPU → CPU → 分布式 → 磁盘

书里给了一条 KV 的"存储食物链"：

```
默认：GPU VRAM（热 KV，decode 每步都要读）
溢出①：CPU 内存（通过 Grace NVLink C2C 快速访问）
溢出②：分布式 KV store（跨副本共享）
溢出③：磁盘（最后手段）
```

| 层级 | 典型容量 | 带宽 | 定位 |
|---|---:|---:|---|
| GPU VRAM | 80–192 GB（H100/H200/B200） | 3.35–8 TB/s | 默认；decode 的每步读取都发生在这里 |
| CPU 内存（Grace C2C） | 数百 GB–1 TB+ | 900 GB/s（双向合计，约 PCIe 5.0 x16 的 7 倍） | 溢出层，可当"大号 GPU 内存"用 |
| 分布式 KV store | TB–PB（多副本） | 网络带宽 | 跨 replica 共享，cache-aware routing 的基础 |
| 磁盘 | 最大 | 很低 | 最后手段：冷数据、重启恢复 |

为什么要分层？01 章的账本：7B MHA 的 KV 是 **512 KB/token**——128K 上下文一个请求就要 64 GB，一张 H100 只够装一两个长请求。如果不允许溢出，长上下文服务在高并发下立刻被显存卡死；允许分层后，KV 预算从"单卡显存"扩展到"整机内存 + 集群"。

层级之间的核心权衡是**热冷**：

```
热前缀（系统提示、高频 few-shot、活跃会话历史）→ 留在 GPU
冷前缀（不活跃会话、低频共享段）→ 降级到 CPU / 分布式 store
层级越高，容量越大、带宽越低；工程上要做"按访问频率迁移"
（FlexGen 的 offload 思想，也是 RadixAttention 论文点名的未来方向）
```

书中配套的 [KV Cache Sizing Calculator](https://inferenceengineering.tech/exercises/kv-cache-calculator/) 就是用来先算"多少 KV、放哪层"的。

---

## 7. Cache-Aware Routing：把请求路由到"有缓存"的副本

生产部署通常有多个模型副本（replica）。路由策略直接决定命中率：

```
朴素路由（轮询 / 最少连接）：
  只看负载，不看缓存 → 同一会话的连续请求被分散到不同副本
  → 每个副本都没有上一轮的 KV → 每轮都 miss、每轮都重新 prefill

Cache-Aware Routing：
  把请求路由到"已持有最长匹配前缀 KV"的副本
  → 命中率 ↑，TTFT ↓（书里的原话）
```

难点：**缓存局部性与负载均衡天然冲突**。如果把请求全塞给"有缓存"的副本，那个副本会过载；分散到所有副本，缓存又全 miss。工程上常用的折中：

```
① Session affinity / 一致性哈希：同一会话/用户固定到同一副本
   → 多轮聊天的前缀自然在固定副本上累积（最简单、最常用）
② 分布式 KV store：公共前缀（系统提示、few-shot）放共享存储，
   任何副本都能取 → 局部性不再绑定单副本
③ 动态路由：NVIDIA Dynamo 按实时负载自动路由（第 8 节），
   并感知 KV 局部性，在"命中"与"负载均衡"之间动态权衡
```

一个数值直觉（详见第 10 节算例 10.3）：会话亲和让命中率从随机路由的 40% 提到 85%，prefill 时间变成原来的 $(1-0.85)/(1-0.4)=0.25$，即 prefill 部分快 4 倍。

---

## 8. Disaggregation：prefill 与 decode 分离

书里的最后一个相关主题是 **disaggregation（分离式部署）**：把 prefill 和 decode 放到**独立扩展**的硬件上：

```
Prefill workers：面向计算优化（高 FLOPS）——prefill 是计算密集的
Decode workers：面向带宽优化——decode 是带宽密集的
两者独立扩缩：流量 prefill-heavy 时加 prefill worker，decode-heavy 时加 decode worker
```

分离带来的新瓶颈是**传输**：prefill worker 算出的 KV cache 必须**转移给 decode worker**。KV 很大：

$$
7\text{B MHA}:\quad 512\ \text{KB/token} \times 2048\ \text{token} = 1\ \text{GB}
$$

所以书里明确说：**KV cache 量化（KV8/KV4，量化系列 08 章）是缓解这个传输瓶颈的手段**。

与前缀缓存协同的视角：

```
系统提示 / 历史轮次的 KV 是"跨请求共享"的 → 可以留在 decode worker
或分布式 KV store 里，不必反复传输；
只有新后缀的 KV 需要从 prefill worker 传过来
→ 命中率越高，跨阶段传输的 KV 越少（两章在此汇合）
```

NVIDIA Dynamo 提供**动态 disaggregation**：按实时负载自动在 prefill/decode worker 之间路由，不需要静态划分。

---

## 9. 组合：分页 + 前缀缓存 + 量化 + 批处理

RadixAttention 论文明确声明与三项技术兼容：**连续批处理（continuous batching）、PagedAttention、张量并行**。把本系列的工具拼起来：

```
连续批处理（05 章）：每步装最多请求，与缓存感知调度直接结合
PagedAttention（05 章）：KV 以物理块存放 → radix 树节点指向一组物理块，
                        共享祖先用引用计数/COW 管理，驱逐即释放块
前缀缓存（本章）：跨请求共享前缀的 KV 不重算
KV 量化（量化系列 08 章）：KV8/KV4 让同块显存放更多 token，
                          也缓解 disaggregation 的传输
Cache-aware 调度（本章）：批内顺序让命中率最大化
```

组合时容易混淆的收益边界：

```
前缀缓存命中 h → 省 h 比例的 prefill 计算（TTFT 的直接收益）
显存释放 → batch 变大 → 吞吐上升（间接收益）
KV 量化 → 显存和传输再缩（与缓存正交，叠加）
推测解码 → 省的是 decode 步数（另一个维度，不重复计算）
```

---

## 10. 数值算例

### 10.1 命中率 → prefill 节省（公式验证）

7B 模型、$L = 4096$、$h = 0.8$（$L_p = 3277$，$L_s = 819$）：

$$
F_{\text{no}} \approx 2LN = 2 \times 4096 \times 7\times10^9 \approx 5.7\times10^{13}\ \text{FLOPs}
$$

$$
F_{\text{cached}} = (1-h)F_{\text{no}} = 0.2 \times 5.7\times10^{13} \approx 1.1\times10^{13}\ \text{FLOPs}
$$

节省 $4.6\times10^{13}$ FLOPs，恰好 80%。

### 10.2 TTFT

H100 FP16 理论 989 TFLOPS，按 50% 实测效率折算约 500 TFLOPS：

$$
t_{\text{prefill}}(0) = \frac{5.7\times10^{13}}{5\times10^{14}} \approx 115\ \text{ms}
$$

$$
t_{\text{prefill}}(0.8) = 0.2 \times 115 \approx 23\ \text{ms}
$$

加上固定开销 10 ms（路由、调度、缓存查找）：TTFT 从约 125 ms 降到约 33 ms——**prefill 部分快 5 倍，端到端约 3.8 倍**。

### 10.3 Cache-Aware Routing

随机路由命中率 40%，会话亲和命中率 85%：

$$
\frac{1-0.85}{1-0.40} = 0.25
$$

prefill 时间缩到原来的 1/4。命中率从 40% 到 85%，看似只涨了一倍多，但 miss 的 token 从 60% 降到 15%——**收益要看"没命中的部分"而不是"命中的部分"**。

### 10.4 多轮对话 / few-shot 显存账

1000 道 MMLU 题，5-shot 示例共 1000 个公共 token，每题 100 个 token，7B MHA（512 KB/token）：

```
无缓存：1000 × (1000 + 100) × 512 KB ≈ 577 GB
有缓存：示例只算一次（1000 × 512 KB ≈ 0.5 GB）
        + 每题新 token（1000 × 100 × 512 KB ≈ 51 GB）
        ≈ 52 GB
省 ≈ 524 GB（≈ 91%）→ 同一张卡能服务的 batch 大一个量级
```

### 10.5 Disaggregation 传输

7B MHA、$L = 2048$：KV = 512 KB × 2048 = 1 GB。不同网络与量化下的传输时间：

| 配置 | 带宽 | 传输时间 | 对比 prefill（约 58 ms @500 TFLOPS） |
|---|---:|---:|---|
| 100 Gbps | 12.5 GB/s | ≈ 85 ms | **超过 prefill，成瓶颈** |
| 400 Gbps | 50 GB/s | ≈ 21 ms | 次要但不可忽略 |
| 400 Gbps + KV4 | 50 GB/s（传 256 MB） | 5 ms | 可忽略 |

结论与书一致：**KV 量化在 disaggregation 里不只是省显存，更是把传输从瓶颈降为噪声。**

---

## 11. 实验数据

论文（SGLang vs vLLM / Guidance / LMQL，A10G / A100）：

| 指标 | 数值 |
|---|---:|
| 吞吐提升 | 最高 **6.4×** |
| 延迟降低 | 最高 **3.7×** |
| 命中率（各 benchmark） | 50%–99% |
| 调度器相对最优命中率 | 平均 ~96% |
| RadixAttention 管理开销 | <0.3%（74.3 s 中 0.2 s） |

覆盖场景：agent control（ReAct）、逻辑推理、few-shot（MMLU/HellaSwag 两级共享）、JSON 解码、RAG 流水线（DSPy）、多轮 chat、自洽采样（self-consistency）。

生产观察（Chatbot Arena 部署一个月）：

```
Vicuna-33B：命中率 74.1%，平均 TTFT 降低 1.7×
LLaVA-NeXT-34B：命中率 52.4%（同一图片的多次提问共享图像 token 的 KV）
命中来源：公共系统消息、高频复用的示例图片、多轮对话历史
```

一个重要的边界：**多轮对话的提速在短输出时最明显**（前缀 prefill 占总延迟比例大）；输出很长时 decode 主导、不同会话间共享又少，提速趋于零。前缀缓存不是万能药——它只对"共享前缀占比高"的负载生效。

---

## 12. 本章小结

1. **问题**：KV 只依赖前缀，但请求结束就被丢弃 → 共享前缀被反复 prefill，既费计算又费显存；杀手场景：多轮对话、代码补全、Agent。
2. **收益模型**：命中率 $h$ 的请求，prefill 计算恰好省 $h$ 比例（第 3 节推导对注意力与 MLP 同时成立）。
3. **RadixAttention**：radix tree 存 KV + LRU leaves-first 驱逐 + 引用计数 + 共享内存池——自动、多级共享、无需手动配置，开销 <0.3%。
4. **调度**：最长共享前缀优先 ≈ DFS，接近最优命中率；FCFS 会缓存抖动。
5. **存储层级**：GPU VRAM → CPU（Grace C2C 900 GB/s）→ 分布式 KV store → 磁盘，按热冷迁移。
6. **路由**：把请求送到持有匹配前缀的副本（session affinity / 一致性哈希 / 动态路由），命中率与负载均衡需要折中。
7. **Disaggregation**：prefill/decode 分离后 KV 传输成为瓶颈，KV 量化（KV8/KV4）直接缓解。
8. **组合**：分页 + 前缀缓存 + 量化 + cache-aware 调度 + 连续批处理，正交叠加。

> 一句话记忆：**"KV 只依赖前缀——别为同一个前缀付两次钱：把 KV 留在 radix tree 里（记住），把请求送到有缓存的地方（找对），把缓存铺到 GPU 之外（扩容），prefill 就能按命中率 $h$ 省下 $h$。"**

---

## 13. 习题与解答

### 题 1（推导）：证明"命中率 = 节省比例"

用单层账本（注意力 $4L^2d$ + MLP $8d^2L$）证明：前缀命中比例 $h = L_p/L$ 的请求，prefill FLOPs 恰好省下 $h$ 比例。指出该结论对哪些成本项成立、对哪一项不成立。

<details>
<summary>题 1 解答</summary>

无缓存 $F_{\text{no}} = 4L^2d + 8d^2L$；有缓存时前缀前向整体跳过，后缀仍 attend 全部 $L$ 个位置：$F_{\text{cached}} = 4L_sLd + 8d^2L_s$。相减得 $(L-L_s)(4Ld+8d^2) = hF_{\text{no}}$。所有 per-token 项（嵌入、投影、MLP）都随 $L$ 线性扩展，同样按 $h$ 节省；唯一不省的是"后缀对前缀 KV 的注意力开销"——它被计入 $F_{\text{cached}}$ 的 $4L_sLd$ 项，因此若后缀很长，实际 TTFT 改善会略低于 $h$（加上固定开销后更明显）。
</details>

### 题 2（计算）：TTFT 收益

7B 模型、$L = 8192$、$h = 0.9$，H100 实测效率约 500 TFLOPS，固定开销 8 ms。求无缓存与有缓存的 TTFT（prefill 部分），以及端到端提升倍数。

<details>
<summary>题 2 解答</summary>

$F_{\text{no}} = 2 \times 8192 \times 7\times10^9 \approx 1.15\times10^{14}$ FLOPs → $t_{\text{prefill}} = 1.15\times10^{14}/5\times10^{14} \approx 230$ ms。有缓存：$0.1 \times 230 = 23$ ms。TTFT：$238 \to 31$ ms，约 **7.7 倍**。
</details>

### 题 3（概念）：为什么驱逐要"先叶子后祖先"、还要引用计数？

<details>
<summary>题 3 解答要点</summary>

叶子是"私有尾巴"，驱逐它不影响其他请求复用公共祖先；祖先被驱逐会一次性杀掉所有后代的共享机会。因此 LRU 只从叶子开始驱逐，祖先只有变成叶子后才能被淘汰——热的公共前缀因此活得最久。引用计数保护运行中的请求：连续批处理下，正在被 batch 使用的节点 refcount > 0，不可驱逐，否则会产生错误结果。共享内存池（缓存与运行请求同一池）则避免"固定缓存分区"造成的显存浪费或不足。
</details>

### 题 4（设计）：cache-aware routing 与负载均衡冲突

两个副本、50% 流量是共享同一系统提示的多轮聊天。给出路由方案，使命中率尽量高又不把某个副本打爆。

<details>
<summary>题 4 解答要点</summary>

方案组合：① 会话亲和（一致性哈希按 session 路由）让同一会话固定在单副本，前缀自然累积；② 公共系统提示/少样本 KV 放分布式 KV store，任何副本可取（局部性不再绑定单副本）；③ 动态路由（如 NVIDIA Dynamo）在"命中收益"与"副本负载"之间实时权衡；④ 热点副本扩容或把热前缀复制到多个副本（多副本各自持有同一份公共 KV）。核心原则：**把"跨请求共享的热前缀"与"单会话私有历史"分开处理**。
</details>

### 题 5（计算）：disaggregation 的传输瓶颈

70B GQA-8 模型（KV 每 token 320 KB，见 06 章题 2）、$L = 8192$ 的 prefill。KV 总量多少？在 200 Gbps（25 GB/s）网络上传输要多久？KV4 量化后呢？什么时候传输会成为主要瓶颈？

<details>
<summary>题 5 解答</summary>

KV = 320 KB × 8192 = 2.7 GB；200 Gbps 下 2.7 GB / 25 GB/s ≈ **107 ms**。KV4 后 0.67 GB → ≈ 27 ms。对比 prefill：$2 \times 8192 \times 70\times10^9 \approx 1.15\times10^{15}$ FLOPs，H100 上约 2.3 s——此时传输（约 100 ms）只占约 4%，不是瓶颈；但如果 prefill 分散到多卡（TP 并行把 prefill 时间除以卡数）或网络更慢，传输占比就会快速上升。**传输是否成为瓶颈取决于"KV 大小 × 请求频率"与"prefill 计算时间"之比**，KV 量化在两者间加了一个 4 倍的保险。
</details>

### 题 6（辨析）：RadixAttention 与 vLLM 前缀共享的区别

05 章讲过 vLLM 的 COW 与简单前缀共享。RadixAttention 多了什么？为什么"自动"很重要？

<details>
<summary>题 6 解答要点</summary>

① **多级树状共享**：vLLM 等早期系统只处理"单一系统提示"这类简单共享；radix tree 支持任意多级共享（系统 → 模板 → 历史轮次 → 分支），节点分裂/合并自动完成；② **LRU 缓存语义**：KV 被当作缓存管理（命中/驱逐/替换），而不仅是引用计数；③ **cache-aware 调度**：执行顺序反过来优化命中率；④ **前端提示**：解释器把"分支会继续"的意图传给运行时，减少误驱逐。自动的意义：论文指出此前的方法需要手动配置（如显式标记共享段），无法覆盖动态树状结构（agent 轨迹、自洽采样分支）；RadixAttention 对普通 serving 请求无需任何配置即可生效，且无命中时开销 <0.3%，所以可以默认开启。
</details>

---

## 14. 延伸阅读

1. [SGLang: Efficient Execution of Structured Language Model Programs（arXiv:2312.07104）](https://arxiv.org/abs/2312.07104)：RadixAttention、cache-aware scheduling、定理 3.1 与全部实验数据。
2. [PagedAttention / vLLM（arXiv:2309.06180）](https://arxiv.org/abs/2309.06180)（05 章）：分页布局与 COW，是 RadixAttention 的物理层基础。
3. [Inference Engineering Ch5（Baseten）](https://inferenceengineering.tech/chapters/techniques/)：前缀缓存、KV 存储层级、cache-aware routing 与 disaggregation 的教材正文。
4. [HydraGen（arXiv:2402.05099）](https://arxiv.org/abs/2402.05099)：从 kernel 侧加速共享前缀的批量注意力。
5. [Prompt Cache（arXiv:2311.04934）](https://arxiv.org/abs/2311.04934)：模块化注意力复用，但可能造成精度下降——注意与"精确复用前缀"的区别。
6. [FlexGen（arXiv:2303.06865）](https://arxiv.org/abs/2303.06865)：KV/权重跨存储层级 offload，本章第 6 节的理论背景。
7. [量化系列 08 章（KIVI）](/notes/llm量化精读笔记-08-kv-cache量化与kivi/)：KV8/KV4 量化，disaggregation 传输瓶颈的解法。
8. [NVIDIA Dynamo 文档](https://developer.nvidia.com/dynamo)：动态 disaggregation 与 KV-cache-aware 路由的工业实现。
9. 上一篇：[07 系统集成与生产验收](/notes/llm注意力内核精读笔记-07-系统集成与生产验收/)——本系列 00–08 的组合框架与验收协议。

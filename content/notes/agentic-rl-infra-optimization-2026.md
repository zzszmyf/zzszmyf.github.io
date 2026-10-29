---
title: "从各家技术文章看26年Agentic RL Infra优化方向"
description: "Agentic RL Infra 优化方向综述：解读 Minimax ForgeRL、ROLL、ThunderAgent、Kimi Seer 四篇大厂技术文章，归纳 AgentServer 抽象、滑动窗口调度、Prefix Tree Merging、推理加速等五大优化方向，附各系统核心贡献对比。"
date: 2026-02-21T00:00:00+08:00
lastmod: 2026-10-18T12:24:51+08:00
draft: false
weight: 100
aliases: ["/posts/agentic-rl-infra-optimization-2026/"]
categories: ["研究笔记"]
tags: ["研究笔记", "Reinforcement Learning", "AI Infra", "Math Theory"]
---

> 原文链接：https://zhuanlan.zhihu.com/p/2007250216227729670
> 作者：attack204（阿里云）
> 发布时间：2026-02-21

Agentic RL 的训练对象变了。以前一条 prompt 对应一条 response，rollout engine 只需要把 token 生成出来；现在一次 rollout 是一个 agent 程序，里面可能有几十轮工具调用、上下文重置、环境准备和长尾等待。Infra 的问题从“怎么把模型推理跑快”变成“怎么让一个有状态的 agent 循环跟无状态的训练器同步”。这是我看 Forge、ROLL、ThunderAgent、Seer 四篇文章时反复出现的同一个矛盾。

四篇里，Seer 针对同步训练，另外几篇更偏异步；场景不同，不能直接把结论拼成一张选型表。拆开看，它们各自把矛盾推到了不同的层。

## 一、Minimax Forge：把 Agent 生命周期拿出来

Forge 先动的是控制面。原来的数据流很直接：

```
RLFramework(Verl/Slime) -> RolloutEngine => [AsyncBuffer] -> Trainer
```

加一个中间层之后：

```
RLFramework(Verl/Slime) -> [AgentServer <-> RolloutEngine] => [AsyncBuffer] -> Trainer
```

AgentServer 把 agent 的生命周期从 rollout engine 里拿出来，负责 sandbox、工具调用和多轮循环；rollout engine 仍然只处理单次生成请求，trainer 从 AsyncBuffer 取 trajectory。这个抽象解决的是状态归属问题：一个多轮 agent 的 token 数、工具状态、上下文版本和当前阶段，不能塞进一次无状态的 generation request 里。

Forge 把 agent 分成黑盒和白盒两种。黑盒 agent 比如在 sandbox 里起一个 Claude Code，Claude Code 和 rollout engine 不断交互产生 trajectory，后面走正常的 RL 流程：算 reward、算优势、算 loss。白盒 agent 则把上下文管理（Context Management）显式建模成 agent action，让模型在训练阶段就知道上下文发生了变化。原文举了两个例子：DeepSeek V3.2 技术报告里，thinking 模型在用户消息到达时会清掉 thinking 内容，避免上下文过长；SearchAgent（比如 BrowseComp）几家都采用 discard-all 策略，token 超过上下文窗口的 80% 就重置整个窗口。示意代码如下：

```python
# discard-all策略示意
if token_usage > 0.8 * context_window:
    reset_context()
```

把 Context Management 当 action 的代价是训练分布更复杂：模型不仅要学会完成任务，还要学会在什么时机接受上下文被清空，并在清空后重新找到状态关键 token。收益是训练和推理的行为对齐；如果上下文管理只发生在推理框架里，模型在训练时根本不知道自己会在某一轮失去历史，长轨迹上的信用分配就会错位。这一层的工程成本不低：AgentServer 成了有状态服务，sandbox 崩溃、工具超时、上下文重置都要能恢复，否则一条长轨迹跑到一半就丢了。

### 滑动窗口：在新鲜度和长尾之间取一个界

异步 RL 一定产生 off-policy。原文把取舍说得很清楚：只取最新鲜的数据，旧版本直接丢，训练样本会偏向“快而简单”的样本；只取最旧的数据，又要等 rollout 的长尾，系统吞吐会掉下来。Forge 的做法是滑动窗口：窗口内的 trajectory 可以任意取来训练，但必须等窗口内的旧数据完成后才推进窗口。这个约束把样本年龄限制在一个界内，同时允许窗口内乱序消费，避免 trainer 被最慢的一条 rollout 卡住。

这个策略的关键参数是窗口大小和推进条件，原文没有给出具体数值。窗口太小，退化成“等最慢”；窗口太大，样本年龄上去，off-policy 程度变高，训练稳定性要重新调。`[待补：Forge 滑动窗口的实际窗口大小、样本年龄上限，以及吞吐与训练稳定性的对比数据]`

### Prefix Tree Merging：把推理侧的 KV 复用搬到训练侧

GRPO 对一个 prompt 生成 n 个 completion（原文举例 20 个），这些 completion 大多共享前缀。推理时可以用 SGLang 的 PrefixCache 复用 KVCache；训练时 Megatron 不会自动做这件事，每个 completion 的前缀都要重新算。Forge 的核心改动是在训练侧也用上前缀：先把 completions 组织成树，再借助 MagiAttention 的 attention mask 做 tree attention。

MagiAttention 原本是为 CP（context parallelism）并行设计的，但它提供了 attention mask 的语义，可以控制每个 sequence 的可见范围，正好能表达“多个 completion 共享一段前缀、在分叉后互相不可见”。AReal 团队也提过类似的 TreeAttention 方案，用 DFS 计算 attention，细节在 AREAL-DTA 论文里。这个优化的收益取决于前缀共享比例和 mask 的实现效率：树形 mask 会让 attention 变成块稀疏或变长模式，kernel 不支持的话，省下的 FLOPs 会被 mask 开销吃掉；训练侧的负载均衡也会变复杂，因为不同分支的长度不一样。`[待补：Prefix Tree Merging 在 Megatron 训练侧的 FLOPs、显存节省和端到端 step time]`

### 推理加速：Dynamic MTP、PD 分离与 L3 KV Cache

Forge 的推理侧有三块改动。第一块是 Dynamic MTP：用 MTP 做投机解码，同时用 Top-K KL loss 在 RL 过程中持续训练一个 detached MTP head，让 draft model 和 RL policy 保持对齐。原文作者补充了一个实现细节：多层 MTP 下，现有的 Eagle 算法不能直接支持，要用一种叫 Vanilla 的变种；SGLang 里已经由 liangsheng 做了实现（issue #15207），但每一层 MTP 单独跑一个 CUDA graph，可以优化成多层 MTP fuse 到一个 CUDA graph，这块阿里云的同学在跟进。这个细节比“用了 MTP”有价值：投机解码的接受率不是静态的，policy 每更新一次，draft head 就要重新对齐；把训练和推理的权重更新节奏绑在一起，才是 Dynamic 的含义。

第二块是 Rollout 侧的 PD 分离。Prefill 是计算密集的，decode 是访存和延迟敏感的，两者混在同一批实例上，MoE 调度会互相干扰。PD 分离给每个实例独立的并行和生成策略，让长尾样本不阻塞 FIFO scheduler。代价是 prefill 和 decode 的队列解耦后，样本年龄和调度复杂度会上升；原文提到这会带来更高的 off-policy，具体影响要看窗口和路由策略。

第三块是全局 L3 KV Cache。多轮、超长上下文的 agent 场景里，请求之间有很高的共享前缀比例；局部 KV cache 受容量限制，prefix cache 命中率上不去，RL batch size 一大就频繁驱逐、重算。Forge 支持全局 L3 KV cache，并用 cost-aware 的调度器权衡排队延迟和缓存传输时间，在不压垮实例的前提下最大化缓存局部性。这三块改动的共同点是都在抢“重复计算”的钱：MTP 抢 decode 的串行步数，PD 分离抢 prefill/decode 的干扰，L3 cache 抢跨请求的前缀重算。

### Dense Reward：Infra 很轻，算法很重

Agentic RL 基本是多轮调用，只对最终结果给 sparse reward，模型容易学会偷工减料。原文给了一个复合奖励：过程奖励监督中间行为，比如惩罚语言混合或特定工具调用错误；任务完成时间奖励把相对完成时间放进信号，因为真实延迟还受工具执行和子 Agent 调用影响；Reward-to-Go 用来标准化回报，降低长周期任务的梯度方差。对 infra 来说，新增一个 RewardServer 类就能接住；难的是算法侧怎么设计权重、怎么避免模型为了拿时间奖励而跳过必要步骤。完成时间奖励如果单独用，会鼓励走捷径，必须和正确性、过程奖励一起看。

## 二、ROLL：环境本身是训练算法的一部分

ROLL 也实现了单独的 AgentServer：用 Rock 作为 sandbox，启动内置的 iFlow CLI 作为 agent，和模型交互。它和 Forge 的抽象方向一致，但 ROLL 把更多篇幅放在环境管理上，这一点在 agentic RL 里很容易被低估。

Agent 在 rollout 过程中会留下中间产物，比如临时文件、日志、装了一半的依赖。这些产物会间接提示模型，甚至让模型直接读到或改掉测试脚本。原文提到一个可观察的信号：测试脚本的调用次数显著上升。这说明模型发现了环境里的漏洞，和模型能力提升没有关系。ROLL 在 rollout 前主动清理环境初始化和 agent 安装过程中产生的中间文件，测试文件只在最终评估阶段上传，和训练阶段严格隔离。它还会在环境里主动引入多样性，比如不同版本的软件包、不同镜像源、不同配置细节，甚至有意移除某个预装依赖、切换到不可用的镜像源。最后这一层看起来像折腾，实际是在防止模型把某一种环境配置背下来；训练环境越单一，模型越容易过拟合到环境的偶然特征。

数据侧也有类似的问题。ROLL 发现大量测试数据存在 false positive：数据不完整，或者本身就是错的。它在洗数据时加了一个 LLM-as-judge 验证模块，让多个 LLM 审查每组测试数据，只有通过验证的实例才进入 RL 训练池。具体做两种检查。如果 golden solution 无法通过全部测试，丢弃该实例。如果在不执行任何有效操作的情况下也能通过测试，也丢弃该实例。第二个检查尤其重要——一个 no-op 就能通过的测试，测不到任何东西，模型只要学会“不动”就能拿奖励。

ROLL 还提了 Chunked MDP。GRPO 在 token level 做重要性采样，GSPO 在 sequence level；Chunked MDP 把一次环境交互到下一次环境交互之间的连续片段定义成 Chunk，在 Chunk 级别计算 reward 和重要性采样。这个改动的动机是多轮轨迹的信用分配：token-level 的比值在长轨迹上噪声很大，sequence-level 又把整条轨迹压成一个动作，Chunk 正好对齐“一次工具调用/一次环境交互”这个语义单位。代价是 Chunk 边界怎么切、Chunk 内部怎么分配信用，需要和具体的 agent 协议一起设计。

## 三、ThunderAgent：给每个 Agent 程序建一个状态对象

ThunderAgent 的思路是在 AgentServer 和 RolloutEngine 之间加一个代理层。原来的系统把每一轮交互看作独立的推理请求，这样的问题是无法实时追踪每个 agent 任务的运行状态：已经用了多少 token、占了哪些工具资源、绑在哪个 backend、当前处在推理还是等待工具的阶段、是否可调度。ThunderAgent 把这些字段收进一个 Program 对象，例如 `P=⟨ID, c=15000 tokens, T={Docker,Bash}, L=backend_2, τ=Reasoning, s=Active⟩`。有了这个对象，调度器才能做“暂停/恢复”和资源回收，而不是只看到一个孤立的 request。

它把任务开销拆成五项：

```
Cost_{total} ≈ Cost_{decode} + Cost_{prefill} + Cost_{recompute} + Cost_{unused} + Cost_{caching}
```

优化目标主要是后三项：重计算、未使用容量和空闲缓存。具体手段是 State-Aware Pausing：当显存水位接近阈值时，暂停一部分程序并释放其 KVCache；水位下降时再恢复。恢复需要重新 prefill，所以暂停不是免费的；系统要尽量把“暂停谁”选对。

原文给了一个反直觉的结论：优先驱逐占 KV Cache 最小的程序。理由是 attention 的计算复杂度是 $N^2$：如果为了释放 $M$ 的显存，驱逐一个占 $M$ 的大程序，重算代价是 $M^2$；驱逐两个各占 $M/2$ 的小程序，重算代价是 $2 \times (M/2)^2 = M^2/2$。在只看重算 FLOPs 的模型里，小块更划算。这个结论有边界：驱逐次数、恢复延迟、工具状态的保存成本都会随“驱逐更多小对象”上升；当这些成本超过重算节省时，策略要反过来。原文的结论建立在它的开销模型上，不能直接当成通用规则。

工具资源管理也遵循同一套状态机。Hook-based garbage collection 在程序到达 Terminated 状态时清理环境资源；异步环境准备则监控全局队列，发现高优先级程序接近恢复阈值时提前 Prepare 环境。后者针对的是环境初始化延迟：装 Docker、拉镜像、起依赖可能比模型推理还慢，等到恢复时才做就来不及。`[待补：ThunderAgent pause/restore 策略的端到端吞吐、P99 与环境准备命中率]`

## 四、Kimi Seer：同步训练下的长尾调度

Seer 的场景是同步训练，原文作者也提醒：他看到的几个 RL case 基本都跑异步训练，所以 Seer 的适用性还要结合自己的同步/异步比例看。这个提醒很重要，这三个改动都建立在“同一组 response 要一起等”这个前提上。

### Divided Rollout：把一组 response 切成 8K 的块

GRPO 会对同一个 prompt 生成 n 个 response，常见调度会把这一组 response 绑到同一台实例上跑完。长尾一出现，两个问题同时发生：实例间负载不均衡；单实例的 KVCache 爆满触发抢占。Seer 的做法是把请求切成 Chunk 粒度，每个 Chunk 8K 长度，按负载调度到不同实例；KV Cache 缓存到 Mooncake，迁移时不需要重新 prefill。这个改动把“整条 response 的绑定”换成“Chunk 的绑定”，用 KV 传输换负载均衡。代价是调度开销和 KV 传输延迟；如果 Chunk 太小，调度次数上去；如果太大，长尾又回来了。8K 是论文选的折中，不是通用常数。`[待补：Seer 8K Chunk 在 Mooncake 上的 KV 传输耗时与重算节省对比]`

### Context-Aware Scheduling：用第一条请求探测组内长度

负载不均衡的根源是不知道哪些 request 会变成长尾，也就是不知道 response 会有多长。Seer 用了一个很便宜的探针：对每个 group，先挑第一条 request 作为 speculative request，优先调度；它完成后，用它的长度作为组内 response 长度的估计；剩下的请求按“长任务优先”调度。大致过程是：先用 SFS（Shortest First）调度所有 speculative requests，短的很快完成，长的暴露成长尾候选；Context Manager 记录每个 group 已完成请求的最大生成长度，作为该组预期长度的在线估计；最后对剩余请求按预测长度降序调度，也就是 approximate LFS。这个策略的假设是组内 response 长度相关；如果同组 response 的长度差异很大，第一条的估计就会偏，需要一个在线更新机制来纠偏。

### DGDS：把 N-gram 的匹配范围扩到多机

投机解码在 RL 里有两个麻烦。用 draft model 做 spec，RL 在 target model 和 draft model 之间的权重更新不同步，draft 的接受率会掉；用传统 N-gram，受限于单机，无法利用 GRPO 同组 response 之间的相似性。Seer 提出一种类似“分布式 N-gram”的做法：把不同 group 内 response 产生的 token 一起写进一棵分布式的压缩后缀树（CST），匹配的信息更多。流程是各实例异步 Append 新 token；DGDS Server 按 group 聚合，形成 group 级的 CST；各实例周期性 Fetch 自己正在处理的 group 的 CST，并且只同步上次 fetch 之后的新增内容。代价是跨实例的同步和过期问题：CST 更新有延迟，draft 可能匹配到旧数据；group 之间的隔离也要设计好，否则会把不相关的 token 混进来。

## 五、四篇文章放在一起看

Forge 和 ThunderAgent 都在加控制面：Forge 用 AgentServer 管 agent 生命周期，ThunderAgent 用 Program 管单个任务的运行状态；两者解决的是“无状态推理系统看不见 agent 状态”的问题。ROLL 把环境清理和数据验证当成训练算法的一部分，解决的是“模型会利用环境漏洞”的问题。Seer 在同步训练里切 Chunk、做长度估计和分布式 spec，解决的是“一组 response 被长尾绑死”的问题。四个系统的优化对象不同，但都在同一个方向上使劲：把 agent 的语义单位（程序、Chunk、轮次）暴露给调度器，让调度器不再把一次生成当成全部。

如果我要在自己的系统里落地，会先建 AgentServer 和 Program 状态；没有这个边界，调度、奖励和资源管理都是盲的。环境清理和数据验证要紧接着补上，否则前面的加速会被 reward hacking 吃掉。prefix/KV 复用和 PD 分离是吞吐的大头，适合在状态边界稳定后做；MTP 和分布式投机解码对 draft 质量、模型结构和推理框架的耦合最深，放到最后。论文里没有这条结论；这个顺序是从“哪一层缺了会让上层优化失效”推出来的。

还缺一块公共证据：四篇文章各自在自己的 workload 上报告收益，没有一组跨系统的端到端对比。Forge 的滑动窗口、ThunderAgent 的 pause/restore、Seer 的 8K Chunk，参数都和工作负载强相关；直接抄参数，很可能在自己的流量上失效。`[待补：四个系统在同一组 agent workload 上的吞吐、P99、样本年龄分布和 KV 命中率对比]`

如果只做一件事，先把 rollout 的样本年龄、长尾分布和 KV cache 命中率打点。没有这三个数，调度策略只能靠拍脑袋；有了它们，至少能判断应该先动窗口、先动 Chunk，还是先动缓存。

参考资料：Forge 的《大规模原生 Agent RL 系统》、ROLL 团队的《苦涩的教训！Agentic RL 训练中的实践经验》、ThunderAgent 的《A Simple, Fast and Program-Aware Agentic Inference System》、Seer 的《Online Context Learning for Fast Synchronous LLM Reinforcement Learning》，以及 AREAL-DTA 的《Dynamic Tree Attention for Efficient Reinforcement Learning of Large Language Models》。

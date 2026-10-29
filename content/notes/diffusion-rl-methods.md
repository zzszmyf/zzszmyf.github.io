---
title: "百花齐放的 Diffusion-RL：技术路线综述"
description: "Diffusion RL 技术路线综述：将扩散模型与强化学习结合的方案归纳为四大类——MDP 套框架（DDPO、Flow-GRPO）、前向过程优化（AWM、DiffusionNFT）、CFGRL 与 Q 函数方法，并比较各方法的代价、适用边界，以及 DanceGRPO 等噪声强度不对齐问题的修正工作。"
date: 2026-08-23T17:25:14+08:00
lastmod: 2026-10-18T12:24:51+08:00
draft: false
weight: 100
aliases: ["/posts/diffusion-rl-methods/"]
categories: ["研究笔记"]
tags: ["研究笔记", "Reinforcement Learning", "Diffusion Model"]
---

Diffusion 模型的 RL 微调在一年多里长出了至少四条路线：把多步降噪直接当 MDP，采样后在前向过程里加权，拿 CFG 当策略改进算子，以及固定原策略只学噪声。它们的差别主要在于把哪一段计算图当成可优化的策略，训练稳定性、reward 设计和算力开销都由此分叉。下面这份整理来自知乎专栏文章，原文链接是 https://zhuanlan.zhihu.com/p/2004562606309020589 ，公式保留原文记号。

## 背景

核心挑战有三个。降噪是一个连续多步的过程，RL 的决策框架要把每一步的噪声或 velocity 当成 action，才能定义状态和轨迹；训练稳定性要求新旧策略的分布不要差太远，否则 reward 信号会被巨大的策略偏移淹没；噪声强度则要和监督训练时保持一致，模型在某个信噪比下学到的 velocity，换到另一个信噪比下未必成立。这三条约束对应后面不同的技术路线，也解释了为什么同一篇论文里既有策略梯度，又有对采样过程的改造。

## 方法一：多步降噪看作 MDP

代表工作是 DDPO（https://arxiv.org/abs/2305.13301 ）和 Flow-GRPO（https://arxiv.org/abs/2502.06737 ）。两者都把多步降噪过程看成马尔可夫决策过程，直接套用 GRPO、策略梯度或 PPO 等 RL 框架；状态是当前的带噪样本，action 是网络预测的 velocity，轨迹是一条完整的降噪路径。

Flow Matching 的采样过程对应一条 ODE 轨迹。为了在每一步注入随机性，把 ODE 转成 SDE，于是每一步的 velocity 服从一个高斯分布：

$$
p(v_t | x_t) = \mathcal{N}\left(v_\theta(x_t), \sigma_t^2\right)
$$

这个高斯分布的均值是网络预测的 flow，方差由 $\sigma_t$ 决定，$\sigma_t$ 是预定义的与 $t$ 相关的量，例如 $\sigma_t = \sigma_{min} + t \cdot \sigma_{max}$。方差项就是注入随机性的入口，也让每一步的概率可以计算。

$v_\theta$ 被当作 RL 里的 action。一次完整降噪产生多条 trace，每条 trace 视为一条 trajectory；每一步根据上面的高斯分布得到 probability；拿到终点的 reward 后，就可以执行策略梯度或 GRPO 更新。DDPO 和 Flow-GRPO 都只在最后降噪结束时评估 reward，中间步骤还是噪声，单独打分意义不大。

Ref model 走同样的过程，只把 $v_\theta$ 换成 $v_{ref}$。两个分布都是高斯，KL 有闭式解，Flow-GRPO 的优化目标里可以直接写 KL 项：

$$
\mathcal{L}_{GRPO} = \mathbb{E}\left[ \frac{\pi_\theta(v_t|x_t)}{\pi_{old}(v_t|x_t)} \cdot A_t \right] - \beta \cdot KL(\pi_\theta || \pi_{ref})
$$

其中概率 ratio 是 actor 和 old policy 在高斯分布下的密度比，不同 $t$ 的 $v_\theta$ 来自同一个网络，reward 只在终点计算。两个同方差高斯的 KL 退化成均值差除以两倍方差，形式如下：

$$
KL(\mathcal{N}(\mu_1, \sigma^2) || \mathcal{N}(\mu_2, \sigma^2)) = \frac{(\mu_1 - \mu_2)^2}{2\sigma^2}
$$

这个形式计算便宜，也让 reference model 的约束可以逐步加在训练里。DDPO 则用策略梯度或 PPO，具体写法如下：

$$
\nabla_\theta J = \mathbb{E}_{x \sim p_\theta(x|c)} \left[ R(x, c) \cdot \nabla_\theta \log p_\theta(x|c) \right]
$$

PPO 则用 clip 限制更新幅度。目标函数如下：

$$
\mathcal{L}_{PPO} = -\mathbb{E}_t \left[ \min\left( r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

其中 $c$ 是输入条件，也就是网络拿到的各类条件。策略梯度把 reward 乘在 log-prob 的梯度上，PPO 再用 clip 限制 ratio 的更新幅度；两条式子对应同一条路线里的不同稳定化手段。

## 方法二：采样后在前向过程中优化

代表工作是 AWM（https://arxiv.org/abs/2410.01808 ）和 DiffusionNFT（https://arxiv.org/abs/2502.05597 ）。AWM 的核心结论是一句很长的论文标题：“DDPO is Secretly Doing Denoising Score Matching with Noisy Data”。它的逻辑链是：DDPO 等价于优化一个噪声加多了的前向过程，原本应该学习 $v_\theta(x_t)$，实际学的是 $v_\theta(\tilde{x}_t)$，其中 $\tilde{x}_t$ 是 $x_t$ 又加了额外噪声的版本。AWM 的实验表明这样不好，优化目标和原来的监督版本不一致。

解决方案很直接：先用噪声做前向，得到一批结果；给结果打分，算出 advantage；再把这些结果当作 $x_t$ 执行一次前向训练；前向过程乘以 reward，强化好结果、抑制坏结果。这样 flow 会远离坏方向，训练速度快，流程也简单。这跟 DPO 的思想类似，都是用偏好样本构造一个加权的回归目标，不去显式训练 reward model。

DiffusionNFT 关注的是 ref model 怎么用。它用同样的条件、不同的噪声生成一批结果，给每个结果打一个 0 到 1 的 optimality probability，得到多个 $(x_t, r)$ 对。然后执行前向过程，构造 positive velocity 和 negative velocity，记法如下：

$$
v_{pos} = v_{old}(x_{pos}), \quad v_{neg} = v_{old}(x_{neg})
$$

$v_{old}$ 是旧策略或 ref model 预测的 flow，$v_\theta$ 是要学的网络。优化目标让网络靠近正例、远离负例，损失写成：

$$
\mathcal{L} = \mathbb{E}\left[ r \cdot ||v_\theta(x_t) - v_{pos}||^2 - (1-r) \cdot ||v_\theta(x_t) - v_{neg}||^2 \right]
$$

梯度方向可以读成“纠正和躲避”：正例项把 $v_\theta$ 拉向 $v_{pos}$，负例项把它推离 $v_{neg}$。稳定性靠 old model 的滑动平均（EMA），old model 不断向新模型靠拢，两者差距不会拉得太大；梯度分析和 AWM 类似，只是多了一个“不要和 old 偏离太远”的约束项。

## 方案三：CFGRL

CFGRL 的代表论文是《Diffusion Guidance Is a Controllable Policy Improvement Operator》（https://arxiv.org/abs/2410.15470 ），它把 Classifier-Free Guidance 看成可控的策略改进算子。核心思想类似 RL 里的 $\pi^*$ 方法，具体推导要回到原论文。这条路线的好处是复用已有的 CFG 机制；代价是改进幅度受 guidance 机制约束，适用范围比前两条路线窄。

## 方案四：使用 Q 函数

这条路线问的是：能不能针对当前场景找到最好的噪声？它先训练一个 $Q(x_t, z)$，评估状态和噪声对的价值；再蒸馏一个噪声生成器 $z_\phi$，用 $\min_\phi ||z_\phi(x_t) - z^*||$ 让它和 Q 函数一致；最后用 $z_\phi$ 针对当前场景生成噪声。原 Flow Matching 策略固定不动，调整只发生在噪声上，因此调整幅度被限制住，不容易训崩，性能下限有原策略兜底。代价是要额外训练 Q 网络和生成器，训练管线比前三条路线长。

## 四条路线的取舍

MDP 路线优化降噪策略本身，把降噪视为 trajectory，用 PPO 或 GRPO 更新；它的优势是能直接套成熟 RL 算法，问题在于噪声强度与监督训练不对齐。前向优化路线优化 flow 预测，采样打分后在前向过程中加权；它简单有效、训练快，但需要设计好正负例的构造。CFGRL 优化引导策略，把 CFG 当策略改进算子；它利用已有 guidance 机制，适用范围却受 guidance 本身限制。Q 函数路线优化噪声生成器，固定原策略、只学最优噪声；它训练稳定、下限有保证，代价是多了 Q 网络和生成器两套训练。

四条路线没有统一的胜负。评价一个方案，先看它优化的是策略、flow、guidance 还是噪声，再看它的 reward 从哪里来、噪声对齐怎么处理；这几项答不清，指标涨了也很难复现。

## 噪声强度不对齐

DDPO 和 Flow-GRPO 的采样方法会让第 $t$ 步的噪声强度与监督训练时的第 $t$ 步强度对不上。模型在监督训练时见过某个信噪比下的 $x_t$，采样时注入的噪声却对应另一个信噪比，velocity 目标就偏了。DanceGRPO（https://arxiv.org/search/?query=DanceGRPO ）选择动态加噪声，根据任务情况和收敛情况改变噪声注入强度；Coefficients-Preserving Sampling（https://arxiv.org/search/?query=COEFFICIENTS-PRESERVING+SAMPLING ）推导强度不一致的原因和程度，再提出修正方法。后者对数学原理的分析更细，值得优先读。

## 还没解决的问题

这四条路线覆盖了目前的主要做法，但两个问题仍然没有统一解。中间步骤没有独立 reward，信用分配只能靠终点回传，长轨迹上的方差压不下去；噪声对齐目前靠 DanceGRPO 和系数保持采样修补，换一个采样器或换一个噪声日程就要重新验证。VLA 这类复杂场景对动作连续性和多模态条件的要求更高，这两块短板会更早暴露。

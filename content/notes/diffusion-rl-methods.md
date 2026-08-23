---
title: "百花齐放的 Diffusion-RL：技术路线综述"
date: 2024-04-03T00:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/diffusion-rl-methods/"]
categories: ["研究笔记"]
tags: ["研究笔记", "Reinforcement Learning", "Diffusion Model"]
---

# 百花齐放的 Diffusion-RL：技术路线综述

> 本文整理自知乎专栏文章，总结了近期 Diffusion Model 与 Reinforcement Learning 结合的多种技术方案。

**原文链接**：[https://zhuanlan.zhihu.com/p/2004562606309020589](https://zhuanlan.zhihu.com/p/2004562606309020589)

## 目录

- [背景](#背景)
- [方法一：多步降噪看作 MDP](#方法一多步降噪看作-mdp套已有rl框架)
- [方法二：采样后在前向过程中优化](#方法二采样后在前向过程中优化)
- [方案三：CFGRL](#方案三-cfgrl)
- [方案四：使用 Q 函数](#方案四使用q函数)
- [方法对比](#方法对比)
- [关键问题与修正](#关键问题与修正)
- [参考论文](#参考论文)

---

## 背景

Diffusion 模型与强化学习（RL）的结合近期涌现出多种方法，技术路线五花八门。本文将主流方案归纳为四大类，帮助读者建立清晰的知识框架。

核心挑战：
- 如何将连续的降噪过程与 RL 的决策框架结合
- 如何保持训练稳定性
- 如何对齐噪声强度（与监督训练一致）

---

## 方法一：多步降噪看作 MDP，套已有RL框架

**代表工作**：
- [DDPO](https://arxiv.org/abs/2305.13301): *Training Diffusion Models with Reinforcement Learning*
- [Flow-GRPO](https://arxiv.org/abs/2502.06737): *Training Flow Matching Models via Online RL*

### 核心思路

将多步降噪过程看作**马尔可夫决策过程（MDP）**，直接套用强化学习框架（GRPO、策略梯度、PPO 等）。

Flow Matching 的过程本质上是一个 ODE 过程。为了引入随机性，将 ODE 转换为 SDE：

$$
p(v_t | x_t) = \mathcal{N}\left(v_\theta(x_t), \sigma_t^2\right)
$$

其中：
- $v_\theta$ 是网络预测的 flow
- $\sigma_t$ 是预定义的与 $t$ 相关的方差（如 $\sigma_t = \sigma_{min} + t \cdot \sigma_{max}$）
- 通过 $\sigma_t$ 注入随机性（标准正态噪声）

### 概率与轨迹

上述公式表达的分布：
- 前两项为**均值**
- 最后一项为**方差**

$v_\theta$ 类比到 RL 中的 **action**。通过这种方式：
1. 获得多条降噪 trace → 视为多个 **trajectory**
2. 根据加入的噪声，利用上述分布获得对应 **probability**
3. 只要有 trajectory 相关的 **reward**，即可执行 RL 算法更新

> **实践注意**：DDPO 和 Flow-GRPO 都使用**最后降噪结束**的 trajectory 进行 reward 评估。中间步骤都是噪声，评估意义不大。

### 参考模型（Ref Model）

Ref model 过程相同，只是将 $v_\theta$ 换成 $v_{ref}$，同样可获得对应概率。由于都是正态分布，**KL 散度**误差也很好计算。

### Flow-GRPO 优化目标

$$
\mathcal{L}_{GRPO} = \mathbb{E}\left[ \frac{\pi_\theta(v_t|x_t)}{\pi_{old}(v_t|x_t)} \cdot A_t \right] - \beta \cdot KL(\pi_\theta || \pi_{ref})
$$

其中：
- 概率 ratio $\frac{\pi_\theta}{\pi_{old}}$：actor 和 ref 的 MDP 都用高斯分布表示，概率可计算
- 不同 $t$ 的 $v_\theta$ 一样，只用最后终点的 trajectory 计算 reward

**KL 散度**：直接套用两个高斯分布的散度公式：

$$
KL(\mathcal{N}(\mu_1, \sigma^2) || \mathcal{N}(\mu_2, \sigma^2)) = \frac{(\mu_1 - \mu_2)^2}{2\sigma^2}
$$

### DDPO 方法

使用策略梯度 / PPO（MDP 对上了，怎么套都行）：

$$
\nabla_\theta J = \mathbb{E}_{x \sim p_\theta(x|c)} \left[ R(x, c) \cdot \nabla_\theta \log p_\theta(x|c) \right]
$$

或：

$$
\mathcal{L}_{PPO} = -\mathbb{E}_t \left[ \min\left( r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

其中 $c$ 为输入条件（即网络中的各类输入）。

---

## 方法二：采样后在前向过程中优化

**代表工作**：
- [AWM](https://arxiv.org/abs/2410.01808): *Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models*
- [DiffusionNFT](https://arxiv.org/abs/2502.05597): *Online Diffusion Reinforcement with Forward Process*

### AWM 核心洞察

**核心结论**：

> **DDPO is Secretly Doing Denoising Score Matching with Noisy Data**

**逻辑链**：
- DDPO 等价于在优化一个**噪声加多了**的前向过程
- 本应该学习目标：$v_\theta(x_t)$
- 实际学习目标：$v_\theta(\tilde{x}_t)$，其中 $\tilde{x}_t$ 是 $x_t$ 加了额外噪声的版本

AWM 实验证明这样不好——优化目标与原来的监督版本不一致。

**解决方案**：

为了使得优化目标与监督学习一致，AWM 采用了简单直接的方法：

1. 先用一堆噪声前向推一下，获得一堆结果
2. 对这些结果打分，获得**优势（advantage）**
3. 将这些结果当作 $x_t$ 再次执行前向训练
4. **关键**：前向过程要乘以 reward $r$，强化好结果，抑制坏结果

这种方法让 flow 远离坏的方向，**训练速度快，简单有效**。

> 与 DPO（Direct Preference Optimization）思想类似。

### DiffusionNFT 方法

启发我们如何正确使用 ref model，通过构建**正例和负例**进行优化。

**算法流程**：

1. 用同样的条件加不同噪声生成一堆结果
2. 给这些结果打分，计算 **optimality probability**（0-1 之间的值，类似 reward）
3. 获得多个 $(x_t, r)$ 对
4. 执行前向过程，构造 **positive velocity** 和 **negative velocity**：

$$
v_{pos} = v_{old}(x_{pos}), \quad v_{neg} = v_{old}(x_{neg})
$$

其中 $v_{old}$ 是旧策略或 ref model 预测的 flow，$v_\theta$ 是要学习的网络。

5. 优化目标：

$$
\mathcal{L} = \mathbb{E}\left[ r \cdot ||v_\theta(x_t) - v_{pos}||^2 - (1-r) \cdot ||v_\theta(x_t) - v_{neg}||^2 \right]
$$

**梯度分析**：

网络实际学习的是一个**纠正和躲避**方向：
- 向正例方向靠近
- 远离负例方向

**稳定性保障**：

old model 不断与新模型做**滑动平均（EMA）**，保证两者差距不太大。梯度分析与 AWM 类似，只是添加了"不要和 old 偏离太远"的约束项。

---

## 方案三：CFGRL

**代表工作**：
- [Diffusion Guidance Is a Controllable Policy Improvement Operator](https://arxiv.org/abs/2410.15470)

将 **Classifier-Free Guidance (CFG)** 视为可控策略改进算子。

核心思想类似 $\pi^*$ 方法（RL 中的最优策略），具体可参见原论文。

---

## 方案四：使用 Q 函数

**代表工作**：
- [Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2502.04320)

### 核心思想

**问题**：能否找到针对当前场景**最好的噪声**？

这是一个 value-based 方法，核心流程：

1. **训练 Q 函数**：$Q(x_t, z)$ 评估状态-噪声对的价值
2. **蒸馏训练噪声生成器**：通过 $\min_\phi ||z_\phi(x_t) - z^*||$ 训练 $z_\phi$，让生成器与 Q 函数一致
3. **优化噪声生成器**：$z_\phi$ 针对当前场景生成最优噪声

### 方法优势

- **固定原策略**：训练好的 Flow Matching 策略固定不动
- **只调整噪声**：通过调整噪声来调整策略效果
- **限制调整幅度**：相当于限制死了调整幅度，不容易训崩
- **性能下限有保证**：原策略能力作为基础保障

---

## 方法对比

| 方案 | 优化对象 | 核心机制 | 优点 | 潜在问题 |
|------|---------|---------|------|---------|
| **MDP套框架** (DDPO/Flow-GRPO) | 降噪策略本身 | 将降噪视为 trajectory，用 PPO/GRPO 更新 | 直接套用成熟 RL 算法 | 噪声强度与监督训练不对齐 |
| **前向优化** (AWM/DiffusionNFT) | Flow 预测 | 采样打分后在前向过程中加权优化 | 简单有效，训练快 | 需要设计好正负例构造 |
| **CFGRL** | 引导策略 | 利用 CFG 作为策略改进算子 | 利用已有 guidance 机制 | 适用范围受限 |
| **Q 函数** (Latent RL) | 噪声生成器 | 固定原策略，学习最优噪声 | 训练稳定，下限有保证 | 需要额外训练 Q 网络和生成器 |

---

## 关键问题与修正

### 噪声强度不对齐问题

**问题**：DPPO 和 Flow-GRPO 的采样方法会导致 $t$ 步的噪声强度与**监督训练时**的 $t$ 步强度不对齐。

**修正工作**：

| 工作 | 贡献 |
|------|------|
| **DanceGRPO** | 考虑**动态加噪声**：根据任务情况、收敛情况动态改变噪声注入强度 |
| **COEFFICIENTS-PRESERVING SAMPLING** | 从根本上推导出强度不一致的原因及程度，并提出解决方法 |

> **推荐阅读**：COEFFICIENTS-PRESERVING SAMPLING 论文写得很好，深入分析了噪声强度不一致的数学原理。

---

## 参考论文

| 论文 | 链接 | 类别 |
|------|------|------|
| DDPO: Training Diffusion Models with Reinforcement Learning | [arXiv:2305.13301](https://arxiv.org/abs/2305.13301) | MDP 框架 |
| Flow-GRPO: Training Flow Matching Models via Online RL | [arXiv:2502.06737](https://arxiv.org/abs/2502.06737) | MDP 框架 |
| DanceGRPO: Unleashing GRPO on Visual Generation | [arXiv](https://arxiv.org/search/?query=DanceGRPO) | 噪声强度修正 |
| COEFFICIENTS-PRESERVING SAMPLING FOR REINFORCEMENT LEARNING WITH FLOW MATCHING | [arXiv](https://arxiv.org/search/?query=COEFFICIENTS-PRESERVING+SAMPLING) | 噪声强度修正 |
| AWM: Advantage Weighted Matching | [arXiv:2410.01808](https://arxiv.org/abs/2410.01808) | 前向优化 |
| DiffusionNFT: Online Diffusion Reinforcement with Forward Process | [arXiv:2502.05597](https://arxiv.org/abs/2502.05597) | 前向优化 |
| Diffusion Guidance Is a Controllable Policy Improvement Operator | [arXiv:2410.15470](https://arxiv.org/abs/2410.15470) | CFGRL |
| Steering Your Diffusion Policy with Latent Space RL | [arXiv:2502.04320](https://arxiv.org/abs/2502.04320) | Q 函数 |

---

## 总结

Diffusion-RL 领域正处于快速发展阶段，各路方法百花齐放：

1. **MDP 派**：直接套用 RL 框架，思路直接但需注意噪声对齐
2. **前向优化派**：采样后优化，简单有效，类似 DPO 思想
3. **CFG 派**：利用已有 guidance 机制做策略改进
4. **Value 派**：固定策略学噪声，稳定性最好

**技术处于早期起步阶段**，特别是在 VLA（Vision-Language-Action）等复杂场景下，仍有很多开放问题待解决。

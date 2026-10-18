---
title: "进一步推进 MuP：从 Muon 到 SSO"
description: "SSO（谱球面优化器）原理解读：在 Muon 只约束参数更新量谱范数的基础上，进一步约束权重矩阵谱范数，完整满足 MuP 的两个 feature learning 条件。实验显示参数量变化 25 倍时最优学习率保持不变，且 loss 下降更快、参数可迁移性更好、MoE 路由负载更均衡。"
date: 2024-04-14T00:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/sso-spectral-sphere-optimization/"]
categories: ["研究笔记"]
tags: ["研究笔记", "Math Theory"]
---

> 原文链接：https://zhuanlan.zhihu.com/p/2008580956940956661　作者：paperplanet（知乎）　收录于：皮皮虾的机器不学习专栏　编辑时间：2026-02-22 17:18（上海）

模型参数量放大 25 倍，SSO 的最优学习率保持不变；Muon 的最优 LR 会跟着规模漂，最终 loss 也更高。这是 SSO 论文里最直接的一组对照，也是它想解决的问题：Muon 在 MuP 框架里只满足了两个 feature learning 条件中的一个，学习率仍然要随宽度重调。SSO 的做法是把权重矩阵的谱范数也约束住，让每一步更新都落在半径为 R 的谱球面上。

![SSO 与 Muon 在不同参数量下的最优学习率对比](https://pic4.zhimg.com/80/v2-a61c59c1e7f4a8a27b7b70a7f85e21ee_720w.webp)

图里右边是 SSO：参数量变化 25 倍，最优 LR 保持不变；Muon 的曲线会移动。

## 1. Muon 缺的那一半

Muon 可以看成参数更新量谱范数限制下的最速梯度下降：每一步在 $\|\Delta \boldsymbol{W}\|_2 = \eta$ 的球内找使梯度下降最多的方向，这个方向由 msign(G) 给出。它在实践中比 AdamW 更有效率，但在 MuP 框架里只满足了两个条件中的一个。另一个条件是权重矩阵的谱范数。不约束它，W 在训练中会漂移，更新量相对 W 的比例随宽度变化，最优学习率就得跟着规模重调；规模一变，Muon 就要重扫 LR，loss 也高于 SSO。

这一系列工作的出发点都是 feature learning：让网络中的神经元处在更有效学习特征的状态。在 MuP 的视角里，只有更新尺度和权重尺度都受控，feature learning 的条件才同时成立；只约束更新量，宽度一变，两者比例就变。

## 2. 约束权重矩阵的谱范数

问题可以写成：在 $\|\Delta \boldsymbol{W}\|_2 = \eta$ 且 $\|\boldsymbol{W} + \Delta \boldsymbol{W}\|_2 = R$ 的条件下，最大化梯度方向上的更新。为方便，取 $\eta = 1$；$R$ 是目标谱范数球的半径，$\boldsymbol{G}$ 是梯度，$\Delta \boldsymbol{W}$ 是要求的更新方向。

为了让更新后的权重仍落在谱球面上，需要对谱范数 $\|\boldsymbol{W}\|_2$ 求梯度。在最大奇异值唯一时，这个梯度是

$$\boldsymbol{\Theta} = \boldsymbol{u}\boldsymbol{v}^T$$

其中 $\boldsymbol{u}, \boldsymbol{v}$ 是最大奇异值对应的左右奇异向量。对 $\|\boldsymbol{W} + \Delta \boldsymbol{W}\|_2$ 在 $\boldsymbol{W}$ 处做一阶泰勒展开：

$$\|\boldsymbol{W} + \Delta \boldsymbol{W}\|_2 \approx \|\boldsymbol{W}\|_2 + \langle \boldsymbol{\Theta}, \Delta \boldsymbol{W} \rangle$$

要让更新后的权重仍落在半径为 $R$ 的谱球面上，一阶项必须为 0：

$$\langle \boldsymbol{\Theta}, \Delta \boldsymbol{W} \rangle = 0$$

也就是说，梯度更新方向与权重矩阵的限制球面相切。

## 3. 拉格朗日乘数法求方向

把这个切面约束和 $\|\Delta \boldsymbol{W}\|_2 = \eta$ 一起写进拉格朗日函数，引入乘子 $\lambda$：

$$\mathcal{L} = \langle \boldsymbol{G}, \Delta \boldsymbol{W} \rangle - \lambda \langle \boldsymbol{\Theta}, \Delta \boldsymbol{W} \rangle$$

在 $\lambda$ 已知时，满足约束的最大值在

$$\Delta \boldsymbol{W} = \eta \cdot \text{msign}(\boldsymbol{G} + \lambda \boldsymbol{\Theta})$$

处取到。剩下的问题是求 $\lambda$。它由切面约束方程决定：

$$h(\lambda) := \langle \boldsymbol{\Theta}, \text{msign}(\boldsymbol{G} + \lambda \boldsymbol{\Theta}) \rangle = 0$$

$h(\lambda)$ 是单调有界函数，可以从 0 开始向数轴两边做数值搜索。论文没有给闭式解；每一步要多次计算 msign，按定义就是对矩阵做奇异值分解，λ 的搜索次数直接决定 SSO 相对 Muon 多出来的开销。具体迭代次数原帖没有给，补测时记 `[待补：λ 搜索的平均迭代次数与每步 Wall-clock 开销]`。

## 4. 二阶项与谱范数缩放

一阶项为 0 只保证当前步相切；二阶项会在训练中累积，$\boldsymbol{W}$ 的谱范数仍可能漂离 $R$。作者的做法是每步显式缩放：

$$\boldsymbol{W} \leftarrow \boldsymbol{W} \cdot \frac{R}{\|\boldsymbol{W}\|_2}$$

谱范数用幂迭代求，顺带得到最大奇异值对应的左右奇异向量，正好就是切面方程里的 $\boldsymbol{\Theta}$。幂迭代的迭代次数和矩阵条件数有关，原帖没有给具体设置。

![SSO 训练流程](https://pic1.zhimg.com/80/v2-754bc2dc73db6354c6a71f2a0f7f1b3f_720w.webp)

训练流程可以按上面四步理解：求梯度、解 λ、算 msign 更新、把 W 缩回谱球面。

## 5. 几何图像

几何上，左下角的实线圆弧是 $\boldsymbol{W}$ 的谱范数球面，右上角的虚线球体是更新量的谱范数球；$\boldsymbol{G}$ 是梯度。Muon 只在更新量球体上取极值，方向可能把 $\boldsymbol{W}$ 推离球面；SSO 先在 $\boldsymbol{W}$ 球面的切面上取更新量球体的极值，再沿同方向把 $\boldsymbol{W}$ 缩回球面。两个约束同时满足后，每一步的更新量相对 $\boldsymbol{W}$ 的尺度就被固定住了。

## 6. 实验结果

论文报告的优势有五项：loss 下降更快、参数可迁移性更好、MoE 路由负载更均衡、异常值更少、激活值被约束在可调范围内。文章只给了定性描述，没有贴出 loss 曲线、可迁移性指标和 MoE 负载的数值。目前能核实的硬数字只有 25 倍参数量范围内最优 LR 不变这一组，它属于实验设置，不在这五项里；这五项优势还缺能复核的数：`[待补：Muon 与 SSO 在相同 token 数下的 loss 曲线和最终 loss 差值]`；`[待补：参数可迁移性的具体指标与跨规模结果]`；`[待补：MoE 路由负载均衡的分位数或最大/均值比]`；`[待补：异常值与激活范围的统计口径]`。

这些优势的因果链是：约束 $\boldsymbol{W}$ 的谱范数和更新的谱范数，每层更新尺度就不随宽度漂移，feature learning 条件才满足，特征表达才更稳定。这条链条在论文里是论证；能复核的量化结果目前只有 25 倍 LR 稳定这一项。工程上要看的是相同 token 预算下的 loss 和下游指标。

## 7. 和 Muon 的账

和 Muon 比，SSO 多约束了 $\boldsymbol{W}$ 的谱范数，因此完全兼容 MuP 的两个条件；代价是每步多一次幂迭代和一次缩放，以及一个一维数值搜索求 λ。Muon 的优势是实现简单、没有额外超参；SSO 把学习率从随规模调变成一次调好，省下的是大规模训练中反复做 LR sweep 的算力。这笔账在小模型上不一定划算：λ 搜索和幂迭代都是每步开销，小模型训练步数少、LR sweep 便宜，SSO 的收益可能抵不过实现复杂度。

## 8. 还没解决的问题

SSO 现在还没有在公开的大规模训练里给出完整账单：每步多出来的开销占多少训练时间，25 倍参数量的结论能不能外推到 10B 以上，都还没有数据。要上线，先在小模型上把这两笔开销测出来，再决定值不值得替换 Muon。

## 参考

原始论文是 [arXiv:2601.08393](https://arxiv.org/pdf/2601.08393)；相关文章是《大模型控制学——谱球优化器》和《流形上的最速下降：4. Muon + 谱球面》。

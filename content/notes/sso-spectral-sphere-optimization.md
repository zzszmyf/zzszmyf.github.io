---
title: "进一步推进 MuP：从 Muon 到 SSO"
date: 2024-04-14T00:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/sso-spectral-sphere-optimization/"]
categories: ["研究笔记"]
tags: ["研究笔记", "Math Theory"]
---

# 进一步推进 MuP：从 Muon 到 SSO

> 原文链接：https://zhuanlan.zhihu.com/p/2008580956940956661  
> 作者：paperplanet（知乎）  
> 收录于：皮皮虾的机器不学习专栏  
> 编辑时间：2026-02-22 17:18（上海）

---

## 1. 背景与动机

众所周知，**Muon** 可以看成是**参数更新量谱范数限制下的最速梯度下降**，在实践中相对于 AdamW 有更高的效率。

在 **MuP（Maximal Update Parametrization）**的框架下，Muon 达到了 feature learning 需要的两个条件中的**一个**。那么在 MuP 的框架下，Muon 能不能更进一步呢？

**答案是可以的**，这个答案就是 **SSO（Controlled LLM Training on Spectral Sphere）**：在**同时约束矩阵谱范数以及参数更新量谱范数**的条件下达到最速梯度下降。

这一系列工作的出发点都是 **feature learning**，让网络中的神经元都处在更有效学习特征的状态，从而获得更加有效的特征表达，同时减少不稳定性并加快收敛速度，减少模型内部空间的特征冲突。

---

## 2. 参数稳定性与 Muon 的对比

![SSO vs Muon 对比图](https://pic4.zhimg.com/80/v2-a61c59c1e7f4a8a27b7b70a7f85e21ee_720w.webp)

**右边是 SSO**：在模型参数量变化 **25 倍**的情况下，最优 LR **保持不变**。

**Muon 的问题**：由于不是完全遵守 MuP 的两个条件（只约束了梯度更新量的谱范数而没有约束参数矩阵的谱范数），最优 LR 会有变化，最终 loss 也大于 SSO。

---

## 3. 约束权重矩阵的谱范数

### 3.1 问题定义

在满足以下两个条件下，最大化梯度方向上的权重更新：

1. **单位权重更新谱范数**：$\|\Delta \boldsymbol{W}\|_2 = \eta$
2. **权重更新后的值**：$\|\boldsymbol{W} + \Delta \boldsymbol{W}\|_2 = R$

其中：
- $\boldsymbol{G}$ 为梯度
- $\Delta \boldsymbol{W}$ 为要求的权重更新方向
- 为表示方便约定 $\eta = 1$（谱范数为 1）
- $R$ 为目标谱范数球体约束半径

### 3.2 求解

为了让更新后的权重依然保持在**半径为 $R$ 的谱球面上**（继续满足 MuP 的权重矩阵谱范数约束），需要对权重的谱范数求梯度。

对于谱范数 $\|\boldsymbol{W}\|_2$，在最大特征值唯一的情况下，可以求梯度得到：

$$\boldsymbol{\Theta} = \boldsymbol{u}\boldsymbol{v}^T$$

其中：
- $\boldsymbol{u}, \boldsymbol{v}$ 是最大奇异值对应的左右奇异向量

考虑权重在某个时刻的一阶泰勒展开：

$$\|\boldsymbol{W} + \Delta \boldsymbol{W}\|_2 \approx \|\boldsymbol{W}\|_2 + \langle \boldsymbol{\Theta}, \Delta \boldsymbol{W} \rangle$$

为了让更新后的权重依然保持在半径为 $R$ 的谱球面上，需要让一阶项为 0：

$$\langle \boldsymbol{\Theta}, \Delta \boldsymbol{W} \rangle = 0$$

即**梯度更新向量与权重矩阵的限制球面相切**。

### 3.3 拉格朗日乘数法

通过拉格朗日乘数法，引入拉格朗日乘子 $\lambda$，对拉格朗日函数：

$$\mathcal{L} = \langle \boldsymbol{G}, \Delta \boldsymbol{W} \rangle - \lambda \langle \boldsymbol{\Theta}, \Delta \boldsymbol{W} \rangle$$

在约束 $\|\Delta \boldsymbol{W}\|_2 = \eta$ 下求极值。

在 $\lambda$ 已知的情况下，满足约束的最大值在：

$$\Delta \boldsymbol{W} = \eta \cdot \text{msign}(\boldsymbol{G} + \lambda \boldsymbol{\Theta})$$

处取到。

### 3.4 求解 λ

接下来的问题就是如何求 $\lambda$，可以由求解 $\lambda$ 的切面约束方程得到：

$$h(\lambda) := \langle \boldsymbol{\Theta}, \text{msign}(\boldsymbol{G} + \lambda \boldsymbol{\Theta}) \rangle = 0$$

而 $h(\lambda)$ 是一个**单调有界函数**，因此可以通过从 0 开始向数轴两边数值搜索逐渐迭代的方式求解。

---

## 4. 梯度更新二阶项的影响

因为梯度更新中的二阶项可能逐渐积累，使得权重矩阵 $\boldsymbol{W}$ 的谱范数漂移离开谱范数=$R$ 的限制，因此作者提出直接对权重矩阵 $\boldsymbol{W}$ 在训练中做**谱范数缩放**，强制限制在 $R$：

$$\boldsymbol{W} \leftarrow \boldsymbol{W} \cdot \frac{R}{\|\boldsymbol{W}\|_2}$$

谱范数由**幂迭代（Power Iteration）**得到，同时能得到最大奇异值对应的左右奇异向量，这两个奇异向量还组成后续计算切面方程用的 $\boldsymbol{\Theta}$。

---

## 5. 整体流程

### 5.1 训练流程示意图

![SSO 训练流程](https://pic1.zhimg.com/80/v2-754bc2dc73db6354c6a71f2a0f7f1b3f_720w.webp)

### 5.2 与 Muon 对比示意图

![SSO vs Muon 几何对比](https://pic3.zhimg.com/80/v2-f4c4e4a5d9f4c3e6c9f4c3e6c9f4c3e6_720w.webp)

**几何解释**：
- **左下角实线圆弧**：表示权重矩阵 $\boldsymbol{W}$ 的谱范数=$R$ 限制球体
- **右上角虚线球体**：权重更新量的谱范数限制球体
- **$\boldsymbol{G}$**：梯度
- **蓝色箭头（Muon）**：只限制了参数更新量的谱范数限制球体
- **绿色箭头（SSO）**：先在阴影的权重谱范数限制球体切面上取得权重更新谱范数球体限制上的极值，然后再方向不变地缩回权重谱范数限制球体上

**核心优势**：同时满足两个 MuP 限制条件的情况下，取得梯度法向上的最大更新量。

---

## 6. 实验结果

作者的实验结果显示，相比 Muon，SSO 具有以下优势：

1. **Loss 下降速度更快**
2. **参数更具有可迁移性**
3. **更好的 MoE 路由负载均衡**
4. **异常值更少**
5. **激活值也约束在可调节范围内**

这些优势很大程度上来自于：符合 MuP 条件限制 → 更好地实现了 feature learning → 得到更好的特征表达。

---

## 7. 参考链接

1. **原始论文**：https://arxiv.org/pdf/2601.08393
2. **相关文章**：
   - 大模型控制学——谱球优化器
   - 流形上的最速下降：4. Muon + 谱球面

---

## 8. 总结

| 特性 | Muon | SSO |
|------|------|-----|
| 约束更新量谱范数 | ✅ | ✅ |
| 约束权重矩阵谱范数 | ❌ | ✅ |
| MuP 完全兼容 | ❌ | ✅ |
| 最优 LR 稳定性 | 随规模变化 | 25倍参数量变化不变 |
| Feature Learning | 部分满足 | 完全满足 |

SSO 通过同时满足 MuP 的两个核心条件，在谱球面上实现了更稳定的训练动态和更优的特征学习效果。

---
title: "TurboQuant 数学原理解析"
date: 2024-04-16T00:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/turboquant-math-principles/"]
categories: ["研究笔记"]
tags: ["研究笔记", "Quantization", "Math Theory"]
---

# TurboQuant 数学原理解析

## 1. Johnson-Lindenstrauss 变换（QJL 基础）

### 核心定理

对于任意高维数据集 $X \subset \mathbb{R}^d$，存在映射 $f: \mathbb{R}^d \to \mathbb{R}^k$（其中 $k = O(\varepsilon^{-2} \log |X|)$），使得对所有点 $u, v \in X$：

$$\|f(u) - f(v)\|^2 \approx \|u - v\|^2$$

即**距离保持性**——降维后点间距离基本不变。

### QJL 的创新

```python
# 标准 JL：随机投影 + 保存浮点数
projected = sign(A · x)  # A 是随机高斯矩阵

# QJL：只保存符号位（1-bit）
qjl_output = sign(A · x) ∈ {-1, +1}^k
```

**关键公式**：
- 原始内积：$\langle q, k \rangle$
- QJL 估计：$\tilde{s} = \frac{1}{m} \sum_{i=1}^{m} \text{sign}(A_i q) \cdot \text{sign}(A_i k) \cdot c_i$

其中 $c_i$ 是校准常数，用于消除量化偏差。

---

## 2. 极坐标量化（PolarQuant）

### 笛卡尔坐标 → 极坐标转换

对于向量 $x \in \mathbb{R}^d$，PolarQuant 将坐标对分组处理：

**第 1 步：配对转换**
$$(x_{2i-1}, x_{2i}) \to (r_i, \theta_i)$$

其中：
- $r_i = \sqrt{x_{2i-1}^2 + x_{2i}^2}$ （半径/模长）
- $\theta_i = \arctan2(x_{2i}, x_{2i-1})$ （角度）

**第 2 步：递归极坐标变换**

将半径继续配对，形成层级结构：

```
Level 0: (r1, θ1), (r2, θ2), (r3, θ3), (r4, θ4)
Level 1: (R12, φ12), (R34, φ34)    ← r1,r2 再转极坐标
Level 2: (R1234, φ1234)             ← 最终单一半径
```

最终存储：**1 个总半径 + d-1 个角度**

### 为什么这能消除开销？

| 传统方法 | PolarQuant |
|---------|-----------|
| 需存储每个块的 min/max 做归一化 | 角度天然集中在 $[0, 2\pi)$，分布已知 |
| 边界动态变化 | 固定"圆形网格"边界 |
| 额外 1-2 位存储常数 | 无需额外存储 |

---

## 3. TurboQuant 的两阶段框架

### 阶段一：PolarQuant（主压缩）

$$\mathbf{v} \xrightarrow{\text{Random Rotation } R} \mathbf{v}' \xrightarrow{\text{PolarQuant}} (\mathbf{r}, \boldsymbol{\theta})$$

- 随机旋转使数据各向同性（isotropic）
- 量化：半径用高位，角度用低位

### 阶段二：QJL（残差修正）

设阶段一的重建向量为 $\hat{\mathbf{v}}$，残差为：

$$\mathbf{\delta} = \mathbf{v} - \hat{\mathbf{v}}$$

QJL 用 1-bit 编码残差：
$$\mathbf{\delta}_{\text{qjl}} = \text{sign}(A \cdot \mathbf{\delta})$$

### 注意力分数计算

$$\text{Attention}(Q, K) = \underbrace{Q \cdot \hat{K}_{\text{polar}}}_{\text{主要项}} + \underbrace{\text{QJL\_Estimator}(Q, \delta_K)}_{\text{偏差修正}}$$

---

## 4. 理论保证

### 无偏估计

$$E[\tilde{s}] = \langle q, k \rangle$$

即期望上，量化后的分数等于真实分数。

### 方差界限

$$\text{Var}(\tilde{s}) \leq \frac{C}{m} \|q\|^2 \|k\|^2$$

$m$ 为 QJL 维度，可通过增加 $m$ 任意减小误差。

### 内存复杂度对比

| 方法 | 每向量比特数 | 存储内容 |
|-----|------------|---------|
| FP32 | 32d | 原始浮点 |
| INT8 | 8d | 量化整数 |
| KIVI | 4d + 开销 | 分组量化 |
| **TurboQuant** | **3d** | 极坐标 + 1-bit 残差 |

---

## 5. 直观理解

```
传统量化：把 3.1415926... 存成 "3.14" + 缩放因子 "100"
         → 需要额外空间存"100"

PolarQuant：把 (3, 4) 存成 "长度5" + "角度53°"
            → 角度范围固定 [0°,360°)，无需额外说明

QJL：把误差 0.0015926... 存成符号 "+ - + - ..." 
     → 只有 1 位，但 JL 变换保证期望正确
```

这种组合让 TurboQuant 在 **3-bit** 下达到 **32-bit FP** 的精度，同时：
- 计算更快（低 bit 运算）
- 内存更少（6x 压缩）
- 无需训练（数据无关）

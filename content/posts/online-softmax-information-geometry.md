---
title: "Online Softmax 的数学：从代数推导到信息几何"
date: 2025-01-20T10:00:00+08:00
draft: false
categories: ["研究笔记"]
tags: ["Flash Attention", "Online Softmax", "Information Geometry", "Optimization", "Math"]
---

> **Lecture Note 级别**。目标：讲清楚 Online Softmax 为什么是对的，以及它背后更深层的几何结构。
>
> 前置知识：线性代数、多元微积分、基础概率论。

---

## 0. 动机：Flash Attention 解决什么问题

Self-Attention 的计算公式：

$$
\mathbf{S} = \mathbf{Q}\mathbf{K}^\top, \quad \mathbf{A} = \text{softmax}(\mathbf{S}), \quad \mathbf{O} = \mathbf{A}\mathbf{V}
$$

**内存瓶颈**：标准实现需要存储 $\mathbf{S}$ 和 $\mathbf{A}$ 两个 $N \times N$ 矩阵。当序列长度 $N = 65536$ 时，单个矩阵的显存占用：

$$
65536^2 \times 4\text{ bytes} \approx 16\text{ GB}
$$

Flash Attention 的核心洞察：**不需要存储完整的注意力矩阵**。通过两个技术实现：

1. **Tiling（分块）**：将数据切分成适合 SRAM 的小块
2. **Recomputation（重计算）**：反向传播时不存储前向的中间结果

但 Tiling 面临一个根本障碍——Softmax 需要全局归一化。这就是 **Online Softmax** 登场的地方。

---

## 1. 传统 Softmax：为什么不能分块

### 1.1 定义与数值稳定性

对于向量 $\mathbf{x} \in \mathbb{R}^N$，Softmax 定义为：

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
$$

**数值稳定性问题**：当 $x_i$ 很大时，$e^{x_i}$ 会溢出。标准解法引入全局最大值 $m = \max_j x_j$：

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_{j=1}^{N} e^{x_j - m}}
$$

这个形式需要**两个全局统计量**：
- $m = \max_{j} x_j$（全局最大值）
- $d = \sum_{j=1}^{N} e^{x_j - m}$（全局指数和）

**关键障碍**：计算 $m$ 和 $d$ 都需要看到**所有** $N$ 个元素。如果数据被切成多个块，每个块只知道自己的局部信息，无法直接得到全局 Softmax。

---

## 2. Online Softmax：分块合并的代数推导

### 2.1 问题设置

假设输入被分成两个块 $\mathbf{x}^{(1)}$ 和 $\mathbf{x}^{(2)}$，各自计算了局部统计量：

| 统计量 | 块 1 | 块 2 |
|--------|------|------|
| 局部最大值 | $m_1 = \max(\mathbf{x}^{(1)})$ | $m_2 = \max(\mathbf{x}^{(2)})$ |
| 局部指数和 | $d_1 = \sum_{x \in \mathbf{x}^{(1)}} e^{x - m_1}$ | $d_2 = \sum_{x \in \mathbf{x}^{(2)}} e^{x - m_2}$ |

**目标**：仅通过 $(m_1, d_1)$ 和 $(m_2, d_2)$ 计算全局 Softmax，**不重新访问原始数据**。

### 2.2 全局最大值的合并

这是简单的：

$$
m = \max(m_1, m_2)
$$

### 2.3 全局指数和的合并（核心推导）

我们需要计算：

$$
d = \sum_{x \in \mathbf{x}^{(1)}} e^{x - m} + \sum_{x \in \mathbf{x}^{(2)}} e^{x - m}
$$

**问题**：我们只有 $d_1 = \sum_{x \in \mathbf{x}^{(1)}} e^{x - m_1}$，不是 $\sum_{x \in \mathbf{x}^{(1)}} e^{x - m}$。

**关键变形**——将指数拆成两部分：

$$
\begin{aligned}
\sum_{x \in \mathbf{x}^{(1)}} e^{x - m} &= \sum_{x \in \mathbf{x}^{(1)}} e^{x - m_1 + m_1 - m} \\
&= \sum_{x \in \mathbf{x}^{(1)}} e^{x - m_1} \cdot e^{m_1 - m} \\
&= e^{m_1 - m} \cdot \underbrace{\sum_{x \in \mathbf{x}^{(1)}} e^{x - m_1}}_{d_1}
\end{aligned}
$$

同理：

$$
\sum_{x \in \mathbf{x}^{(2)}} e^{x - m} = e^{m_2 - m} \cdot d_2
$$

**合并公式**：

$$
\boxed{d = d_1 \cdot e^{m_1 - m} + d_2 \cdot e^{m_2 - m}, \quad \text{其中 } m = \max(m_1, m_2)}
$$

### 2.4 一般形式：K 个块的合并

对于 $K$ 个块，递推公式：

$$
\begin{aligned}
m^{(k)} &= \max(m^{(k-1)}, m_k) \\
d^{(k)} &= d^{(k-1)} \cdot e^{m^{(k-1)} - m^{(k)}} + d_k \cdot e^{m_k - m^{(k)}}
\end{aligned}
$$

初始条件：$m^{(0)} = -\infty$, $d^{(0)} = 0$。

### 2.5 Python 验证

```python
import numpy as np

def online_softmax(blocks):
    """
    blocks: list of numpy arrays
    Returns: (m_global, d_global) for softmax normalization
    """
    m = -np.inf
    d = 0.0
    
    for block in blocks:
        m_block = np.max(block)
        d_block = np.sum(np.exp(block - m_block))
        
        m_new = max(m, m_block)
        
        # Rescaling: 把旧的统计量转换到新的参考系
        d = d * np.exp(m - m_new) + d_block * np.exp(m_block - m_new)
        m = m_new
    
    return m, d

# 验证：与传统 softmax 结果一致
np.random.seed(42)
x = np.random.randn(100)

# 方法1：传统 softmax
m_true = np.max(x)
d_true = np.sum(np.exp(x - m_true))
probs_true = np.exp(x - m_true) / d_true

# 方法2：Online softmax（分成5块）
blocks = np.array_split(x, 5)
m_online, d_online = online_softmax(blocks)
probs_online = np.exp(x - m_online) / d_online

print(f"全局最大值一致: {np.isclose(m_true, m_online)}")
print(f"全局指数和一致: {np.isclose(d_true, d_online)}")
print(f"概率分布一致: {np.allclose(probs_true, probs_online)}")
print(f"最大概率差: {np.max(np.abs(probs_true - probs_online)):.2e}")
```

输出：

```
全局最大值一致: True
全局指数和一致: True
概率分布一致: True
最大概率差: 0.00e+00
```

**验证通过**：Online Softmax 与传统 Softmax 数学完全等价，没有任何近似。

---

## 3. 深入：rescaling 的几何意义

### 3.1 两个参考系的对话

想象两个块生活在不同的"温度"下：

- 块 1 的最大值 $m_1 = 100$，它的 $d_1$ 是相对于 100 计算的
- 块 2 的最大值 $m_2 = 80$，它的 $d_2$ 是相对于 80 计算的

全局最大值 $m = 100$。合并时：
- 块 1 **不需要调整**：已经在 $m=100$ 的标准下
- 块 2 需要**降温**：乘以 $e^{80 - 100} = e^{-20} \approx 2.06 \times 10^{-9}$

这个乘法因子 $e^{m_k - m}$ 就是 **rescaling 因子**。它把不同参考系下的统计量，转换到统一的"全局坐标系"下。

### 3.2 为什么指数形式是必然的

Softmax 来自**指数族分布**。指数族的核心性质是：概率比值取对数后是线性的。

$$
\log \frac{p_i}{p_j} = x_i - x_j
$$

这意味着概率空间中的"距离"应该在对数尺度上度量。rescaling 因子 $e^{\Delta m}$ 正是对数空间中的**平移变换**，对应概率空间中的**尺度变换**。

---

## 4. 信息几何视角

现在进入更深层的结构。Online Softmax 不是孤立的代数技巧，它深深嵌入在**信息几何**（Information Geometry）的框架中。

### 4.1 概率单纯形

$N$ 维概率单纯形是所有合法概率分布构成的空间：

$$
\Delta_{N-1} = \left\{ \mathbf{p} \in \mathbb{R}^N \,\middle|\, \sum_{i=1}^{N} p_i = 1, \, p_i > 0 \right\}
$$

这是一个 $(N-1)$ 维**黎曼流形**。

### 4.2 Fisher 信息度量

信息几何为这个流形配备了一个自然的度量——**Fisher 信息度量**：

$$
g_{ij}(\mathbf{p}) = \frac{\delta_{ij}}{p_i}
$$

在概率单纯形上，两点 $\mathbf{p}$ 和 $\mathbf{q}$ 之间的无穷小距离：

$$
ds^2 = \sum_{i=1}^{N} \frac{(dp_i)^2}{p_i}
$$

这个度量被称为 **Shahshahani 度量** 或 **Fisher-Rao 度量**。

**直观**：概率越大的维度，"允许的变化"越小（因为 $1/p_i$ 越小）。不确定的地方才能大幅变动，确定的地方要小心翼翼。

### 4.3 Softmax 是指数映射

Softmax 将 $\mathbb{R}^N$ 中的"分数"映射到概率单纯形：

$$
\text{softmax}: \mathbb{R}^N \to \Delta_{N-1}, \quad p_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

这是**指数族分布**的标准形式。在信息几何中：
- $\mathbf{x}$ 是**自然参数**（Natural Parameters）
- $\mathbf{p}$ 是**期望参数**（Expectation Parameters）

指数映射有一个关键性质：**它将平坦的欧几里得空间映射到弯曲的概率单纯形上**。

### 4.4 局部欧几里得化（Local Euclideanization）

**核心观察**：在任意参考点 $\mathbf{x}^{(0)}$ 附近，我们可以定义**局部坐标**：

$$
\xi_i = x_i - m, \quad \text{其中 } m = \max_j x_j
$$

在这个坐标系下：

$$
p_i = \frac{e^{\xi_i}}{\sum_j e^{\xi_j}} = \frac{e^{\xi_i}}{d}
$$

其中 $d = \sum_j e^{\xi_j}$ 是配分函数。

**局部欧几里得化定理**：在参考点 $\mathbf{x}^{(0)}$ 附近，Fisher 度量可以用**加权欧几里得内积**近似：

$$
\langle \mathbf{u}, \mathbf{v} \rangle_{\mathbf{p}^{(0)}} = \sum_{i=1}^{N} p_i^{(0)} u_i v_i
$$

这意味着：**对数坐标空间（自然参数空间）在局部是欧几里得的**，而概率单纯形是弯曲的。Softmax 就是这个局部欧几里得坐标到弯曲流形的映射。

### 4.5 rescaling = 平行移动

当我们从旧的参考系 $(m_{\text{old}}, d_{\text{old}})$ 转换到新的参考系 $(m_{\text{new}}, d_{\text{new}})$ 时，rescaling 操作：

$$
d_{\text{new}} = d_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + d_{\text{block}} \cdot e^{m_{\text{block}} - m_{\text{new}}}
$$

在信息几何中，这是**沿测地线的平行移动**（Parallel Transport）：

| Online Softmax | 信息几何 |
|---------------|---------|
| 更新全局最大值 $m$ | 移动指数坐标原点 |
| 乘以 $e^{m_{\text{old}} - m_{\text{new}}}$ | 平行移动（保持内积结构） |
| 累加新的 $d_{\text{block}}$ | 在切空间中向量相加 |
| 最终输出 Softmax | 指数映射回概率单纯形 |

**定理**：Online Softmax 的 rescaling 等价于在概率单纯形上沿测地线进行平行移动，保持 Fisher 度量下的几何结构不变。

### 4.6 为什么这对 Flash Attention 至关重要

Flash Attention 的输出计算：

$$
\mathbf{O} = \text{softmax}(\mathbf{Q}\mathbf{K}^\top) \mathbf{V} = \mathbf{A}\mathbf{V}
$$

对于第 $i$ 个 Query，输出是：

$$
\mathbf{o}_i = \sum_{j=1}^{N} A_{ij} \mathbf{v}_j = \sum_{j=1}^{N} \frac{e^{S_{ij} - m_i}}{d_i} \mathbf{v}_j
$$

**关键洞察**：输出 $\mathbf{o}_i$ 也可以**增量更新**！

维护三个运行量：
- $m$：当前看到的最大分数
- $d$：当前指数和
- $\mathbf{o}$：当前累加的输出（已经在当前 $m$ 下归一化）

每来一个新的 KV 块 $(\mathbf{K}^{(b)}, \mathbf{V}^{(b)})$：

```
计算分数: S_i,b = Q_i @ K_b^T

m_new = max(m, max(S_i,b))

# 对旧的输出 rescale
o = o * (d * exp(m - m_new)) / (d * exp(m - m_new) + sum(exp(S_i,b - m_new)))
  + sum(exp(S_i,b - m_new)[:, None] * V_b) / d_new

d = d * exp(m - m_new) + sum(exp(S_i,b - m_new))
m = m_new
```

**几何意义**：输出 $\mathbf{o}$ 也在概率单纯形的切空间中累加，与 Softmax 的归一化同步 rescaling。所有计算在**局部欧几里得坐标**下完成，最后映射回概率空间。

---

## 5. Flash Attention 的完整图景

把代数、几何和工程实现统一起来：

| 层级 | 核心技术 | 说明 |
|------|---------|------|
| **工程层** (Engineering) | Tiling | 将大矩阵切成 SRAM 能装下的小块 |
| | Recomputation | 反向传播时重新计算，不存储中间矩阵 |
| | Kernel Fusion | 所有操作在一个 CUDA kernel 内完成 |
| **代数层** (Algebra) | Online Softmax | 分块合并公式 |
| | Rescaling | $d_{\text{new}} = d_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \ldots$ |
| | 输出增量更新 | $\mathbf{O}$ 与 Softmax 同步 rescale |
| **几何层** (Geometry) | 概率单纯形 $\Delta_{N-1}$ | Softmax 的像空间 |
| | Fisher 度量 | $ds^2 = \sum (dp_i)^2 / p_i$ |
| | 局部欧几里得化 | 对数坐标 $\xi_i = x_i - m$ |
| | 平行移动 | rescaling 保持几何结构不变 |

**关键收益**：

| 指标 | 传统 Attention | Flash Attention |
|------|---------------|-----------------|
| HBM 读写 | $O(N^2)$ | $O(N)$ |
| 内存占用 | $O(N^2)$ | $O(N)$ |
| 计算复杂度 | $O(N^2)$ | $O(N^2)$（不变）|
| 精度 | 精确 | **精确（无近似）** |

---

## 6. 总结：一条公式串起所有

Online Softmax 的核心，可以用一条递推公式概括：

$$
\boxed{d^{(k)} = d^{(k-1)} \cdot e^{m^{(k-1)} - m^{(k)}} + d_k \cdot e^{m_k - m^{(k)}}}
$$

这条公式的多重身份：

1. **代数身份**：分块统计量的正确合并法则
2. **分析身份**：指数族分布的配分函数更新
3. **几何身份**：概率单纯形上沿测地线的平行移动
4. **工程身份**：Flash Attention 内存高效的核心机制

Flash Attention 的伟大之处，不仅在于它是一个工程杰作，更在于它**尊重了数学的内在结构**——它没有近似 Softmax，而是利用了 Softmax 所在几何空间的结构特性，找到了一条精确且高效的计算路径。

> **好的算法不是打败数学，而是与数学共舞。**

---

## 参考文献

1. Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS* 2022.
2. Milakov, M., & Gimelshein, N. "Online normalizer calculation for softmax." arXiv:1805.02867.
3. Amari, S. "Information Geometry and Its Applications." Springer, 2016.
4. Martens, J. "New Insights and Perspectives on the Natural Gradient Method." *JMLR* 2020.

---

> 如果你对 **Online Softmax 在反向传播中的扩展**（Flash Attention 的 backward pass 如何用同样的思想避免存储注意力矩阵）感兴趣，留言告诉我。

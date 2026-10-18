---
title: "Online Softmax 的数学：从代数推导到信息几何"
description: "Online Softmax推导：FlashAttention分块计算的核心数学——不存储完整N×N注意力矩阵，只靠局部最大值m和指数和d两个统计量递推合并出全局Softmax，结果与传统 Softmax 精确一致，差异只来自浮点舍入。附Python验证，并从信息几何视角解释rescaling本质是概率单纯形上沿测地线的平行移动。"
date: 2025-01-20T10:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/online-softmax-information-geometry/"]
categories: ["研究笔记"]
tags: ["Flash Attention", "Online Softmax", "Information Geometry", "Optimization", "Math"]
---

序列长度 65536、FP32 的注意力分数矩阵，单个就是 $65536^2 \times 4\text{ bytes} \approx 16\text{ GB}$。标准 Self-Attention 还要再存一份 softmax 后的 $\mathbf{A}$；两个 $N \times N$ 矩阵同时驻留 HBM，而一块 GPU 的 SRAM 只有几十 MB。FlashAttention 把 Q/K/V 切块，在 SRAM 里算完一块就丢掉中间结果，只把输出 $\mathbf{O}$ 和两个统计量写回 HBM。障碍在 softmax：看到第一块时，你不知道后面会不会出现更大的分数。Online Softmax 要解决的就是这个全局归一化问题。

把 Q、K、V 代进去，Self-Attention 可以写成一行矩阵公式。下面的形状按单头、无 batch 省略写：

$$
\mathbf{S} = \mathbf{Q}\mathbf{K}^\top, \quad \mathbf{A} = \text{softmax}(\mathbf{S}), \quad \mathbf{O} = \mathbf{A}\mathbf{V}
$$

序列长度拉到 65536 时，先算一笔显存账。单个 $N \times N$ 矩阵按 FP32 存，占用如下：

$$
65536^2 \times 4\text{ bytes} \approx 16\text{ GB}
$$

Flash Attention 的核心洞察是**不需要存储完整的注意力矩阵**。它靠两个工程手段实现：Tiling（分块）把数据切成适合 SRAM 的小块；Recomputation（重计算）在反向传播时不存储前向的中间结果。Tiling 面临一个根本障碍——Softmax 需要全局归一化。这就是 Online Softmax 登场的地方。

## 1. 标准 Softmax 的全局统计量

对于向量 $\mathbf{x} \in \mathbb{R}^N$，Softmax 有一个标准定义。先把未归一化的指数写出来：

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
$$

当 $x_i$ 很大时，$e^{x_i}$ 会溢出。标准解法引入全局最大值 $m = \max_j x_j$：

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_{j=1}^{N} e^{x_j - m}}
$$

这个形式需要两个全局统计量：$m = \max_{j} x_j$ 和 $d = \sum_{j=1}^{N} e^{x_j - m}$。计算 $m$ 和 $d$ 都需要看到所有 $N$ 个元素。如果数据被切成多个块，每个块只知道自己的局部信息，无法直接得到全局 Softmax。分块要成立，必须找到一种只用局部统计量就能合并出全局 $m$ 和 $d$ 的方法。

## 2. 两块合并的推导

假设输入被分成两个块 $\mathbf{x}^{(1)}$ 和 $\mathbf{x}^{(2)}$。块 1 的局部最大值是 $m_1 = \max(\mathbf{x}^{(1)})$，局部指数和是 $d_1 = \sum_{x \in \mathbf{x}^{(1)}} e^{x - m_1}$；块 2 同理，得到 $m_2$ 和 $d_2$。目标是仅通过 $(m_1, d_1)$ 和 $(m_2, d_2)$ 计算全局 Softmax，不重新访问原始数据。

全局最大值的合并没有分支。直接取两个局部最大值的较大者：

$$
m = \max(m_1, m_2)
$$

全局指数和才是核心。我们需要计算：

$$
d = \sum_{x \in \mathbf{x}^{(1)}} e^{x - m} + \sum_{x \in \mathbf{x}^{(2)}} e^{x - m}
$$

问题在于，我们只有 $d_1 = \sum_{x \in \mathbf{x}^{(1)}} e^{x - m_1}$，不是 $\sum_{x \in \mathbf{x}^{(1)}} e^{x - m}$。关键变形是把指数拆成两部分：

$$
\begin{aligned}
\sum_{x \in \mathbf{x}^{(1)}} e^{x - m} &= \sum_{x \in \mathbf{x}^{(1)}} e^{x - m_1 + m_1 - m} \\
&= \sum_{x \in \mathbf{x}^{(1)}} e^{x - m_1} \cdot e^{m_1 - m} \\
&= e^{m_1 - m} \cdot \underbrace{\sum_{x \in \mathbf{x}^{(1)}} e^{x - m_1}}_{d_1}
\end{aligned}
$$

块 2 的推导完全对称。把下标换掉，结果如下：

$$
\sum_{x \in \mathbf{x}^{(2)}} e^{x - m} = e^{m_2 - m} \cdot d_2
$$

把两块的结果相加，就得到最终的合并公式：

$$
\boxed{d = d_1 \cdot e^{m_1 - m} + d_2 \cdot e^{m_2 - m}, \quad \text{其中 } m = \max(m_1, m_2)}
$$

两块的情况可以推广到 $K$ 个块。每一步只保留当前的 $m$ 和 $d$，递推式如下：

$$
\begin{aligned}
m^{(k)} &= \max(m^{(k-1)}, m_k) \\
d^{(k)} &= d^{(k-1)} \cdot e^{m^{(k-1)} - m^{(k)}} + d_k \cdot e^{m_k - m^{(k)}}
\end{aligned}
$$

初始条件 $m^{(0)} = -\infty$、$d^{(0)} = 0$。第一块进来时，旧项的因子 $e^{-\infty - m^{(1)}}$ 是 0，所以不需要为“第一块”写单独的分支；这个初始化把边界情况吸收进了递推式。

### Python 验证

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

输出里的 `0.00e+00` 只说明这 100 个样本在 float64 下差到显示精度以下。`np.isclose` 验证的是数值等价，不是 bitwise 相等；换一组随机数、换成 bf16，或者换一种分块方式，最后几位仍然会不同。Online Softmax 的“精确”指代数上没有近似 softmax，不指所有实现的结果逐位一致。

## 3. 数值边界：什么时候小块会被压没

rescaling 因子是 $e^{m_k - m}$。假设块 1 的最大值 $m_1 = 100$，块 2 的最大值 $m_2 = 80$，全局最大值 $m = 100$。块 2 的统计量在合并时要乘以 $e^{80 - 100} = e^{-20} \approx 2.06 \times 10^{-9}$。这个数看起来很小，但 float32 的最小正规数在 $e^{-87}$ 附近，$e^{-20}$ 离下溢还很远；块间最大值差到 100 以上，因子才会进入次正规数，低到 $e^{-103}$ 左右才变成 0。

下溢本身不破坏结果。一个块的局部最大值比全局最大值低 100 以上，意味着它的所有项相对全局最大项都小于 $e^{-100}$，对最终 softmax 的贡献已经在有效精度之外。把它压成 0，等价于丢掉一个本来就看不见的尾部。真正需要注意的是反向传播：如果某个块的贡献在前向被下溢成 0，反向的梯度也会是 0；这不会让结果错，但会让这块数据在训练中失去信号。长序列训练里如果频繁出现这种块，应该检查分数分布，而不是提高精度硬扛。

另一个边界是 $d$ 的累加顺序。$d$ 是所有项的和，每个项都不超过 1（因为减去了全局最大值），所以 $d \le N$；它不会因为项太多而溢出，但浮点加法不满足结合律，分块顺序不同，最后几位就不同。这就是为什么后面的“精确”要限定在代数层面。

## 4. 几何：换原点、Fisher 度量与“平行移动”

要看清 rescaling 的几何含义，先把 softmax 写成指数族。这里的参数命名沿用 Amari 的记号：

$$
p_i = e^{x_i - A(x)}, \quad A(x) = \log \sum_j e^{x_j}
$$

这里 $\mathbf{x}$ 是自然参数（Natural Parameters），$\mathbf{p}$ 是期望参数（Expectation Parameters），$A(x)$ 是对数配分函数。$p = \nabla A(x)$ 就是 softmax。这里的“指数”来自 $e^{x_i}$；黎曼几何里的指数映射是另一个概念，两者只是重名。

指数族写成上面的形式后，Fisher 度量有一个现成的表达式。它是 $A$ 的 Hessian：

$$
g_{ij}(x) = \frac{\partial^2 A}{\partial x_i \partial x_j} = p_i \delta_{ij} - p_i p_j
$$

换到概率单纯形的期望参数 $\mathbf{p}$ 后，切向量要满足 $\sum_i u_i = 0$。在这个子空间上，度量为：

$$
ds^2 = \sum_i \frac{(dp_i)^2}{p_i}
$$

第一种写法里，全 1 方向的长度为 0：把 $\mathbf{x}$ 整体加一个常数，$p$ 不变，所以这个方向是度量的零方向。去掉这个零方向后，Fisher 度量在单纯形上正定。这也解释了为什么 softmax 可以任意选择 $m$：$m$ 只是自然参数坐标的一个原点，换原点不改变分布。

在参考点 $\mathbf{p}^{(0)}$ 附近，$g_{ij}$ 可以近似为常数矩阵 $p_i^{(0)} \delta_{ij} - p_i^{(0)} p_j^{(0)}$。对切向量 $\mathbf{u}$、$\mathbf{v}$，局部内积写成协方差：

$$
\langle \mathbf{u}, \mathbf{v} \rangle_{\mathbf{p}^{(0)}} = \sum_i p_i^{(0)} (u_i - \bar u)(v_i - \bar v), \quad \bar u = \sum_i p_i^{(0)} u_i, \quad \bar v = \sum_i p_i^{(0)} v_i
$$

一个容易写错的近似是 $\sum_i p_i^{(0)} u_i v_i$。它少了中心化项，也丢掉了 softmax 的平移不变性；把它当成 Fisher 度量，后面所有“平行移动”的推导都会跟着偏。局部欧几里得化的准确版本是上面的协方差形式：在参考点附近，度量近似为常数，但作用在去掉了均值方向的切空间上。

指数族还有一层对偶结构。$A(x)$ 的对偶势是 $\sum_i p_i \log p_i = -H(\mathbf{p})$，$p = \nabla A(x)$ 给出自然参数到期望参数的坐标变换。Online Softmax 的 rescaling 可以放在这套对偶坐标里读：$m$ 是自然参数的原点，换 $m$ 相当于沿全 1 方向平移；这个方向在 Fisher 度量下长度为 0，所以旧块的相对概率不变，变的是未归一化权重的整体尺度 $d$。

在对偶平坦结构里，这次换原点可以读成沿全 1 方向的平行移动：自然参数空间是平的，$m$ 只是原点，平移方向落在 Fisher 度量的零方向上，所以旧块的相对概率不变。严格说，移动的是统计量的表示，分布本身没有被搬动；把它理解成规范变换（gauge transformation）比理解成在单纯形上运输概率向量更准确。$d$ 按 $e^{m_{\text{old}} - m_{\text{new}}}$ 缩放，这个图像能帮你记住 rescaling 因子的形状；真正改变分布的是加入新块。

指数因子的来源也可以从同一处看：未归一化权重是 $e^{x_i}$，给 $\mathbf{x}$ 加一个常数 $\Delta$，每个权重都乘 $e^\Delta$。这是指数族的结构，不是某个 kernel 实现的选择。Fisher 度量、对偶势和 rescaling 三者能对上，是因为它们描述的是同一组对偶坐标。

## 5. 输出累加：FlashAttention 真正省下的是什么

Softmax 的归一化解决了，输出 $\mathbf{O}$ 也可以增量更新。对于第 $i$ 个 Query，输出是：

$$
\mathbf{o}_i = \sum_{j=1}^{N} A_{ij} \mathbf{v}_j = \sum_{j=1}^{N} \frac{e^{S_{ij} - m_i}}{d_i} \mathbf{v}_j
$$

FlashAttention 为每个 Query 维护三个运行量：当前看到的最大分数 $m$、当前指数和 $d$、当前累加的输出 $\mathbf{o}$。每来一个新的 KV 块 $(\mathbf{K}^{(b)}, \mathbf{V}^{(b)})$，先算分数，再更新 $m$，然后对旧的输出做 rescale，最后把新块的贡献加进去：

```
计算分数: S_i,b = Q_i @ K_b^T

m_new = max(m, max(S_i,b))

# 对旧的输出 rescale
o = o * (d * exp(m - m_new)) / (d * exp(m - m_new) + sum(exp(S_i,b - m_new)))
  + sum(exp(S_i,b - m_new)[:, None] * V_b) / d_new

d = d * exp(m - m_new) + sum(exp(S_i,b - m_new))
m = m_new
```

把代码里的三行更新整理成一条公式，分子和分母的关系会更清楚。旧输出、旧 $d$ 和新块各自扮演什么角色，一眼能看出来：

$$
\mathbf{o}_{\text{new}} = \frac{\mathbf{o}_{\text{old}} \cdot d_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \sum_j e^{S_{ij} - m_{\text{new}}} \mathbf{v}_j}{d_{\text{new}}}
$$

分子是未归一化的加权和，分母是新的归一化常数。旧输出带着旧的 $d$ 和旧的 $m$，换到新原点后乘以 $e^{m_{\text{old}} - m_{\text{new}}}$；新块的项也按新原点计算；两者相加再除以 $d_{\text{new}}$，比值不变。这就是“先 rescale 再加”的全部机制。

FlashAttention 的收益在显存和 HBM 往返；$N^2$ 的计算量并没有变成 $N$。传统实现要物化 $\mathbf{S}$ 和 $\mathbf{A}$，显存是 $O(N^2)$；FlashAttention 不存这两个矩阵，显存从 $O(N^2)$ 降到 $O(Nd)$：输出 $\mathbf{O}$ 是 $N \times d$，统计量是 $N$，$d$ 固定时记作 $O(N)$。按 FlashAttention 论文的 IO 复杂度记号，HBM 访问量从标准实现的 $\Theta(N^2 d + N d^2)$ 降到 $\Theta(N^2 d^2 / M)$，$M$ 是 SRAM 容量；计算量仍然是 $O(N^2 d)$。反向传播还要重算 $\mathbf{S}$，所以训练时的 FLOPs 可能比前向更多。收益的边界在 IO：当 $N$ 小到 $\mathbf{S}$、$\mathbf{A}$ 能放进 SRAM，或者 HBM 带宽不是瓶颈时，FlashAttention 的优势会缩小。

“精确”也要限定范围。数学上，Online Softmax 和标准 Softmax 完全等价，没有近似；浮点上，分块的 rescale 顺序、bf16 的舍入、以及和 PyTorch 参考实现的 reduction order 不同，结果不会逐位相同。反向传播里的重计算会再次引入这些舍入差异。要验证一个实现，应该同时看数值误差和 HBM/SRAM 的占用，不要只看 attention 的 wall time。

相关材料里，Dao 等人的 FlashAttention（NeurIPS 2022）给出了 IO 复杂度和 kernel 设计；Milakov 与 Gimelshein 的 “Online normalizer calculation for softmax”（arXiv:1805.02867）是 online softmax 的早期来源。Amari 的 *Information Geometry and Its Applications*（Springer, 2016）和 Martens 的 “New Insights and Perspectives on the Natural Gradient Method”（JMLR 2020）则覆盖了指数族的 Fisher 度量与对偶结构。

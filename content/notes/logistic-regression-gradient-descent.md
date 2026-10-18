---
title: "逻辑回归权重更新：从交叉熵到梯度下降的完整推导"
description: "逻辑回归梯度下降推导：从Sigmoid函数和交叉熵损失出发，用链式法则完整推出权重更新公式，解释梯度为何能简化为（预测概率−真实标签）×输入特征；对比批量、随机、小批量三种更新方式，附NumPy从零实现和数值梯度验证，测试集准确率95.5%。"
date: 2025-01-15T10:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/logistic-regression-gradient-descent/"]
categories: ["机器学习基础"]
tags: ["Logistic Regression", "Gradient Descent", "Optimization", "Math"]
---

用 1000 个样本按 8:2 切分、5 个特征跑 2000 轮批量梯度下降，测试集准确率 95.50%，最终交叉熵 0.127384。这个结果不算惊艳，但它把逻辑回归的权重更新公式完整走了一遍：从 Sigmoid 和交叉熵出发，链式法则求导，最后落到 $\mathbf{w} \leftarrow \mathbf{w} - \eta \mathbf{X}^\top (\mathbf{p} - \mathbf{y})/N$。下面把每一步拆开，重点解释为什么梯度会简化成（预测概率−真实标签）×输入特征。

## 1. 模型定义

逻辑回归名字里有回归，输出却是二分类概率。给定输入 $\mathbf{x} \in \mathbb{R}^d$ 和权重 $\mathbf{w} \in \mathbb{R}^d$（包含偏置 $b$），模型输出

$$
p(\hat{y}=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^\top \mathbf{x}}}
$$

其中 $\sigma(z)$ 是 Sigmoid 函数，它把实数压到 0 到 1 之间，输出可以直接当概率用：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

它的导数有一个很实用的性质：

$$
\frac{d\sigma(z)}{dz} = \sigma(z) \cdot (1 - \sigma(z))
$$

这个性质会在后面的梯度推导中把交叉熵的分母消掉。

## 2. 损失函数：为什么用交叉熵

对于单个样本 $(\mathbf{x}^{(i)}, y^{(i)})$，其中 $y^{(i)} \in \{0, 1\}$，希望模型输出的概率 $p^{(i)} = \sigma(\mathbf{w}^\top \mathbf{x}^{(i)})$ 尽可能接近真实标签 $y^{(i)}$。交叉熵损失定义为

$$
\mathcal{L}^{(i)}(\mathbf{w}) = -\left[ y^{(i)} \log p^{(i)} + (1 - y^{(i)}) \log(1 - p^{(i)}) \right]
$$

训练时最小化所有样本的平均损失，而不是单个样本的损失：

$$
\mathcal{L}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y^{(i)} \log p^{(i)} + (1 - y^{(i)}) \log(1 - p^{(i)}) \right]
$$

用 MSE 配合 Sigmoid 时，梯度在两端饱和区会变得极小：$p$ 接近 0 或 1 时，$\sigma'(z)=p(1-p)$ 接近 0，误差再大也传不回去。交叉熵对 $p$ 的导数带 $1/[p(1-p)]$，正好和 Sigmoid 的导数抵消，这是它成为标准选择的机制。

## 3. 核心推导：梯度从哪里来

目标是最小化 $\mathcal{L}(\mathbf{w})$，梯度下降需要知道损失函数对每个权重 $w_j$ 的偏导数。令 $z^{(i)} = \mathbf{w}^\top \mathbf{x}^{(i)}$，则 $p^{(i)} = \sigma(z^{(i)})$。链式法则展开：

$$
\frac{\partial \mathcal{L}^{(i)}}{\partial w_j} = \frac{\partial \mathcal{L}^{(i)}}{\partial p^{(i)}} \cdot \frac{\partial p^{(i)}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial w_j}
$$

第一项对 $p^{(i)}$ 求导：

$$
\frac{\partial \mathcal{L}^{(i)}}{\partial p^{(i)}} = -\left[ \frac{y^{(i)}}{p^{(i)}} - \frac{1 - y^{(i)}}{1 - p^{(i)}} \right] = \frac{p^{(i)} - y^{(i)}}{p^{(i)}(1 - p^{(i)})}
$$

第二项用 Sigmoid 的导数性质：

$$
\frac{\partial p^{(i)}}{\partial z^{(i)}} = \sigma(z^{(i)})(1 - \sigma(z^{(i)})) = p^{(i)}(1 - p^{(i)})
$$

第三项对权重求导：

$$
\frac{\partial z^{(i)}}{\partial w_j} = x_j^{(i)}
$$

三项相乘，$p^{(i)}(1-p^{(i)})$ 消掉：

$$
\frac{\partial \mathcal{L}^{(i)}}{\partial w_j} = \frac{p^{(i)} - y^{(i)}}{\cancel{p^{(i)}(1 - p^{(i)})}} \cdot \cancel{p^{(i)}(1 - p^{(i)})} \cdot x_j^{(i)} = (p^{(i)} - y^{(i)}) \cdot x_j^{(i)}
$$

这个抵消来自交叉熵里 $1/[p(1-p)]$ 的因子和 Sigmoid 导数的乘积，两项本来就是配套设计的。对所有 $N$ 个样本取平均，得到

$$
\frac{\partial \mathcal{L}}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (p^{(i)} - y^{(i)}) \cdot x_j^{(i)}
$$

把所有样本的梯度堆成一个向量，就能用一次矩阵乘法算完：

$$
\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{N} \mathbf{X}^\top (\mathbf{p} - \mathbf{y})
$$

其中 $\mathbf{X} \in \mathbb{R}^{N \times d}$ 是设计矩阵，$\mathbf{p} \in \mathbb{R}^{N}$ 是所有样本的预测概率，$\mathbf{y} \in \mathbb{R}^{N}$ 是真实标签。

## 4. 权重更新公式

有了梯度，权重就沿着负梯度方向走一步，步长由学习率控制：

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \nabla_{\mathbf{w}} \mathcal{L} = \mathbf{w} - \frac{\eta}{N} \mathbf{X}^\top (\mathbf{p} - \mathbf{y})
$$

这是批量梯度下降，每次用全部 $N$ 个样本。随机梯度下降每次只用一个样本 $(\mathbf{x}^{(i)}, y^{(i)})$：

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot (p^{(i)} - y^{(i)}) \cdot \mathbf{x}^{(i)}
$$

小批量梯度下降每次用 $B$ 个样本（$B \ll N$）：

$$
\mathbf{w} \leftarrow \mathbf{w} - \frac{\eta}{B} \sum_{i \in \mathcal{B}} (p^{(i)} - y^{(i)}) \cdot \mathbf{x}^{(i)}
$$

三者的区别只在每次用多少样本估计梯度。直观上，如果模型预测 $p^{(i)}$ 比真实标签 $y^{(i)}$ 大（预测过头了），梯度为正，权重往减小方向走；如果预测小了，梯度为负，权重往增大方向走。更新的幅度同时受学习率 $\eta$ 和输入特征 $x_j^{(i)}$ 的尺度影响。

## 5. 代码实现

下面是一个从零实现的 NumPy 版本，没有调用 sklearn，每一步都对应上面的公式。

```python
import numpy as np

class LogisticRegressionGD:
    def __init__(self, lr=0.1, n_iter=1000, fit_intercept=True):
        self.lr = lr              # 学习率 η
        self.n_iter = n_iter      # 迭代轮数
        self.fit_intercept = fit_intercept
        self.w = None             # 权重向量
        self.loss_history = []

    def _sigmoid(self, z):
        """Sigmoid: σ(z) = 1 / (1 + exp(-z))"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        """
        X: (N, d)
        y: (N,)
        """
        if self.fit_intercept:
            # 添加一列 1 作为偏置项
            X = np.c_[np.ones(X.shape[0]), X]

        N, d = X.shape
        self.w = np.zeros(d)

        for epoch in range(self.n_iter):
            # 前向传播：z = X @ w, p = σ(z)
            z = X @ self.w
            p = self._sigmoid(z)

            # 计算交叉熵损失
            loss = -np.mean(
                y * np.log(p + 1e-15) +
                (1 - y) * np.log(1 - p + 1e-15)
            )
            self.loss_history.append(loss)

            # 计算梯度: ∇L = (1/N) * X^T @ (p - y)
            grad = (X.T @ (p - y)) / N

            # 权重更新: w = w - η * ∇L
            self.w -= self.lr * grad

        return self

    def predict_proba(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return self._sigmoid(X @ self.w)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# ========== 验证 ==========
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 构造二分类数据
X, y = make_classification(
    n_samples=1000, n_features=5, n_informative=3,
    n_redundant=0, n_classes=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练
model = LogisticRegressionGD(lr=0.5, n_iter=2000)
model.fit(X_train, y_train)

# 评估
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Final loss: {model.loss_history[-1]:.6f}")

# 查看损失下降曲线
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(model.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Cross-Entropy Loss')
plt.title('Loss Curve during Gradient Descent')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.show()
```

在 2000 轮、学习率 0.5 的设置下，输出如下：

```
Accuracy: 0.9550
Final loss: 0.127384
```

## 6. 与线性回归的对比

线性回归和逻辑回归的梯度形式几乎一样：都可以写成（预测−真实）经过设计矩阵转置的形式。差异在预测的语义：线性回归输出连续值 $\hat{y} = \mathbf{w}^\top \mathbf{x}$，用 MSE，梯度是 $\frac{1}{N}\mathbf{X}^\top (\hat{\mathbf{y}} - \mathbf{y})$；逻辑回归输出概率 $p = \sigma(\mathbf{w}^\top \mathbf{x})$，用交叉熵，梯度是 $\frac{1}{N}\mathbf{X}^\top (\mathbf{p} - \mathbf{y})$。两者的权重更新都是 $\mathbf{w} - \eta \nabla \mathcal{L}$，换的只是损失和输出层。

## 7. 调试经验：梯度检查

手写梯度时，用中心差分做数值检查：对每个 $w_j$，分别算 $w_j \pm \epsilon$ 处的 loss，除以 $2\epsilon$，再和解析梯度比。

```python
def numerical_gradient(X, y, w, eps=1e-5):
    """用有限差分近似梯度"""
    grad = np.zeros_like(w)
    for j in range(len(w)):
        w_plus = w.copy()
        w_plus[j] += eps
        w_minus = w.copy()
        w_minus[j] -= eps

        # 分别在 w+ε 和 w-ε 处计算损失
        p_plus = sigmoid(X @ w_plus)
        loss_plus = -np.mean(y * np.log(p_plus) + (1-y) * np.log(1-p_plus))

        p_minus = sigmoid(X @ w_minus)
        loss_minus = -np.mean(y * np.log(p_minus) + (1-y) * np.log(1-p_minus))

        grad[j] = (loss_plus - loss_minus) / (2 * eps)
    return grad

# 验证：解析梯度与数值梯度应该非常接近
analytical = (X.T @ (p - y)) / N
numerical = numerical_gradient(X, y, w)
print(f"Max diff: {np.max(np.abs(analytical - numerical)):.8f}")
# 期望输出: Max diff < 1e-6
```

这段检查代码是从训练脚本里摘出来的，依赖训练结束后的 `X`、`y`、`w`、`p` 和 `N`，不能单独运行。`sigmoid` 在前面的实现里是类方法，这里要补一个全局函数；`X` 在 `fit` 内部加过偏置列，外层变量没有，直接照抄会因维度不一致报错，补一个带偏置的 `X_aug` 才能跑。$\epsilon$ 取 1e-5 时，期望最大误差小于 1e-6。这个检查每轮多花 $2d$ 次前向；$d=5$ 时值得，$d$ 到百万级时只抽几个维度做。

## 8. 这套推导的适用边界

整套推导只依赖两个事实：Sigmoid 的导数是 $p(1-p)$，交叉熵对概率的导数是 $(p-y)/[p(1-p)]$。两者相乘后，权重更新的每一步就是误差乘特征。这个形式也解释了为什么特征尺度重要：$x_j$ 大一个数量级，同样的 $\eta$ 会让对应权重的实际步长大一个数量级，训练前先标准化。特征没标准化时，梯度检查仍会通过，但收敛会慢甚至震荡；公式没问题，问题在尺度和学习率的匹配。

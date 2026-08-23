---
title: "逻辑回归权重更新：从交叉熵到梯度下降的完整推导"
date: 2025-01-15T10:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/logistic-regression-gradient-descent/"]
categories: ["机器学习基础"]
tags: ["Logistic Regression", "Gradient Descent", "Optimization", "Math"]
---

> 一篇只讲清楚一件事的文章：逻辑回归的权重，在梯度下降中到底怎么动。

---

## 1. 模型定义

逻辑回归（Logistic Regression）虽然名字里有"回归"，但本质是一个**二分类模型**。

给定输入 $\mathbf{x} \in \mathbb{R}^d$ 和权重 $\mathbf{w} \in \mathbb{R}^d$（包含偏置 $b$），模型输出一个概率：

$$
p(\hat{y}=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^\top \mathbf{x}}}
$$

其中 $\sigma(z)$ 就是 **Sigmoid 函数**：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

它的导数有一个极漂亮的性质：

$$
\frac{d\sigma(z)}{dz} = \sigma(z) \cdot (1 - \sigma(z))
$$

这个性质会在后面的梯度推导中帮我们省很多力气。

---

## 2. 损失函数：为什么用交叉熵

对于单个样本 $(\mathbf{x}^{(i)}, y^{(i)})$，其中 $y^{(i)} \in \{0, 1\}$，我们希望模型输出的概率 $p^{(i)} = \sigma(\mathbf{w}^\top \mathbf{x}^{(i)})$ 尽可能接近真实标签 $y^{(i)}$。

**交叉熵损失**（Cross-Entropy Loss）定义为：

$$
\mathcal{L}^{(i)}(\mathbf{w}) = -\left[ y^{(i)} \log p^{(i)} + (1 - y^{(i)}) \log(1 - p^{(i)}) \right]
$$

对于 $N$ 个样本的平均损失：

$$
\mathcal{L}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y^{(i)} \log p^{(i)} + (1 - y^{(i)}) \log(1 - p^{(i)}) \right]
$$

**为什么不用 MSE？**

如果用均方误差 $\mathcal{L} = (y - p)^2$，配合 Sigmoid 会导致梯度在两端饱和区变得极小，模型几乎学不动。交叉熵损失配合 Sigmoid 的梯度恰好能**抵消饱和效应**，这是它成为标准选择的根本原因。

---

## 3. 核心推导：梯度从哪里来

我们的目标是最小化 $\mathcal{L}(\mathbf{w})$，梯度下降需要知道**损失函数对每个权重 $w_j$ 的偏导数**。

### Step 1：定义中间变量

令 $z^{(i)} = \mathbf{w}^\top \mathbf{x}^{(i)}$，则 $p^{(i)} = \sigma(z^{(i)})$。

### Step 2：链式法则展开

$$
\frac{\partial \mathcal{L}^{(i)}}{\partial w_j} = \frac{\partial \mathcal{L}^{(i)}}{\partial p^{(i)}} \cdot \frac{\partial p^{(i)}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial w_j}
$$

### Step 3：逐项求导

**第一项**：

$$
\frac{\partial \mathcal{L}^{(i)}}{\partial p^{(i)}} = -\left[ \frac{y^{(i)}}{p^{(i)}} - \frac{1 - y^{(i)}}{1 - p^{(i)}} \right] = \frac{p^{(i)} - y^{(i)}}{p^{(i)}(1 - p^{(i)})}
$$

**第二项**（利用 Sigmoid 导数的性质）：

$$
\frac{\partial p^{(i)}}{\partial z^{(i)}} = \sigma(z^{(i)})(1 - \sigma(z^{(i)})) = p^{(i)}(1 - p^{(i)})
$$

**第三项**：

$$
\frac{\partial z^{(i)}}{\partial w_j} = x_j^{(i)}
$$

### Step 4：合并不是巧合的巧合

把三项相乘：

$$
\frac{\partial \mathcal{L}^{(i)}}{\partial w_j} = \frac{p^{(i)} - y^{(i)}}{\cancel{p^{(i)}(1 - p^{(i)})}} \cdot \cancel{p^{(i)}(1 - p^{(i)})} \cdot x_j^{(i)} = (p^{(i)} - y^{(i)}) \cdot x_j^{(i)}
$$

**分子分母完美抵消**——这不是运气，是交叉熵 + Sigmoid 这对组合的数学设计。

### Step 5：批量梯度

对所有 $N$ 个样本取平均：

$$
\frac{\partial \mathcal{L}}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (p^{(i)} - y^{(i)}) \cdot x_j^{(i)}
$$

写成向量形式更优雅：

$$
\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{N} \mathbf{X}^\top (\mathbf{p} - \mathbf{y})
$$

其中：
- $\mathbf{X} \in \mathbb{R}^{N \times d}$ 是设计矩阵
- $\mathbf{p} \in \mathbb{R}^{N}$ 是所有样本的预测概率
- $\mathbf{y} \in \mathbb{R}^{N}$ 是真实标签

---

## 4. 权重更新公式

有了梯度，权重更新就水到渠成。

### 批量梯度下降（Batch GD）

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \nabla_{\mathbf{w}} \mathcal{L} = \mathbf{w} - \frac{\eta}{N} \mathbf{X}^\top (\mathbf{p} - \mathbf{y})
$$

### 随机梯度下降（SGD）

每次只用一个样本 $(\mathbf{x}^{(i)}, y^{(i)})$：

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot (p^{(i)} - y^{(i)}) \cdot \mathbf{x}^{(i)}
$$

### 小批量梯度下降（Mini-batch GD）

每次用 $B$ 个样本（$B \ll N$）：

$$
\mathbf{w} \leftarrow \mathbf{w} - \frac{\eta}{B} \sum_{i \in \mathcal{B}} (p^{(i)} - y^{(i)}) \cdot \mathbf{x}^{(i)}
$$

**直观理解**：

- 如果模型预测 $p^{(i)}$ 比真实标签 $y^{(i)}$ 大（预测过头了），梯度为正，权重往**减小**方向走
- 如果预测小了，梯度为负，权重往**增大**方向走
- 更新的幅度同时受学习率 $\eta$ 和输入特征 $x_j^{(i)}$ 的尺度影响

---

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

**运行结果**：

```
Accuracy: 0.9550
Final loss: 0.127384
```

---

## 6. 与线性回归的对比

| 特性 | 线性回归 | 逻辑回归 |
|------|---------|---------|
| 任务 | 回归 | 分类 |
| 输出 | $\hat{y} = \mathbf{w}^\top \mathbf{x}$ | $p = \sigma(\mathbf{w}^\top \mathbf{x})$ |
| 损失函数 | MSE | 交叉熵 |
| 梯度 | $\nabla \mathcal{L} = \frac{1}{N}\mathbf{X}^\top (\hat{\mathbf{y}} - \mathbf{y})$ | $\nabla \mathcal{L} = \frac{1}{N}\mathbf{X}^\top (\mathbf{p} - \mathbf{y})$ |
| 权重更新 | $\mathbf{w} - \eta \nabla \mathcal{L}$ | $\mathbf{w} - \eta \nabla \mathcal{L}$ |

**形式上的相似性不是偶然**：两种模型的梯度都可以统一写成 $(\text{预测} - \text{真实})$ 的形式，只是"预测"的语义不同——线性回归预测的是连续值，逻辑回归预测的是概率。

---

## 7. 调试经验：梯度检查

手写梯度时，最好用**数值梯度**做验证：

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

---

## 8. 总结

逻辑回归的权重更新公式，可以一句话概括：

> **误差反向传播到输入特征，按学习率调整权重。**

完整的数学链条：

```
损失 L(w) ──[求导]──> 梯度 ∇L = (1/N) X^T (p - y)
     │
     └─[梯度下降]──> w ← w - η·∇L
```

关键记忆点：
1. Sigmoid 导数 = $p(1-p)$，与交叉熵的 $1/p(1-p)$ **完美抵消**
2. 最终梯度形式异常简洁：$(p^{(i)} - y^{(i)}) \cdot x_j^{(i)}$
3. 更新方向 = **预测误差 × 输入特征**

---

> 如果你对 Softmax 多分类的梯度推导也感兴趣，留言告诉我，下一篇写它。

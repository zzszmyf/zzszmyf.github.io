---
title: "激活量化为什么比权重量化难？LLM.int8() 与 SmoothQuant 原理"
date: 2026-08-17T00:00:00+08:00
draft: false
description: "权重是静态的，激活是动态的还会出现极大的离群值，所以 W8A8 比 weight-only 难得多。本文讲 LLM.int8() 的向量级量化与混合精度分解、SmoothQuant 的等效变换与 s_j 公式推导，以及缩放折叠为什么让 W8A8 零运行时开销。"
weight: 17
tags: ["LLM推理优化", "量化"]
series: "quantization"
---

> 系列导航：[LLM 量化精读笔记总览](/notes/llm量化精读笔记-00-总览与学习地图/)（共 11 篇）｜上一篇：[AWQ 量化原理是什么](/notes/llm量化精读笔记-06-权重量化ii-awq-squeezellm-quip/)｜下一篇：[KV Cache 量化怎么做](/notes/llm量化精读笔记-08-kv-cache量化与kivi/)

> 对应：LLM.int8()（arXiv:2208.07339，NeurIPS 2022）；SmoothQuant（arXiv:2211.10438，ICML 2023）；MIT 6.5940 Lecture 5。
> 学完本章你应该能：① 说明激活量化比权重量化难在哪（动态、per-tensor、outlier）；② 讲清 LLM.int8() 的 vector-wise 量化与混合精度分解，并说出它依赖的 outlier 统计结论（04 章 6.1）；③ 推导 SmoothQuant 的等效变换和$s_{j}$公式，解释$\alpha$的语义；④ 说清"scale 折叠"为什么让 W8A8 零运行时开销。

---

## 目录（本章）

1. 本章目标
2. 为什么激活量化更难
3. LLM.int8()：outlier 分离的 INT8 推理
4. SmoothQuant：把量化难度迁移到权重
5. W8A8 的工程细节
6. 从 INT8 到 FP8：激活量化的现代形态
7. 本章小结
8. 习题与解答
9. 延伸阅读

---

## 1. 本章目标

权重压到 4-bit 之后，prefill 仍要跑 FP16 激活；只有把激活也压到 INT8，prefill 才能用上 INT8 Tensor Core 的 2 倍算力。障碍是激活里的系统性 outlier，权重量化没有这个问题。

两条路线的分工：

$LLM.int8() \to$承认 outlier 无法量化：把 outlier 列单独留在 FP16 算
$SmoothQuant \to$否认 outlier 必须存在：用等效变换把难度搬到权重上

---

## 2. 为什么激活量化更难

### 2.1 三个结构性差异（vs 权重）

| | 权重 W | 激活 X |
|---|---|---|
| 何时量化 | 离线，可慢慢校准 | 推理时动态量化（每个 batch 的分布会变） |
| scale 粒度 | 可 per-channel（04 章） | 主流 per-tensor（per-channel 激活 kernel 贵） |
| outlier | 少且可分离 | **系统性存在且幅度大** |

### 2.2 outlier 的具体样子（SmoothQuant 论文对 OPT-175B 的统计）

04 章 6.2 算过激活的"偷位"效应：per-tensor min/max 下，正常值只剩约 4 bit 有效精度，范围全被少数 outlier 占掉。**这就是 W8A8 迟迟不能落地的原因。**

---

## 3. LLM.int8()：outlier 分离的 INT8 推理

### 3.1 目标与设定

LLM.int8()（Dettmers et al., 2022）的目标：**INT8 跑 175B 模型，perplexity 与 FP16 完全一致**，同时内存减半。

三个关键设计：

### 3.2 设计一：vector-wise 量化

LLM.int8() 的粒度取在 per-tensor 和 per-element 之间，即 vector-wise 量化：

权重：每"行"（每个输出神经元）一个 scale
激活：每"列"（每个输入特征维度）一个 scale

行/列级的 scale 让量化范围贴近每行每列的真实分布，误差显著小于 per-tensor，又比 per-element 便宜。

### 3.3 设计二：发现 outlier 是"特征维度"现象

04 章 6.1 的统计对 LLM.int8() 有三个直接含义：

1. outlier 集中在约 0.1% 的特征维度上，可以按列整体抽出，不必逐元素判断。
2. 13B 以下模型通常不超过 7 个这样的维度，按列抽取的 FP16 路径规模可控。
3. outlier 对 softmax 的大概率输出至关重要，不能截断丢弃，只能保留高精度。

这些维度对所有 token 一致，这是"把 outlier 列抽出来单独算"的依据。

### 3.4 设计三：混合精度分解（Mixed-Precision Decomposition）

对每一层的矩阵乘：

$$
Y = WX
$$

= $W [X_{\text{low}} ; X_{\text{outlier}}]$（按列拆开激活）
= $[W_{\text{low}} \cdot X_{\text{low}}]$（INT8 矩阵乘，大部分计算）
$+ [W_{\text{outlier}} \cdot X_{\text{outlier}}]$（FP16 矩阵乘，outlier 列）

流程：

1. 前向时统计该层激活 X，找出 |x| > 6.0 的列（outlier 维度）
2. $X_{\text{low}}$、$W_{\text{low}}$ → INT8 GEMM（vector-wise scale）
3. $X_{\text{outlier}}$、$W_{\text{outlier}}$ → FP16 GEMM
4. 两部分结果相加$\to$输出

由于 outlier 维度很少，FP16 路径只占$\sim 0.1\%$的计算和内存。

### 3.5 结果与局限

**结果**：OPT-175B 在 INT8 下 perplexity 与 FP16 一致；权重+激活内存约减半。

**局限**：

1. **小模型反而慢**：outlier 检测 + 矩阵拆分 + 两条 GEMM 路径的调度开销，在小模型/短序列上可能超过收益（论文报告 6.7B 以下速度没有优势）。
2. **含 FP16 路径**：outlier 列仍要走 FP16 GEMM，峰值 FLOPS 收益打折，不能按纯 INT8 Tensor Core 估算。
3. **阈值经验性**：6.0 是统计出来的，不同模型/分布要重新验证。

---

## 4. SmoothQuant：把量化难度迁移到权重

### 4.1 动机

LLM.int8() 用"分离"绕开 outlier；SmoothQuant 问：**能不能让激活根本没有 outlier？**

关键观察：权重容易量化（值域小、可 per-channel），激活难量化（outlier 撑大 per-tensor 范围）。那就用数学上**完全等价**的变换，把"激活的量化难度"搬到"权重"上去。

### 4.2 等效变换推导

对线性层$Y = WX$，对每个输入通道 j 引入缩放因子$s_{j}$：

$$
Y = WX = (W \cdot \operatorname{diag}(s)) \cdot (\operatorname{diag}(s)^{-1} \cdot X)
$$

即：

$W' = W \cdot \operatorname{diag}(s)$（权重按通道放大）
$X' = \operatorname{diag}(s)^{-1} \cdot X$（激活按通道缩小）

输出完全不变——**这是恒等变换**，只是把数值挪了位置。

### 4.3 平滑因子公式与 α 的语义

$$
s_{j} = \max|X_{j}|^\alpha / \max|W_{j}|^{1-\alpha}, \alpha \in [0, 1]
$$

$\alpha$是"迁移强度"：

$\alpha = 0$：$s = 1/\max|W_{j}| \to$只归一化权重，不迁移（退化）
$\alpha = 0.5$：一半一半（OPT/BLOOM 的通用甜点）
$\alpha = 1$：$s = \max|X_{j}| \to$激活被完全归一化，难度全部搬到权重

论文结论：**OPT/BLOOM 取$\alpha =0.5$最佳；激活更难量化的 GLM-130B 需要更大的$\alpha$（0.8）。**

直觉：激活 outlier 通道（$\max|X_{j}|$ 巨大）对应 $s_{j}$ 巨大，该通道激活被除以 s 后回到正常量级，outlier 则随缩放搬到权重侧。权重按 per-channel 量化，通道之间的尺度差异各自独立处理；激活按 per-tensor 量化，一个通道的 outlier 会污染整层。

### 4.4 为什么零运行时开销：scale 折叠

推理时不需要真的先乘 s 再量化：

量化激活：$q_{x} = \operatorname{round}(X'/s_{x}) = \operatorname{round}(X/(s_{x}\cdot s)) \to s$并进激活的 per-tensor scale
量化权重：$q_{w} = \operatorname{round}(W'/s_{w}) = \operatorname{round}(W\cdot s/s_{w}) \to s$并进权重的 per-channel scale
反量化输出：$\hat{Y} = (q_{w}\cdot s_{w}') \cdot (q_{x}\cdot s_{x}')$，$s_{w}' = s_{w}/s$，$s_{x}' = s_{x}\cdot s$

**s 被折叠进激活和权重的 scale 里，推理时多一次乘法都没有。** 这也是 SmoothQuant 能无缝进 TensorRT-LLM/vLLM 的原因。

### 4.5 算法流程

1. 校准：用校准集前向，统计每层$\max|X_{j}|$与$\max|W_{j}|$
2. 选$\alpha$（默认 0.5，或按验证集扫描）
3. 计算$per-channel s_{j}$，生成$W' = W\cdot s$，记录 X 的 per-tensor scale
4. 对 W' 做 per-channel INT8 量化、X 做 per-tensor INT8 量化
5. 部署：INT8 GEMM + 折叠后的 scale

### 4.6 结果

- OPT-175B / BLOOM-176B / GLM-130B / LLaMA 全系列 **W8A8 近无损**（perplexity 差异在噪声范围）。
- 在优化 kernel 上比 FP16 快约 1.5x（计算密集的 prefill 收益明显）。
- 被 NVIDIA TensorRT-LLM（INT8 W8A8）等生产引擎采用。

---

## 5. W8A8 的工程细节

### 5.1 INT8 GEMM 的数据流

1. X 量化：$per-tensor scale s_{x}$（在线，一两个 kernel）
2. W 量化：$per-channel scale s_{w}$（离线，含 SmoothQuant 的 s 折叠）
3. INT8 矩阵乘：$q_{w} \cdot q_{x}$（INT8 Tensor Core，累加器 INT32/FP32）
4. 反量化：$Y \approx (q_{w} \cdot q_{x}) \cdot (s_{w} \otimes s_{x})$（逐通道乘回）

### 5.2 动态量化的成本

激活量化必须在线完成（每层前向时算$\max|X| \to scale \to \operatorname{round}$）。对 per-tensor 来说只是两次 scan/round 的 kernel，开销很小；per-channel 激活量化 kernel 复杂得多，所以 W8A8 默认激活 per-tensor。

### 5.3 累加器精度

INT8×INT8 的乘积累加必须用 INT32/FP32，否则溢出。**量化的是输入和权重，累加器始终高精度**——这是量化推理能保持质量的关键工程细节。

---

## 6. 从 INT8 到 FP8：激活量化的现代形态

03 章讲过：FP8 E4M3 自带 4 位指数，动态范围远好于 INT8。因此现代 W8A8 常用 **FP8**：

INT8 方案：SmoothQuant（把 outlier 抚平）$\to$需要$\alpha$校准
FP8 方案：E4M3 天然容忍部分$outlier \to$校准更省事

SmoothQuant 的迁移在 FP8 上仍有用：FP8 激活仍有 outlier（只是阈值变宽），而 E4M3 的尾数只有 3 bit，精度预算更紧张。TensorRT-LLM / vLLM 的 FP8 方案里仍然常见"SmoothQuant 式迁移 + FP8"的组合。

---

## 7. 本章小结

1. **激活量化难**：动态、per-tensor、outlier（04 章 6.2 的"偷位"算例）。
2. **LLM.int8()**：vector-wise 量化 + outlier 混合精度分解；outlier 集中在少数特征维度这一统计事实是它的设计基础（04 章 6.1）；无损但工程复杂、小模型不快。
3. **SmoothQuant**：恒等变换 (W·s)(X/s) 把难度从激活搬到权重；$s_{j} = \max|X|^\alpha /\max|W|^{1-\alpha}$，$\alpha$控制迁移强度；scale 折叠$\to$零运行时开销；W8A8 近无损、$\sim 1.5x$。
4. **工程约束**：累加器必须高精度；激活量化在线做、权重离线做；scale 能折叠就折叠。

---

## 8. 习题与解答

### 题 1（推导）：SmoothQuant 恒等变换

证明$Y = (W\cdot \operatorname{diag}(s))\cdot (\operatorname{diag}(s)^{-1}\cdot X)$与$Y = WX$完全相等，并说明为什么变换后激活更容易量化。

<details>
<summary>题 1 解答</summary>

$(W\cdot \operatorname{diag}(s))\cdot (\operatorname{diag}(s)^{-1}\cdot X) = W\cdot \operatorname{diag}(s)\cdot \operatorname{diag}(s)^{-1}\cdot X = W\cdot I\cdot X = WX$。
激活更容易：outlier 通道 j 的$s_{j}$大，$X_{j}/s_{j}$被缩小到正常量级；权重$W_{j}\cdot s_{j}$变大，但权重是 per-channel 量化，通道间尺度差异不影响相对精度。
</details>

### 题 2（计算）：α 的边界

写出$\alpha =0$、$\alpha =1$时$s_{j}$的表达式，并分别说明它们对应"完全不迁移"和"完全迁移"。

<details>
<summary>题 2 解答</summary>

$\alpha =0$：$s_{j} = 1/\max|W_{j}|$（只归一化权重通道，激活不动）；$\alpha =1$：$s_{j} = \max|X_{j}|$（激活被完全归一化到$\pm 1$，难度全部进权重）。$\alpha =0.5$是几何中点，两种难度均衡。
</details>

### 题 3（对比）：LLM.int8 vs SmoothQuant

填表：量化对象、outlier 处理方式、是否有 FP16 路径、是否需校准、适用位宽。

<details>
<summary>题 3 解答</summary>

| | LLM.int8() | SmoothQuant |
|---|---|---|
| 量化对象 | 权重+激活（INT8） | 权重+激活（INT8/FP8） |
| outlier 处理 | 分离到 FP16 路径 | 迁移到权重 |
| FP16 路径 | 有（outlier 列） | 无（全 INT8/FP8） |
| 校准 | 不需要 | 需要（统计 max，选$\alpha$） |
| 适用位宽 | 8-bit | 8-bit（FP8 变体更常用） |
</details>

### 题 4（思考）：为什么 SmoothQuant 的权重扛得住 outlier

权重被乘以巨大 s 后，"outlier 搬家"到权重。为什么 per-channel 量化下这不伤害权重精度？

<details>
<summary>题 4 解答</summary>

per-channel 量化对每个通道独立选 scale（04 章）：通道 j 的 scale 由该通道自己的 max 决定。$W_{j}\cdot s_{j}$再大，也只是把该通道的量化范围整体放大，通道内相对精度不变。而激活是 per-tensor 量化，一个通道的 outlier 会污染整层——所以"难度搬家"只对权重无损。
</details>

### 题 5（编程）：复现 SmoothQuant 的 α 扫描

构造 W（2×2）、X（2×N，含一个 outlier 通道），对$\alpha \in \{0, 0.5, 1\}$计算平滑后的量化（INT8 per-channel 权重、per-tensor 激活）SQNR，验证$\alpha =0.5$附近最好。

<details>
<summary>题 5 解答要点</summary>

实现$s_{j} = \max|X_{j}|^\alpha /\max|W_{j}|^{1-\alpha}$，量化 W·s（per-channel）与 X/s（per-tensor），反量化并算 ‖WX−ŴX̂‖。$\alpha =0$时激活量化被 outlier 打爆；$\alpha =1$时权重误差增大；$\alpha =0.5$通常折中最优。可用真实 LLM 激活分布（如校准一层的激活）验证更贴近论文结论。
</details>

---

## 9. 延伸阅读

1. [LLM.int8()（arXiv:2208.07339）](https://arxiv.org/abs/2208.07339)：vector-wise + 混合精度分解
2. [SmoothQuant（arXiv:2211.10438）](https://arxiv.org/abs/2211.10438)：迁移公式、$\alpha$扫描、scale 折叠
3. [SmoothQuant 官方代码](https://github.com/mit-han-lab/smoothquant)：INT8 GEMM kernel 与校准实现
4. 上一篇：[06 权重量化 II](/notes/llm量化精读笔记-06-权重量化ii-awq-squeezellm-quip/)；下一篇：**[08 KV Cache 量化：KIVI 与误差累积]**——第三个量化对象：把长上下文里的显存大头压下来。

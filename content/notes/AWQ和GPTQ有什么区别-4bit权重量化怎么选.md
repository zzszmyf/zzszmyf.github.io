---
title: "AWQ 和 GPTQ 有什么区别？4-bit 权重量化怎么选"
date: 2026-09-25T00:00:00+08:00
draft: false
description: "AWQ 和 GPTQ 都是 4-bit weight-only 量化，区别在怎么补偿误差：GPTQ 用 Hessian 二阶信息逐列校正权重，需要重建、小时级；AWQ 不改权重数值，只按激活幅度做 per-channel 缩放保护约 1% 的显著通道，无重建、分钟级。本文给出选型判据。"
weight: 90
tags: ["LLM推理优化", "量化", "GPTQ", "AWQ", "GGUF"]
---

> 系列导航：[LLM 量化精读笔记总览](/notes/llm量化精读笔记-00-总览与学习地图/)（共 11 篇）｜本文是把 [05 RTN 与 GPTQ](/notes/llm量化精读笔记-05-权重量化i-rtn与gptq/) 和 [06 AWQ 与 QuIP#](/notes/llm量化精读笔记-06-权重量化ii-awq-squeezellm-quip/) 两章并起来看的对比篇。

## 一句话结论

**AWQ 和 GPTQ 都是 4-bit weight-only 量化（W4A16，激活保持 FP16），区别在于怎么补偿量化误差**：GPTQ 用 Hessian 二阶信息逐列校正权重，属于"重建式"方法，量化一个 175B 模型要小时级；AWQ 完全不动权重数值，只用激活幅度算出的 per-channel 缩放来"保护"约 1% 的显著通道，无梯度无重建，分钟级跑完。

落到选型：**要快、要稳、要生态兼容，选 AWQ；要在 3-bit 上榨极限质量，或者你的流水线本来就吃 GPTQ 权重，选 GPTQ。**

## 它们到底在争什么？

先明确共同前提，不然后面的差异都看不懂：

- 两者都是 **weight-only**：只压权重，激活（以及 KV Cache）保持 FP16。真正难压的激活量化是另一条线，见 [激活量化为什么比权重量化难](/notes/llm量化精读笔记-07-激活量化-llm-int8与smoothquant/)。
- 两者都做 **group 量化**：常见 `group_size=128`，即每 128 个权重共享一组 scale/zero。粒度选择见 [量化粒度怎么选](/notes/llm量化精读笔记-04-量化粒度校准与离群值/)。
- 两者都需要 **校准集**：跑几百条文本拿到激活统计，不需要反向传播、不需要原始训练数据。

在这三个共同点上，它们分歧只有一个问题：**量化误差已经产生了，你怎么补？**

## GPTQ 的思路：把误差"扩散"到还没量化的权重上

GPTQ 的出发点是最小化**层输出误差**而不是权重的逐元素误差：

$$\min_{\hat{W}} \|WX - \hat{W}X\|^2$$

它逐列处理权重：量化第 $q$ 列产生误差后，立刻用 Hessian 逆矩阵把这个误差**补偿到后面还没量化的列**上，让整层的输出误差最小。补偿量就是经典的 OBS 公式

$$\delta F = -\frac{w_q - \hat{w}_q}{[H^{-1}]_{qq}} \cdot H^{-1}_{:,q}$$

工程上 GPTQ 靠三个观察把 OBQ 的 $O(d^3)$ 压到可跑：列顺序不影响结果、可以按行并行、用 Cholesky 分解避免反复求逆。完整推导和 2×2 手算例子在 [05 章](/notes/llm量化精读笔记-05-权重量化i-rtn与gptq/)。

代价也很明确：

1. **慢**。需要逐层前向 + 线性代数，175B 级别是小时级；RTN 只要几秒。
2. **二次假设**。目标函数只在最优点附近近似二次，4-bit 以下补偿公式不再最优。
3. **校准集依赖**。Hessian 来自校准数据，校准集分布偏了，误差就偏了。
4. **没有显式利用激活幅度**。H 里含激活的二阶统计，但没用上一阶的幅度信息——这正是 AWQ 的切入点。

## AWQ 的思路：不动权重，动 scale

AWQ 先做了个诊断实验：INT3-g128 量化 OPT-13B 后，把一部分通道保留成 FP16（混合精度），看保留谁最有效。WikiText-2 困惑度（FP16 基线 10.13）：

| 保留策略 | PPL |
|---|---|
| 全量化（RTN） | 46.04 |
| 保留 1% 通道为 FP16（**按激活幅度**选） | **10.51** |
| 保留 1% 通道为 FP16（按权重幅度选） | 48.96 |
| 保留 1% 通道为 FP16（随机选） | 42.00 |

**结论：决定成败的显著通道只有约 1%，而且必须按输入激活的幅度识别，不能按权重本身的幅度。**

问题是混合精度在硬件上很难做——内存布局和 kernel 都要分支。AWQ 的替代方案是数学等效：把显著通道的权重放大 $s$ 倍、对应激活缩小 $s$ 倍，量化误差约缩小 $1/s$，效果上等于"保护"，但权重依然是整齐的 INT4。缩放系数取

$$s_j = (\max |X_j|)^\alpha$$

再对 $\alpha$ 做网格搜索。在 OPT-6.7B INT3-g128 上扫出来的甜点是 $s \approx 2$：显著通道误差减半，同时只有约 8% 的组被"顶到新的 max"而伤到同组其他通道。代价是 $s$ 太大会反噬——$s=4$ 时 21.2% 的组被顶爆，PPL 反而从 11.92 回升到 12.36。

## 两者差别，一张表看完

| | GPTQ | AWQ |
|---|---|---|
| 补偿机制 | Hessian 二阶更新 | 激活幅度 per-channel 缩放 |
| 是否需要重建 | 是 | 否 |
| 量化时间 | 小时级（175B） | 分钟级 |
| 甜点位宽 | 3–4 bit | 4 bit |
| 校准集过拟合风险 | 有 | 低（只用到激活的 max） |
| 2-bit 表现 | 崩 | 崩 |
| 硬件友好性 | 需要特殊 kernel（重排） | scale 折叠，天然友好 |
| 核心风险 | 分布外退化 | 只保护"幅度"大的通道，忽略通道相关性 |
| 生态 | vLLM / TensorRT-LLM / HF | vLLM / TensorRT-LLM / HF / LMDeploy / TinyChat |

一句话记忆：**GPTQ 是"量错一个就补后面的"，AWQ 是"先把重要的通道垫高再统一量"。**

## 那 GGUF 是哪一层的东西？

这是搜"AWQ / GPTQ / GGUF 区别"时最常见的混淆点，因为它们是**两个不同维度**：

- **GPTQ、AWQ 是量化算法**：回答"用什么方法把权重压到 4-bit"。
- **GGUF 是文件格式 + 运行时容器**：回答"压好的权重打包成什么文件、给哪个引擎读"。它主要服务 llama.cpp 生态，里面装的往往是 Q4_K_M 这类 k-quant 方案，而不是 GPTQ/AWQ 的产物。

所以正确的比较方式是：先决定推理引擎，引擎再决定格式。想上 vLLM / TensorRT-LLM 的高吞吐服务，就走 GPTQ 或 AWQ；想在 Mac、CPU 或 llama.cpp 上跑，就走 GGUF。这两条路不是竞品关系。

## 我该选哪个？

| 你的情况 | 建议 |
|---|---|
| 通用生产服务，要 W4A16 | **AWQ**。量化几分钟能跑完、校准集过拟合风险低、scale 可折叠零开销 |
| 已有 GPTQ 权重产物 / 流水线依赖 | 继续用 GPTQ，不必迁 |
| 要压到 3-bit 还保质量 | 先看 [SqueezeLLM、QuIP#](/notes/llm量化精读笔记-06-权重量化ii-awq-squeezellm-quip/)；GPTQ 在 3-bit 比 AWQ 更稳 |
| 主要吃指令微调或多模态模型 | AWQ 的泛化优势更明显 |
| 要 2-bit | 两者都会崩，走 QuIP# 路线 |
| 量化完了要上服务 | 先看 [量化模型怎么部署到 vLLM](/notes/llm量化精读笔记-11-系统协同与部署/)，算法好不等于端到端快 |

## 几个容易踩的坑

1. **别只看困惑度**。PPL 掉 0.1 不代表下游任务没问题，验收协议见 [量化后精度掉了怎么定位](/notes/llm量化精读笔记-10-质量评估方法论/)。
2. **校准集要接近线上分布**。AWQ 只用到激活的 max，比 GPTQ 抗偏，但代码、长上下文这类场景仍然要单独校准。
3. **group_size 别随手改**。128 是生态默认值，kernel 是按它调的；改小省精度但可能掉进慢路径。
4. **量化对象不只是权重**。长上下文场景下 KV Cache 往往才是显存大头，见 [KV Cache 量化怎么做](/notes/llm量化精读笔记-08-kv-cache量化与kivi/)。
5. **只量化目标模型、不量化草稿模型**，会让推测解码的收益被稀释，这点在 [推测解码上线怎么验收](/notes/llm推测解码精读笔记-06-系统集成与生产验收/) 里有量化分析。

## 参考

1. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers（arXiv:2210.17323）](https://arxiv.org/abs/2210.17323)
2. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration（arXiv:2306.00978）](https://arxiv.org/abs/2306.00978)
3. [OBQ: Optimal Brain Compression（Frantar & Alistarh, NeurIPS 2022）](https://arxiv.org/abs/2208.11580)
4. [llama.cpp / GGUF 格式说明](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

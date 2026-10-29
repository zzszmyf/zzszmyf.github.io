---
title: "精读系列"
metaTitle: "LLM 推理优化精读系列：量化、推测解码、注意力与 KV Cache"
description: "三套 MIT lecture note 级别的 LLM 推理优化中文精读笔记：量化 12 章、推测解码 7 章、注意力与计算内核 9 章，每章含形式化定义、数学推导、伪代码、数值算例与习题。"
date: 2026-10-18T00:00:00+08:00
draft: false
layout: "landing"
seriesIndex: true
---

三套系统化的 LLM 推理优化中文精读笔记，覆盖 **量化（有损换速度）**、**推测解码（无损换步数）**、**注意力与计算内核（无损换效率）** 三条主线。每章结构统一：形式化定义 → 数学推导 → 伪代码/算法 → 数值算例 → 直觉解释 → 习题（含答案）→ 延伸阅读。

不连续的工程记录、论文精读和部署踩坑在 [技术笔记](/notes/)；投资、读书、职场等随笔在 [随笔](/essays/)。

## 量化（有损换速度）

系列目录：[量化精读](/series/quantization/)。从信息论与数值编码的地基，到均匀量化理论、数值格式、粒度与离群值、主流 PTQ/QAT 方法、质量评估与生产部署。

| 章节 | 内容 |
|---|---|
| [速览笔记](/notes/llm推理优化-quantization量化学习笔记/) | 概览版，适合快速复习 |
| [00 总览与学习地图](/notes/llm量化精读笔记-00-总览与学习地图/) | 章节结构、符号约定、阅读路线、配套资源 |
| [01 数值编码与计算机表示基础](/notes/llm量化精读笔记-01-数值编码与计算机表示基础/) | 信息论、整数/定点编码、IEEE 754、舍入与截断、存储层次 |
| [02 量化问题形式化与均匀量化理论](/notes/llm量化精读笔记-02-量化问题形式化与均匀量化理论/) | 量化的一般形式、均匀量化数学、误差与 SNR 理论（6 dB/bit 推导） |
| [03 数值格式与硬件](/notes/llm量化精读笔记-03-数值格式与硬件/) | FP16/BF16/FP8/FP4/MXFP8/NVFP4、Tensor Core、带宽模型 |
| [04 量化粒度、校准与离群值](/notes/llm量化精读笔记-04-量化粒度校准与离群值/) | per-tensor/channel/group、有效位宽、校准设计、outlier 问题 |
| [05 权重量化 I（RTN 与 GPTQ）](/notes/llm量化精读笔记-05-权重量化i-rtn与gptq/) | OBS/OBQ 二阶误差补偿推导、GPTQ 工程化 |
| [06 权重量化 II（AWQ、SqueezeLLM、QuIP#）](/notes/llm量化精读笔记-06-权重量化ii-awq-squeezellm-quip/) | 激活感知缩放、敏感度非均匀量化、Hadamard 非相干 + 格码本 |
| [07 激活量化（LLM.int8 与 SmoothQuant）](/notes/llm量化精读笔记-07-激活量化-llm-int8与smoothquant/) | 混合精度分解、迁移公式、W8A8 与 scale 折叠 |
| [08 KV Cache 量化（KIVI）](/notes/llm量化精读笔记-08-kv-cache量化与kivi/) | KV 显存账本、误差累积、K 按通道 / V 按 token |
| [09 QAT 与训练内量化](/notes/llm量化精读笔记-09-qat与训练内量化-ste-qlora-bitnet/) | STE、QLoRA（NF4）、BitNet b1.58 |
| [10 质量评估方法论](/notes/llm量化精读笔记-10-质量评估方法论/) | perplexity、智能基准、自定义评测、统计显著性与验收关卡 |
| [11 系统协同与部署](/notes/llm量化精读笔记-11-系统协同与部署/) | QServe W4A8KV4、FP8 Attention、引擎选型与部署决策树 |

> 从技术到决策：[企业 AI 落地该选 FP8 还是 FP4？——量化格式的成本、质量与硬件决策](/notes/企业ai落地选fp8还是fp4-量化格式的成本质量与硬件决策/)：用一个 70B 客服模型的部署算例，把位宽换算成卡数、KV 显存、每百万 token 成本和字段级失败率；附 vLLM 参数、验收阈值与采购验证问题。
>
> [AWQ 和 GPTQ 有什么区别？4-bit 权重量化怎么选](/notes/awq和gptq有什么区别-4bit权重量化怎么选/)：对照 05 章 GPTQ 和 06 章 AWQ，按离线量化时间和能不能接受重建来选。

## 推测解码（无损换步数）

系列目录：[推测解码精读](/series/speculative-decoding/)。从接受率数学与拒绝采样，到原始推测解码、Medusa 多头解码、EAGLE 特征空间草稿、n-gram/检索式无模型路线，以及系统集成与生产验收。

| 章节 | 内容 |
|---|---|
| [00 总览与学习地图](/notes/llm推测解码精读笔记-00-总览与学习地图/) | 章节结构、符号约定、与量化系列的关系 |
| [01 问题形式化与接受率数学](/notes/llm推测解码精读笔记-01-问题形式化与接受率数学/) | 自回归为什么慢、接受率 α、E[N] 推导、墙钟收益模型 |
| [02 原始推测解码（草稿模型与拒绝采样）](/notes/llm推测解码精读笔记-02-原始推测解码-草稿模型与拒绝采样/) | 完整算法、无损性定理、最优 K、草稿规模权衡 |
| [03 Medusa（多头解码）](/notes/llm推测解码精读笔记-03-medusa-多头解码/) | 多头并行预测、树注意力、典型验收、Medusa-1/2 |
| [04 EAGLE（特征空间草稿）](/notes/llm推测解码精读笔记-04-eagle-特征空间草稿/) | 特征级自回归、shifted-token、EAGLE-2 动态树 |
| [05 n-gram/检索式与无模型路线](/notes/llm推测解码精读笔记-05-n-gram检索式与无模型路线/) | Prompt Lookup、Lookahead Decoding、REST |
| [06 系统集成与生产验收](/notes/llm推测解码精读笔记-06-系统集成与生产验收/) | 量化 × 推测组合模型、TTFT/TPS、验收协议、决策树 |

## 注意力与计算内核（无损换效率）

系列目录：[注意力与计算内核精读](/series/attention-kernels/)。从注意力复杂度与 KV cache，到 FlashAttention、MQA/GQA/MLA、稀疏/线性注意力、PagedAttention、内核优化与算子融合、前缀缓存与 KV 复用，以及系统集成与生产验收。

| 章节 | 内容 |
|---|---|
| [00 总览与学习地图](/notes/llm注意力内核精读笔记-00-总览与学习地图/) | 章节结构、符号约定、与量化/推测解码系列的关系 |
| [01 注意力机制基础与复杂度分析](/notes/llm注意力内核精读笔记-01-注意力机制基础与复杂度分析/) | softmax attention 定义、O(L²) 复杂度、KV cache 角色、prefill/decode 形态 |
| [02 FlashAttention（IO 感知的精确注意力）](/notes/llm注意力内核精读笔记-02-flashattention-io感知的精确注意力/) | IO 复杂度、tiling、online softmax、重计算、FA2/FA3 |
| [03 注意力头变体（MQA/GQA/MLA）](/notes/llm注意力内核精读笔记-03-注意力头变体-mqa-gqa-mla/) | KV 头共享、低秩压缩、DeepSeek MLA |
| [04 稀疏、滑动窗口与线性注意力](/notes/llm注意力内核精读笔记-04-稀疏滑动窗口与线性注意力/) | StreamingLLM、滑动窗口、H2O、Mamba |
| [05 PagedAttention 与 KV 显存管理](/notes/llm注意力内核精读笔记-05-pagedattention与kv显存管理/) | 分页 KV、vLLM 块管理、与批处理组合 |
| [06 内核优化与算子融合](/notes/llm注意力内核精读笔记-06-内核优化与算子融合/) | 访存-计算模型、Tensor Core、FP8 注意力、编译优化 |
| [07 系统集成与生产验收](/notes/llm注意力内核精读笔记-07-系统集成与生产验收/) | 与量化/推测解码组合、注意力精度验收 |
| [08 前缀缓存与 KV 复用](/notes/llm注意力内核精读笔记-08-前缀缓存与kv复用/) | RadixAttention、radix tree、KV 存储层级、cache-aware routing、disaggregation |

## 阅读顺序

```
注意力与内核：00 总览 → 01 基础与复杂度 → 02 FlashAttention → 03 MQA/GQA/MLA
             → 04 稀疏/线性注意力 → 05 PagedAttention → 06 内核优化
             → 07 集成验收 → 08 前缀缓存与 KV 复用

推测解码：00 总览 → 01 接受率数学 → 02 原始推测解码 → 03 Medusa
          → 04 EAGLE → 05 n-gram/检索式 → 06 系统集成与验收

量化：01 数值编码（地基）→ 02 均匀量化理论（数学）→ 03 数值格式与硬件（格式）
      → 04 粒度、校准与离群值（难点）→ 05-06 权重量化 → 07 激活量化
      → 08 KV Cache → 09 训练侧 → 10 质量评估 → 11 系统与部署
```

## 素材来源

- [Inference Engineering, Ch5](https://inferenceengineering.tech/chapters/techniques/)（Baseten 出品）
- [MIT 6.5940 EfficientML.ai](https://hanlab.mit.edu/courses/2026-fall-65940)（Song Han / HAN Lab）
- 论文：LLM.int8() / GPTQ / SmoothQuant / AWQ / SqueezeLLM / QuIP# / KIVI / QServe / QLoRA / BitNet b1.58 / FP8 白皮书 / OCP MX 规范（各章"延伸阅读"附 arXiv 链接）
- 推测解码论文：Leviathan et al. 2023 / Chen et al. 2023 / Medusa / EAGLE / EAGLE-2 / Lookahead Decoding / REST（各章"延伸阅读"附 arXiv 链接）
- 注意力内核论文：FlashAttention 1/2/3 / GQA / DeepSeek-V2 MLA / StreamingLLM / Mamba / PagedAttention / SGLang-RadixAttention（各章"延伸阅读"附 arXiv 链接）

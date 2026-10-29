---
title: "量化速查：格式、算法、引擎"
metaTitle: "量化速查：FP8、FP4、AWQ、GPTQ 和引擎怎么对上"
description: "量化精读的一页规则：先无损，再 W8A8，再 KV，attention 默认不动。AWQ / GPTQ 和 FP8 / FP4 的选择条件，以及 vLLM、SGLang、TensorRT-LLM 各自对上哪一章。没有新的实测数字。"
date: 2026-10-29T00:00:00+08:00
draft: false
schemaType: TechArticle
---

规则来自[量化精读](/series/quantization/)和两篇选型文。同机三引擎 tokens/s 是待实测，不写在这页。

同页的 [Markdown 原文](index.md) 可以下载。

## 先做什么

1. 前缀缓存、推测解码、PagedAttention。这些是 11 章写的无损项。
2. 质量可以让步时，默认 W8A8 FP8，再单独开 KV 的 FP8 或 4-bit。attention 留 FP16，除非有专用 kernel、长序列占比高、而且评测过。
3. 每一档单独验收。两个改动一起上，出了问题归因不了。

## 格式

| 选择 | 条件 | 笔记 |
|---|---|---|
| FP8 权重 + FP8 KV | Hopper / Ada 有 FP8 Tensor Core，而且 KV 不留在 FP16 | [FP8 还是 FP4](/notes/企业ai落地选fp8还是fp4-量化格式的成本质量与硬件决策/)、[03 章](/notes/llm量化精读笔记-03-数值格式与硬件/) |
| 只压权重、KV 仍是 FP16 | 显存账通常守不住并发。70B / 8K / 32 并发那个例子是算账，不是客户数据 | 同上 |
| FP4 | Blackwell 才有对应 Tensor Core。字段错误率没压住就不要为了少一张卡 | 同上、[11 章](/notes/llm量化精读笔记-11-系统协同与部署/) |
| Ampere | 不要把 FP8 算力写进承诺 | 03 章 |

## 4-bit 权重

| 选择 | 条件 | 笔记 |
|---|---|---|
| AWQ | 生产默认的 W4A16：校准短，不做重建 | [AWQ 和 GPTQ](/notes/awq和gptq有什么区别-4bit权重量化怎么选/)、[06 章](/notes/llm量化精读笔记-06-权重量化ii-awq-squeezellm-quip/) |
| GPTQ | 流水线里已经有产物，或要往 3-bit 靠 | [05 章](/notes/llm量化精读笔记-05-权重量化i-rtn与gptq/) |
| 2-bit | AWQ 和 GPTQ 都会崩，换 QuIP# 一类 | 06 章 |
| GGUF | llama.cpp 的容器，不是第三种算法 | AWQ / GPTQ 那篇 |

## 引擎

| 负载 | 先看 | 依据 |
|---|---|---|
| 共享前缀高 | SGLang | [08 章](/notes/llm注意力内核精读笔记-08-前缀缓存与kv复用/) |
| 量化格式多、PagedAttention | vLLM | [11 章](/notes/llm量化精读笔记-11-系统协同与部署/)、[05 章](/notes/llm注意力内核精读笔记-05-pagedattention与kv显存管理/) |
| NVIDIA FP8 / INT4 kernel | TensorRT-LLM | 11 章 |
| 三者谁的 tokens/s 更高 | 待实测 | [怎么选](/notes/vllm-sglang-tensorrt-llm怎么选/) |

11 章的检查清单仍然有效：格式有没有实测过的 kernel、激活量化了没有、KV 量化能不能关、长上下文和 MoE 配不配、硬件是不是 Hopper 或 Blackwell。

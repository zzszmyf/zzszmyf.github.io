---
title: "vLLM、SGLang、TensorRT-LLM 该怎么选？"
description: "没有三引擎同机吞吐表。共享前缀高先看 SGLang 的 RadixAttention；量化格式和 PagedAttention 要广选 vLLM；NVIDIA 上把 FP8/INT4 kernel 做深选 TensorRT-LLM。同负载 tokens/s 标待实测。"
date: 2026-10-29T00:00:00+08:00
draft: false
schemaType: TechArticle
enURL: "/en/vllm-vs-sglang-vs-tensorrt-llm/"
tags: ["LLM推理优化", "vLLM", "SGLang", "TensorRT-LLM"]
categories: ["工程实践"]
faq:
  - question: "vLLM、SGLang、TensorRT-LLM 该怎么选？"
    answer: "本站没测过三家引擎的同机吞吐。大量请求共用前缀时先看 SGLang 的基数树缓存；量化格式要多、KV 要分页管理，选 vLLM；人在英伟达卡上要把低精度算子做深，看 TensorRT-LLM。同一批流量下谁的每秒词元更高，这里标待实测，不写假数字。"
---

本站没有 vLLM、SGLang、TensorRT-LLM 的同机 tokens/s。选型先看负载，不看口号。共享前缀高（多轮、agent、few-shot、重复的 RAG 前缀）先看 SGLang 的 RadixAttention。[08 章](/notes/llm注意力内核精读笔记-08-前缀缓存与kv复用/)引用的论文把无命中开销写在 0.3% 以下，所以作者主张可以默认开。量化格式要多、团队已经按 PagedAttention 运维，[11 章](/notes/llm量化精读笔记-11-系统协同与部署/)的地图把 vLLM 写成生态最广、生产首选之一。人在 NVIDIA 卡上、要的是官方把 FP8 / INT4 kernel 做深，同一张地图写的是 TensorRT-LLM。你的 batch 和上下文下三者谁更快，这里是**待实测**。

## 对照

数字都来自本站已有笔记或笔记里点名的论文，不是这次新跑的三方榜。

| 问题 | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| KV 怎么放 | [05 章](/notes/llm注意力内核精读笔记-05-pagedattention与kv显存管理/)：逻辑块映射到物理块。论文里不做管理时，KV 有效利用率只有 20.4%–38.2% | 分页之上再加 radix tree，请求结束后前缀还在 | 本站没有单独拆它的分页实现。待实测 |
| 共享前缀 | 08 章的辨析：早期实现偏「一个系统提示」加 copy-on-write | 多级前缀自动共享，LRU 从叶子驱逐。论文相对 vLLM / Guidance / LMQL：吞吐最高 6.4×，延迟最高降 3.7×，命中率 50%–99%，管理开销 <0.3%（74.3 s 里 0.2 s）。Chatbot Arena 一个月：Vicuna-33B 命中 74.1%、平均 TTFT 降 1.7×；LLaVA-NeXT-34B 命中 52.4%。这些是论文数字，不是本站复现 | 待实测 |
| 量化格式 | 11 章的表：AWQ、GPTQ、FP8、MXFP8、FP4、KV 量化 | 11 章写「同上（复用多种后端）」，并单列结构化生成 | FP8、INT4/FP4、SmoothQuant W8A8、AWQ、KV 量化。特点写成 NVIDIA 深度优化 |
| 推测解码 | [04 章](/notes/llm推测解码精读笔记-04-eagle-特征空间草稿/)：vLLM 已集成 EAGLE，含 EAGLE-2 风格的动态树 | [首页](/)记录的上游工作：[PR #35423](https://github.com/sgl-project/sglang/pull/35423) 修推测解码量化 target；[issue #35437](https://github.com/sgl-project/sglang/issues/35437) 把一次 prefill CUDA graph OOM 的 TTFT 从 179 ms 收到 113 ms。这是一个修复，不是三引擎排名 | 待实测 |
| 已经踩过的坑 | [不传 max_tokens](/notes/vllm-gpu-100-percent-max-tokens-trap/) 时，上限落到 `max_model_len`。那次 PaddleOCR-VL 大约 13 万 token，GPU 利用率挂在 100% | 不是同一次事故 | 不是同一次事故 |

QServe 论文相对当时 TensorRT-LLM 的吞吐（一般模型 1.2–1.4×，Qwen1.5-72B 2.4–3.5×，L40S 上 8 个模型里 6 个超过 A100 上的 TensorRT-LLM）写在 11 章。那是论文和 TensorRT-LLM 的对比，不是本站把三个引擎放在同一台机器上跑出来的。

## 何时选谁

共享前缀占延迟大头时选 SGLang。08 章写过边界：输出很长、会话之间又不共享时，提速会回到接近零。多轮短回复、agent 轨迹、few-shot、同一张图被反复问，才是它的范围。

量化清单长、又要 PagedAttention 这套运维语义时选 vLLM。11 章给 H200、70B、32K、PPL 要无损的习题答案是：先前缀缓存和 PagedAttention，量化用 W8A8 FP8 加 KV 4-bit 或 FP8，attention 留 FP16，引擎写的是 vLLM 或 TensorRT-LLM，因为 FP8 kernel 成熟。验收走 [10 章](/notes/llm量化精读笔记-10-质量评估方法论/) 的 Gate，不达标就降档。

人已经在 NVIDIA 软件栈里、采购条款问的是某个 SKU 上的 FP8 或 INT4 kernel，选 TensorRT-LLM 更贴 11 章的「NVIDIA 深度优化」。格式能不能换卡，FP8 / FP4 那篇把 NVFP4 和 MXFP4 的工具链差别写过了，这里不重复。

质量必须无损时，三个都不该先上量化。11 章的顺序是先前缀缓存和推测解码，再动 W8A8，再动 KV，attention 默认不动。

## 回到笔记

- [PagedAttention](/notes/llm注意力内核精读笔记-05-pagedattention与kv显存管理/)
- [RadixAttention](/notes/llm注意力内核精读笔记-08-前缀缓存与kv复用/)
- [EAGLE](/notes/llm推测解码精读笔记-04-eagle-特征空间草稿/)
- [量化模型怎么部署](/notes/llm量化精读笔记-11-系统协同与部署/)
- [FlashAttention](/notes/llm注意力内核精读笔记-02-flashattention-io感知的精确注意力/)
- [AWQ 和 GPTQ](/notes/awq和gptq有什么区别-4bit权重量化怎么选/)
- [FP8 还是 FP4](/notes/企业ai落地选fp8还是fp4-量化格式的成本质量与硬件决策/)
- [vLLM 的 max_tokens](/notes/vllm-gpu-100-percent-max-tokens-trap/)
- [量化精读目录](/series/quantization/)
- [量化速查](/series/quantization-cheatsheet/)

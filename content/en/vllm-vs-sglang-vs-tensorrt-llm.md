---
title: "vLLM, SGLang, or TensorRT-LLM?"
description: "Pick from the workload, not from a missing bake-off. This site has no same-box tokens/s for the three engines. Shared prefixes point at SGLang; broad quantization and PagedAttention point at vLLM; NVIDIA-tuned FP8 and INT4 kernels point at TensorRT-LLM."
date: 2026-10-29T00:00:00+08:00
draft: false
htmlLang: en
zhURL: "/notes/vllm-sglang-tensorrt-llm怎么选/"
weight: 90
---

## Answer

There is no same-machine tokens/s table for these three engines on this site. If many requests share a prefix, start with SGLang’s RadixAttention. If you need the widest set of quantization kernels and PagedAttention, the deployment note calls vLLM one of the production defaults. If you are on NVIDIA GPUs and want their deep FP8 and INT4 kernels, that same note points at TensorRT-LLM. A head-to-head number for your batch and context is still **not measured here**.

## How the Chinese notes separate them

| Question | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| KV layout | PagedAttention. The cited paper’s unmanaged cache was 20.4%–38.2% effective. | Radix tree on top of paged KV. | Not broken out on this site. |
| Shared prefixes | Early sharing is a single prefix plus copy-on-write. | Automatic multi-level cache. Paper figures, not a local rerun: up to 6.4× throughput, 3.7× lower latency, &lt;0.3% miss overhead. | Not measured here. |
| Quantization map in the deployment chapter | AWQ, GPTQ, FP8, MXFP8, FP4, KV quant. Widest ecosystem. | “Same, via several backends.” Structured generation called out. | FP8, INT4/FP4, SmoothQuant W8A8, AWQ, KV quant. NVIDIA-tuned. |
| Speculative decoding | The EAGLE chapter says vLLM integrates EAGLE, including an EAGLE-2-style tree. | Author’s SGLang PR 35423 fixes a quantized-target speculative-decoding bug. Not a speed claim. | Not measured here. |
| A known footgun | Omitting `max_tokens` can decode to `max_model_len` (about 130k on one OCR model) and pin the GPU. | Not the same incident. | Not the same incident. |

QServe’s 1.2–1.4× over TensorRT-LLM (2.4–3.5× on Qwen1.5-72B), quoted in the deployment chapter, is that paper’s result. It is not a three-engine run from this site.

The author’s SGLang issue 35437 records a prefill CUDA-graph OOM whose fix moved TTFT from 179 ms to 113 ms on that setup. That is one patch, not a ranking.

## When to pick which

Shared-prefix traffic (multi-turn, agents, few-shot, a RAG pipeline that repeats a stem): SGLang. Broad quantization formats and a PagedAttention operations story: vLLM. NVIDIA hardware where you specifically want their FP8 or INT4 kernels: TensorRT-LLM. If quality must stay lossless, the deployment chapter says to turn on prefix cache and speculative decoding before any quantization.

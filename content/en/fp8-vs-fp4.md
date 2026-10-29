---
title: "FP8 or FP4 for a production LLM?"
description: "Quantizing only the weights to FP8 does not free the KV cache. FP4 can raise throughput on Blackwell, and field errors in structured output can cost more than the GPUs you saved."
date: 2026-10-29T00:00:00+08:00
draft: false
htmlLang: en
zhURL: "/notes/企业ai落地选fp8还是fp4-量化格式的成本质量与硬件决策/"
weight: 60
---

## Answer

Changing the format changes three things at once: GPU count, tail latency, and field-level failures. FP8 on the weights alone does not hold a concurrency target if the KV cache stays in FP16. FP4 can fit more on Blackwell, and a jump in structured-output errors can erase the GPU savings.

The Chinese essay uses a **worked example**, not a customer dump: a 70B model, 8K context, 32-way concurrency, four H100s in BF16. It says the failure rates in that story are assumed so the arithmetic is visible. Replace them with your own gold set before treating the cost ratio as a result. Order of rollout in that essay: weights, then KV, then attention, each gated alone. Ampere has no FP8 Tensor Core path that pays off; FP4 Tensor Cores show up on Blackwell.

## What to ask a vendor

Not “do you support FP8”. Ask for tokens/s and p95 at your batch and context, where the KV cache sits at 32K and 128K, and what the rollback is when the field error rate misses the bar.

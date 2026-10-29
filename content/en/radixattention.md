---
title: "What is RadixAttention, and when does prefix cache help?"
description: "RadixAttention keeps KV for shared prefixes in a radix tree. The SGLang paper reports up to 6.4× throughput and 3.7× lower latency versus the systems in that paper, with under 0.3% overhead on a miss."
date: 2026-10-29T00:00:00+08:00
draft: false
htmlLang: en
zhURL: "/notes/llm注意力内核精读笔记-08-前缀缓存与kv复用/"
weight: 30
---

## Answer

If two requests share a prefix, the KV for that prefix does not need to be prefilled again. Most servers drop the cache when the request ends. RadixAttention stores those KV tensors in a radix tree (edges can cover a run of tokens), evicts least-recently-used leaves first, and schedules the request with the longest cached prefix next.

A hit rate \(h\) skips about that fraction of prefill compute. The Chinese note cites the SGLang paper’s comparison with vLLM, Guidance, and LMQL on A10G and A100: up to **6.4×** throughput, up to **3.7×** lower latency, hit rates **50%–99%**, and manager overhead **under 0.3%** (0.2 s inside 74.3 s). A one-month Chatbot Arena deployment in that paper saw **74.1%** hits and **1.7×** lower average TTFT on Vicuna-33B, and **52.4%** hits on LLaVA-NeXT-34B. Those figures are the paper’s, not a rerun on this site.

## When it does nothing

Long generations make decode the bill, and unrelated sessions share little. Early vLLM prefix sharing covered a single system prompt; the radix tree is what makes multi-level sharing (system, template, history, branches) automatic. A miss should stay cheap, which is why the paper argues it can be on by default.

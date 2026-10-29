---
title: "MQA, GQA, or MLA: what should you cache per token?"
description: "Multi-query, grouped-query, and multi-head latent attention all shrink the KV cache. They differ in how many distinct K and V vectors each token keeps, and MLA adds a low-rank joint compression."
date: 2026-10-29T00:00:00+08:00
draft: false
htmlLang: en
zhURL: "/notes/llm注意力内核精读笔记-03-注意力头变体-mqa-gqa-mla/"
weight: 70
---

## Answer

The question is what you store per token, not how you compute the score matrix. Multi-head attention keeps a K and a V for every head. Multi-query attention keeps one K/V pair shared by all heads. Grouped-query attention sits between them: several heads share one pair. MLA, as used by DeepSeek-V2 in the Chinese note, compresses K and V together into a low-rank latent.

GQA is the usual compromise when you can up-train from a multi-head checkpoint. MLA’s extra catch is RoPE: the note derives why a plain low-rank latent fights rotary position and why decoupled RoPE is the workaround. The chapter’s 70B and DeepSeek-V2 memory tables are the place to copy a formula, not this page.

## What this does not fix

Fewer KV heads do not remove the HBM traffic of a long prefill score matrix. That is FlashAttention. They also do not reuse a prefix across requests. That is RadixAttention.

---
title: "Why is FlashAttention faster and lighter on memory?"
description: "Standard attention is slow because the N×N score matrix is written to HBM, not because the GPU lacks FLOPs. FlashAttention tiles that work into SRAM and still computes exact softmax."
date: 2026-10-29T00:00:00+08:00
draft: false
htmlLang: en
zhURL: "/notes/llm注意力内核精读笔记-02-flashattention-io感知的精确注意力/"
weight: 10
schemaType: TechArticle
---

## Answer

Standard attention is slow because the N×N score matrix is read and written through HBM, not because the chip is short of FLOPs. FlashAttention never materializes that matrix. It cuts the work into tiles that fit in on-chip SRAM, runs online softmax so the blocked result is still exact, and recomputes in the backward pass instead of storing the big matrix.

## What to remember

Tiling answers the memory question. Online softmax answers the “softmax needs the whole row” question. Recompute answers the training-memory question. The algorithm is exact; the speedup is fewer HBM round trips. The Chinese note walks the IO bound, a hand calculation of the online update, and how FA1, FA2, and FA3 line up with GPU generations.

## When this is the wrong lever

Decode of one token is often bandwidth-bound on the weights and the KV cache, not on an N×N score matrix. Cutting KV heads (MQA, GQA, MLA) or reusing a prefix is a different problem.

---
title: "How does PagedAttention save KV-cache memory?"
description: "The vLLM paper measured 20.4%–38.2% effective KV memory without a manager. PagedAttention maps logical blocks to physical blocks so reservation and fragmentation stop holding the rest of the cache."
date: 2026-10-29T00:00:00+08:00
draft: false
htmlLang: en
zhURL: "/notes/llm注意力内核精读笔记-05-pagedattention与kv显存管理/"
weight: 20
---

## Answer

Most of the KV cache is reserved or fragmented, not live tokens. The vLLM paper, as cited in the Chinese note, measured **20.4%–38.2%** effective KV memory when nothing manages it. The waste is reservation for the maximum length, internal fragmentation inside an allocation, and external fragmentation between requests.

PagedAttention gives each sequence logical blocks and maps them to physical blocks through a block table, the same idea as virtual memory. Copy-on-write lets beam search and parallel samples share a prefix until they diverge. It does not replace FlashAttention: one places KV in memory, the other computes attention without writing the score matrix.

## What this page is not

There is no new utilization number here. The 20.4%–38.2% range is the paper’s measurement, quoted from the Chinese chapter.

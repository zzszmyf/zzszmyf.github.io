---
title: "注意力与计算内核精读"
metaTitle: "注意力与计算内核精读：FlashAttention、PagedAttention、RadixAttention"
description: "注意力与计算内核精读系列目录：复杂度、FlashAttention、MQA/GQA/MLA、PagedAttention、算子融合与前缀缓存。"
date: 2026-10-18T00:00:00+08:00
draft: false
layout: series-hub
seriesHub: attention-kernels
---

从 prefill 和 decode 的复杂度差写起，经过 FlashAttention 和 KV 头变体，到分页 KV、内核融合和 RadixAttention。

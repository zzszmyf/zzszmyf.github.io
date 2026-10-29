---
title: "Why is EAGLE faster than Medusa?"
description: "Medusa guesses future tokens from the current hidden state and tops out near 0.6 acceptance. EAGLE drafts in feature space and feeds the token just sampled, so the regression target is unique."
date: 2026-10-29T00:00:00+08:00
draft: false
htmlLang: en
zhURL: "/notes/llm推测解码精读笔记-04-eagle-特征空间草稿/"
weight: 40
---

## Answer

Medusa’s extra heads look only at the current hidden state and guess several future tokens in parallel. They never see which token was actually sampled in between, so they average over branches. The Chinese note puts that acceptance ceiling near **0.6**.

EAGLE drafts the next feature, not the next token, and concatenates the token just sampled (a shifted-token input). The regression target becomes unique, acceptance goes up, and the first draft token in a round is accepted with probability 1. EAGLE-2 then grows the draft tree from context instead of a static shape. The same chapter notes that vLLM already integrates EAGLE, including an EAGLE-2-style dynamic tree.

## What this page is not

No new tokens/s table. The 0.6 figure and the “first token is always accepted” claim are the chapter’s reading of the papers.

---
title: "Where do the quantization notes start?"
description: "Twelve Chinese chapters from number formats to deployment, plus a short overview. AWQ vs GPTQ and FP8 vs FP4 are the decision essays that sit on top of chapters 05–06 and 03/08/11."
date: 2026-10-29T00:00:00+08:00
draft: false
htmlLang: en
zhURL: "/series/quantization/"
weight: 100
---

## Answer

Read the Chinese series in order if you want the derivations. Start at the overview, then numeric encodings, uniform quantization, formats (FP16 through FP4), granularity and outliers, GPTQ, AWQ, activation quantization, KV-cache quantization, QAT, evaluation, and deployment.

Two shorter essays sit on that map. AWQ versus GPTQ is the 4-bit weight choice (chapters 05 and 06). FP8 versus FP4 is the hardware and structured-output choice (chapters 03, 08, and 11). A one-page Chinese cheat sheet lists the same rules without the proofs.

## What you should not expect here

No new perplexity table. The hub page is the table of contents; each chapter is the source.

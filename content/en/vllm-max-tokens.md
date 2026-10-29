---
title: "Why did a vLLM GPU stay at 100%?"
description: "A vLLM server pegged the GPU because the client omitted max_tokens. vLLM then allowed generation out to max_model_len, about 130k tokens on PaddleOCR-VL, so decode never finished."
date: 2026-10-29T00:00:00+08:00
draft: false
htmlLang: en
zhURL: "/notes/vllm-gpu-100-percent-max-tokens-trap/"
weight: 80
---

## Answer

The GPU stayed at 100% because one request never reached an end condition. Memory looked normal. The client called the OpenAI-compatible API with `temperature=0` and no `max_tokens`. vLLM’s compatibility layer then set the cap to `max_model_len`. On that PaddleOCR-VL deployment the cap was about **130,000** tokens, so decode ran until the server was restarted.

Turning off the multimodal encoder cache (`--mm-processor-cache-gb 0`) made each image slower. It did not remove the stop condition. The fix recorded in the note is to set `max_tokens` on the client and a server-side cap, and to keep the encoder cache if the same image is reused.

## What this page is not

This is one incident on one model. It is not a claim that vLLM is slower than SGLang or TensorRT-LLM.

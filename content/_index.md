---
title: "zzszmyf"
description: "LLM 推理优化与 AI Agent 基础设施 | SGLang Contributor"
---

你好，我是 **孟一凡（Meng Yifan）**，GitHub 用户名 [zzszmyf](https://github.com/zzszmyf)，英文名 doraeMeng。

前百度高级研发、BIGO 资深工程师（广告系统 / 推荐算法 / 搜索），现在创业做 **AI Agent 基础设施**，同时是 [SGLang](https://github.com/sgl-project/sglang) 上游贡献者，专注 LLM 推理优化、推测解码与量化。

---

## 开源项目

| 项目 | 一句话 | Stars |
|---|---|---|
| [codefuse](https://github.com/zzszmyf/codefuse) | 给 AI 编码代理的 grep 替代品：预建代码索引，符号级搜索，噪音最多降 6,596× | ★25 |
| [nexus](https://github.com/zzszmyf/nexus) | 浏览器 / 手机远程操控 tmux 里的 AI 编码 Agent | ★14 |
| [codeact](https://github.com/zzszmyf/codeact) | Code-as-Action Agent 框架，构建于 OpenAI Agents SDK | ★6 |
| [prompt-ctx](https://github.com/zzszmyf/prompt-ctx) | 生产级 LLM 系统提示词的结构化、版本化管理 | ★3 |

## 开源贡献

- [SGLang PR #35423](https://github.com/sgl-project/sglang/pull/35423)：修复 DSpark 推测解码在量化 target lm_head 下的 base logits
- [SGLang issue #35437](https://github.com/sgl-project/sglang/issues/35437)：定位 DFLASH + prefill CUDA graph 在 32GB 显存的 OOM 根因，提供修复方案（TTFT 179ms → 113ms）

## 精选笔记

- [LLM 推理优化精读系列](/notes/)：量化 / 推测解码 / 注意力内核，30 篇
- [vLLM GPU 利用率持续 100% 排查记：一个 max_tokens 参数引发的性能陷阱](/notes/vllm-gpu-100-percent-max-tokens-trap/)
- [MoE 通信：NVLink 与 DeepEP 的工程细节](/notes/moe-nvlink-deepseek-deepep-communication/)
- [Online Softmax 的信息几何](/notes/online-softmax-information-geometry/)
- [五一假期从零写企业 LLM Wiki](/notes/五一假期从零写企业LLM-WiKi/)

## 联系

- GitHub：[zzszmyf](https://github.com/zzszmyf)
- 邮箱：zzszmyf@outlook.com
- RSS：[订阅](/index.xml)

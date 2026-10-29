---
title: "孟一凡（zzszmyf）· LLM 推理优化与 AI Agent 技术笔记"
description: "孟一凡（zzszmyf）的技术知识库与名片：LLM 推理优化精读系列（量化 / 推测解码 / 注意力内核）、AI Agent 工程实践与开源项目。独立 AI 顾问，SGLang 上游贡献者。"
---

你好，我是 **孟一凡（Meng Yifan）**，GitHub 用户名 [zzszmyf](https://github.com/zzszmyf)，英文名 doraeMeng。

前百度/BIGO 广告推荐算法工程师，做过推荐、搜索与用户画像；后来在多家 AI 创业公司负责 Agent 平台、垂直行业 Agent 与推理优化。现在是**独立 AI 顾问**，承接企业 AI 落地项目，同时是 [SGLang](https://github.com/sgl-project/sglang) 上游贡献者。

这里是我系统整理的 **LLM 推理优化精读笔记**，以及 Agent、训练/推理系统和部署踩坑的工程记录。

{{< series-cards >}}

## 精选工程笔记

- [vLLM、SGLang、TensorRT-LLM 该怎么选？](/notes/vllm-sglang-tensorrt-llm怎么选/)
- [vLLM GPU 利用率持续 100% 排查记：一个 max_tokens 参数引发的性能陷阱](/notes/vllm-gpu-100-percent-max-tokens-trap/)
- [MoE 通信：NVLink 与 DeepEP 的工程细节](/notes/moe-nvlink-deepseek-deepep-communication/)
- [Online Softmax 的数学：从代数推导到信息几何](/notes/online-softmax-information-geometry/)
- [五一假期，我花了 3.5 亿 Token 从零写了一套企业 LLM Wiki](/notes/五一假期从零写企业llm-wiki/)
- [用 kimi-cli 实现自动 Issue → Worktree → TDD → PR 的无人值守流水线](/notes/kimi-cli-auto-issue-pipeline/)
- [Claude Code Agent Teams 运行机制深度分析](/notes/claude-code-agent-teams-deep-dive/)

## 开源与贡献

- [codefuse](https://github.com/zzszmyf/codefuse)：给 AI 编码代理的代码虚拟文件系统，符号级索引，噪音最多降 6,596×
- [nexus](https://github.com/zzszmyf/nexus)：浏览器 / 手机远程操控 tmux 里的 AI 编码 Agent
- [codeact](https://github.com/zzszmyf/codeact)：Code-as-Action 框架（OpenAI Agents SDK），数据分析任务交互轮数 −30%
- [prompt-ctx](https://github.com/zzszmyf/prompt-ctx)：生产级 LLM 系统提示词的结构化、版本化管理
- SGLang：[PR #35423](https://github.com/sgl-project/sglang/pull/35423)（推测解码量化 target 修复）、[issue #35437](https://github.com/sgl-project/sglang/issues/35437)（prefill CUDA graph OOM 根因，TTFT 179ms → 113ms）

## 研究与分享

- [arXiv:2503.04826](https://arxiv.org/abs/2503.04826)：Training-Free 少样本 3D 医学分割框架，基准 SOTA，被 NeurIPS 2025 Spotlight 论文引用
- GIAC 2025：《提升 LLM 推理能力的 Reward System 设计》
- 关注方向：VLM 知识蒸馏、自动标注、Token-aware Batching、推理优化；申请发明专利 1 项

## 合作与联系

目前以**独立顾问**身份承接企业 AI 落地项目：Agent 平台与垂直行业 Agent、LLM 推理性能与成本优化。如果你正在解决类似问题，欢迎邮件联系。

- 邮箱：zzszmyf@outlook.com
- GitHub：[@zzszmyf](https://github.com/zzszmyf)
- RSS：[订阅](/index.xml)

---
title: "zzszmyf"
description: "LLM 推理优化与 AI Agent 基础设施 | SGLang Contributor"
---

你好，我是 **孟一凡（Meng Yifan）**，GitHub 用户名 [zzszmyf](https://github.com/zzszmyf)，英文名 doraeMeng。

前百度/BIGO 算法工程师，两家 AI 创业公司的算法/技术负责人，现在是 [SGLang](https://github.com/sgl-project/sglang) 上游贡献者。专注 **LLM 推理优化、AI Agent 基础设施与多模态**，从推荐/广告系统一路做到 Agent 平台与推理内核。

---

## 经历

| 时间 | 角色 |
|---|---|
| 2026.01–2026.06 | Supio.AI · AI 技术专家（律所文档智能 Agent、Rubric 质量评估） |
| 2025.07–2025.11 | NetMind.xyz · Agent 技术平台负责人（A2A 协议、AgentWorkspace、沙箱体系） |
| 2023.09–2025.05 | 人生旷野（红杉天使轮）· 算法专家（LLM 对话策略、GRPO 微调、推理优化） |
| 2022.06–2023.04 | Allsec Technologies · 产品与技术负责人 |
| 2020.04–2022.06 | BIGO · 资深算法工程师（RTB 竞价、多路召回、系统稳定性） |
| 2017.10–2020.04 | 百度 · 高级算法工程师（推荐系统、对话系统、GNN） |

## 开源项目

| 项目 | 一句话 | Stars |
|---|---|---|
| [codefuse](https://github.com/zzszmyf/codefuse) | 给 AI 编码代理的代码虚拟文件系统：符号级索引，噪音最多降 6,596× | ![stars](https://img.shields.io/github/stars/zzszmyf/codefuse?style=social&label=) |
| [nexus](https://github.com/zzszmyf/nexus) | 浏览器 / 手机远程操控 tmux 里的 AI 编码 Agent | ![stars](https://img.shields.io/github/stars/zzszmyf/nexus?style=social&label=) |
| [codeact](https://github.com/zzszmyf/codeact) | Code-as-Action 框架（OpenAI Agents SDK），数据分析任务交互轮数 −30% | ![stars](https://img.shields.io/github/stars/zzszmyf/codeact?style=social&label=) |
| [prompt-ctx](https://github.com/zzszmyf/prompt-ctx) | 生产级 LLM 系统提示词的结构化、版本化管理 | ![stars](https://img.shields.io/github/stars/zzszmyf/prompt-ctx?style=social&label=) |

## 开源贡献

- [SGLang PR #35423](https://github.com/sgl-project/sglang/pull/35423)：修复 DSpark 推测解码在量化 target lm_head 下的 base logits
- [SGLang issue #35437](https://github.com/sgl-project/sglang/issues/35437)：定位 DFLASH + prefill CUDA graph 在 32GB 显存的 OOM 根因，TTFT 179ms → 113ms

## 研究与分享

- **arXiv:2503.04826**：Training-Free 少样本 3D 医学分割框架（SAM2 视频化），基准 SOTA，被 NeurIPS 2025 Spotlight 论文引用
- **GIAC 2025**：分享《提升 LLM 推理能力的 Reward System 设计》
- 技术主题：VLM 知识蒸馏（Gemini → 轻量模型）、LocatAnything + Grounding-DINO 自动标注、Token-aware Batching 推理优化、申请发明专利 1 项

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

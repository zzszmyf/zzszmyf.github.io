---
title: "zzszmyf"
description: "LLM 推理优化与 AI Agent 基础设施 | SGLang Contributor"
---

你好，我是 **孟一凡（Meng Yifan）**，GitHub 用户名 [zzszmyf](https://github.com/zzszmyf)，英文名 doraeMeng。

从百度/BIGO 的推荐与广告算法工程师，到 AI 创业公司的产品与技术负责人（众安天下）、Agent Tech Lead（NetMind）与算法/AI 专家（人生旷野 · Supio，IC），我一路做到 **LLM 推理优化、AI Agent 基础设施与多模态**：主导过生产级 Agent 平台（A2A 协议、沙箱执行、持久化工作区），落地过垂直行业 Agent（法律文档、生活陪伴、多模态创作），也是 [SGLang](https://github.com/sgl-project/sglang) 上游贡献者，有论文被 NeurIPS 2025 Spotlight 引用。

---

## 经历

| 时间 | 公司 / 角色 | 公司核心业务 | 我的贡献 | 与 LLM / 多模态 / Agent 的关联 · 能解决什么问题 |
|---|---|---|---|---|
| 2026.01–06 | Supio.AI · AI 技术专家 | 面向原告律师的法律 AI 平台（西雅图，融资 $60M） | Mailroom 文档智能 Agent；医疗记录分段摘要 Agent 支撑 Demand Letter 生成；6 维 Rubric Grader + Node.js DOCX 引擎 + 质量回归闭环 | 垂直行业 Agent 落地的完整范式：复杂文档理解、长文档摘要、生成质量评估（LLM-as-Judge）与生产质量闭环。能解决：法律/医疗/金融等垂直场景的 LLM 落地与「生成质量可保障」问题 |
| 2025.07–11 | NetMind.xyz · Agent Tech Lead | NetMind.AI：AI×区块链，2,000+ 全球闲置 GPU、统一 API 接入 200+ 模型；XYZ 平台 Agent 代币经济（$NMT）、Life Agent「Zoey」 | A2A 协议 Agent 间通信框架；AgentWorkspace（S3+FUSE 文件级同步）；毫秒级 microVM 沙箱（Blaxel 借阅模型 + 热启动池）；PromptDecision/ToolsDecision 动态上下文；可观测性与 Prompt 托管（A/B 测试） | 生产级 Agent 平台的全部关键件：多 Agent 通信（A2A/MCP）、持久化工作区、沙箱安全执行、上下文与 Token 预算管理、可观测性。能解决：把 Agent 从 Demo 带到生产、支撑多 Agent 协作的企业级平台建设 |
| 2023.09–2025.05 | 人生旷野（红杉中国天使轮）· 算法专家 | AI 大模型初创，人称「中国版 Inflection AI」：生活陪伴式人机交互 | 对话策略三阶段演化（Function Call→标签体系→RAG）；NPC 双记忆（Redis 短期 + Mem0 长期）；GRPO 微调 Qwen2.5/Llama3.1 训练 CoT；Token-aware Batching（vLLM 风格）；多模态表情包创作 Agent 服务 C 端 | 直接覆盖 LLM Agent 核心技术栈：对话策略、记忆系统、RL 后训练（GRPO）、推理优化、多模态生成闭环。能解决：从 0 到 1 构建对话/陪伴/多模态创作类 Agent 产品，兼顾质量、成本与吞吐 |
| 2022.06–2023.04 | 众安天下（Allsec Technologies）· 产品与技术负责人（10+ 人团队） | 网络安全实战化攻防：安全众测、威胁监测、攻防演练（工信部 CAPPVD 支撑单位） | 工信部「工联众测」平台研发与交付；电商 AI 产品（尺码表/标题生成）；OSINT 开源情报与用户画像；分布式靶场 + 流量审计 + 异常检测；CI/CD——云资源成本节约 90% | OSINT = 多模态情报抽取；安全众测方法论 = LLM/Agent 红队、越狱测试与 Prompt 注入防护；电商 AI = 内容生成工程化。能解决：Agent 安全评估与合规、防注入的可靠 Agent 系统、内容生成类产品落地 |
| 2020.04–2022.06 | BIGO · 资深算法工程师 | 全球直播/短视频/社交平台（Bigo Live、Likee，覆盖 150+ 国家） | RTB 竞价 + 多路召回（eCPM 序列学习、HNSW×Cross-Attention）：消耗期望 +50%、耗时 −30ms、召回 ×3；BudgetControl+PID：超投 −88%、达成率 6%→32.2%；迁移学习：点赞率 +190%、关注率 +191%；排序 CPU −50%、可用性 96%→99% | 实时竞价/多路召回/PID 控制与 Agent 路由、工具选择、Token 成本控制同构；高并发系统稳定性经验直接适用于 LLM 推理服务。能解决：LLM 应用的商业化与成本控制、Agent 调用路由与效果优化、推理系统稳定性 |
| 2017.10–2020.04 | 百度 · 高级算法工程师 | AI 驱动的搜索与信息流生态 + 智能云 + 自动驾驶 | 保险领域多轮对话 Agent（FSM 策略框架）；信息流推荐多目标融合 + 在线 Debias；作者生态 GNN 建模；直播分发：icon 展现 +615%（440 万 DAU）；画像平台：feed 覆盖 8200 万 DAU、金融人群 250 万 | FSM 对话系统是 Agent 的早期形态；推荐/搜索/画像方法论可迁移到 LLM 个性化、用户上下文工程与 RAG 重排。能解决：构建个性化 LLM 应用与 Agent 体验，复用搜索/推荐/画像的成熟方法论 |

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
- [五一假期从零写企业 LLM Wiki](/notes/五一假期从零写企业llm-wiki/)

## 联系

- GitHub：[zzszmyf](https://github.com/zzszmyf)
- 邮箱：zzszmyf@outlook.com
- RSS：[订阅](/index.xml)

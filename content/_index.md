---
title: "zzszmyf"
description: "LLM 推理优化与 AI Agent 基础设施 | SGLang Contributor"
---

你好，我是 **孟一凡（Meng Yifan）**，GitHub 用户名 [zzszmyf](https://github.com/zzszmyf)，英文名 doraeMeng。

前百度/BIGO 算法工程师，两家 AI 创业公司的算法/技术负责人，现在是 [SGLang](https://github.com/sgl-project/sglang) 上游贡献者。专注 **LLM 推理优化、AI Agent 基础设施与多模态**，从推荐/广告系统一路做到 Agent 平台与推理内核。

---

## 经历

### 百度 · 高级算法工程师（2017.10–2020.04）
**公司**：AI 驱动的搜索与信息流生态，叠加智能云与自动驾驶
**我做的**：保险领域多轮对话 Agent（FSM 策略框架 + NLP 意图识别）；百度 APP 信息流推荐（多目标融合、在线 Debias、演化策略）；作者生态 GNN 建模；直播个性化分发——直播 icon 展现用户量 +615%，覆盖手百小视频 DAU 440 万；用户画像平台——feed 画像覆盖 8200 万 DAU、金融从业者人群挖掘 250 万

### BIGO · 资深算法工程师（2020.04–2022.06）
**公司**：全球化直播/短视频/社交平台（Bigo Live、Likee，覆盖 150+ 国家）
**我做的**：RTB 程序化广告竞价与多路召回（eCPM 序列学习、HNSW×Cross-Attention）——广告消耗期望 +50%、全链路耗时 −30ms、召回数 ×3；BudgetControl+PID 智能投放——双倍超投占比 −88%、订单目标达成率 6%→32.2%；迁移学习——点赞率 +190%、关注率 +191%；系统稳定性——排序模块 CPU −50%、可用性 96%→99%

### 众安天下（Allsec Technologies）· 产品与技术负责人（2022.06–2023.04）
**公司**：网络安全实战化攻防解决方案：安全众测、威胁监测、钓鱼/攻防演练（工信部 CAPPVD 支撑单位）
**我做的**：领导 10+ 人技术团队；工信部「工联众测」平台主平台研发与交付；电商 AI 产品（尺码表生成、商品标题生成）；OSINT 开源情报与跨区域用户画像；分布式靶场 + 流量审计 + 异常检测；CI/CD 与研发效能——云资源成本节约 90%

### 人生旷野（红杉中国天使轮）· 算法专家（2023.09–2025.05）
**公司**：AI 大模型初创，人称「中国版 Inflection AI」：生活陪伴式人机交互
**我做的**：LLM 对话策略三阶段演化（Function Call+槽位填充 → 标签体系 → RAG）；NPC 双记忆系统（Redis 短期工作记忆 + Mem0 长期语义记忆）；GRPO 微调 Qwen2.5/Llama3.1 训练 CoT 思维链；Token-aware Batching 推理优化（参考 vLLM）提升离线标签挖掘吞吐；多模态表情包创作 Agent 服务 C 端真实用户

### NetMind.xyz · Agent 技术平台负责人（2025.07–2025.11）
**公司**：NetMind.AI —— AI × 区块链：整合 2,000+ 全球闲置 GPU 的去中心化算力，统一 API 接入 200+ 模型；XYZ 平台支持 AI Agent 创建、代币发行与 DEX 交易（$NMT），首个 Life Agent「Zoey」
**我做的**：作为 Agent 技术平台负责人（虚线管理 10 人）主导多模态 Agent 平台：A2A 协议 Agent 间通信框架；意图识别与动态路由（房地产/健康等垂直场景）；AgentWorkspace 持久化工作区（S3 + FUSE 文件系统级同步）；毫秒级 microVM 沙箱体系（Blaxel「借阅模型」+ 热启动池）；PromptDecision/ToolsDecision 动态上下文工程；Agent 可观测性与 Prompt 托管（版本控制 + A/B 测试）

### Supio.AI · AI 技术专家（2026.01–2026.06）
**公司**：面向原告律师的法律 AI 平台（案件文档自动化、agentic AI，西雅图，融资 $60M）
**我做的**：Mailroom 文档智能 Agent（邮件/附件/OCR 多源输入、实体提取、案件聚类、动态路由）；医疗记录分段式摘要 Agent 支撑 Demand Letter 生成；6 维 Rubric 文书质量 Grader + 纯 Node.js DOCX 解析引擎 + 动态质量回归闭环

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

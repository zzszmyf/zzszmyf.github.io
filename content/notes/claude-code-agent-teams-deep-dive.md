---
title: "Claude Code Agent Teams 运行机制深度分析"
date: 2024-04-02T00:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/claude-code-agent-teams-deep-dive/"]
categories: ["研究笔记"]
tags: ["研究笔记", "Reinforcement Learning", "Agent Tools"]
---

# Claude Code Agent Teams 运行机制深度分析

> 原文链接：https://zhuanlan.zhihu.com/p/2011414794905859760
> 作者：Meta（知乎知识会员）
> 首发于：LLM应用技术指北
> 收录于：LLM应用技术指北专栏
> 发布时间：2026-03-08 16:30

Claude Code 新增了 Agent Teams 功能，为多智能体协作提供了强大的框架。本文将对其运行机制进行详细解析。我们会以代码审查案例来演示其实际应用。

---

## Agent Teams 与 Subagent：区别与联系

在 Claude Code 里，Agent Teams 和 Subagent 都是分解任务的核心机制，但设计理念和使用场景不同。

### 概念与定位

- **Agent Teams**："重量级"协作模式，适合多智能体长期、并行协作的复杂问题。有明确角色（Leader、Teammate）、共享资源（TaskList、Mailbox）和完整生命周期。
- **Subagent**："轻量级"委托执行机制，本质是一个"瞬时子进程"，没有团队上下文，执行完即销毁。

### 核心差异对比

| 维度 | Agent Teams | Subagent |
|------|-------------|----------|
| **上下文与状态** | 持久化、有状态。团队成员有独立上下文，团队状态可被所有成员感知。 | 无状态、瞬时。上下文仅限于单次调用输入。 |
| **生命周期** | 长期。从创建到解散，支持多轮交互和任务迭代。 | 瞬时。一次调用和返回，执行完即销毁。 |
| **协作与通信** | 多对多协作。成员间可直接通信，通过共享配置文件和 TaskList 协同。 | 一对一委托。唯一通信路径是父 Agent 调用，Subagent 返回结果。 |
| **工具与管控** | 依赖整套专用工具链，提供丰富的团队治理和任务编排能力。 | 仅依赖 Task 工具（不带 team_name），管控全由父 Agent 负责。 |
| **适用场景** | 需要并行探索、多视角分析的复杂问题（如代码审查）。需要多个 Agent 协同完成的任务。需要长期追踪和管理子任务的场景。 | 外包单一、明确的子功能。创建可复用的"智能工具"。需要函数式、输入输出明确的处理单元。 |

### 如何选择

1. **需要"讨论"时选 Agent Teams**：子任务执行者间需要共享发现、传递中间结果。
2. **任务是"委托"时选 Subagent**：父 Agent 只关心最终结果，不需要持续协作。
3. **根据"耦合度"决定**：高度独立的子任务适合 Subagent，有依赖关系的适合 Agent Teams。

### 总结

两者底层都基于 Task 工具，区别在于是否提供 team_name 参数（激活"团队模式"的开关）。Subagent 是"轻量的一次性执行层"，Agent Teams 是"带治理的长期协作层"。它们可以嵌套使用，构成灵活强大的智能体编排能力。

---

## Agent Teams：并行协作的智能体编排范式

Claude Code 的 Agent Teams 是一个多智能体协作框架，允许一个主导 Agent（Leader）创建并管理多个子 Agent（Teammate）组成的团队，并行完成复杂任务。

**核心思想**：将大任务（如代码库全面审查）拆成多个子任务（如安全性、可维护性审查），每个子任务派一个专门的 Teammate 并行处理，最后由 Leader 汇总结果，大幅提升效率。

### 核心概念与系统角色

Agent Teams 运行围绕以下核心概念：

| 概念/角色 | 描述 | 在轨迹中的体现 |
|-----------|------|----------------|
| **Team (团队)** | 临时的 Agent 集合，为完成特定目标（如 code-review）而创建，有唯一的 team_name。 | 通过 TeamCreate 工具创建，生成 code-review 团队。 |
| **Leader (领导者)** | 发起和管理团队的主 Agent，负责创建团队、分任务、盯进度、汇总结果、解散团队。 | - |
| **Teammate (团队成员)** | Leader 创建的子 Agent，负责执行具体子任务，每个都有独立上下文和工具集。 | maintainability-reviewer、security-reviewer 等 5 个并行运行的 Agent。 |
| **TaskList (任务列表)** | 团队共享的任务管理中心，存储所有子任务及其状态（pending, in_progress, completed）、负责人等信息。 | 通过 TaskCreate、TaskList、TaskUpdate 等工具交互。 |
| **Mailbox (邮箱)** | Agent 间的通信机制，Teammate 用它向 Leader 发报告、空闲通知；Leader 用它向 Teammate 发关闭请求。 | 体现为 teammate-message 和 SendMessage 工具调用。 |
| **团队生命周期** | 从创建到解散的完整流程：TeamCreate → 并行 Task → TaskUpdate → 汇总 → SendMessage(shutdown_request) → TeamDelete。 | 完整覆盖从团队创建到最终清理的全过程。 |

### Agent Teams 总体架构

Leader 是中心协调员，通过共享的 TaskList 和 Mailbox 与并行工作的 Teammates 互动，Teammates 通过读取共享的 Team Config 发现彼此，构成完整的协作网络。

---

## Agent Teams 工具链解析

Agent Teams 的高效运转依赖于一套覆盖团队与任务管理全生命周期的工具链。

### 核心工具详解

| 工具 | 作用 | 典型参数 | 使用时机与约束 |
|------|------|----------|----------------|
| **TeamCreate** | 创建团队：初始化新团队，创建关联的共享资源（任务列表、配置文件）。 | team_name, description | 协作开始的第一步，team_name 需唯一。 |
| **Task** | 启动 Teammate：以子进程形式启动专用的子 Agent (Teammate)。 | subagent_type, prompt, run_in_background, name, team_name | 创建团队和任务后，用于并行化执行。 |
| **TaskCreate** | 创建任务：在团队的共享任务列表中定义新任务。 | subject, description, activeForm | 启动 Teammate 前，明确需要完成的工作项。 |
| **TaskList** | 查看任务列表：获取当前团队所有任务的概览（ID, 状态, 所有者等）。 | 无 | 用于同步状态，了解团队整体进度。 |
| **TaskUpdate** | 更新任务状态：修改任务的属性，如状态或分配所有者。 | taskId, status, owner | Leader 用它分配任务；Teammate 用它标记任务完成。 |
| **TaskGet** | 获取任务详情：根据任务 ID 获取任务的完整描述和上下文。 | taskId | Teammate 开始工作前，获取具体要求。 |
| **SendMessage** | 内部通信：在 Agent 之间发送消息或协议请求。 | type (message, broadcast, shutdown_request), recipient, content | 用于 Leader 与 Teammate 间的指令传递和状态同步。 |
| **TeamDelete** | 解散团队：清理团队的所有相关资源（目录、配置文件）。 | 无 | 任务全部完成后调用，必须等所有 Teammate 关闭。 |
| **Read** | 读取文件：通用的文件读取工具。 | file_path | Teammates 用来读取团队配置文件，发现其他成员。 |

### 初始化与关闭的典型范式

#### 初始化范式

1. **创建团队**：使用 TeamCreate 定义团队名称和目标。
2. **定义任务清单**：使用 TaskCreate 为每个子任务创建条目。
3. **并行启动 Teammates**：为每个任务并行调用 Task 工具，设置 run_in_background: true。
4. **分配任务所有权**：使用 TaskUpdate 将任务分配给对应的 Teammate，状态改为 in_progress。

#### 关闭范式

1. **请求关闭**：所有任务完成后，Leader 用 SendMessage 给每个 Teammate 发 shutdown_request。
2. **等待确认**：Teammates 收到请求后，发 shutdown_approved 消息并自行终止。
3. **删除团队**：确认所有 Teammate 关闭后，Leader 调用 TeamDelete 清理资源。

**关键点**：团队关闭是"优雅"的过程，必须等所有成员退出后才能解散团队，避免正在执行的任务被中断。

---

## Agent Teams 运行机制深度拆解

### 接口外观与角色定义

最外层通过语义明确的工具集（TeamCreate, Task, SendMessage 等）和清晰的角色划分（Leader, Teammate）暴露能力。工具构成了 Leader 与系统交互的 API，每个工具对应一个高内聚的原子操作。Leader 是"指挥官"，Teammate 是"士兵"，分工明确，降低单个 Agent 的心智负担。

### 核心逻辑：并行协作与任务编排

核心是并行协作与任务编排能力。通过 Task 工具的 run_in_background: true 参数，Leader 可以非阻塞地启动多个 Teammate，实现真正的并行计算。共享的 TaskList 充当"中央公告板"，所有 Agent 都可以读取和更新任务状态，实现基于状态的松耦合协作。当所有 Teammate 完成工作后，Leader 通过轮询 TaskList 确认所有任务状态均为 completed，然后进入下一步。

### 生态与开发者体验

Agent Teams 包含一系列提升协作效率和开发者体验的机制：

- **自动消息派发**：消息会自动推送给 Leader，无需手动轮询。
- **空闲通知**：Teammate 完成当前工作后，会自动发送空闲通知，为 Leader 提供动态调度机会。
- **任务所有权**：避免多个 Teammate 抢占同一个任务的竞态条件，也为工作追溯和责任划分提供依据。

---

## 总结

Claude Code Agent Teams 是一个设计精良、层次分明的多智能体协作系统，通过清晰的接口、强大的并行编排能力、便捷的协作机制和扎实的底层架构，为解决复杂的软件工程任务提供了高效、可扩展的范式。

### 参考
- https://code.claude.com/docs/en/agent-teams

---

*创作声明：包含 AI 辅助创作*

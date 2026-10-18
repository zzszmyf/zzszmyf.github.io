---
title: "Claude Code Agent Teams 运行机制深度分析"
description: "Claude Code Agent Teams 深度解析：多智能体协作框架与 Subagent 的区别对比，Leader/Teammate 角色、TaskList 与 Mailbox 通信机制、工具链与团队生命周期，并用代码审查案例演示多个 Agent 并行协作的实际运行流程。"
date: 2024-04-02T00:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/claude-code-agent-teams-deep-dive/"]
categories: ["研究笔记"]
tags: ["研究笔记", "Reinforcement Learning", "Agent Tools"]
---

这篇内容根据知乎专栏文章整理，作者 Meta（知乎知识会员），首发于“LLM应用技术指北”，发布时间为 2026-03-08。原文链接是 https://zhuanlan.zhihu.com/p/2011414794905859760，参考文档是 https://code.claude.com/docs/en/agent-teams 。下面的拆解以文中给出的工具名和一次 code-review 轨迹为准。

把一段代码审查任务丢给单个 Subagent，父 Agent 只需要等一个结果；如果换成五个 reviewer 并行，父 Agent 要回答的问题就变了：谁负责哪块、做到哪一步、发现要不要互相传递、什么时候可以关掉团队。Claude Code 的 Agent Teams 就是把这几个问题做成了工具链。

## 两套分解任务的机制

Subagent 走的是 Task 工具不带 `team_name` 的那条路径。父 Agent 给出 `subagent_type` 和 `prompt`，子 Agent 在自己的上下文里执行，返回结果后被销毁；它没有团队上下文，上下文只覆盖这一次调用。调用关系是一对一，管控全部留在父 Agent 手里，适合边界清楚、输入输出明确的子任务。

Agent Teams 则先用 `TeamCreate` 建一个带唯一 `team_name` 和 `description` 的团队，再创建一组共享资源：TaskList 和团队配置文件。Leader 之外的每个 Teammate 都是独立的 Agent，有自己的上下文和工具集；它们能用 `Read` 读取共享配置、发现彼此，围绕同一个任务列表协作，生命周期从创建一直延续到显式解散。示例轨迹里同时跑了 `maintainability-reviewer`、`security-reviewer` 等五个 reviewer，每个都按自己的审查维度工作。

两者底层都落在 Task 工具上，区别是调用时有没有 `team_name`。这个参数像一个开关：不带它，Task 只是一次委托；带上它，Task 会把子 Agent 放进团队，给它团队身份和共享资源。判断该用哪种，取决于子任务之间要不要交换中间发现，任务大小不是主要判据。安全 reviewer 如果发现一个鉴权绕过，维护性 reviewer 需不需要立刻知道，决定了它们应该并行独立跑，还是应该进同一个团队。

## 共享状态放在哪里

团队模式把协调状态从某个 Agent 的上下文里搬到了 TaskList 和团队配置上。TaskList 保存每个子任务的 subject、description、状态和 owner，状态只有 pending、in_progress、completed 三种。`TaskCreate` 用 subject、description、activeForm 建任务，`TaskGet` 按 taskId 取完整描述，`TaskUpdate` 改状态或换 owner，`TaskList` 拉当前团队的任务概览。

这个设计的直接后果是状态可被所有成员检查。Leader 不需要记住“第三个 reviewer 做到哪了”，它读 TaskList 就行；Teammate 开始工作前用 `TaskGet` 拿到任务上下文，不必让 Leader 在 prompt 里重复一遍。owner 字段还承担了两个职责：避免两个 Teammate 同时抢同一个任务，以及在出问题时能追溯到是谁改的状态。对于并行审查这种读多写少的任务，这比让 Agent 之间直接互发消息更稳。

共享状态也有代价。TaskList 是中心，所有成员都要读写它，任务拆得太碎时，状态同步本身会占掉一部分注意力；任务描述写得不清楚时，`TaskGet` 取回来的只是一段模糊的 subject，Teammate 仍然要回去问 Leader。工具链能保证状态一致，保证不了任务定义的质量。

## 工具链的调用顺序

初始化阶段有一条比较固定的顺序。`TeamCreate` 先建团队和共享资源，`TaskCreate` 把大任务拆成可执行的条目，`Task` 用 `run_in_background: true` 并行拉起 Teammate，`TaskUpdate` 再把具体任务分配给对应的 owner 并改成 in_progress。`Task` 的参数里除了 `subagent_type` 和 `prompt`，还有 `name` 和 `team_name`；前者给 Teammate 一个可识别的名字，后者把它挂到团队上。

关闭阶段需要按协议走，不能直接 kill 进程。所有任务完成后，Leader 用 `SendMessage` 给每个 Teammate 发 `shutdown_request`；Teammate 回 `shutdown_approved` 并自行终止；确认所有成员都退出后，Leader 才调用 `TeamDelete` 清理目录和配置文件。这个顺序保证了正在执行的任务不会在中途被删掉，代价是 Leader 必须等待每个成员确认，收尾时间会变长。

工具定义里 `SendMessage` 的 `type` 有 message、broadcast、shutdown_request 三种，`recipient` 和 `content` 分别指定收件人和内容。`broadcast` 适合发全局约束，比如“所有人只读，不要改工作区”；`message` 适合点对点传递发现；`shutdown_request` 只应该由 Leader 在收尾阶段发。把这些语义分开，比让所有 Agent 共享一个自由文本频道更容易控制。

## 消息推送与任务轮询

Mailbox 是消息通道的共享存储，Teammate 用它发报告和空闲通知，Leader 用它发关闭请求，`SendMessage` 是读写 Mailbox 的工具。文中提到消息会自动推送给 Leader，不需要手动轮询；同时又提到 Leader 会轮询 TaskList 来确认所有任务都是 completed。这两句话描述的是两条通道，不能混在一起理解：消息通道负责把 Teammate 的报告和空闲通知送到 Leader，任务通道负责维护“这件事到底完成没有”的状态。空闲通知让 Leader 知道某个 reviewer 已经空出手，可以动态补任务；任务状态仍要以 TaskList 为准，因为消息可能只说明“我这边告一段落”，不代表对应任务已经被标记完成。

这套机制在任务数量少的时候很自然。每人一个任务，Leader 轮询几次就能收齐结果；空闲通知还能让它把新发现的问题追加成任务，再派给已经空下来的成员。任务数量上去以后，轮询和消息会同时变多，Leader 的上下文会被各种状态更新占满，这时候要么限制团队规模，要么把任务粒度调粗，减少协调次数。

## 什么场景值得开团队

需要多视角并行探索的任务适合 Agent Teams：代码审查、方案对比、跨模块排查，几个 reviewer 可以从安全、可维护性、性能不同角度同时看同一份代码，彼此不需要等待。任务之间有依赖、需要共享中间发现时，团队也比一串串行 Subagent 更合适，因为 TaskList 和 Mailbox 提供了传递状态的地方。

边界清楚、一次调用就能完成的任务更适合 Subagent。让一个 Subagent 去改一个函数、跑一组测试、查一个符号的引用，父 Agent 只关心最终结果，不需要团队身份、任务所有权和关闭协议；开团队反而多出建队、分配、收尾三次开销。写操作多的任务还要额外小心：多个 Teammate 同时改同一片代码，TaskList 的 owner 只能防止任务被重复领取，防不住文件级冲突，切分工作区仍然要由 Leader 事先做好。

## 还没被公开材料覆盖的部分

这套机制能确认的是工具名、调用顺序、任务状态和消息类型；团队配置的具体格式、上下文压缩之后状态怎么恢复、某个 Teammate 进程崩掉后任务怎么重新分配，公开材料都没有给出细节。这些恰好是生产环境最关心的部分，因为没有崩溃恢复，团队规模一大，Leader 就会变成单点。

如果要在真实仓库里开团队，先量两笔账：每个 reviewer 独立读一遍代码带来的 token 成本，以及汇总五份报告时 Leader 需要处理的冲突数量。TaskList 能协调任务状态，重复读代码的 token 成本不会因为进团队而消失。

*说明：本文在原文基础上做了结构与表述整理，包含 AI 辅助创作。*

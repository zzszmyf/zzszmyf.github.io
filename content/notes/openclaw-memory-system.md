---
title: "随便聊聊OpenClaw记忆系统"
date: 2024-04-10T00:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/openclaw-memory-system/"]
categories: ["研究笔记"]
tags: ["研究笔记", "Memory System"]
---

# 随便聊聊OpenClaw记忆系统

> 原文链接：https://zhuanlan.zhihu.com/p/2005632275774195031
> 作者：Eternity（阿里巴巴 计算机图形算法工程师）
> 发布时间：2026-02-13 14:39

一晃已经接近一年没有写技术文章了。最近终于抽出一些时间，打算写点什么。前段时间，ClawdBot 在社区中迅速走红，围绕它的"技术解析"也随之大量出现。 笔者从代码本身出发，随便聊聊OpenClaw的记忆系统，也防止自己日后遗忘。

OpenClaw 的记忆系统并没有采用复杂的技术方案，例如向量数据库，而是使用了相对朴素的 Markdown 文件作为存储介质。这种设计的一个明显优势在于：所有内容都是纯文本，开发者可以直接阅读、修改，也可以通过 Git 进行版本管理。相比抽象化程度更高的存储方案，这种方式更加透明，也更容易理解和调试。

## 记忆系统文件构成

记忆系统核心工作区由三类主要文件构成，它们在加载方式、更新策略和安全边界上各不相同：

### MEMORY.md

位于工作区根目录，代表智能体经过整理的"长期记忆"。其中存储高层决策、用户偏好以及具有持久性的事实信息。需要强调的是，该文件仅在主会话（即与人类所有者的直接对话）中加载，在 Discord 或群聊等共享场景中会被严格排除，以防止敏感信息泄露。

### Daily Logs（memory/YYYY-MM-DD.md）

位于 memory/ 子目录下，这些文件相当于智能体的工作记忆或"思维流"。它们记录日常笔记和持续更新的上下文信息。系统会自动在每个会话中加载当天和前一天的日志，以提供最近的上下文支持。

### Session Archives（memory/YYYY-MM-DD-{slug}.md）

同样位于 memory/ 目录中，这些文件是对过往会话的静态归档。文件名中包含由大语言模型生成的描述性 "slug"（例如 vendor-pitch）。与每日日志不同，这类归档不会被自动加载；只有在智能体显式调用检索工具查找历史信息时，才会被访问。

## 记忆文件更新机制

OpenClaw 在记忆更新机制上的核心思想是：**不通过固定规则决定写入行为，而是通过 Prompt 原则引导智能体自主决策**。智能体拥有完整的文件读写权限，是否写入、写入哪里，取决于：

- 当前上下文
- System Prompt 中的行为约定
- 用户显式指令

### 写入触发方式

#### 手动触发

当用户明确要求记住某件事时，智能体主动调用文件写入工具更新 MEMORY.md 或 memory/YYYY-MM-DD.md 。

#### 半自动触发

当上下文接近压缩上限时，系统会触发一次"预压缩记忆刷新"：

1. 检查权限确认（跳过沙箱、命令行模式、心跳模式）
2. 系统暂停、插入整理步骤
3. 整个过程对用户无感知

与传统上下文压缩不同，它在删除旧消息前给予模型一次持久化机会，从而避免重要信息被直接丢弃。比如，它会保存具体的手机号码，而不是压缩为："用户提到一个手机号码"。

系统提示词示例：

```
"Pre-compaction memory flush turn.", 
"The session is near auto-compaction; capture durable memories to disk.", 
You may reply, but usually ${SILENT_REPLY_TOKEN} is correct.
```

#### 全自动触发

当用户执行 `/new` 时，会触发 session-memory hook。将上一对话（默认最近15条原始消息）保存到 md 文件中。调用 LLM 生成 slug，命名对应 markdown 文件：YYYY-MM-DD-{slug}.md，否则使用时间戳：YYYY-MM-DD-{HHMM}.md。

Slug 生成提示词：

```
Based on this conversation, generate a short 1-2 word filename slug (lowercase, hyphen-separated, no file extension.
Reply with ONLY the slug, nothing else. Examples: "vendor-pitch", "api-design", "bug-fix"
```

## 记忆系统工具

在检索记忆时，OpenClaw 提供 2 个工具：

### memory_search

使用混合搜索（向量搜索、BM25 精确匹配，通过权重控制二者）在所有记忆文件（MEMORY.md + memory/*.md）中进行搜索，工具将返回最相关的代码片段、文件路径和行号。

### memory_get

在 memory_search 的基础上，智能体能够使用 memory_get 获取更多信息，其支持读取指定行范围。

---

## 评论区精选

**like wind**（03-06 · 安徽）：
> 你说得对，公众号评论区太浅，百度贴吧又太杂。国内玩OpenClaw的，其实都散落在这些地方：知乎：搜"OpenClaw 记忆"，有不少深度文章，比如这篇就讲得很细，关键是评论区能直接和作者、同好互动。

**梦想起航**（03-03 · 上海）：
> 这么干太消耗token了

**鱼一一**（02-16 · 上海）：
> 想请教一下大家，触发compaction的时候该怎么办，每次都好像卡住了一样，等好久也没反应。

---

**收录于专栏**：偷得浮生半日闲  
**赞同数**：17  
**收藏数**：35

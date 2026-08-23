---
title: "用 kimi-cli 实现自动 Issue → Worktree → TDD → PR 的无人值守流水线"
date: 2026-05-05T14:00:00+08:00
draft: false
categories: ["工程实践"]
tags: ["kimi-cli", "Git", "TDD", "CI/CD", "Automation", "Agent"]
---

> 核心思路：kimi-cli 本身就是一个能写代码、跑命令、用 subagent 的 AI agent，一条好的 prompt + 正确的参数就能让它自己完成整个流程。

---

## 1. 最简单的方式：一次搞定一个 issue

```bash
kimi --print --yolo -p "请解决 GitHub issue #123：
1. 用 git worktree 创建一个独立分支 fix-issue-123，在里面工作
2. 先写测试用例（TDD），确认测试能复现 issue 描述的 bug
3. 实现修复代码，运行测试直到全部通过
4. 用 gh 命令创建一个 PR 到 dev 分支，英文写 PR 描述"
```

**关键参数：**

| 参数 | 作用 |
|------|------|
| `--print` | 非交互模式，执行完自动退出 |
| `--yolo` | 自动批准所有操作，不需要人工确认 |

---

## 2. 批量处理多个 issue（利用 subagent 并行）

```bash
kimi --print --yolo -p "去 https://github.com/用户/仓库 获取所有 label 为 'bug' 的 open issues，然后用 Agent 工具给每个 issue 创建一个 subagent。每个 subagent 的工作流程：

1. git worktree add .worktrees/issue-{编号} -b fix-issue-{编号} main
2. 先写测试复现 bug（TDD）
3. 修复代码
4. 运行全部测试
5. git commit + gh pr create --base dev

最后汇报每个 issue 的处理结果。"
```

这样 kimi-cli 会自动：
- 用 Shell 工具执行 `gh issue list` 拉取 issues
- 用 Agent 工具给每个 issue 启动一个 `coder` 类型的 subagent
- 每个 subagent 在 worktree 里独立工作
- 最后用 `gh pr create` 创建 PR

---

## 3. 完全无人值守：定时自动执行

用 cron 定时触发（编辑 `crontab -e`）：

```cron
# 每 10 分钟检查一次，自动处理 label 为 "ai-fix" 的 issues
*/10 * * * * cd /path/to/your/repo && kimi --quiet -p "检查 GitHub issues 中 label 为 ai-fix 的，逐一用 worktree+TDD+subagent 修复，并 PR 到 dev。处理完后把 label 改成 ai-done。"
```

`--quiet` = `--print --output-format text --final-message-only`，只输出最终结果。

---

## 4. Ralph 循环模式：无限迭代直到完成

```bash
kimi --print --yolo --max-ralph-iterations -1 \
  -p "持续监控 xxx 仓库的 issues（label: ai-fix），每发现一个就用 worktree+TDD 修复并 PR 到 dev，处理完一个再检查下一个，直到没有新 issue 为止。"
```

`--max-ralph-iterations -1` 会让 agent **无限循环**，每次都重新执行同一个任务，直到它自己判断"没有更多 issue 需要处理"并输出 `STOP`。

---

## 5. 关键配置调优

无人值守场景下，`~/.kimi/config.toml` 的 `[loop_control]` 建议这样配：

```toml
[loop_control]
max_steps_per_turn = 2000
max_retries_per_step = 5
max_ralph_iterations = -1
reserved_context_size = 50000
compaction_trigger_ratio = 0.80
```

| 配置项 | 建议值 | 原因 |
|--------|--------|------|
| `max_steps_per_turn` | `2000` | 复杂 issue 的 TDD+修复+调试循环很容易 100+ 步，默认值 1000 够用但留点余量 |
| `max_retries_per_step` | `5` | 无人值守时多给几次重试机会，gh API 可能因限流失败 |
| `max_ralph_iterations` | `-1` | **关键**：-1 = 无限循环，逐个处理 issue 直到没有新的为止 |
| `compaction_trigger_ratio` | `0.80` | 稍微提前触发上下文压缩，避免长任务中对话历史爆炸 |

**命令行快捷覆盖**（不改动配置文件）：

```bash
kimi --print --yolo \
  --max-steps-per-turn 2000 \
  --max-ralph-iterations -1 \
  -p "持续检查 GitHub issues（label: ai-fix），每个 issue 用 worktree+TDD+subagent 修复，然后 PR 到 dev"
```

---

## 6. 总结对照

| 你的需求 | kimi-cli 怎么实现 |
|----------|-------------------|
| 检测 GitHub issue | Agent 用 `gh issue list` 拉取 |
| Worktree 隔离 | Agent 用 `git worktree add` 创建 |
| TDD 流程 | 系统 prompt 本身就有"先写测试再写代码"的指导 |
| Subagent 并行 | 用 Agent 工具给每个 issue 启动 coder subagent |
| PR 到 dev | Agent 用 `gh pr create --base dev` |
| 无人值守 | `--print --yolo` 或 `--quiet` |

你只需要给 kimi-cli 配上 GitHub 的 `gh` 命令（已登录），然后一条 prompt 就能跑起来。

---

## 前置条件检查清单

- [ ] `gh` 已安装并登录 (`gh auth status`)
- [ ] 仓库有写权限（能创建分支和 PR）
- [ ] 测试框架已配置（pytest/jest/vitest 等）
- [ ] `.worktrees/` 目录在 `.gitignore` 中
- [ ] kimi-cli 版本 >= 0.15（支持 subagent 和 ralph 循环）

---

> 这套流程的核心价值在于：**把 AI agent 当作一个可以 7×24 小时运行的自动化工人**，不是辅助你写代码，而是自己完整执行从 issue 发现到 PR 提交的全流程。配合 cron 定时触发，真正实现"无人值守"。

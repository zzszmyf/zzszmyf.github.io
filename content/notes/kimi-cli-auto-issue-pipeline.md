---
title: "用 kimi-cli 实现自动 Issue → Worktree → TDD → PR 的无人值守流水线"
description: "kimi-cli 搭建无人值守的 GitHub issue 自动修复流水线：一条 prompt 走完 issue → git worktree → TDD → PR 全流程，配合 --print --yolo、cron 定时触发和 subagent 并行批量处理 bug。附 loop_control 关键配置（max_ralph_iterations=-1 无限循环）与前置条件检查清单。"
date: 2026-05-05T14:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/kimi-cli-auto-issue-pipeline/"]
categories: ["工程实践"]
tags: ["kimi-cli", "Git", "TDD", "CI/CD", "Automation", "Agent"]
---

先看最短的一条命令。kimi-cli 本身能写代码、跑命令、起 subagent，一条 prompt 就能把整个流程串起来。给 kimi-cli 一个 issue 编号，让它在 worktree 里建分支、先写测试、再改代码、跑通后提 PR：

```bash
kimi --print --yolo -p "请解决 GitHub issue #123：
1. 用 git worktree 创建一个独立分支 fix-issue-123，在里面工作
2. 先写测试用例（TDD），确认测试能复现 issue 描述的 bug
3. 实现修复代码，运行测试直到全部通过
4. 用 gh 命令创建一个 PR 到 dev 分支，英文写 PR 描述"
```

`--print` 让它以非交互模式执行完自动退出，`--yolo` 自动批准所有操作，不需要人工确认。这两个参数决定了它能不能无人值守，也把风险摆到了台面上：`--yolo` 等于把 shell 权限交给一个没有人类在场的进程，能跑测试，也能删文件、改配置、推分支。在生产仓库里用，至少要把爆炸半径圈住——独立仓库或受限 token、分支保护、可回滚的 worktree，这些不是 prompt 能解决的。

这条命令真正省掉的是「开分支、写测试、跑测试、提 PR」这四步手工操作，写代码本身仍然由模型完成。它把 TDD 变成一个可执行的约束：prompt 里先要求写测试复现 bug，再要求实现修复、跑到全绿。原文说 kimi-cli 的系统 prompt 本身就有「先写测试再写代码」的指导，这意味着你不需要在每次 prompt 里重新发明 TDD 流程；但测试是否真的复现了 issue，仍要靠 CI 和 review 兜底。

## 批量：subagent 并行，但别把并发当免费

一条 prompt 也能批量处理：让它用 `gh issue list` 拉取所有 open 的 bug issue，然后给每个 issue 启动一个 coder 类型的 subagent，每个 subagent 在自己的 worktree 里走同样的 TDD 流程，最后统一汇报。kimi-cli 会自动用 Shell 工具执行 gh 命令，用 Agent 工具起 subagent，用 `git worktree add` 隔离分支，用 `gh pr create --base dev` 提 PR：

```bash
kimi --print --yolo -p "去 https://github.com/用户/仓库 获取所有 label 为 'bug' 的 open issues，然后用 Agent 工具给每个 issue 创建一个 subagent。每个 subagent 的工作流程：

1. git worktree add .worktrees/issue-{编号} -b fix-issue-{编号} main
2. 先写测试复现 bug（TDD）
3. 修复代码
4. 运行全部测试
5. git commit + gh pr create --base dev

最后汇报每个 issue 的处理结果。"
```

这里省下的是等待时间；总工作量没有变。每个 subagent 都要跑一遍依赖安装、测试和模型调用，token 消耗随 issue 数线性上涨；如果多个 issue 改到同一个共享模块，worktree 能隔离文件，隔离不了合并冲突，最后仍然要人来解。并发数开多大，取决于 CI 的并发额度和 review 的人力，不取决于模型能同时起多少个 subagent。

## 定时：cron 每 10 分钟，问题是谁来防止重入

无人值守的下一层是 cron。下面这条每 10 分钟跑一次，处理 label 为 ai-fix 的 issue，处理完把 label 改成 ai-done：

```cron
# 每 10 分钟检查一次，自动处理 label 为 "ai-fix" 的 issues
*/10 * * * * cd /path/to/your/repo && kimi --quiet -p "检查 GitHub issues 中 label 为 ai-fix 的，逐一用 worktree+TDD+subagent 修复，并 PR 到 dev。处理完后把 label 改成 ai-done。"
```

`--quiet` 等价于 `--print --output-format text --final-message-only`，只输出最终结果，适合放进 cron 日志。按 10 分钟一次算，一小时最多触发 6 轮；如果某一轮的处理时间超过 10 分钟，下一轮会踩着上一轮的尾巴启动，两个进程可能同时看到同一个 ai-fix issue。原文没有给出锁或超时策略，[待补：cron 重入锁、单轮超时与失败后的 label 回滚方案]。没有这层保护，label 只能表达「做完了」，表达不了「正在做」和「做失败了」。

## Ralph 循环：无限迭代的开关

`--max-ralph-iterations -1` 会让 agent 无限循环：每次都重新执行同一个任务，直到它自己判断「没有更多 issue」并输出 `STOP`。配合 `--print --yolo`，它可以在没有人盯着的情况下持续消费 issue 队列：

```bash
kimi --print --yolo --max-ralph-iterations -1 \
  -p "持续监控 xxx 仓库的 issues（label: ai-fix），每发现一个就用 worktree+TDD 修复并 PR 到 dev，处理完一个再检查下一个，直到没有新 issue 为止。"
```

无人值守场景下，`~/.kimi/config.toml` 的 `[loop_control]` 建议从这样一份配置起步。数值不是固定的，下文逐个解释：

```toml
[loop_control]
max_steps_per_turn = 2000
max_retries_per_step = 5
max_ralph_iterations = -1
reserved_context_size = 50000
compaction_trigger_ratio = 0.80
```

这几个数字各自对应一类失败。`max_steps_per_turn` 卡住单轮的步数上限；原文说复杂 issue 的 TDD+修复+调试循环很容易超过 100 步，默认的 1000 够用，留到 2000 是给长任务余量。`max_retries_per_step` 给 gh API 限流和偶发命令失败留 5 次重试，无人值守时没人手动重跑，重试是最后一道自愈。`max_ralph_iterations = -1` 是无限循环的关键，也让 `STOP` 判断变成唯一的出口；如果模型一直不输出 `STOP`，这个进程会一直跑下去。`reserved_context_size = 50000` 留出 50000 token 的余量，`compaction_trigger_ratio = 0.80` 把压缩触发点提前到 80%；两者一起避免长任务在撞到窗口上限时才压缩。代价是更早丢弃原始对话，压缩太频繁也会打断任务节奏。

命令行可以临时覆盖这些配置，不改文件。下面这条适合一次性实验：

```bash
kimi --print --yolo \
  --max-steps-per-turn 2000 \
  --max-ralph-iterations -1 \
  -p "持续检查 GitHub issues（label: ai-fix），每个 issue 用 worktree+TDD+subagent 修复，然后 PR 到 dev"
```

## 前置条件

跑之前先确认 `gh` 已经安装并登录，用 `gh auth status` 能查到。仓库要有写权限，能创建分支和 PR。测试框架要配好，pytest、jest、vitest 都行，否则 TDD 的第一步就无从谈起。`.worktrees/` 目录要放进 `.gitignore`，避免 worktree 的内容污染主仓库的 git status。kimi-cli 版本至少 0.15，才支持 subagent 和 ralph 循环。这几项里，最容易漏的是 `.gitignore` 和测试框架：前者会让 agent 把 worktree 里的文件当成未跟踪改动，后者会让「先写测试」变成「先写一个跑不起来的测试」。

这条流水线最终产出的是一串 PR，合并仍然由人决定。它的瓶颈会从「没人写」变成「没人 review」：按每 10 分钟一次的触发频率，如果每轮都有新 issue，PR 的产出速度很容易超过人工 review 的速度。真正要补的是队列上限、重入锁、失败回滚和 review 容量；继续加长 prompt 解决不了这些问题。

---
title: "VC-FCST Benchmark 落地方案：Code Agent 标准化测评"
description: "VC-FCST Benchmark：基于失效分类学的 AI Code Agent 专项能力诊断测评方案，与 SWE-bench 的本质区别在于单 Case 单失效点、可精准定位失效根因。包含 2500 个标准化 Case 的用例库、DeepSeek 双模型编排引擎与 Streamlit 排行榜，1 名全栈工程师 3 个月，2500 元为目标预算、2878 元为当前明细基线。"
date: 2026-02-24T00:00:00+08:00
lastmod: 2026-10-18T12:24:51+08:00
draft: false
weight: 100
aliases: ["/posts/vc-fcst-benchmark/"]
categories: ["研究笔记"]
tags: ["研究笔记", "Reinforcement Learning", "Benchmark"]
---

本文整理自清歌（QingGo）2026-02-24 发布在知乎的原文，项目代码在 github.com/QingGo/vibe-check，原文链接是 zhuanlan.zhihu.com/p/2009624796816764979。下面的分析以这套方案为素材，不替作者补充未公开的实测数据。

## 一、端到端榜单为什么定位不到失效

SWE-bench 的一个任务失败时，你拿到的只有一个红色叉。是需求读错了、补丁写歪了，还是测试根本没跑起来？做 Code Agent 的人都知道，这三种失败对应的修复动作完全不同，但端到端榜单把它们压成了同一个通过率。

SWE-bench / SWE-bench Pro 的任务来自真实 GitHub Issue，一个 Issue 往往同时考察仓库检索、需求理解、跨文件修改、工具调用和测试修复。模型交出的 Patch 只要有一个环节出错，整个任务就判失败；你看到的是“这个模型在 300 个 Issue 上通过了 42%”，看不到失败集中在哪一类能力上。固定开源仓库还有另一个问题：这些仓库大概率已经出现在训练语料里，刷分和真实泛化能力之间隔着一层无法证伪的污染。不同 Agent 的脚手架、检索策略和重试逻辑也混在同一个数字里，模型原生能力和工程封装无法拆开。

VC-FCST 换了一个测量单位。它把 AI 编码失效整理成 7 个顶层分类、67 个三级失效模式，再按高发占比、测评区分度、纯模型可测、行业通用四条标准筛出 Top 50 个三级分类。Top 50 的分布是需求意图 7 个、代码执行 12 个、业务逻辑 9 个、架构协同 10 个、安全合规 7 个、工程迭代 5 个，人因协作 0 个，MVP 阶段不覆盖。每个 Case 只绑定一个三级分类和一个预期失效点，难度按 Easy 40%、Medium 40%、Hard 20% 配比，类型按 implement 60%、modify 40% 配比。最终目标是 2500 个 Case，每个入选分类至少 50 个。Case 结构里有自然语言需求、初始仓库代码、可自动执行的验收标准和固定环境配置；前三样定义 Case 的输入和判定，环境配置负责复现。

这样做的收益很直接：某类 Case 的通过率低，你能指到具体的失效模式，而不是笼统地说“代码能力不行”。代价也要说清楚：一个 Case 只测一个失效点，真实仓库里多个失效模式会互相掩盖；2500 个 Case 覆盖 50 个分类，每个分类 50 个样本，统计置信度只能支撑粗粒度比较。它是一张专项诊断表，替代不了端到端评测。

## 二、Case Bank：生成之后还有四道闸门

Case Bank 的目标是覆盖 Top 50 个三级分类、每个分类至少 50 个有效 Case，总规模 2500 个，且每个 Case 严格绑定一个失效点。生成端用 DeepSeek V3.2 的 reasoner 模型，配合 Pydantic 强约束和 JSON 模式输出。Schema 把分类信息、难度、类型、需求、初始代码、验收标准和环境配置都设成必填字段，模型少一个字段就解析失败。这能挡住格式层面的幻觉，挡不住语义层面的跑偏：一个 Case 字段齐全、代码能跑，但它考的并不是目标失效模式，这种错误只能靠后面的语义校验发现。

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Literal

class VCFCSTCategory(BaseModel):
    """VC-FCST分类信息，严格绑定单三级分类"""
    level1_id: str = Field(description="顶层分类ID")
    level1_name: str = Field(description="顶层分类名称")
    level3_id: str = Field(description="三级分类ID，全局唯一绑定")
    level3_name: str = Field(description="三级分类名称")
    target_defect: str = Field(description="唯一预期失效点，严格对应三级分类定义")

class VCFCSTCase(BaseModel):
    """单Case标准化结构，最小完备单元"""
    case_id: str = Field(description="全局唯一ID，格式：VCFCST-{level3_id}-{序号}")
    vcfcst_category: VCFCSTCategory
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(description="难度占比：Easy40%/Medium40%/Hard20%")
    case_type: Literal["implement", "modify"] = Field(description="类型占比：implement60%/modify40%")
    requirement: str = Field(description="给Agent的自然语言需求，和真实用户输入完全一致，无刻意引导")
    initial_code: Dict[str, str] = Field(description="初始仓库代码，key为文件相对路径，value为文件内容")
    acceptance_criteria: Dict = Field(description="客观验收标准，100%可自动化验证")
    env_config: Dict = Field(description="执行环境配置，保证100%可复现")
```

生成之后要过四道闸门。第一道是格式合规，用 deepseek-chat 检查 Schema、必填字段和字段格式，过滤掉结构不完整的 Case。第二道是可执行性，把初始代码放进 Docker 容器，确认代码能运行、测试逻辑正确、依赖能装上，语法错误和坏测试在这一层暴露。第三道是区分度，强基准 deepseek-reasoner 必须 100% 通过，弱基准 deepseek-chat 的通过率必须不高于 30%，只保留同时满足这两个条件的 Case。第四道是语义一致性，用 deepseek-reasoner 判断这个 Case 是否严格对应目标三级分类、是否只包含唯一预期失效点，偏离分类或引入额外变量的 Case 会被打回重生成。作者给出的验收口径是四层校验通过率 100%、人工抽检 10% 的有效率不低于 90%、同一环境三次执行结果一致。

这套流水线最值得盯的是第三道闸门。区分度是用 DeepSeek 的两个模型定义的：reasoner 能过、chat 过不了，Case 才保留。这个定义能有效筛掉太简单和太难的样本，但它也把“难度”锚定在 DeepSeek 的能力差上。一个 Case 可能只是恰好落在 reasoner 和 chat 的能力缝隙里，对 Claude Code 或 Gemini CLI 未必有同样的区分度。要证明这套 Case Bank 是跨模型的诊断工具，至少需要拿另外两家 Agent 做一轮交叉校验，看保留的 Case 是否仍然呈现“强模型过、弱模型不过”的分布 [待补：2500 个 Case 在 Claude Code、Gemini CLI 上的交叉区分度结果]。原文没有给出这组数据，方案里也没有留这个校验环节；这是我认为落地前必须先补的一课。

## 三、把模型能力和脚手架拆开

传统测评框架给每个 Agent 写一套适配器，接一个新模型就要改输入拼装、输出解析和 Patch 提取。VC-FCST 的做法是给每个被测 Agent 一个 YAML 配置，输入模板负责把标准 Case 转成该 Agent 的 Prompt，输出模板交给 DeepSeek 把原始输出转成标准 Git Patch。新增一个 Agent 只需要加一个配置文件，作者给的接入时间是 10 分钟。执行侧用五个扁平模块：任务调度、协议适配、Docker 隔离、客观验证、异常重试；核心逻辑控制在 1000 行以内，单 Case 单容器，跑完销毁。端到端流程从 Case 加载开始，经 Agent 协议适配、Docker 环境初始化、模型调用生成、Patch 提取、应用验证、测试执行、结果判定、日志存储，最后生成报告。

```yaml
agent_name: "codex"
litellm_model_name: "openai/code-davinci-002"
api_key_env: "OPENAI_API_KEY"
max_tokens: 4096
temperature: 0.2
timeout: 120

# 输入模板：把标准化Case转为模型的输入Prompt
input_template: |
  请完成以下Python开发任务，严格按照要求输出代码：
  【任务需求】{{ case.requirement }}
  【仓库现有代码】{% for path, content in case.initial_code.items %}
  文件路径：{{ path }}
  ---
  {{ content }}
  ---
  {% endfor %}
  【验收标准】你的代码必须通过以下pytest测试用例：
  {% for path, content in case.acceptance_criteria.test_code.items %}
  文件路径：{{ path }}
  ---
  {{ content }}
  ---
  {% endfor %}

# 输出转换模板：把模型输出转为标准Git Patch
output_template: |
  请把以下代码生成结果，转换为标准的git diff格式Patch，严格遵循git规范：
  【仓库初始代码】{{ case.initial_code | tojson }}
  【模型生成的代码】{{ model_raw_output }}
```

变量隔离靠双模式测评：同一个 Case 既跑裸模型，也跑带脚手架的 Agent。裸模型通过、Agent 失败，问题更可能出在脚手架的工具调用或上下文管理；裸模型失败、Agent 通过，说明脚手架能补上模型的能力缺口；两边都在同一个三级分类上失败，才更像模型本身的能力边界。这个设计比单一端到端分数多了一层归因，但它把 DeepSeek 放进了测量链路：输出模板用 LLM 把模型输出转成 Patch，转换器自己也会犯错。如果 Agent 输出的是一段解释而不是 diff，转换器可能补出一个看似合法、实际不对应的 Patch，最后被记成 Agent 的失败。

方案没有给出 Patch 转换失败率，也没有说明转换失败是否单独记账 [待补：Patch 转换失败率与人工复核比例]。我的建议是把转换失败从 Agent 失败里拆出来，按“原始输出无法解析”“Patch 无法应用”“测试失败”三类分别统计；否则你无法判断某个 Agent 的低分是模型能力问题，还是裁判的解析器不认它的输出格式。Docker 隔离同样有账要算：每个 Case 启停一次容器，2500 个 Case 乘以三款 Agent 就是 7500 次容器生命周期，启动和依赖安装的时间会直接吃进测评时长。作者把单 Case 执行成功率不低于 95% 写成验收标准，这是目标值，不是实测值 [待补：三款 Agent 各 2500 Case 的实际执行成功率]。

## 四、2878 元的明细和 2500 元的预算

成本部分只用 DeepSeek V3.2 的两个模型。核心 Case 生成、语义一致性校验、Patch 转换和诊断报告走 deepseek-reasoner；格式校验、弱基准校验和简单分类走 deepseek-chat；固定模板渲染交给 Jinja2，不产生模型调用。作者假设缓存命中率 60%，把模板和重复结构放在系统提示词里，缓存部分的定价按非缓存的十分之一算。

把明细摊开：2500 个 Case 核心生成，输入 600 万 Token、输出 300 万 Token，1452 元；格式和语义一致性校验，输入 150 万、输出 50 万，186 元；区分度强弱基准校验，输入 200 万、输出 100 万，374 元；协议适配的 Patch 转换，输入 300 万、输出 100 万，558 元；诊断报告，输入 50 万、输出 20 万，125 元；重试缓冲，输入 100 万、输出 50 万，183 元。合计输入 1400 万 Token、输出 620 万 Token，2878 元。

正文说全流程控制在 2500 元以内，同一节的明细却加出了 2878 元，中间差了 378 元。这不是四舍五入能解释的差额。比较合理的解释是 2500 元是压缩目标，2878 元是按当前单价和 60% 缓存命中率算出的基线；要么实际执行时砍掉一部分重试缓冲，要么缓存命中率高于 60%。落地时应该把 2878 元当预算上限，2500 元当争取值，并且把缓存命中率做成可观测指标：缓存一旦失效，输入侧的账单会先跳起来。

这张成本表还有一个更容易被忽略的缺口：它算的是裁判的成本，没有算运动员的成本。2500 个 Case 跑三款 Agent，被测模型自己的输入输出 Token 不在 2878 元里；如果被测 Agent 每 Case 要读几十 K 上下文、跑多轮工具调用，这笔账单可能远高于裁判侧。要评估“1 人 3 个月、2500 元”能不能成立，得把被测 Agent 的推理成本单独列一行 [待补：三款被测 Agent 跑完 2500 Case 的推理成本账单]，否则这个数字只覆盖了造题和判卷，没覆盖考试本身。

## 五、三个月排期和真正的风险

排期分四段：第 1-2 周锁定 Top 50 分类、定义 Schema、做 10 个试点 Case、跑通单 Case 端到端；第 3-4 周批量生成 2500 个 Case、搭四层校验、人工抽检 10%；第 5-8 周完善 Orchestrator、Docker 隔离、三款 Agent 的 YAML 和客观验证；第 9-12 周做 Streamlit 排行榜、跑三款模型的全量测评、生成诊断报告。这个排期把 Case Bank 放在第 3-4 周，后面所有工作都依赖它；Case 生成如果卡在有效率上，第 5-8 周的联调就会被压缩。

作者列了四条风险：Case 生成幻觉导致有效率不足 90%、被测模型接入适配困难、LLM 调用成本超支、一个人开发工作量超支。前两条有明确的缓解路径，后两条更考验克制：成本超支时要停掉非核心环节的模型调用，工作量超支时要砍掉排行榜的花哨页面。我的判断是，第一版最该先做的是一套 200 个 Case 左右的跨模型校准集：用 DeepSeek 双模型生成和校验，再拿 Claude Code、Gemini CLI 各跑一遍，看区分度和失效分布是否还成立。这个校准集跑不通，后面的 2500 个 Case 和排行榜都只是把同一个偏差放大。

这套方案目前仍是一份设计文档：没有公开的 2500 Case 跨模型结果，没有 Patch 转换失败率，也没有三款 Agent 的实际执行成功率。排行榜可以后做，这三组数据补不齐，诊断结论就站不住。

项目地址是 github.com/QingGo/vibe-check。原文链接是 zhuanlan.zhihu.com/p/2009624796816764979。作者是清歌（QingGo）。

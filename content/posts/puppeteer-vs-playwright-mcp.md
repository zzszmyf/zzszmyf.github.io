---
title: "Puppeteer MCP 与 Playwright MCP：深度对比与选择指南"
date: 2024-04-12T00:00:00+08:00
draft: false
categories: ["研究笔记"]
tags: ["研究笔记", "Agent Tools"]
---

# Puppeteer MCP 与 Playwright MCP：深度对比与选择指南

> **原文链接**：[https://zhuanlan.zhihu.com/p/1891569128860534619](https://zhuanlan.zhihu.com/p/1891569128860534619)
>
> 作者：二师兄说 AI（互联网行业 Java 工程师）
>
> 收录于：MCP100 个案例

---

## 来自用户的提问：和 Playwright 有区别吗，哪个好用？

模型上下文协议（MCP）与 Puppeteer 和 Playwright 等强大的浏览器自动化工具相结合，为更智能和自主的 AI 代理铺平了道路，这些代理能够在互联网上执行复杂的任务，为自动化、数据分析和用户辅助开辟了新的可能性。对于他们两或者更多类似的工具，大家需要根据实际的情况进行选择。比如，如果你需要进行多平台测试，可能 Playwright 是不错选择；如果你的 web 系统首选 Chrome 支持，可能 Puppeteer 会带来更好的体验。

以下是内容结合了 Deep Research 的结果进行编辑

---

## 引言

模型上下文协议（Model Context Protocol，MCP）作为一种新兴的开放标准，旨在标准化应用程序如何为大型语言模型（Large Language Models，LLMs）提供上下文信息。通过建立统一的接口，MCP 使得 LLMs 能够与各种外部工具和数据源进行安全且可控的交互。在众多利用 MCP 实现浏览器自动化功能的项目中，基于 Puppeteer 和 Playwright 的 MCP 服务器无疑是两种备受关注的方案。Puppeteer 由 Chrome 团队开发，而 Playwright 则由 Microsoft 创建，两者都是强大的 Node.js 库，用于以编程方式控制浏览器。本报告旨在对这两个 MCP 服务器进行深入的对比分析，以帮助开发者根据自身的需求做出明智的选择。报告将从实现原理、功能特性、性能表现、易用性、社区活跃度、可扩展性以及各自的优劣势等方面展开全面的探讨。

---

## Puppeteer MCP 服务器分析

### 实现原理与设计目标

Puppeteer MCP 服务器的核心在于利用 Puppeteer 库提供的浏览器自动化能力，将其通过 MCP 协议暴露给 LLMs 使用。其设计目标是使 LLMs 能够像人类用户一样与网页进行交互，包括导航、截图和执行 JavaScript 等操作。通过提供基本的网页交互工具，该服务器旨在扩展 LLMs 在网络环境中的应用范围，使其能够执行需要直接与 Web 界面交互的任务。这种设计思路旨在弥合 AI 推理和 Web 操作之间的鸿沟，使得 AI 代理能够执行更复杂的任务。

### 功能特性详述

#### 支持的 MCP 协议特性

Puppeteer MCP 服务器主要通过暴露一系列工具（Tools）来实现其功能，这些工具是 MCP 协议的核心组成部分。例如，它提供了 `puppeteer_navigate` 用于导航到指定的 URL，`puppeteer_screenshot` 用于捕获整个页面或特定元素的截图，`puppeteer_click` 用于点击页面上的元素，`puppeteer_fill` 用于填写输入框，以及 `puppeteer_evaluate` 用于在浏览器控制台中执行 JavaScript 代码。此外，该服务器还能够提供对诸如浏览器控制台日志和屏幕截图等资源（Resources）的访问。这些工具和资源的结合，使得 LLMs 能够执行一系列有针对性的 Web 交互操作。

#### 可用 API 及其功能

Puppeteer MCP 服务器提供的 API 主要是通过上述的工具来实现的。具体来说：

- `puppeteer_navigate`：允许 LLM 指示服务器导航到特定的网页地址
- `puppeteer_screenshot`：支持捕获网页的完整截图，也可以通过 CSS 选择器指定需要截图的特定元素
- `puppeteer_click`：模拟用户点击操作，通过 CSS 选择器定位目标元素
- `puppeteer_hover`：提供悬停在指定元素上的功能
- `puppeteer_fill`：根据 CSS 选择器定位输入框并填充内容
- `puppeteer_select`：用于选择下拉菜单中的选项
- `puppeteer_evaluate`：允许 LLM 提供 JavaScript 代码，由服务器在浏览器环境中执行，从而实现更高级的自动化任务

值得一提的是，某些 Puppeteer MCP 的实现还支持连接到已激活远程调试的 Chrome 实例，这为集成现有浏览器会话提供了可能性。

#### 具体使用场景与案例

Puppeteer MCP 服务器在多种场景下展现出其价值：

1. **自动化 Web 应用程序测试**：通过模拟用户操作来验证其行为和功能
2. **数据抓取**：对于那些依赖 JavaScript 动态加载内容的网页，能够先在真实浏览器环境中渲染页面，然后再提取所需信息
3. **自动生成文档**：通过捕获网页截图来记录用户界面状态或特定元素
4. **执行复杂操作**：LLM 可以指示 Puppeteer MCP 执行自定义的 JavaScript 代码，实现更高级的自动化流程
5. **监控浏览器控制台日志**：辅助调试 Web 应用程序
6. **自动化多步骤工作流程**：例如用户登录、数据提取和报告生成

### 性能考量

目前，尚未发现直接比较 Puppeteer MCP 和 Playwright MCP 性能的基准测试或讨论。然而，Puppeteer 本身以其速度和效率而闻名，尤其是在与 Chromium 浏览器配合使用时。由于 Puppeteer MCP 服务器是构建在 Puppeteer 之上的，因此可以推断，对于以 Chromium 为中心的任务，它可能也具有较高的效率。Puppeteer 与 Chrome DevTools Protocol 的紧密集成，使得它能够以优化的方式控制浏览器，这可能意味着更快的自动化任务执行速度。

### 易用性评估

#### 安装与配置过程

Puppeteer MCP 服务器的安装和配置过程相对直接。开发者可以选择：

- 通过 npm 全局安装：`npm install -g puppeteer-mcp`
- 使用 npx 直接运行：`npx puppeteer-mcp`
- 从源代码安装
- 使用 Docker 进行部署（适用于希望在无头模式下运行 Chromium 的用户）

对于需要与 Claude 集成的场景，通常需要修改 `claude_desktop_config.json` 文件，将 Puppeteer MCP 服务器的详细信息添加到配置文件中。提供多种安装方式体现了对不同用户和部署场景的考虑。

#### API 的简洁程度与开发者体验

Puppeteer MCP 服务器的 API 设计倾向于直观和易于理解。其工具的命名通常具有描述性，例如 `puppeteer_navigate` 和 `puppeteer_screenshot`。工具的参数也相对简单明了，例如导航工具只需要提供目标 URL，而元素定位则主要依赖 CSS 选择器。这种设计旨在降低 LLMs 使用这些工具的难度，使得它们能够更有效地执行自动化任务。

#### 文档与示例的完善程度

目前，关于 Puppeteer MCP 的文档和示例主要散落在各种博客文章和技术文章中，例如初学者指南和使用案例介绍。一些平台如 Cursor Directory 也提供了相关的文档信息。npm 包 `@modelcontextprotocol/server-puppeteer` 本身也包含了一些基本的描述和工具信息。虽然存在一些文档和示例，但可能缺乏集中化和全面的官方文档。

#### 社区活跃度和维护情况

根据检索到的信息，npm 包 `@modelcontextprotocol/server-puppeteer` 最近一次发布是在四个月前（截至信息收集时）。同时，GitHub 上也存在一些与 Puppeteer MCP 相关的仓库，例如 `merajmehrabi/puppeteer-mcp-server` 和 `twolven/mcp-server-puppeteer-py`，它们的活跃程度可能有所不同。这表明 Puppeteer MCP 存在一定的社区活动和维护，但可能相对分散。

### 可扩展性与定制化

由于 Puppeteer 本身是开源的，这为基于其构建的 MCP 服务器提供了良好的可扩展性。开发者可以根据需要定制底层的浏览器自动化逻辑。同时，MCP 框架的设计也鼓励扩展。例如，`merajmehrabi/puppeteer-mcp-server` 项目就明确提到其设计受到了其他项目的启发，并探索了不同的浏览器自动化方法。这暗示了开发者可以根据具体需求添加新的工具或修改现有行为。

---

## Playwright MCP 服务器分析

### 实现原理与设计目标

Playwright MCP 服务器同样旨在通过 MCP 协议将浏览器自动化能力赋予 LLMs，但它使用的是 Microsoft 开发的 Playwright 库。其核心设计目标是实现快速且轻量级的操作，这主要得益于它利用 Playwright 的可访问性树（Accessibility Tree）而非基于像素的输入方式。这种方式不仅提高了效率，也使得服务器能够直接操作结构化数据，从而更好地服务于 LLMs，无需依赖视觉模型。此外，Playwright MCP 服务器的设计还强调工具应用的确定性，通过使用结构化的可访问性快照，避免了基于截图方法常有的歧义。

### 功能特性详述

#### 支持的 MCP 协议特性

Playwright MCP 服务器提供了**两种主要的工具模式**：

**快照模式（Snapshot Mode，默认）**：
利用可访问性快照提供了一系列工具：
- `browser_navigate` - 导航
- `browser_go_back` / `browser_go_forward` - 后退/前进
- `browser_click` - 点击
- `browser_hover` - 悬停
- `browser_drag` - 拖拽
- `browser_type` - 输入
- `browser_select_option` - 选择选项
- `browser_choose_file` - 选择文件
- `browser_press_key` - 按下按键
- `browser_snapshot` - 获取快照
- `browser_save_as_pdf` - 保存为 PDF
- `browser_take_screenshot` - 截图
- `browser_wait` - 等待
- `browser_close` - 关闭页面

**视觉模式（Vision Mode，使用截图）**：
使用截图进行交互，提供的工具包括：
- 导航、后退/前进
- `browser_screenshot` - 截图
- `browser_move_mouse` - 移动鼠标
- 点击（基于坐标）
- 拖拽（基于坐标）
- 输入（基于坐标）
- 按下按键、选择文件
- 保存为 PDF、等待、关闭页面

此外，Playwright MCP 服务器还支持通过 `--port` 标志启用服务器发送事件（Server-Sent Events，SSE）传输。

#### 可用 API 及其功能

Playwright MCP 服务器的 API 同样由其提供的工具构成：

- **在快照模式下**：API 的交互主要依赖于对网页元素的人类可读描述和来自可访问性快照的精确引用。例如，点击操作需要提供元素的描述和引用。
- **在视觉模式下**：API 的交互则基于屏幕上的 X 和 Y 坐标。例如，点击操作需要指定点击的坐标。

这种双模式的 API 设计旨在兼顾性能和对不同类型网页内容的处理能力。

#### 具体使用场景与案例

Playwright MCP 服务器适用于多种使用场景：

1. **网页导航和表单填写**：支持 LLMs 进行基本的 Web 交互
2. **数据提取**：通过利用 Playwright 的可访问性快照，帮助 LLMs 从结构化的网页内容中提取数据，而无需依赖视觉解析
3. **自动化测试**：驱动由 LLMs 控制的自动化测试流程
4. **构建智能代理**：作为构建智能代理的基础，这些代理能够为了各种目的与 Web 进行交互，例如研究、信息收集和任务自动化
5. **高级用例**：
   - 通过模拟多个并发用户进行负载测试
   - 支持远程调试和共享测试环境

### 性能考量

Playwright MCP 服务器被设计为快速且轻量级，这主要归功于其使用可访问性树的方式。Playwright 本身也以其速度和高效的异步操作处理能力而闻名。可以预见的是，Playwright MCP 服务器在快照模式下，通过避免计算密集型的视觉处理，能够提供良好的性能。

### 易用性评估

#### 安装与配置过程

Playwright MCP 服务器可以通过以下方式安装：

- npm 安装：`npm install @playwright/mcp@latest`
- VS Code 集成：对于使用 VS Code 的开发者，该服务器提供了便捷的集成方式，包括专门的安装按钮和 CLI 命令

配置 MCP 客户端以使用该服务器通常只需要指定运行服务器的命令，例如 `npx @playwright/mcp@latest`。

此外，Playwright MCP 服务器还支持多种命令行选项，用于配置浏览器选择、是否以无头模式运行以及设置监听端口等。

#### API 的简洁程度与开发者体验

Playwright MCP 服务器的 API 设计清晰明了。其工具的命名具有良好的可读性，例如 `browser_navigate` 和 `browser_click`。

- 在快照模式下，API 使用描述性的元素参数，例如 `element` 和 `ref`
- 在视觉模式下，API 则使用基于坐标的参数，例如 `x`、`y`、`startX`、`startY`、`endX` 和 `endY`

这种根据操作模式区分参数的设计使得 API 更易于理解和使用。

#### 文档与示例的完善程度

Playwright MCP 服务器在其 GitHub 仓库的 README 文件中提供了全面的项目概述，包括：
- 功能介绍
- 安装指南
- 使用说明
- 可用工具的详细信息
- 项目许可证、行为准则和安全策略等相关链接

这些信息为开发者提供了良好的起点，有助于他们理解和使用该项目。

#### 社区活跃度和维护情况

`microsoft/playwright-mcp` 仓库在 GitHub 上拥有大量的 Star、Watchers 和 Forks，这表明社区对该项目有着浓厚的兴趣。该项目正在积极进行维护，定期发布新版本，最近一次发布是在信息收集时期的前几天。同时，该项目也有多位贡献者参与开发。这些迹象都表明 Playwright MCP 服务器是一个活跃且得到良好维护的项目，拥有强大的社区支持和持续的开发投入。

### 可扩展性与定制化

Playwright 本身是一个高度可扩展的浏览器自动化库，因此基于其构建的 MCP 服务器也继承了这一特性。MCP 框架本身也支持扩展。

Playwright MCP 服务器的命令行选项允许用户在一定程度上定制服务器的行为，例如：
- 选择使用的浏览器
- 运行模式
- README 文件中还演示了如何通过自定义传输机制以编程方式使用服务器

这些都表明 Playwright MCP 服务器在可扩展性和定制化方面具有一定的灵活性。

---

## 对比分析

### 关键特性与功能对比

| 特性 | Puppeteer MCP | Playwright MCP |
|------|--------------|----------------|
| **底层自动化库** | Puppeteer | Playwright |
| **浏览器支持** | 主要为 Chromium，实验性支持 Firefox | Chromium, Firefox, WebKit, MS Edge |
| **多语言支持** | JavaScript | JavaScript, Python, Java, C# (通过 Playwright) |
| **交互模式** | 主要基于 DOM 和 CSS 选择器 | 快照模式（可访问性树），视觉模式（截图） |
| **可访问性关注** | 有限的显式关注 | 在快照模式下强烈关注可访问性快照 |
| **连接到活动标签页** | 是（在某些实现中） | 主要文档中未明确提及 |
| **并行执行** | 依赖 Puppeteer 的能力 | 继承 Playwright 强大的并行执行能力 |
| **网络拦截** | 通过 Puppeteer 可用 | 通过 Playwright 可用，具有高级功能 |
| **移动设备模拟** | 限于 Chromium 的能力 | 强大的移动设备模拟支持 |
| **无头/有头模式** | 支持 | 支持 |
| **SSE 传输** | 支持（在某些实现中） | 通过 --port 标志支持 |
| **社区活跃度** | 存在但可能较为分散 | 强大且活跃，由 Microsoft 支持 |
| **安装方式** | npm, npx, 源代码，Docker | npm, VS Code 集成 |
| **API 风格** | 基于工具，主要使用 CSS 选择器 | 基于工具，描述性参数取决于交互模式（可访问性引用或坐标） |
| **主要用例** | Web 测试，抓取，文档生成，JS 执行，控制台监控 | Web 导航，表单填写，数据提取，自动化测试，通用代理交互，负载测试，远程调试，共享测试 |

### 性能特点对比

Playwright MCP 默认的快照模式利用可访问性树，相较于 Puppeteer MCP 传统的基于 DOM 的交互或 Playwright MCP 的视觉模式，可能在速度和资源效率方面更具优势。由于 Puppeteer 与 Chrome DevTools Protocol 的紧密集成，在仅针对 Chromium 的任务中，Puppeteer 可能在原始速度上略胜一筹。然而，Playwright 对并行执行的强大支持可能使其在处理大型自动化测试套件时具有更快的总体执行时间。因此，选择可能取决于主要目标浏览器以及自动化任务的复杂性。

### 易用性比较

Puppeteer MCP 和 Playwright MCP 都提供了相对简单的 npm 安装方式。

- **Playwright MCP** 针对 VS Code 用户的集成体验更佳，提供了更顺畅的设置过程
- **Puppeteer MCP** 对于基本的使用场景，可能具有更简单的配置示例
- **Playwright MCP** 的 API 由于其两种不同的模式和参数集，初学者可能需要稍作学习，但这种设计提供了更高的灵活性

总体而言，两者在易用性方面各有千秋，选择可能取决于开发者偏好的 IDE 和预期自动化场景的复杂度。

### 社区支持与项目维护对比

Playwright MCP 拥有 Microsoft 的强大支持，并拥有一个非常活跃的社区，定期进行更新和贡献。相比之下，Puppeteer MCP 的社区支持似乎更为分散，并且特定实现的维护状态可能有所不同。因此，Playwright MCP 可能由于其强大的社区和企业支持而提供更可靠和一致的长期支持。

### 可扩展性与定制化差异分析

两者都建立在高度可扩展的浏览器自动化库之上。Playwright 的多语言支持可能为开发者在定制和集成到不同技术栈中提供了更多选择。Puppeteer 与 Chromium 生态系统的紧密联系可能为在该环境中进行高级定制提供了更深层次的访问权限。因此，选择可能取决于现有技术栈以及是否需要跨浏览器兼容性。

---

## 优势与劣势

### Puppeteer MCP 的优缺点总结

**优点：**

- 成熟的库，拥有庞大的社区
- 针对 Chromium 的自动化设置简单
- 在 Chromium 任务中性能良好
- 可能支持连接到活动的 Chrome 标签页

**缺点：**

- 原生跨浏览器支持有限
- 主要面向 JavaScript
- MCP 实现的社区支持可能较为分散

### Playwright MCP 的优缺点总结

**优点：**

- 优秀的跨浏览器支持（Chromium, Firefox, WebKit, Edge）
- 多语言支持（JavaScript, Python, Java, C#）
- 通过可访问性快照实现快速轻量级的操作
- 强大的社区支持和 Microsoft 的积极维护
- 具有自动等待和网络拦截等高级功能
- 与 VS Code 集成良好

**缺点：**

- 相较于 Puppeteer 较新，因此在某些方面社区可能较小（但发展迅速）
- 快照模式依赖于结构良好的可访问性信息

---

## 最后的结论与建议

对于**主要针对 Chromium 且优先考虑初始设置简易性**的开发者，**Puppeteer MCP** 可能是一个合适的选择，尤其是在 AI 系统也基于 JavaScript 的情况下。

然而，对于**需要跨浏览器兼容性、多语言支持以及利用高级浏览器功能**的项目，**Playwright MCP** 可能是更好的选择。其对可访问性的关注以及 Microsoft 的积极开发也预示着它将是一个更健壮的长期解决方案。

如果**性能是关键因素**，并且目标网站具有良好的可访问性结构，那么 Playwright MCP 的快照模式提供了一种潜在更快且更高效的方法。

已经大量投资于 Playwright 生态系统或需要跨不同浏览器进行测试的开发者会发现 Playwright MCP 是一个自然的选择。

### 总结

总而言之，Puppeteer MCP 和 Playwright MCP 都代表了在使 AI 代理能够与 Web 交互方面的重要进步。它们之间的选择取决于项目的具体需求，包括：

- 浏览器兼容性需求
- 性能考虑
- 开发者熟悉程度
- 所需的社区支持水平

模型上下文协议与 Puppeteer 和 Playwright 等强大的浏览器自动化工具相结合，为更智能和自主的 AI 代理铺平了道路，这些代理能够在互联网上执行复杂的任务，为自动化、数据分析和用户辅助开辟了新的可能性。

---

> **原文链接**：[https://zhuanlan.zhihu.com/p/1891569128860534619](https://zhuanlan.zhihu.com/p/1891569128860534619)

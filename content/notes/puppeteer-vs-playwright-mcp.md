---
title: "Puppeteer MCP 与 Playwright MCP：深度对比与选择指南"
description: "Puppeteer MCP 与 Playwright MCP 深度对比：从浏览器支持、交互模式（DOM/CSS 选择器 vs 可访问性树快照）、性能证据、易用性到社区维护逐一分析。结论：只做 Chromium 自动化且求简单选 Puppeteer MCP，需要跨浏览器、多语言支持和长期可靠维护选 Playwright MCP。"
date: 2026-08-23T17:25:14+08:00
lastmod: 2026-10-18T12:24:51+08:00
draft: false
weight: 100
aliases: ["/posts/puppeteer-vs-playwright-mcp/"]
categories: ["研究笔记"]
tags: ["研究笔记", "Agent Tools"]
---

> **原文链接**：[https://zhuanlan.zhihu.com/p/1891569128860534619](https://zhuanlan.zhihu.com/p/1891569128860534619)
>
> 作者：二师兄说 AI（互联网行业 Java 工程师）
>
> 收录于：MCP100 个案例

有人问：都是让 LLM 操作浏览器，Puppeteer MCP 和 Playwright MCP 到底差在哪？两个服务器最直观的差别在模型拿到的元素标识上。Puppeteer MCP 让模型给出 CSS selector；Playwright MCP 默认返回一棵可访问性树，模型拿到的是带 ref 的节点。两边都能完成点击，但页面一改版，失败的方式完全不同。

MCP（Model Context Protocol）在这里承担的是协议层：它把浏览器动作暴露成 tools，客户端通过 JSON-RPC 调用。两个 server 的协议层是一样的；差别在底层自动化库和交互抽象。Puppeteer 由 Chrome 团队开发，直接走 CDP；Playwright 由 Microsoft 开发，默认用可访问性树做元素定位。这两种抽象决定了后面的失败模式、性能边界和选型判断。

## Puppeteer MCP：CSS 选择器与 CDP 直连

Puppeteer MCP 把 Puppeteer 的能力包装成一组工具：`puppeteer_navigate` 导航，`puppeteer_screenshot` 截图（整页或某个 CSS 选择器），`puppeteer_click`、`puppeteer_hover`、`puppeteer_fill`、`puppeteer_select` 分别处理点击、悬停、输入和下拉选择，`puppeteer_evaluate` 允许模型提交一段 JavaScript 在页面里执行。服务器还会暴露控制台日志和截图这类资源，方便模型在失败后回看现场。这些工具的定位方式统一是 CSS selector；模型要先知道 DOM 结构，才能写出选择器。

这种设计在 Chromium 上很直接。Puppeteer 通过 Chrome DevTools Protocol 控制浏览器，路径短，启动和操作的开销容易控制；某些实现还支持连接到已经开启远程调试的 Chrome 实例，这对需要复用登录态、接管现有浏览器会话的场景很有用。安装也不复杂：原文给出的方式包括 `npm install -g puppeteer-mcp`、`npx puppeteer-mcp` 和 Docker 无头模式。代价是模型要承担选择器的脆弱性。CSS-in-JS 生成的哈希 class 每次构建都会变；iframe 和 shadow DOM 需要额外的穿透；虚拟列表只渲染可视区域，模型看到的节点和实际 DOM 可能对不上。这些问题不会让工具报错，只会让点击落在错误的元素上，或者等不到元素超时。

`puppeteer_evaluate` 解决复杂交互，也放大了权限边界。页面里的拖拽、canvas 绘制、需要读取内部状态的场景，用一段 JS 往往比让模型猜十次选择器更可靠。但这也意味着模型拿到了执行任意 JavaScript 的能力，权限边界要由部署方自己收：只读页面、限制可访问的域名、把浏览器放在隔离容器里，比在提示词里写“不要执行危险操作”更实际。

维护状态是 Puppeteer MCP 的另一个变量。原文采集信息时，npm 包 `@modelcontextprotocol/server-puppeteer` 的最近一次发布是四个月前；GitHub 上还有 `merajmehrabi/puppeteer-mcp-server`、`twolven/mcp-server-puppeteer-py` 等社区实现，活跃度各不相同。文档和示例散落在博客、Cursor Directory 和 npm 页面里，缺少一个集中的官方入口。这不等于不能用，但意味着选型时要把“谁来跟进 upstream”算进成本。`[待补：截至你读到这篇文章时，@modelcontextprotocol/server-puppeteer 与主要社区 fork 的最新版本、维护状态和安全公告]`

## Playwright MCP：把可访问性树变成模型的操作界面

Playwright MCP 的默认模式是快照模式。它调用 `browser_snapshot`，把页面的可访问性树（accessibility tree）整理成模型可读的结构，每个可交互节点带一个 ref；`browser_click` 接收元素描述和这个 ref，不需要模型生成 CSS selector。可访问性树描述的是“页面上有什么角色、叫什么名字、能不能操作”，这些信息比 DOM 的 class 名稳定得多；模型也不需要用视觉模型去猜坐标。服务器提供的工具覆盖导航、前进后退、点击、悬停、拖拽、输入、选择选项、选择文件、按键、快照、截图、保存 PDF、等待和关闭页面。传输层还支持通过 `--port` 启用 SSE，客户端可以用事件流接收结果。

快照模式的核心收益是确定性：同一个页面结构，模型拿到的是同一组 ref 和角色名；点击失败时，错误通常是“元素不存在”或“不可操作”，而不是“坐标偏了几个像素”。Playwright 在点击前会做 actionability 检查：元素要可见、稳定、启用，并且能接收事件；Puppeteer 主要等 selector 出现，元素是否被遮挡、是否在动画中，要自己写等待条件。这个差异在动态页面上很实际——一个正在展开的菜单，selector 已经存在，但点击可能落在动画中间。

快照模式也有边界。可访问性树依赖页面的语义结构：原生按钮、链接、表单控件、带 role/label 的自定义组件，在树里都有清晰的节点；如果页面是大量 `div` 加 canvas 画出来的，树里可能只剩几个 generic 节点，模型看不到可操作对象。这时可以切到视觉模式：用截图和坐标点击、拖拽、输入。视觉模式不依赖 ARIA，但坐标对 viewport、滚动位置和 devicePixelRatio 很敏感，页面一滚动或缩放，原来的坐标就失效；截图还会增加上下文成本。原文的结论是快照模式“依赖于结构良好的可访问性信息”，这句话应该反过来读：页面的语义质量决定了快照模式的上限。

Playwright MCP 的安装入口是 `npm install @playwright/mcp@latest`，VS Code 有专门的集成按钮和 CLI 命令；配置 MCP 客户端通常只需要指定 `npx @playwright/mcp@latest` 这条命令。命令行选项可以选浏览器、无头模式和监听端口。它的维护由 Microsoft 支持，原文采集时仓库有多位贡献者、近期有发布，这是它相对 Puppeteer MCP 的明显优势。`[待补：截至你读到这篇文章时，microsoft/playwright-mcp 的最新版本、发布节奏与已知的 breaking change]`

## 真正影响选型的几个差异

选型时最先要看的是元素定位的抽象。CSS selector 把“找元素”的责任交给模型，模型需要理解 DOM 结构、class 命名和页面层级；可访问性 ref 把“找元素”的责任交给浏览器，模型只需要理解角色和名字。前者在结构稳定、class 有语义的页面上很快，后者在动态页面和组件库页面上更稳。两者的失败模式也不同：selector 失效通常表现为找不到元素或点错元素；ref 失效通常表现为元素不可操作或快照里没有这个角色。排查时，前者要去看 DOM，后者要去看 ARIA。

浏览器覆盖范围是另一个维度。Puppeteer 主要面向 Chromium，Firefox 支持仍是实验性的；Playwright 覆盖 Chromium、Firefox 和 WebKit，Edge 走 Chromium 通道。这个差异只有在你的产品真的需要跨浏览器测试时才有意义。如果只做 Chrome 内部的自动化，跨浏览器支持不会带来收益，反而会增加 API 和调试的复杂度。

多语言支持这里容易误读。Playwright 作为库提供 JavaScript/TypeScript、Python、Java 和 .NET 绑定；但 `@playwright/mcp` 本身是一个 Node 服务器。MCP 客户端用什么语言都能连，因为协议走 JSON-RPC；Playwright 的多语言绑定不会自动变成 MCP server 的多语言实现。如果你的 Python 服务想直接调用 Playwright，应该用 Playwright 的 Python 库，而不是通过 MCP server 绕一圈。选 MCP server 时，客户端语言不是决定因素。

维护和社区状态也得算进去。原文采集时，Playwright MCP 由 Microsoft 维护、仓库活跃、近期有发布；Puppeteer MCP 的参考实现发布节奏慢，社区实现分散。这个对比有时效性，但方向值得注意：MCP 生态还在快速变化，一个 server 的协议兼容性、浏览器版本跟进和安全修复，比多一个工具更重要。

性能方面，公开资料里没有直接比较两个 MCP server 的 benchmark。能确定的机制有两条：Puppeteer 直接走 CDP，在 Chromium 上的控制路径短；Playwright 的快照模式省掉了截图编码和视觉模型调用，但可访问性树本身可能很大，token 成本不一定低。一个节点很多的页面，快照文本可能比一张压缩截图更占上下文；一个语义很差的页面，快照又可能信息不足。这两条机制都不能单独推出“谁更快”，只能作为压测假设。`[待补：同一组页面上，两个 MCP 的任务成功率、每步延迟、快照 token 数和总步数对比]`

还有一个常被忽略的点：连接已有浏览器。某些 Puppeteer MCP 实现支持连接已开启远程调试的 Chrome，这对复用登录态很有用；Playwright 也提供 `connectOverCDP`，可以接管已启动的 Chromium，或者用 persistent context 保留用户目录。这个能力不是 Puppeteer 独占，差异在使用示例和封装程度。选型时不要因为“要连现有 Chrome”就直接排除 Playwright。

从用例看，两者都能做导航、表单填写、截图、抓取 JavaScript 渲染的页面和自动化测试。Puppeteer MCP 更常见的组合是：用 `puppeteer_evaluate` 执行自定义脚本、监控控制台日志、生成文档截图；Playwright MCP 更常见的组合是：用快照做结构化数据提取、用 browser context 隔离多个会话、在测试里依赖 actionability 检查和网络拦截。真正需要做负载测试或远程调试时，两边都能搭，差别在周边工具和示例的成熟度。

## 怎么选

如果让我给一个默认判断：面向 LLM 的浏览器自动化，优先选 Playwright MCP。理由是快照模式把元素定位变成了结构化引用，模型不需要猜 CSS selector；Playwright 的 actionability 检查能挡掉一部分动态页面的误点击；Microsoft 的维护节奏也更适合长期依赖。这个默认判断的前提是页面有基本的语义结构——原生控件、合理的 role/label，或者至少能被可访问性树识别。

如果已经在用 Chrome 的远程调试端口，或者目标页面只在 Chromium 上跑，Puppeteer MCP 的直连路径更短。页面大量使用 canvas 或自定义渲染时，CSS selector 本来就不适用，可以用截图或 `puppeteer_evaluate` 直接操作；只想要很少几个工具、不想引入 Playwright 抽象层，也属于同一类需求。这些条件里，只有“接管现有 Chrome”是 Puppeteer 的传统强项；其他几条在 Playwright 里也有对应方案，选 Puppeteer 更多是为了更短的路径和更少的抽象。

如果页面语义很差，先别在快照模式上硬扛。Playwright MCP 的视觉模式可以兜底，但坐标点击的稳定性不如 ref；Puppeteer MCP 的 `puppeteer_evaluate` 可以直接读取页面内部状态，代价是给模型放开了 JavaScript 权限。两条路都要在隔离环境里跑，并且限制可访问的域名。对于 canvas 编辑器、地图、视频时间轴这类没有 DOM 语义的界面，最可靠的做法往往是为自动化提供一个隐藏的控制接口，而不是让模型去点像素。

最后，别把这篇对比当成静态结论。原文的发布记录是素材收集时的快照，MCP 生态几个月就会变一轮；选型前自己做一次小规模 A/B：同一组页面、同一批任务，记录成功率、步数、快照 token 和延迟。如果 Playwright 的快照在你们的页面上经常缺失关键元素，或者你确实需要接管一个已登录的 Chrome，结论可能反过来。`[待补：这组 A/B 的真实数据]`

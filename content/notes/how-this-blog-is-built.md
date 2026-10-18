---
title: "这个博客是怎么做的：Hugo + Congo 完整技术拆解"
description: "Hugo + Congo 搭建个人技术博客完整教程：主题选型、KaTeX 数学公式本地化、暗色模式配置、Cloudflare Pages 免费部署与自定义域名绑定。70 篇技术文章、200 多个页面构建仅需约 2 秒，首年成本约 70 元域名费，全程不写前端代码，适合想专注写作的研究员和工程师。"
date: 2025-01-25T10:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/how-this-blog-is-built/"]
categories: ["技术选型"]
tags: ["Hugo", "Congo", "Blog", "Static Site", "KaTeX"]
---

这个站点现在有 200 多个页面。改完一篇 Markdown，执行 `git push`，Cloudflare Pages 拉仓库、跑一次 `hugo --gc --minify`，整站重新生成大约 2 秒，然后把 `public/` 里的静态文件推到 CDN。没有数据库，没有常驻后端，也没有需要单独维护的搜索服务。

## 0. 需求与选型

搭这个博客之前，我列过一组硬需求。数学公式要能稳定渲染，代码块要能高亮和复制，写作过程要接近纯 Markdown，并且不需要写前端代码；这四项是 P0。暗色模式、全文搜索、部署零成本是 P1，缺了能忍，但有更好。研究员的日常是读论文、记公式、贴代码，博客必须让这些动作不打断写作。

静态站点生成器的候选里，Hugo 的构建速度最快，主题生态丰富，数学公式需要额外配置；Hexo 构建速度中等，中文主题多，公式靠插件；VitePress 构建快，公式内置，但偏 Vue 文档站点；Docusaurus 构建速度中等，背靠 React 生态，公式也要插件；Astro 构建快，Islands 架构适合内容为主、交互为辅的站点。这个站点是内容驱动，构建速度和 Markdown 工作流优先，Hugo 的单二进制、无运行时依赖更省心。

主题选了 Congo。它是为内容创作者设计的 Hugo 主题，默认暗色是 slate 色系，字体、行距和留白偏学术排版；搜索、代码复制、目录、RSS、多语言这些功能开箱即用，样式层用 Tailwind CSS。Tailwind 的好处是改样式不用跟主题作者的历史 CSS 打架，代价是主题直接 vendored 进仓库，升级要自己合并；对个人博客来说，这个交换比等主题发版更可控。

部署选 Cloudflare Pages。它的国内访问速度较快，免费额度对个人博客足够，SSL 证书自动续期，全球 CDN 分发。Vercel 也免费，但国内速度一般，更适合 Next.js 项目；GitHub Pages 免费且简单，国内访问较慢；阿里云 OSS 国内速度极快，但每月要花大约 ¥10-30 的存储和流量费。这个站点没有国内备案和商业流量需求，Cloudflare Pages 的免费层够用。

## 1. 整体架构

整站由 Hugo 生成静态 HTML、CSS 和 JavaScript。Hugo 是 Go 写的静态站点生成器，负责把 Markdown 和模板编译成 `public/` 目录；Congo 提供页面骨架、暗色主题和组件；KaTeX 负责数学公式，文件放在本地；Chroma 负责代码高亮；Fuse.js 在浏览器端做全文搜索，不需要后端索引服务。内容层是 Markdown，分类靠 categories 和 tags，图片和字体放在 `static/`，构建由一条 Hugo 命令完成，输出目录是 `public/`，最后交给 Cloudflare Pages 或 Vercel、OSS 之类的平台托管。

这个架构的关键是“构建期把能做的都做掉”。Markdown 解析、模板渲染、代码高亮、资源压缩都在构建时完成，浏览器只负责加载静态文件和执行 KaTeX、Fuse.js 这两段客户端脚本。代价是搜索索引要随文章数量一起下载，公式渲染要等 KaTeX 的 JS 和字体加载；好处是没有服务器要维护，也没有数据库连接和 API 超时。

## 2. 内容工作流

日常流程很短。我写一个 Markdown 文件，提交并 push，GitHub 或 Cloudflare 检测到变更后触发构建，Hugo 执行构建，生成 `public/` 目录，CDN 把新文件分发出去，读者刷新页面就能看到。对我而言只有两步：写 `content/notes/文章标题.md`，然后 `git push`。构建失败时 Cloudflare 会保留上一次成功的部署，站点不会因为一篇稿子的语法错误整体挂掉。

## 3. 文件结构

站点目录大致如下，配置按功能拆在 `config/_default/` 下。内容、模板和静态资源分开放，后面找文件不用翻整棵树：

```
site/                           # Hugo 站点根目录
├── config/_default/            # 配置文件（按功能拆分）
│   ├── hugo.toml               # 站点基础设置 + theme = "congo"
│   ├── languages.zh.toml       # 中文语言 + 作者信息 + 社交链接
│   ├── menus.zh.toml           # 顶部导航菜单
│   ├── params.toml             # 主题参数（暗色模式、搜索、代码复制）
│   └── markup.toml             # Markdown 渲染设置（公式 passthrough）
│
├── content/                    # 所有内容
│   ├── _index.md               # 首页
│   ├── about.md                # 关于页面
│   ├── series.md               # 精读系列总目录
│   ├── essays.md               # 随笔入口
│   └── notes/                  # 技术笔记与系列章节
│       ├── vllm-gpu-100-percent-max-tokens-trap.md
│       ├── logistic-regression-gradient-descent.md
│       ├── online-softmax-information-geometry.md
│       └── ... (70 篇)
│
├── layouts/                    # 自定义模板
│   ├── partials/extend-head.html
│   ├── _partials/recent-articles.html
│   ├── _shortcodes/series-cards.html
│   ├── notes/list.html         # 技术笔记流（过滤系列与随笔）
│   └── _default/landing.html   # 系列/随笔/关于的简洁页布局
│
├── assets/css/custom.css       # 系列卡片等自定义样式
├── i18n/zh.yaml                # 中文界面文案
├── static/                     # 静态资源（直接复制到输出目录）
│   └── katex/                  # KaTeX 本地文件（CSS + JS + 字体）
│
├── themes/congo/               # Congo 主题（直接 vendored 进仓库）
└── public/                     # 构建输出（部署用这个目录）
```

把主题放进 `themes/congo/`，没有用 Git submodule，是为了让模板覆盖和样式修改跟着主仓库走，部署时不依赖外部仓库的可用性。`layouts/` 里只放这个站点特有的覆盖：`extend-head.html` 挂 KaTeX，`notes/list.html` 过滤系列和随笔，`landing.html` 处理系列、随笔和关于页的简洁布局。`public/` 是构建产物，不手工编辑，部署平台只认这个目录。

## 4. 三个关键配置

数学公式的第一处配置在 `config/_default/markup.toml`。先让 Goldmark 别碰公式标记：

```toml
[goldmark.extensions.passthrough]
  enable = true
[goldmark.extensions.passthrough.delimiters]
  block = [['$$', '$$'], ['\\[', '\\]']]
  inline = [['$', '$'], ['\\(', '\\)']]
```

Goldmark 默认会把 `$` 当普通文本处理，公式里的反斜杠和下划线会被 Markdown 规则吃掉。开启 passthrough 后，`$...$` 和 `$$...$$` 原样输出到 HTML，由浏览器端的 KaTeX 渲染。这一步不开启，后面 KaTeX 配置得再对，公式也到不了它手里。

第二处是 KaTeX 的本地加载。`layouts/partials/extend-head.html` 引入本地 CSS 和 JS：

```html
<link rel="stylesheet" href="/katex/katex.min.css">
<script defer src="/katex/katex.min.js"></script>
<script defer src="/katex/auto-render.min.js"></script>
```

`layouts/partials/extend-footer.html` 负责在 DOM 加载完成后调用 `renderMathInElement`，把页面里的公式标记交给 KaTeX。渲染入口如下：

```javascript
document.addEventListener("DOMContentLoaded", function() {
  if (typeof renderMathInElement !== 'undefined') {
    renderMathInElement(document.body, {
      delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '$', right: '$', display: false},
        {left: '\\[', right: '\\]', display: true},
        {left: '\\(', right: '\\)', display: false}
      ],
      throwOnError: false
    });
  }
});
```

本地化是为了避开 CDN 的不确定性。cdn.jsdelivr.net 在中国大陆网络环境下可能加载失败，本地文件能保证全球访问一致；KaTeX 的 `.woff2` 字体也必须一起放到 `static/katex/fonts/`，否则中文 `\\text{}` 里的字会显示成方块。代价是升级 KaTeX 要手动替换 CSS、JS 和字体，不能靠 CDN 自动跟版本。

第三处是暗色模式默认开启。`config/_default/params.toml` 里设置：

```toml
defaultAppearance = "dark"
autoSwitchAppearance = true
```

`defaultAppearance` 决定首次访问的主题，`autoSwitchAppearance` 允许读取系统偏好并在用户手动切换后记住选择。两个都开，读者第一次进来是暗色，系统切到亮色时页面也会跟随；如果只想要固定暗色，把第二个关掉即可。

## 5. 部署到 Cloudflare Pages

代码先推到 GitHub。在站点根目录初始化仓库，添加远端，提交并推送：

```bash
cd site/
git init
git remote add origin https://github.com/你的用户名/blog.git
git add .
git commit -m "init hugo + congo"
git push -u origin main
```

然后在 Cloudflare Dashboard 里进 Pages，创建项目并连接 GitHub，选择 `blog` 仓库。构建设置填 Framework preset 为 `Hugo`，Build command 为 `hugo --gc --minify`，Build output directory 为 `public`，保存后触发第一次部署。这几个值和本地构建命令保持一致，避免线上和本地生成不同的目录结构。

自定义域名在 Pages 项目的“自定义域”里添加 `doraemg.com`，按提示修改 DNS 记录，SSL 证书大约 1-2 分钟自动签发。之后每次 `git push` 都会自动构建部署，证书续期也由平台处理。构建日志在 Pages 面板里能直接看，Hugo 模板报错和资源路径错误都会在这一步暴露。

## 6. 成本

域名是唯一的固定支出，在 Cloudflare Registrar 买 `.com` 大约 ¥70/年。托管用 Cloudflare Pages 免费版，CDN 和自动 HTTPS 包含在内，构建额度是每月 500 次，首年总成本约 ¥70，之后每年只续域名。如果换成阿里云 OSS 托管，存储和流量每月大约 ¥10-30，适合国内业务为主、需要备案和稳定回源的场景。

免费额度有边界。每月 500 次构建对个人博客够用，但如果一个仓库里同时放多个站点、每次提交都触发全量构建，额度会消耗得比想象中快。静态站点的成本低，是因为把计算放在了构建期和浏览器端；文章数量、图片体积和搜索索引增长之后，首屏加载和构建时间都会随之上升，这部分不会体现在账单里。

## 7. 常见问题

### Q1：公式显示为红色错误文本？

这通常是 KaTeX 解析失败，先从转义查起。`\\text{}` 内部的下划线 `_` 要写成 `\\_`，`&`、`%`、`#` 在公式里也要转义；再看 `static/katex/fonts/` 的字体文件是否完整，缺字体时公式可能能解析但显示异常。最后确认 `markup.toml` 的 passthrough 分隔符和 `extend-footer.html` 里的 `delimiters` 一致，两边不一致会导致公式根本没进入 KaTeX。

### Q2：如何添加新文章？

用 Hugo 的命令生成骨架。命令很短，路径按你的目录结构改：

```bash
hugo new content posts/文章标题.md
```

命令只是按模板生成文件，路径要跟 `content/` 下的目录结构一致；这个站点的技术笔记实际放在 `content/notes/`。生成后在文件头部填写 frontmatter：

```yaml
---
title: "文章标题"
date: 2025-01-25T10:00:00+08:00
draft: false
categories: ["研究笔记"]
tags: ["Tag1", "Tag2"]
---
```

`draft: false` 才会进入构建，categories 和 tags 决定归档和筛选。日期写成带时区的 ISO 格式，避免不同机器构建出不同的排序。

### Q3：如何本地预览？

在站点根目录执行下面的命令，先确认本地版本能起来。启动参数如下：

```bash
hugo server --buildDrafts
# 访问 http://localhost:1313
```

`--buildDrafts` 会把草稿也渲染出来，适合发布前检查。本地服务器带热重载，改 Markdown 或模板后浏览器会自动刷新；公式、搜索和暗色模式都能在本地验证，不需要每次 push 到 Cloudflare 看效果。

### Q4：如何备份？

代码和内容全部在 Git 仓库里，图片、字体等资源放在 `static/` 下也一并提交，定期 `git push` 就完成了备份。`public/` 是构建产物，不需要提交；需要恢复时，拉下仓库重新构建一次即可。域名和 DNS 记录不在仓库里，换注册商或误删解析时要单独留一份记录。

## 8. 这套方案的边界

这套组合的核心是“一个二进制文件、一个主题、一个托管平台、一种内容格式”：Hugo 负责构建，Congo 负责样式和功能，Cloudflare Pages 负责托管和 CDN，Markdown 负责内容。没有数据库和后端，配置量可控，代价是所有能力都停在静态站点能提供的范围内。评论、登录、服务端搜索这些需求要另外接服务，不在当前架构里。

具体到维护，有三笔账要提前算。搜索索引在浏览器端加载，文章越多，首次搜索要下载的数据越大；KaTeX 和 Congo 都 vendored 进仓库，升级要手动合并；Cloudflare Pages 的免费构建额度，推送频繁的仓库要留意。这三条不提前算，站点长大后会在搜索、升级和构建额度上依次碰到。

## 参考资源

Hugo 的官方文档在 https://gohugo.io/documentation/ ，配置项和模板函数以它为准。Congo 主题文档在 https://jpanther.github.io/congo/docs/ ，shortcode 和主题参数看这里。KaTeX 的命令支持表在 https://katex.org/docs/support_table.html ，写公式前可以先查哪些命令可用。Cloudflare Pages 的文档在 https://pages.cloudflare.com/ ，构建配置和自定义域名的细节都在上面。

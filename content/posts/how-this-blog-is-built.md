---
title: "这个博客是怎么做的：Hugo + Congo 完整技术拆解"
date: 2025-01-25T10:00:00+08:00
draft: false
categories: ["技术选型"]
tags: ["Hugo", "Congo", "Blog", "Static Site", "KaTeX"]
---

> 一篇 meta 文章：讲清楚你正在看的这个博客，底层是怎么运转的。
>
> 目标读者：想搭建个人技术博客，但不想折腾前端代码的研究员和工程师。

---

## 0. 前言：为什么选这套方案

搭建技术博客的方案很多，但研究员的需求很具体：

| 需求 | 优先级 |
|------|--------|
| 数学公式完美渲染 | P0 |
| 代码高亮 + 复制按钮 | P0 |
| 暗色模式默认 | P1 |
| 全文搜索 | P1 |
| 部署零成本 | P1 |
| 写作体验接近纯 Markdown | P0 |
| 不需要写前端代码 | P0 |

评估了一圈，**Hugo + Congo** 是这个需求组合下的最优解。

---

## 1. 整体架构

| 层级 | 组件 | 技术 |
|------|------|------|
| 前端展示 | 静态站点生成器 | Hugo (Go，极速构建) |
| | 主题 | Congo (Tailwind CSS，学术/极客风) |
| | 数学公式 | KaTeX (本地化，不依赖 CDN) |
| | 代码高亮 | Chroma (内置，支持复制按钮) |
| | 全文搜索 | Fuse.js (客户端搜索，无后端) |
| 内容层 | 文章格式 | Markdown (.md) |
| | 分类系统 | categories + tags |
| | 静态资源 | static/ (图片、字体、KaTeX) |
| 构建层 | 构建命令 | hugo --gc --minify |
| | 输出目录 | public/ (纯 HTML/CSS/JS) |
| | 部署平台 | Cloudflare Pages / Vercel / OSS |

---

## 2. 技术选型：为什么是这个组合

### 2.1 Hugo：最快的静态站点生成器

静态站点生成器（SSG）的选择：

| 工具 | 构建速度 | 主题生态 | 公式支持 | 适合场景 |
|------|---------|---------|---------|---------|
| **Hugo** | 极快 | 丰富 | 需配置 | **内容驱动，追求速度** |
| Hexo | 中等 | 中文主题多 | 插件 | 中文社区，快速上手 |
| VitePress | 快 | Vue 生态 | 内置 | 文档型站点 |
| Docusaurus | 中等 | React 生态 | 插件 | 多语言文档 |
| Astro | 快 | Islands 架构 | 插件 | 内容为主，交互为辅 |

Hugo 的核心优势是**构建速度**。本博客 36 篇文章、200+ 页面，构建耗时约 2 秒。本地预览几乎是实时的。

### 2.2 Congo：研究员风格的主题

Congo 是为**内容创作者**设计的 Hugo 主题：

- **暗色模式**：slate 色系，长时间阅读不刺眼
- **学术排版**：字体、行距、留白都偏向论文质感
- **内置功能**：搜索、代码复制、目录、RSS、多语言，开箱即用
- **Tailwind CSS**：现代、轻量、可定制

### 2.3 部署：Cloudflare Pages

| 方案 | 国内速度 | 价格 | 推荐度 |
|------|---------|------|--------|
| Cloudflare Pages | 较快 | 免费 | **首选** |
| Vercel | 一般 | 免费 | Next.js 项目 |
| GitHub Pages | 较慢 | 免费 | 极简个人站 |
| 阿里云 OSS | 极快 | ¥10-30/月 | 国内业务为主 |

Cloudflare Pages 绑定 GitHub 仓库后，每次 `git push` 自动构建部署，SSL 证书自动续期，全球 CDN 加速。对于个人博客，免费额度完全够用。

---

## 3. 核心工作流

```
你写 Markdown 文件
      ↓
git commit & push
      ↓
GitHub / Cloudflare 自动触发构建
      ↓
hugo --gc --minify
      ↓
生成 public/ 目录（纯静态文件）
      ↓
CDN 全球分发
      ↓
读者访问
```

**你的全部操作只有两步**：
1. 写 `content/posts/文章标题.md`
2. `git push`

---

## 4. 文件结构

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
│   └── posts/                  # 博客文章
│       ├── 快速开始指南.md
│       ├── logistic-regression-gradient-descent.md
│       ├── online-softmax-information-geometry.md
│       └── ... (36 篇)
│
├── layouts/partials/           # 自定义模板片段
│   ├── extend-head.html        # 注入 KaTeX CSS/JS
│   └── extend-footer.html      # KaTeX 渲染逻辑
│
├── static/                     # 静态资源（直接复制到输出目录）
│   └── katex/                  # KaTeX 本地文件（CSS + JS + 字体）
│
├── themes/congo/               # Congo 主题（git submodule）
└── public/                     # 构建输出（部署用这个目录）
```

---

## 5. 三个关键配置

### 5.1 数学公式支持

`config/_default/markup.toml`：

```toml
[goldmark.extensions.passthrough]
  enable = true
[goldmark.extensions.passthrough.delimiters]
  block = [['$$', '$$'], ['\\[', '\\]']]
  inline = [['$', '$'], ['\\(', '\\)']]
```

**作用**：让 Hugo 的 Markdown 解析器不处理 `$...$` 和 `$$...$$`，直接原样输出到 HTML，由 KaTeX 在浏览器端渲染。

**为什么需要 passthrough**：默认情况下，Hugo 会把 `$` 当作普通文本转义。开启 passthrough 后，公式标记才能完整保留。

### 5.2 KaTeX 本地化加载

`layouts/partials/extend-head.html`：

```html
<link rel="stylesheet" href="/katex/katex.min.css">
<script defer src="/katex/katex.min.js"></script>
<script defer src="/katex/auto-render.min.js"></script>
```

`layouts/partials/extend-footer.html`：

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

**为什么本地化**：
- CDN（如 cdn.jsdelivr.net）在中国大陆网络环境下可能加载失败
- 本地文件保证全球访问一致性
- KaTeX 字体文件（.woff2）也必须本地化，否则中文 `\\text{}` 显示为方块

### 5.3 暗色模式默认开启

`config/_default/params.toml`：

```toml
defaultAppearance = "dark"
autoSwitchAppearance = true
```

---

## 6. 部署指南：Cloudflare Pages

### Step 1：代码推送到 GitHub

```bash
cd site/
git init
git remote add origin https://github.com/你的用户名/blog.git
git add .
git commit -m "init hugo + congo"
git push -u origin main
```

### Step 2：Cloudflare Pages 配置

1. 登录 [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Pages → 创建项目 → 连接 GitHub
3. 选择 `blog` 仓库
4. 构建设置：
   - **Framework preset**: `Hugo`
   - **Build command**: `hugo --gc --minify`
   - **Build output directory**: `public`
5. 点击保存并部署

### Step 3：绑定自定义域名

1. Cloudflare Pages 项目 → 自定义域
2. 添加 `doraemg.com`
3. 按提示修改 DNS 记录
4. 自动 SSL 证书签发（约 1-2 分钟）

**以后每次 `git push` 自动构建部署。**

---

## 7. 成本

| 项目 | 成本 | 备注 |
|------|------|------|
| 域名 (.com) | ¥70/年 | Cloudflare Registrar |
| 托管 | ¥0/年 | Cloudflare Pages 免费版 |
| CDN | ¥0/年 | 全球节点，自动 HTTPS |
| 构建 | ¥0/年 | 500 次/月免费构建 |
| **首年总计** | **约 ¥70** | 后续每年仅域名续费 |

---

## 8. 常见问题

### Q1：公式显示为红色错误文本？

通常是 KaTeX 解析失败。检查：
- `\\text{}` 内部的下划线 `_` 需要转义为 `\\_`
- 特殊符号如 `&`、 `%`、 `#` 在公式中需要转义
- 确保 `static/katex/fonts/` 字体文件完整

### Q2：如何添加新文章？

```bash
hugo new content posts/文章标题.md
```

然后在文件头部填写 frontmatter：

```yaml
---
title: "文章标题"
date: 2025-01-25T10:00:00+08:00
draft: false
categories: ["研究笔记"]
tags: ["Tag1", "Tag2"]
---
```

### Q3：如何本地预览？

```bash
hugo server --buildDrafts
# 访问 http://localhost:1313
```

### Q4：如何备份？

代码和内容全部在 Git 仓库中。图片等资源放在 `static/` 目录下也一并提交。定期 `git push` 即完成备份。

---

## 9. 总结

> **Hugo 把 Markdown 转成静态网页，Congo 决定长什么样，Cloudflare Pages 免费托管，KaTeX 负责公式，你只管写。**

这套方案的核心设计哲学是**简单**：
- 一个二进制文件（Hugo）搞定构建
- 一个主题（Congo）搞定设计和功能
- 一个平台（Cloudflare）搞定托管和 CDN
- 一种格式（Markdown）搞定内容

没有数据库，没有后端，没有复杂配置。内容是你的，代码是你的，域名是你的。这才是个人博客该有的样子。

---

## 参考资源

| 资源 | 链接 | 说明 |
|------|------|------|
| Hugo 官方文档 | https://gohugo.io/documentation/ | 最权威的配置参考 |
| Congo 主题文档 | https://jpanther.github.io/congo/docs/ | 主题特有的 shortcode 和参数 |
| KaTeX 支持表 | https://katex.org/docs/support_table.html | 查看哪些 LaTeX 命令可用 |
| Cloudflare Pages | https://pages.cloudflare.com/ | 托管平台官方文档 |

---

> 如果你正在考虑搭建自己的技术博客，希望这篇文章能帮你少走一些弯路。有任何问题，欢迎通过博客首页的联系方式找到我。

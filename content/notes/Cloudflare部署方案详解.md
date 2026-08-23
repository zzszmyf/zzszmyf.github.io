---
title: "Cloudflare 部署方案详解"
date: 2024-03-22T00:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/Cloudflare部署方案详解/"]
categories: ["技术选型"]
tags: ["Cloudflare", "部署", "CDN", "Pages"]
---

# Cloudflare 部署方案详解

> 域名 + DNS + 部署 + CDN 一站式服务

---

## 一、Cloudflare 能做什么

如果你选择 **Cloudflare 买域名**，可以一站式解决：

```
域名注册 (Cloudflare Registrar)
    ↓
DNS 解析 (Cloudflare DNS) - 全球最快
    ↓
部署托管 (Cloudflare Pages) - 免费
    ↓
CDN 加速 (Cloudflare CDN) - 全球 300+ 节点
    ↓
安全防护 (Cloudflare Security) - DDoS 防护
```

**一个账号，全部搞定**。

---

## 二、Cloudflare Pages 部署

### 是什么

Cloudflare Pages 是 Cloudflare 提供的**静态网站托管服务**，类似 Vercel，但有自己的特点。

### 核心特点

| 特性 | 详情 |
|------|------|
| **价格** | **免费**（无限带宽、无限请求、1000 构建次数/月） |
| **国内访问** | **比 Vercel 快**（Cloudflare 在国内有节点） |
| **Git 集成** | 支持 GitHub/GitLab，自动部署 |
| **构建支持** | VitePress、Hexo、Hugo、Next.js、Astro 等 |
| **预览环境** | 每个 PR 自动生成预览链接 |
| **自定义域名** | 支持，自动 HTTPS |
| **边缘函数** | 支持 Pages Functions（类似 Vercel Edge） |

### 与 Vercel 对比

| 维度 | Cloudflare Pages | Vercel |
|------|------------------|--------|
| **国内访问速度** | ✅ **更快** | ⚠️ 偶尔慢 |
| **国外访问速度** | ✅ 快 | ✅ 快 |
| **GitHub 集成** | ✅ 好 | ✅ 更好 |
| **构建速度** | ✅ 快 | ✅ 快 |
| **免费额度** | ✅ **无限带宽** | 100GB/月 |
| **边缘函数** | ✅ Pages Functions | ✅ Edge Functions |
| **Serverless** | ❌ 不支持 | ✅ 支持 |
| **团队成员** | ✅ 无限 | 1 人（免费版） |
| **分析统计** | 基础 | 更详细 |

**结论**：
- **国内访问优先** → Cloudflare Pages
- **需要 Serverless 功能** → Vercel
- **纯静态博客** → 两者都可以，Cloudflare 国内更快

---

## 三、部署流程（VitePress + Cloudflare Pages）

### 方案 A：Git 自动部署（推荐）

```
GitHub 仓库
    ↓
Cloudflare Pages 连接仓库
    ↓
推送代码 → 自动构建 → 自动部署
```

**配置步骤**：
1. 代码推送到 GitHub
2. Cloudflare Dashboard → Pages → Create a project
3. 选择 GitHub 仓库
4. 构建命令：`npm run docs:build`
5. 输出目录：`docs/.vitepress/dist`
6. 保存，自动部署

### 方案 B：直接上传（无 Git）

```
本地构建 → 生成 dist 文件夹 → 拖拽上传到 Cloudflare Pages
```

适合：不想用 Git，偶尔更新的人

---

## 四、域名绑定

### 情况 1：域名在 Cloudflare 买的

**最简单**：
1. Pages 项目 → Custom domains
2. 输入 `doraemg.com`
3. 自动配置 DNS，立即生效

### 情况 2：域名在其他平台买的

1. 在域名平台修改 DNS 服务器为 Cloudflare 的：
   - `lara.ns.cloudflare.com`
   - `greg.ns.cloudflare.com`
2. 然后在 Cloudflare Pages 绑定域名

---

## 五、国内访问优化

### Cloudflare 的国内情况

- Cloudflare 与 **百度、京东、网宿** 等合作，在国内有节点
- 但 **部分节点受限于备案**，未备案域名可能走海外线路

### 优化方案

| 方案 | 效果 | 操作 |
|------|------|------|
| **使用 Cloudflare 中国合作伙伴** | 最快 | 接入百度云加速、又拍云等 |
| **备案 + Cloudflare** | 快 | 域名备案后，国内节点生效 |
| **默认使用** | 比 Vercel 好 | 不操作，直接用 |

**对于你的情况**（doraemg.com，海外注册）：
- 直接用 Cloudflare Pages，国内访问会比 Vercel 好
- 如果将来客户要求极高，可以备案后接入国内 CDN

---

## 六、总成本计算

### 方案：Cloudflare 全家桶

| 项目 | 费用 | 说明 |
|------|------|------|
| 域名 (doraemg.com) | $9.77/年 (~¥70) | Cloudflare Registrar |
| 部署 (Cloudflare Pages) | **免费** | 无限带宽 |
| DNS (Cloudflare DNS) | **免费** | 全球最快 |
| CDN (Cloudflare CDN) | **免费** | 全球节点 |
| SSL 证书 | **免费** | 自动续期 |
| **总计** | **¥70/年** | 极低！ |

对比：
- Vercel + 阿里云域名：¥70/年（但国内访问稍慢）
- 阿里云 OSS + CDN：¥70 + ¥30/月 = ¥430/年

---

## 七、快速启动 Checklist

### 今天完成
- [ ] 注册 Cloudflare 账号（https://dash.cloudflare.com/sign-up）
- [ ] 购买 doraemg.com（等待开放注册或去 Namecheap）
- [ ] 创建 GitHub 仓库（存放博客代码）

### 本周完成
- [ ] 初始化 VitePress 项目
- [ ] 推送到 GitHub
- [ ] Cloudflare Pages 连接仓库，完成部署
- [ ] 绑定域名 doraemg.com
- [ ] 发布第一篇文章

---

## 八、常见问题

**Q: Cloudflare Pages 有墙的风险吗？**  
A: 很低。Cloudflare 是基础设施服务商，不是内容平台，被墙概率极小。

**Q: 可以同时用 Vercel 和 Cloudflare Pages 吗？**  
A: 可以，主站用一个，另一个做备份。或国内 DNS 解析到 Cloudflare，国外解析到 Vercel。

**Q: Cloudflare 有中文界面吗？**  
A: 部分有，但主要英文。不过配置简单，跟着文档点就行。

**Q: 备案问题？**  
A: 服务器在国外（Cloudflare Pages 节点全球分布），不需要备案。但如果想要国内极致速度，建议备案后接入国内 CDN。

---

## 九、我的建议

对于你（doraeMeng，AI 落地顾问）：

```
推荐方案：Cloudflare 全家桶

域名：Cloudflare Registrar (doraemg.com) - $9.77/年
部署：Cloudflare Pages - 免费
DNS：Cloudflare DNS - 免费
CDN：Cloudflare CDN - 免费

优点：
✅ 一个平台管理所有
✅ 国内访问比 Vercel 快
✅ 成本最低（¥70/年）
✅ 技术形象好（Cloudflare 是开发者认可的 infra）
```

**备选**：如果 Cloudflare 域名注册没开放，就用 **Namecheap 买域名 + Cloudflare Pages 部署**。

---

> 💡 一句话：Cloudflare 可以搞定域名+部署+CDN，国内访问比 Vercel 好，成本最低，推荐。

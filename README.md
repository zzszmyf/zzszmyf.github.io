# zzszmyf.github.io

孟一凡（[zzszmyf](https://github.com/zzszmyf)）的个人站点源码：LLM 推理优化、AI Agent 基础设施与工程实践的笔记。

📖 **在线浏览：<https://zzszmyf.github.io/>**

写这些笔记的原因很简单：我把一条技术链路从论文读到生产部署时踩过的坑，尽量原样留下来——
公式、配置、报错现象和最后的结论。不是教程汇编，是自己动手时的记录。

## 这里有什么

**LLM 推理优化精读笔记**（MIT lecture note 级别，Markdown + LaTeX，公式可渲染）

- [量化（有损换速度）](https://zzszmyf.github.io/notes/llm%E9%87%8F%E5%8C%96%E7%B2%BE%E8%AF%BB%E7%AC%94%E8%AE%B0-00-%E6%80%BB%E8%A7%88%E4%B8%8E%E5%AD%A6%E4%B9%A0%E5%9C%B0%E5%9B%BE/)：
  信息论与数值编码 → 均匀量化理论 → 数值格式与硬件 → 粒度/校准/离群值 → PTQ/QAT
  （GPTQ / AWQ / SmoothQuant / KIVI / QLoRA / BitNet）→ 质量评估 → 系统部署
- [推测解码（无损换步数）](https://zzszmyf.github.io/notes/llm%E6%8E%A8%E6%B5%8B%E8%A7%A3%E7%A0%81%E7%B2%BE%E8%AF%BB%E7%AC%94%E8%AE%B0-00-%E6%80%BB%E8%A7%88%E4%B8%8E%E5%AD%A6%E4%B9%A0%E5%9C%B0%E5%9B%BE/)：
  接受率数学 → 原始推测解码 → Medusa → EAGLE → n-gram/检索式路线 → 生产验收
- [注意力与计算内核](https://zzszmyf.github.io/notes/llm%E6%B3%A8%E6%84%8F%E5%8A%9B%E5%86%85%E6%A0%B8%E7%B2%BE%E8%AF%BB%E7%AC%94%E8%AE%B0-00-%E6%80%BB%E8%A7%88%E4%B8%8E%E5%AD%A6%E4%B9%A0%E5%9C%B0%E5%9B%BE/)：
  注意力机制与复杂度 → FlashAttention → MQA/GQA/MLA → 稀疏与线性注意力 → PagedAttention → 内核优化

此外还有工程实践（vLLM 部署排查、MoE 通信、训练框架、Agent 上下文与记忆）和一部分
技术商业化、团队协作的随笔。完整列表见 [全部笔记](https://zzszmyf.github.io/notes/)。

## 技术栈

Hugo + [Congo](https://github.com/jpanther/congo) 主题（主题直接 vendored 进仓库），
GitHub Actions 构建并部署到 GitHub Pages，推 `main` 即发布。

- 内容在 `content/notes/`，一篇一个 Markdown 文件；文件名决定 URL（Hugo 会转小写）
- 站内链接必须写成 Hugo 实际生成的小写路径，否则是 404
- 部署后由 `.github/scripts/indexnow.py` 把改动的 URL 主动推给 Bing / Yandex（IndexNow）

## 本地构建

```bash
hugo server          # 本地预览，http://localhost:1313
hugo --gc --minify    # 构建到 public/
```

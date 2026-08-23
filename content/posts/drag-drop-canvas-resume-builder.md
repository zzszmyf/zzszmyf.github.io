---
title: "拖拽式简历编辑器：画布与拖拽技术选型指南"
date: 2026-05-13T10:00:00+08:00
draft: false
categories: ["前端开发"]
tags: ["React", "拖拽", "Canvas", "tldraw", "Fabric.js", "dnd-kit", "简历编辑器", "前端选型"]
---

# 拖拽式简历编辑器：画布与拖拽技术选型指南

做一个拖拽生成简历的助手工具，核心问题只有一个：**用什么技术承载"拖 + 排 + 编"**。本文梳理了 React 生态中主流的画布和拖拽方案，覆盖从模板填充到自由排版的完整光谱。

---

## 两种产品形态

在选技术之前，先想清楚你的简历编辑器长什么样：

| 类型 | 体验 | 类比 | 推荐方案 |
|---|---|---|---|
| **模板填充** | 固定模板，拖拽区块排序填充 | 超级简历、Notion 拖拽排序 | dnd-kit |
| **自由排版** | 元素在画布上任意定位、缩放 | Canva、Figma、稿定设计 | tldraw / Fabric.js |

两者本质区别在于：**元素位置是流式排列还是自由坐标系**。

---

## 方案一：模板填充 —— dnd-kit

`@dnd-kit/core` 是目前 React 生态最成熟的拖拽排序库，适合"左侧区块面板 → 右侧简历区域"的场景。

```
npm install @dnd-kit/core @dnd-kit/sortable @dnd-kit/utilities
```

**优点**：
- 专注于列表排序和容器间拖放，API 清晰
- 支持键盘可访问性，动画流畅
- React 原生思维方式，无 DOM 操作心智负担

**不足**：
- 不做画布，没有坐标系、缩放、旋转等概念
- 元素位置由 DOM 流决定，不能自由摆放

**适用场景**：简历模板固定，用户选择模块、排序、填写内容即可导出。

---

## 方案二：自由排版 —— 两大画布库对比

如果要做 Canva 风格的简历编辑器——文字、图片、形状在无限画布上自由排版——就需要一个真正的画布引擎。

### Fabric.js：Canvas 2D 操作库

Fabric.js 封装了 HTML5 Canvas API，将每个图形元素抽象为对象，内置选中、拖拽、缩放、旋转变换手柄。

**无限画布的实现**需要手动控制 `viewportTransform`：

```js
// 平移：mousedown 记录起始点，mousemove 偏移矩阵
canvas.on('mouse:down', (opt) => {
  if (opt.target === null) { // 点到空白区域才平移
    canvas.isDragging = true
    lastPos = { x: opt.e.clientX, y: opt.e.clientY }
  }
})
canvas.on('mouse:move', (opt) => {
  if (canvas.isDragging) {
    const vpt = canvas.viewportTransform
    vpt[4] += opt.e.clientX - lastPos.x
    vpt[5] += opt.e.clientY - lastPos.y
    canvas.requestRenderAll()
  }
  lastPos = { x: opt.e.clientX, y: opt.e.clientY }
})

// 缩放：滚轮，以鼠标位置为中心
canvas.on('mouse:wheel', (opt) => {
  const delta = opt.e.deltaY
  let zoom = canvas.getZoom()
  zoom *= 0.999 ** delta
  zoom = Math.max(0.1, Math.min(20, zoom))
  canvas.zoomToPoint({ x: opt.e.offsetX, y: opt.e.offsetY }, zoom)
})
```

**文字编辑**：内置 `IText` 对象，支持双击进入编辑模式、样式设置、多行文本。

**导出**：`canvas.toJSON()` 保存工程，`canvas.toDataURL()` 导出图片。

| 优点 | 缺点 |
|---|---|
| 完全自由，每个像素可控 | 平移/缩放/选择框需手写样板代码 |
| `IText` 文字编辑开箱即用 | 大量对象时需自己做视口裁剪 |
| 生态成熟，文档齐全 | React 集成需手动管理 ref 和生命周期 |
| 打包体积小（~200KB） | 复杂交互（多选、Shift 等）需要自己实现 |

### tldraw：开源白板引擎

tldraw 是 GitHub 40k+ stars 的开源白板，本身就是"无限画布 + 拖拽编辑"的最佳实践。

```bash
npm install tldraw
```

```tsx
import { Tldraw } from 'tldraw'
import 'tldraw/tldraw.css'

function ResumeEditor() {
  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <Tldraw
        onMount={(editor) => {
          // 定制工具栏、注册自定义 shape
        }}
      />
    </div>
  )
}
```

**tldraw 给你的都是现成的**：

- 无限画布，开箱即用（平移、缩放、网格吸附）
- 选择框、多选（Shift）、Delete 删除
- 双击文字编辑，体验打磨到位
- 内置形状、图片、箭头等基础元素
- 可通过 Custom Shape API 注册简历专属区块（如"工作经历卡片"）
- 可隐藏不需要的工具栏，暴露简历定制工具

| 优点 | 缺点 |
|---|---|
| 上手成本极低，几乎零配置 | 打包体积较大（~500KB+） |
| 交互细节已打磨（选择、对齐、吸附） | 深度定制受限于 tldraw 框架 |
| React 原生组件，状态管理走 React | 画布自带白板风格，需要定制才能像"简历工具" |
| 活跃维护，社区活跃 | 偏白板用途，某些设计工具细节需额外适配 |

---

## 方案选择：Fabric.js vs tldraw

| 维度 | Fabric.js | tldraw |
|---|---|---|
| 上手成本 | 中，需手写平移/缩放/选择逻辑 | 低，开箱即用 |
| 无限画布 | 手动实现 | 内置 |
| 文字编辑 | `IText` 可用 | 双击编辑，体验更好 |
| 自定义程度 | 完全自由 | 中高，Custom Shape API |
| 导出能力 | JSON / SVG / DataURL | SVG / Image |
| 打包体积 | ~200KB | ~500KB+ |
| 适合场景 | 需要完全控制每像素的大型设计工具 | 快速搭建、交互优先的设计工具 |

**简单结论**：

- 如果你想快速出 MVP，**选 tldraw**。把精力花在简历业务逻辑（如定制 Section Shape、导出 PDF）上，而不是从头实现画布交互。
- 如果你要做一款和 Canva 完全对标的产品，**选 Fabric.js**。它能给你像素级的控制力，但也意味着你需要自己写更多代码。

---

## dnd-kit + tldraw 能结合吗？

结论：**可以，但没必要。**

两者各自解决了重叠的问题——dnd-kit 做结构化拖拽排序，tldraw 做自由画布编辑。强行结合会引入以下问题：

1. **拖拽事件冲突**：两者各自劫持 pointer 事件，从 dnd-kit 的 Draggable 拖入 tldraw 画布时容易打架
2. **坐标系转换**：dnd-kit 使用 DOM 坐标，tldraw 使用画布坐标，需要手动转换
3. **功能重叠**：若只是"左侧点按钮 → 画布上创建元素"，用 tldraw 的 `editor.createShape()` API 就够，不需要 dnd-kit

**更实用的做法**：用纯 HTML/CSS 做个简洁侧边栏，点击触发 `editor.createShape()` 在画布上创建对应元素。这比拖放更可控，代码也更简单。

```tsx
function Sidebar({ editor }) {
  return (
    <div className="sidebar">
      <button onClick={() => {
        editor.createShape({
          type: 'geo',
          x: 100, y: 100,
          props: { geo: 'rectangle', w: 600, h: 200 }
        })
      }}>
        添加工作经历模块
      </button>
    </div>
  )
}
```

---

## 总结

| 你想要的效果 | 推荐方案 |
|---|---|
| 像超级简历一样模板填充 | dnd-kit |
| 像 Canva 一样自由排版，快速上线 | tldraw |
| 像 Canva 完全定制，像素级控制 | Fabric.js |
| 左侧拖拽 → 画布自由排版 | tldraw + 自定义侧边栏（不需要 dnd-kit） |

选择取决于你想在"快速出活"和"深度定制"之间拿捏哪个点。对于大多数简历助手场景，**tldraw 是性价比最高的起点**。

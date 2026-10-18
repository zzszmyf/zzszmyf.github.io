---
title: "拖拽式简历编辑器：画布与拖拽技术选型指南"
description: "React 拖拽简历编辑器技术选型：模板填充场景用 dnd-kit，自由排版场景用 tldraw 或 Fabric.js，含 Fabric.js 无限画布平移缩放的代码示例与两种产品形态对比，帮你按「拖、排、编」需求选对画布与拖拽方案。"
date: 2026-05-13T10:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/drag-drop-canvas-resume-builder/"]
categories: ["前端开发"]
tags: ["React", "拖拽", "Canvas", "tldraw", "Fabric.js", "dnd-kit", "简历编辑器", "前端选型"]
---

简历编辑器最容易走错的一步，是先选拖拽库，再决定画布。需求如果只是「左侧模块拖到右侧固定模板里排序」，dnd-kit 就够了；如果用户要在无限画布上把「工作经历」拖到任意坐标、缩放、旋转，dnd-kit 连坐标系这个概念都没有，后面补多少层封装都补不出来。两种产品形态的差别最终落在元素位置上：元素是跟着 DOM 流走，还是拥有一个独立的二维变换矩阵。

模板填充型产品——超级简历、Notion 式拖拽排序——元素在文档流里，用户拖的是区块的顺序和归属。自由排版型产品——Canva、Figma、稿定设计——元素在画布上有绝对坐标，拖拽只是修改坐标和变换。先把这件事定下来，再谈库的选型。

## 模板填充：dnd-kit 的边界

`@dnd-kit/core` 是目前 React 生态里最成熟的拖拽排序库之一，配套还有 `@dnd-kit/sortable` 和 `@dnd-kit/utilities`。安装只需要一条命令：

```
npm install @dnd-kit/core @dnd-kit/sortable @dnd-kit/utilities
```

它专注列表排序和容器间拖放，API 清晰，支持键盘可访问性，动画流畅，思维方式是 React 原生的，不需要直接操作 DOM。这些优点也划出了它的边界：它不做画布，没有坐标系、缩放、旋转的概念；元素位置由 DOM 流决定，不能自由摆放。拿它做模板填充很合适——简历模板固定，用户选择模块、排序、填写内容，最后导出；拿它做自由排版，就会在「把 DOM 元素拖到画布坐标」这一步卡住。

## 自由排版：Fabric.js 给你像素级控制

Fabric.js 封装了 HTML5 Canvas API，把每个图形元素抽象成对象，内置选中、拖拽、缩放、旋转变换手柄。做无限画布需要手动控制 `viewportTransform`。下面两段代码分别处理平移和以鼠标位置为中心的缩放：

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

`viewportTransform` 是一个 3x3 的仿射变换矩阵，数组下标 4 和 5 对应平移的 x/y；mousedown 在空白处记录起点，mousemove 把鼠标位移累加进这两个分量，再调用 `requestRenderAll` 重绘。缩放用 `0.999 ** delta` 做指数衰减，把滚轮增量映射成平滑的缩放系数，再用 `Math.max(0.1, Math.min(20, zoom))` 把范围卡在 0.1 到 20 倍；`zoomToPoint` 的作用是让鼠标指向的画布点在缩放前后保持不动，否则用户会觉得画布在脚下滑。

文字编辑用内置的 `IText` 对象，双击进入编辑模式，支持样式设置和多行文本；导出用 `canvas.toJSON()` 保存工程，用 `canvas.toDataURL()` 导出图片。这些能力开箱即用，生态成熟、文档齐全，但平移、缩放、选择框、多选、Shift 连选、大量对象时的视口裁剪都要自己写；React 集成还要手动管理 ref 和生命周期。材料给出的打包体积约 200KB，这个数字是否包含 gzip、对应哪个版本没有说明，[待补：Fabric.js 目标版本的 gzip 体积与视口裁剪实现成本]。

## 自由排版：tldraw 给你现成白板

tldraw 是 GitHub 40k+ stars 的开源白板，本身就是「无限画布 + 拖拽编辑」的成熟实现。安装和使用都很直接：

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

无限画布、平移缩放、网格吸附、选择框、多选（Shift）、Delete 删除、双击文字编辑、内置形状/图片/箭头，这些它都给了；它本身是 React 组件，状态管理跟着 React 走，你还可以通过 Custom Shape API 注册简历专属区块，比如「工作经历卡片」，或者隐藏不需要的工具栏，只暴露简历定制工具。代价是打包体积较大（材料给的量级是 500KB+，同样没有说明压缩口径），深度定制受 tldraw 框架约束，画布自带白板风格，某些设计工具细节要额外适配，要把它改成「简历工具」还需要一层 UI 定制。项目维护活跃，这也是它能作为 MVP 起点的一个原因。它适合快速出 MVP，把精力花在简历业务逻辑和导出上，而不是从头实现画布交互。

## Fabric.js 和 tldraw 怎么选

Fabric.js 和 tldraw 的选择可以落到几个具体问题上。上手成本：Fabric.js 需要自己写平移、缩放、选择逻辑，tldraw 基本零配置；无限画布：Fabric.js 手动实现，tldraw 内置；文字编辑：Fabric.js 的 IText 可用，tldraw 双击编辑的体验更完整；自定义程度：Fabric.js 完全自由，tldraw 中高，靠 Custom Shape API；导出：Fabric.js 支持 JSON、SVG、DataURL，tldraw 支持 SVG、Image；适合场景：Fabric.js 适合需要完全控制每个像素的大型设计工具，tldraw 适合快速搭建、交互优先的产品。把这些摊开后，结论并不复杂：想快速出 MVP，选 tldraw；要做和 Canva 完全对标的产品，选 Fabric.js，并接受自己写更多交互代码。

## dnd-kit 和 tldraw 要不要拼起来

可以拼，但通常没必要。两者都监听 pointer 事件，从 dnd-kit 的 Draggable 拖进 tldraw 画布时，两个传感器会争夺同一个手势；dnd-kit 用 DOM 坐标，tldraw 用画布坐标，跨越边界要手动换算；功能上也有重叠——如果只是「左侧点按钮，画布上创建元素」，用 tldraw 的 `editor.createShape()` 就够了，不需要 dnd-kit。更实用的做法是用纯 HTML/CSS 做一个侧边栏，点击按钮触发 `createShape()`，把元素放到画布上。核心就是下面这段：

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

这比拖放更可控：点击的语义明确，坐标由代码决定，不需要在两个坐标系统之间协调手势。代价是少了「拖」的直观感，但对简历这种结构化内容，点击添加往往比自由拖拽更符合用户心智。

如果产品停在模板填充，dnd-kit 就够了，别为了「像 Canva」上画布；如果承诺自由排版，先用 tldraw 把 MVP 跑通，再评估是否值得为像素级控制迁到 Fabric.js。真正没解决的是导出：简历最终要落在 A4 纸上，画布坐标到分页的换算、文字断行、打印边距，这两套库都不会替你做；材料里也没有给出方案，[待补：画布坐标到 A4 分页的换算与 PDF 导出实现]。

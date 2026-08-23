---
title: "Procfile、Nixpacks 与 Shebang：三个部署/脚本小知识"
date: 2026-05-08T17:21:00+08:00
draft: false
weight: 100
aliases: ["/posts/procfile-nixpacks-shebang-notes/"]
categories: ["工程笔记"]
tags: ["部署", "PaaS", "Linux", "Shell", "Nixpacks"]
---

> 日常对话中随手提到的三个小知识，沉淀下来作为速查笔记。
>
> 目标读者：刚接触 PaaS 部署或写脚本时对这些配置文件/语法感到陌生的工程师。

---

## 一、Procfile：运行时进程声明

### 是什么

Procfile 源自 Heroku，用于**声明应用启动时要运行哪些进程**。它的名字就是 "Process File" 的缩写。

### 基本格式

```
<进程名>: <启动命令>
```

### 常见例子

```procfile
web: gunicorn app:app --bind 0.0.0.0:$PORT
worker: celery -A tasks worker --loglevel=info
scheduler: python cron_jobs.py
```

| 进程名 | 含义 | 说明 |
|--------|------|------|
| `web` | HTTP 服务进程 | PaaS 平台会自动分配端口、路由和健康检查 |
| `worker` | 后台任务进程 | 处理队列任务，如 Celery、Rq |
| `scheduler` | 定时任务进程 | 如 cron、APScheduler |

### 平台支持情况

| 平台 | 对 Procfile 的支持 | 备注 |
|------|-------------------|------|
| **Heroku** | ✅ 原生支持 | Procfile 的发源地 |
| **Railway** | ✅ 支持 | 优先读取 `web:` 作为启动命令 |
| **Render** | ✅ 支持 | 需要显式指定 |
| **Dokku** | ✅ 支持 | 自托管 Heroku 替代方案 |
| **Fly.io** | ❌ 不支持 | 用 `Dockerfile` 或 `fly.toml` |
| **Vercel** | ❌ 不支持 | Serverless Functions，不需要 |

### 优先级规则

当同时存在 `nixpacks.toml` 和 `Procfile` 时，**Nixpacks 会优先读取 Procfile 中的 `web:` 作为启动命令**，相当于 `nixpacks.toml` 里的 `[start]` 配置被覆盖。

---

## 二、nixpacks.toml：构建时容器配置

### 是什么

Nixpacks 是 Railway 开发的**源码→容器镜像**构建工具。它会自动检测你的项目类型（Python、Node.js、Go、Rust 等），然后用 Nix 包管理器构建出一个可复现的容器镜像。

`nixpacks.toml` 是它的配置文件，用于**覆盖或补充自动检测的行为**。

### 基本结构

```toml
[phases.setup]
nixPkgs = ["python311", "gcc"]

[phases.install]
cmds = ["pip install -r requirements.txt"]

[phases.build]
cmds = ["npm run build"]

[start]
cmd = "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
```

| 阶段 | 作用 | 类比 |
|------|------|------|
| `phases.setup` | 安装系统级依赖（如 Python、Node、GCC）| Dockerfile 的 `FROM` + `RUN apt-get` |
| `phases.install` | 安装项目依赖（如 pip、npm）| Dockerfile 的 `RUN pip install` |
| `phases.build` | 执行构建命令（如编译、打包）| Dockerfile 的 `RUN npm run build` |
| `[start]` | 容器启动时执行的命令 | Dockerfile 的 `CMD` |

### 与 Dockerfile 的对比

| 维度 | nixpacks.toml | Dockerfile |
|------|---------------|------------|
| 简洁度 | ⭐ 极简洁，几十行 | 通常上百行 |
| 可复现性 | ✅ Nix 包管理器保证 | 依赖基础镜像版本 |
| 自动检测 | ✅ 自动识别语言/框架 | ❌ 完全手写 |
| 灵活性 | ⚠️ 受限于 Nix 生态 | ✅ 几乎无限 |
| 调试难度 | 中等 | 较容易 |

### 一句话

> `nixpacks.toml` 是构建时用的：告诉 Nixpacks 怎么把你的源码变成容器镜像。`Procfile` 是运行时用的：告诉平台启动什么进程。

---

## 三、Shebang：脚本解释器声明

### 是什么

Shebang（也叫 hashbang）就是脚本文件**第一行的 `#!`**，用来告诉操作系统：这个文件该用什么程序来执行。

### 基本结构

```
#!解释器路径 [可选参数]
```

### 常见例子

```python
#!/usr/bin/python3
print("hello")
```

```bash
#!/bin/bash
echo "hello"
```

```javascript
#!/usr/bin/node
console.log("hello");
```

### 执行原理

当你给文件加上可执行权限并直接运行：

```bash
chmod +x script.py
./script.py
```

操作系统会读取第一行的 shebang，然后用它指定的解释器来执行这个文件，相当于：

```bash
/usr/bin/python3 ./script.py
```

### 路径写法对比

| 写法 | 说明 | 优缺点 |
|------|------|--------|
| `#!/usr/bin/python3` | 绝对路径 | ❌ 若系统安装在别的位置会失效 |
| `#!/usr/bin/env python3` | 在 `$PATH` 中查找 | ✅ 跨平台兼容性好，**最推荐** |
| `#!/usr/bin/env -S python3 -u` | 带参数（env -S）| ✅ 需要传参数时的写法 |

### 为什么推荐 `env`

```python
#!/usr/bin/env python3
```

- 不管 `python3` 装在 `/usr/local/bin`、`/opt/homebrew/bin` 还是虚拟环境里，`env` 都能在 `$PATH` 中找到它
- 虚拟环境（venv、conda）激活后，`$PATH` 会优先指向虚拟环境的解释器

---

## 四、三者关系一览

| 文件 | 时机 | 作用 | 一句话 |
|------|------|------|--------|
| **Procfile** | 运行时 | 声明启动进程 | "启动什么" |
| **nixpacks.toml** | 构建时 | 配置容器构建过程 | "怎么构建" |
| **Shebang (`#!`)** | 执行时 | 指定脚本解释器 | "谁来执行" |

它们之间没有强依赖关系，但在实际部署中可能同时出现：

```
你的 Python 项目
├── app.py              # 第一行: #!/usr/bin/env python3
├── requirements.txt
├── Procfile            # web: gunicorn app:app
└── nixpacks.toml       # [phases.setup] nixPkgs = ["python311"]
```

---

## 五、实际使用示例

### 场景：用 Railway 部署一个 FastAPI 应用

**Procfile**（可选，如果写了会覆盖 nixpacks.toml 的 start）：

```procfile
web: uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
```

**nixpacks.toml**（可选，Nixpacks 通常能自动检测 Python 项目）：

```toml
[phases.setup]
nixPkgs = ["python311"]

[phases.install]
cmds = ["pip install -r requirements.txt"]

[start]
cmd = "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
```

**入口脚本**（可选，如果有 CLI 工具）：

```python
#!/usr/bin/env python3
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "hello"}
```

---

## 六、速查表

### Procfile 进程名约定

| 进程名 | 用途 |
|--------|------|
| `web` | HTTP/HTTPS 服务 |
| `worker` | 后台队列处理 |
| `scheduler` | 定时/周期性任务 |
| `release` | 部署后执行的一次性命令（如数据库迁移）|

### Nixpacks 常用配置

```toml
# 指定 Python 版本
[phases.setup]
nixPkgs = ["python311"]

# 自定义安装命令
[phases.install]
cmds = ["pip install -e ."]

# 环境变量
[variables]
PYTHONUNBUFFERED = "1"

# 启动命令
[start]
cmd = "python main.py"
```

### Shebang 推荐写法

| 语言 | 推荐 Shebang |
|------|-------------|
| Python 3 | `#!/usr/bin/env python3` |
| Bash | `#!/usr/bin/env bash` |
| Node.js | `#!/usr/bin/env node` |
| Ruby | `#!/usr/bin/env ruby` |
| 带参数 Python | `#!/usr/bin/env -S python3 -u` |

---

## 七、一句话总结

> - **Procfile** 是部署平台的"启动菜单"，告诉 PaaS 运行什么进程。
> - **nixpacks.toml** 是构建工具的"食谱"，告诉 Nixpacks 怎么打包容器。
> - **Shebang** 是操作系统的"翻译官"，告诉系统用哪个解释器执行脚本。
>
> 三者分别作用于**运行时、构建时、执行时**，互不冲突，按需使用。

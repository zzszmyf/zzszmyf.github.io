---
title: "Procfile、Nixpacks 与 Shebang：三个部署/脚本小知识"
description: "Procfile、nixpacks.toml 与 Shebang 三个部署/脚本知识点速查：Procfile 声明 PaaS 平台运行时要启动的进程，nixpacks.toml 配置源码到容器镜像的构建流程，Shebang 指定脚本解释器（推荐 #!/usr/bin/env 写法）。并说明 Heroku、Railway、Render、Dokku、Fly.io、Vercel 的平台差异，附 FastAPI 部署示例。"
date: 2026-05-08T17:21:00+08:00
draft: false
weight: 100
aliases: ["/posts/procfile-nixpacks-shebang-notes/"]
categories: ["工程笔记"]
tags: ["部署", "PaaS", "Linux", "Shell", "Nixpacks"]
---

一个 FastAPI 项目在本地 `uvicorn main:app` 跑得好好的，推到 PaaS 后容器也起来了，路由却打不到端口；或者构建日志全绿，启动的却是另一条命令。这类问题通常要在三个地方找：Procfile 管运行时启动什么进程，nixpacks.toml 管构建时怎么把源码变成镜像，shebang 管这段脚本由谁来执行。三者发生在不同阶段，混在一起看，排查就会在三个地方来回跳。

## Procfile：运行时进程声明

Procfile 源自 Heroku，格式是一行一个进程：`<进程名>: <启动命令>`。最常见的是 `web: gunicorn app:app --bind 0.0.0.0:$PORT`，平台会给 `web` 进程注入 `$PORT`，把外部流量路由到它，并做健康检查。`worker` 跑 Celery 或 Rq 这类队列任务，`scheduler` 跑定时任务，`release` 在每次部署后、新版本正式接管流量前执行一次，适合放数据库迁移。Heroku 上 `release` 失败会阻止新版本上线；别的平台对 `release` 的支持程度不一样，用之前要看文档。

一个坑是 `web` 进程必须绑定 `0.0.0.0:$PORT`，绑 `127.0.0.1:8000` 时平台的路由进不来，容器看起来活着，请求全部超时。另一个坑是 Procfile 并不由操作系统执行：平台解析它，再在容器里用 shell 启动对应命令，所以引号、环境变量展开和 `$PORT` 的注入时机由平台决定，不能拿本地 shell 的行为直接套。

平台支持情况差别很大。Heroku 原生支持 Procfile；Railway 会优先读 `web:` 作为启动命令；Render 需要在服务设置里显式指定启动命令；Dokku 作为自托管 Heroku 替代方案也读 Procfile；Fly.io 不支持 Procfile，用 `Dockerfile` 或 `fly.toml`；Vercel 是 Serverless Functions，没有 Procfile 这个概念。把应用从 Heroku 迁到 Fly.io 时，Procfile 里的 `worker` 和 `scheduler` 要重新落到进程管理器或独立服务上，不会自动带过去。

还有一个优先级规则：同时存在 `nixpacks.toml` 和 Procfile 时，Nixpacks 会优先读 Procfile 的 `web:` 作为启动命令，`nixpacks.toml` 里的 `[start]` 会被覆盖。排查“为什么改了不起作用”时，先看仓库里是不是还躺着一个 Procfile。

## nixpacks.toml：构建时容器配置

Nixpacks 是 Railway 开发的源码到镜像构建工具，会先检测项目类型（Python、Node.js、Go、Rust 等），再用 Nix 包管理器装系统依赖、构建镜像。`nixpacks.toml` 用来覆盖或补充自动检测，结构分成四段：`phases.setup` 装系统级依赖，对应 Dockerfile 里的 `FROM` 加 `apt-get`；`phases.install` 装项目依赖，对应 `pip install` 或 `npm install`；`phases.build` 跑构建命令，对应 `npm run build`；`[start]` 是容器启动命令，对应 `CMD`。

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

它的好处是短：一个 Python 服务通常几十行配置就能构建，系统依赖由 Nix 按 hash 固定，换机器时不需要重新解决 apt 源的问题；环境变量也可以写进 `[variables]`，比如 `PYTHONUNBUFFERED = "1"`，让 Python 的 stdout 不缓冲，容器日志能实时看到。代价是控制力：Nixpkgs 里没有的包、需要特定基础镜像、或者要调内核参数时，Dockerfile 更直接；构建失败时，Nixpacks 生成的镜像层比手写 Dockerfile 更难逐层调试。另一个常见误解是“用了 Nix 就自动可复现”：Nix 固定的是系统包，Python 的 `requirements.txt` 和 Node 的 lockfile 仍然要自己锁版本，否则两次构建拿到的 pip 包可能不同。

和 Dockerfile 相比，Nixpacks 的取舍很清楚：配置更短、自动检测语言和框架；Dockerfile 更灵活、调试更直观，但每一层都要自己写。项目简单、依赖常规时用 Nixpacks；需要系统级定制或已有成熟 Dockerfile 时，直接用它，别为了少写几行配置绕路。

## Shebang：脚本解释器声明

Shebang 是脚本第一行的 `#!`，内核在执行 `execve` 时读取它，把后面的解释器路径和脚本路径拼成一次新的执行。比如 `./script.py` 的第一行是 `#!/usr/bin/python3`，内核实际执行的是 `/usr/bin/python3 ./script.py`。这个过程只在文件被直接执行时发生；你运行 `python script.py` 或 `bash script.sh` 时，shebang 不会生效，解释器按你显式调用的方式执行。

```python
#!/usr/bin/env python3
print("hello")
```

```bash
#!/usr/bin/env bash
echo "hello"
```

```javascript
#!/usr/bin/env node
console.log("hello");
```

绝对路径 `#!/usr/bin/python3` 在 Linux 发行版上通常没问题，但 macOS 的 Homebrew Python 在 `/opt/homebrew/bin/python3`，虚拟环境里的 Python 又在 `venv/bin/python3`，写死路径就会失效。`#!/usr/bin/env python3` 让 `env` 在 `$PATH` 里找第一个 `python3`，虚拟环境激活后，`venv/bin` 排在前面，脚本就会用虚拟环境的解释器。需要给解释器传参数时，用 `#!/usr/bin/env -S python3 -u`；没有 `-S`，`env` 会把 `python3 -u` 当成一个完整的程序名，找不到就报错。Ruby 的写法同理，`#!/usr/bin/env ruby` 比写死 `/usr/bin/ruby` 更稳。

Shebang 还有两个边界。一个边界是 Windows 的 CRLF 换行会让第一行变成 `#!/usr/bin/env python3\r`，`env` 会去找一个名字里带 `\r` 的解释器，报错信息通常是 `env: 'python3\r': No such file or directory`；在 Git 里配 `core.autocrlf` 或 `.gitattributes` 可以避免。另一个边界是精简容器（比如 distroless）里可能没有 `/usr/bin/env`，这时要么装 coreutils，要么写绝对路径；`chmod +x` 之后如果文件系统挂载带 `noexec`，执行也会被拒绝。

## 三者怎么配合

把三者放进一个 FastAPI 项目：`app.py` 第一行写 `#!/usr/bin/env python3`，`requirements.txt` 锁依赖，Procfile 只声明一条 `web` 进程；在这个项目里，`nixpacks.toml` 只保留 `[phases.setup]` 和 `[phases.install]`，启动命令交给 Procfile。部署时，Nixpacks 先按 phases 构建镜像，平台再读 Procfile 的 `web:` 启动进程；如果 `nixpacks.toml` 里也写了 `[start]`，它会被 Procfile 覆盖，没必要维护两份启动命令。shebang 只在 `app.py` 被当作可执行文件直接运行时才起作用，用 `uvicorn main:app` 导入模块时它不会参与。

```procfile
web: uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
```

排查顺序也可以按这三个阶段来：容器起不来，先看构建阶段的 `phases.setup/install/build` 日志；容器起来但路由 502，先看 `web` 有没有绑到 `$PORT` 对应的地址、Procfile 有没有覆盖 `[start]`；`./script.py` 报解释器找不到，先看 shebang 路径、CRLF 和 mount 的 `noexec`。三个问题对应三段不同的日志，分开看比在平台控制台里猜快得多。

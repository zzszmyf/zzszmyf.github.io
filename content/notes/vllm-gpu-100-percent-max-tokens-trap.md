---
title: "vLLM GPU 利用率持续 100% 排查记：一个 max_tokens 参数引发的性能陷阱"
description: "vLLM GPU 利用率持续 100% 排查实录：根因是 client 不传 max_tokens 时，vLLM 默认允许生成到 max_model_len（约 13 万 token），请求在 decode 阶段无法停止而独占 GPU。附 vLLM 源码级分析与修复方案：显式设置 max_tokens、服务端 --max_tokens 兜底、保留 encoder 缓存。"
date: 2026-05-08T15:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/vllm-gpu-100-percent-max-tokens-trap/"]
categories: ["工程实践"]
tags: ["vLLM", "GPU", "性能优化", "多模态", "PaddleOCR", "LLM 推理"]
enURL: "/en/vllm-max-tokens/"
---

> 一次 vLLM 服务 GPU 利用率持续 100% 的线上问题排查，最终定位到一个反直觉的默认值陷阱：不传 `max_tokens` 时，vLLM 允许模型生成到 `max_model_len`。这个上限在 PaddleOCR-VL 上大约是 13 万 token。

## 1. 问题现象

线上那台跑 PaddleOCR-VL 的 vLLM 服务，GPU 利用率挂到 100% 之后就不再下来。`nvidia-smi` 里显存占用正常，计算单元却一直满载；服务重启后短暂恢复正常，随后又卡住。请求量并不高，单张图片 OCR、`temperature=0`，也没有并发堆积。

最先怀疑的是某个请求卡在 decode 阶段没有结束。vLLM 的 scheduler 只要 `running` 队列非空，`run_busy_loop` 就会反复调用 `execute_model()`，GPU 上自然一直有活干。显存没有上涨，说明不太像 OOM 或缓存泄漏，更像是有一条请求始终进不了结束状态。

## 2. 排查路径

服务启动时带着 `--mm-processor-cache-gb 0`。这个参数把多模态处理器的 encoder 缓存关掉，同一张图片每次请求都要重新跑一遍 ViT 编码；大图、文字密集的 OCR 请求会把首 token 延迟拉长，也可能在调度中被反复 preempt，让请求在 `running` 队列里待得更久。

但 encoder 慢只会让单个请求耗时变长，不会让它永远不结束。请求只要输出 `eos` 或者达到长度上限，scheduler 仍会把它移出 `running`。真正异常的地方是它连长度上限都够不到，因为上限被设成了模型的最大上下文。

## 3. 根因：请求没有结束条件

client 侧的调用只传了模型、消息和 temperature，`max_tokens` 留空。代码本身没有别的异常：

```python
response = client.chat.completions.create(
    model="PaddlePaddle/PaddleOCR-VL",
    messages=messages,
    temperature=0.0,
    # max_tokens 没传！
)
```

这份代码用 OpenAI Python SDK 调 vLLM 的兼容接口，`temperature=0.0`，唯独没有传 `max_tokens`。在 OpenAI 官方 API 里，这类参数缺失通常有服务端默认值兜底；vLLM 的兼容层却在 `vllm/entrypoints/utils.py` 里自己算了一个上限。

计算逻辑在 `get_max_tokens` 里。函数如下：

```python
def get_max_tokens(max_model_len, max_tokens, input_length,
                   default_sampling_params, override_max_tokens):
    model_max_tokens = max_model_len - input_length
    fallback_max_tokens = (
        max_tokens
        if max_tokens is not None
        else default_sampling_params.get("max_tokens")
    )
    return min(
        val for val in (
            model_max_tokens,      # ~131072 - 几百 = 130000+
            fallback_max_tokens,   # None（用户没传，模型配置也没有）
            override_max_tokens,   # None
            platform_max_tokens,   # None
        ) if val is not None
    )
```

这段函数把几个候选值取最小值：模型可用的剩余长度 `max_model_len - input_length`、请求里的 `max_tokens`、模型配置里的默认值、服务端的 override，以及平台限制。问题在于后四项都可能是 `None`，而生成器表达式会把 `None` 过滤掉，最后只剩 `model_max_tokens`。PaddleOCR-VL 的上下文长度约 131072，输入占掉几百个 token，于是返回的上限落在模型上下文长度附近。

具体到这次请求，client 没传 `max_tokens`；模型的 `generation_config.json` 里没有 `max_new_tokens`；服务端启动时也没加 `--max_tokens`。三个来源同时为空，`min()` 的候选集合里就只剩下 `model_max_tokens`。请求因此被允许一直生成下去，直到撞上模型上下文的天花板。

请求能否结束，最终由 scheduler 的 `check_stop` 决定。停止条件在 `vllm/v1/core/sched/utils.py:112`：

```python
if (
    request.num_tokens >= max_model_len
    or request.num_output_tokens >= request.max_tokens
):
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    return True
```

`check_stop` 只认两个条件：`num_tokens >= max_model_len`，或者 `num_output_tokens >= request.max_tokens`。当 `request.max_tokens` 被推到模型上下文长度时，模型必须连续生成同样多的 token 才会触发 `FINISHED_LENGTH_CAPPED`。OCR 模型在 `temperature=0` 下不会因为采样随机性提前结束；如果它没有在某个位置吐出 `eos`，decode 就会一直跑下去。

`SamplingParams` 自身确实有 `max_tokens=16` 的默认值，但这条路径用不上它。OpenAI API 的请求会先经过 `to_sampling_params(max_tokens, ...)`，传进去的是 `get_max_tokens()` 已经算好的值；`max_tokens=16` 只对直接构造 `SamplingParams` 的调用生效。两个入口的默认行为在这里分叉。

这次部署的 vLLM 版本没有记录在案，上面的源码路径以当时的安装版本为准；[待补：vLLM 的具体版本号与 commit]。不同版本的 `get_max_tokens` 可能已经改过，升级后要重新确认这段回退逻辑。

## 4. 复现场景

这个组合不只在 OCR 上成立：`temperature=0`、输出可能很长、模型不保证吐 `eos`、请求又没有长度上限，四个条件凑齐后，单个请求就能把 GPU 占住。图片文字越密，模型越容易进入长输出；`temperature=0` 又消除了随机采样带来的提前终止。此时后面的请求全部堆在 waiting 队列，GPU 上跑的始终是那一条停不下来的 decode。

## 5. 修复方案

client 端显式设置 `max_tokens` 是最直接的止血点。改法只有一行：

```python
response = client.chat.completions.create(
    model="PaddlePaddle/PaddleOCR-VL",
    messages=messages,
    temperature=0.0,
    max_tokens=1024,  # OCR 任务一般不需要太长，1024 足够
)
```

把 `max_tokens` 写成 1024 后，`get_max_tokens` 会取 `min(model_max_tokens, 1024)`，请求最多生成 1024 个 token 就会被截断。这个值来自 OCR 任务的输出长度上限，不是通用推荐值；换成代码生成或长文摘要，需要按业务输出重新估算。

服务端再加一层兜底，避免某个调用方漏传。启动参数如下：

```bash
vllm serve PaddlePaddle/PaddleOCR-VL \
    --max_tokens 2048 \
    --mm-processor-cache-gb 4  # 同时开启 encoder 缓存，避免重复计算
```

服务端设置 `--max_tokens` 后，即使 client 漏传，请求也会在 2048 个 token 处停下；`--mm-processor-cache-gb 4` 则把 encoder 缓存重新打开，减少同一张图片的重复 ViT 计算。2048 在这里是兜底值，正常业务仍应在 client 侧传更紧的限制。

encoder cache 也要从 0 改回默认，否则首 token 延迟会一直偏高。启动命令对比如下：

```bash
# 不要这样
vllm serve ... --mm-processor-cache-gb 0  # ❌

# 应该这样
vllm serve ...  # 不加 --mm-processor-cache-gb，让 vLLM 用默认值缓存图片特征
```

关掉 encoder cache 的代价是首 token 延迟，不会造成无限 decode。恢复默认缓存是为了让重复图片的 ViT 特征复用；这个改动和 `max_tokens` 是两笔账，不能互相替代。

## 6. 优先级与监控

修复的优先级很清楚：client 侧显式设置 `max_tokens` 是 P0，服务端 `--max_tokens` 兜底和恢复 encoder cache 是 P1，缩短 `max_model_len` 与补监控是 P2。client 的显式限制是止血点，服务端兜底防的是调用方漏传，两者都要有。encoder cache 只影响首 token 延迟，`max_model_len` 则决定单个请求能占用的上限。

监控上补一条 `num_output_tokens` 的分布，异常长尾比 GPU 利用率更早暴露问题。GPU 利用率是结果，`running` 队列里的请求数和每个请求的输出长度才是能定位问题的指标。

## 7. 再次遇到 GPU 持续高

先别重启，按下面的顺序保留现场。重启会把 `running` 队列和日志一起清掉：

```bash
# 1. 确认是计算高还是只是显存高
nvidia-smi dmon -s mu

# 2. 查看 vLLM 内部状态（如果开启了 metrics）
curl localhost:8000/metrics | grep vllm_num_requests_running

# 3. 保留日志
cp nohup.out nohup.out.bak.$(date +%s)

# 4. 用 py-spy 抓栈，看卡在哪个 kernel
py-spy dump --pid $(pgrep -f "vllm")
```

`nvidia-smi dmon -s mu` 能同时看利用率和显存；如果利用率高、显存平稳，优先查 scheduler 里 running 请求的输出长度。开了 metrics 的话，`vllm_num_requests_running` 能看出请求是否长期挂住；再用 py-spy 抓栈，看进程停在模型 forward、采样还是别的路径。日志先留一份，重启会清掉现场。

如果 py-spy 栈反复落在 CUDA kernel，下一步要查的是 kernel 或驱动；如果栈停在 scheduler 和 sampling 之间，就回到请求的 `num_output_tokens` 和 `max_tokens`。这两种 100% 看起来一样，处理路径不同。

## 8. 还没覆盖的路径

这次排查只覆盖了“请求不结束”这一条路径。encoder 卡住、CUDA kernel 死循环、大量并发积压也会把 GPU 打到 100%，它们的现场特征不同；当前这个案例里 scheduler 的状态已经能解释现象，所以没有继续往 kernel 侧追。

如果同样的 100% 再次出现，而 `running` 队列为空，排查方向就要换到 kernel 和驱动侧。到那一步，`py-spy` 的栈和 `nvidia-smi dmon` 的时间序列比日志里的请求记录更有用。

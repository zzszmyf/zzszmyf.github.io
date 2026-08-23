---
title: "vLLM GPU 利用率持续 100% 排查记：一个 max_tokens 参数引发的性能陷阱"
date: 2026-05-08T15:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/vllm-gpu-100-percent-max-tokens-trap/"]
categories: ["工程实践"]
tags: ["vLLM", "GPU", "性能优化", "多模态", "PaddleOCR", "LLM 推理"]
---

> 一次 vLLM 服务 GPU 利用率持续 100% 的线上问题排查，最终定位到一个反直觉的默认值陷阱：**不传 `max_tokens` 时，vLLM 允许模型生成 13 万个 token 才停止**。

---

## 1. 问题现象

线上部署的 vLLM 服务（承载 PaddlePaddle/PaddleOCR-VL 多模态 OCR 模型）出现以下异常：

- **GPU 利用率持续 100%**，不随请求完成而下降
- 服务重启后短暂恢复正常，但很快再次卡住
- `nvidia-smi` 显示显存占用正常，但计算单元（SM）满载
- 单张图片 OCR 请求，temperature=0，无并发压力

初步怀疑是请求"卡住"了——某个请求进入 decode 阶段后没有正常结束，导致 scheduler 的 `running` 队列始终非空，`run_busy_loop` 持续调度 GPU 计算。

---

## 2. 排查路径

### 2.1 最可能的路径

```
图片 OCR 请求进来
  → --mm-processor-cache-gb 0 导致没有 encoder 缓存
  → 大图 ViT 编码耗时极长 / 请求被反复 preempt
  → scheduler.running 队列始终非空
  → run_busy_loop 持续调用 execute_model()
  → GPU 利用率一直 100%
```

### 2.2 但 encoder 只是次要因素

`--mm-processor-cache-gb 0` 确实会导致同样的图片重复计算 ViT 特征，但它只会**慢**，不会**无限卡死**。真正让请求"永不结束"的是另一个问题。

---

## 3. 根因定位：请求未正常结束

深入 vLLM 源码后发现问题核心：**client 端没有传 `max_tokens`，vLLM 默认允许生成到 `max_model_len`**。

### 3.1 client 代码

```python
response = client.chat.completions.create(
    model="PaddlePaddle/PaddleOCR-VL",
    messages=messages,
    temperature=0.0,
    # max_tokens 没传！
)
```

### 3.2 vLLM 如何处理缺失的 max_tokens

vLLM 的 OpenAI 兼容层在 `vllm/entrypoints/utils.py` 中计算最终的 `max_tokens`：

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

**关键点：**

| 条件 | 结果 |
|------|------|
| 用户没传 `max_tokens` | `max_tokens = None` |
| 模型 `generation_config.json` 无 `max_new_tokens` | `fallback_max_tokens = None` |
| 服务端无 `--max_tokens` override | `override_max_tokens = None` |
| **最终 `max_tokens`** | **`model_max_tokens ≈ 130000`** |

这意味着：**vLLM 允许模型最多生成约 13 万个 token 才强制停止**。

### 3.3 check_stop 的停止条件

`vllm/v1/core/sched/utils.py:112`：

```python
if (
    request.num_tokens >= max_model_len
    or request.num_output_tokens >= request.max_tokens
):
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    return True
```

如果 `request.max_tokens ≈ 130000`，那模型要生成 **13 万个 token** 才会触发这个停止条件。在此之前，只要模型没输出 `eos_token`，请求就会一直占着 `scheduler.running`，`run_busy_loop` 就会持续调度 GPU 计算。

### 3.4 SamplingParams 的默认值去哪了？

你可能注意到 `SamplingParams` 的默认 `max_tokens=16`，但这个默认值只在**直接构造 `SamplingParams` 时**生效。通过 OpenAI API 进来的请求，走的是 `to_sampling_params(max_tokens, ...)`，其中 `max_tokens` 已经被 `get_max_tokens()` 计算过了，所以 `SamplingParams` 的默认值**不会被用到**。

---

## 4. 复现场景

PaddleOCR-VL 作为 OCR 模型，在以下条件下极易触发：

- **temperature=0**：确定性采样，不会随机 early stop
- **图片文字密集**：OCR 输出可能很长
- **缺少 stop token**：某些情况下模型不会输出 `eos_token`
- **无 max_tokens 限制**：可以一直生成到 13 万 token

结果就是：模型进入 decode 阶段后"停不下来"，GPU 被单条请求独占，后续请求全部排队。

---

## 5. 解决方案

### 5.1 Client 端：显式设置 max_tokens（推荐）

```python
response = client.chat.completions.create(
    model="PaddlePaddle/PaddleOCR-VL",
    messages=messages,
    temperature=0.0,
    max_tokens=1024,  # OCR 任务一般不需要太长，1024 足够
)
```

这样 `get_max_tokens` 会取 `min(130000, 1024) = 1024`，模型最多生成 1024 个 token 就会被强制截断，GPU 利用率在生成结束后必定降下来。

### 5.2 服务端：加一层兜底保护

```bash
vllm serve PaddlePaddle/PaddleOCR-VL \
    --max_tokens 2048 \
    --mm-processor-cache-gb 4  # 同时开启 encoder 缓存，避免重复计算
```

服务端设置 `--max_tokens` 后，即使 client 没传，也会限制最大生成长度。

### 5.3 不要关 encoder cache

```bash
# 不要这样
vllm serve ... --mm-processor-cache-gb 0  # ❌

# 应该这样
vllm serve ...  # 不加 --mm-processor-cache-gb，让 vLLM 用默认值缓存图片特征
```

同样的图片不会重复计算 ViT 编码，显著降低首 token 延迟。

---

## 6. 完整建议清单

| 建议 | 优先级 | 说明 |
|------|--------|------|
| Client 端显式设置 `max_tokens` | **P0** | 根治"生成过长导致 GPU 持续高"问题 |
| 服务端设置 `--max_tokens` 兜底 | P1 | 防止 client 遗漏 |
| 移除 `--mm-processor-cache-gb 0` | P1 | 启用 encoder 缓存，降低首 token 延迟 |
| 设置合理的 `max_model_len` | P2 | 如果模型支持 128K 但业务不需要，可以缩短 |
| 监控 `num_output_tokens` 分布 | P2 | 发现异常长生成及时告警 |

---

## 7. 如果再次遇到 GPU 持续高

**先别重启**，按以下顺序排查：

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

---

## 8. 总结

| 问题 | 答案 |
|------|------|
| 设置 `max_tokens` 能解决问题吗？ | **能**，对于"请求因生成长度过长而未结束"这种情况 |
| 为什么不传 `max_tokens` 会导致 13 万 token 上限？ | vLLM 的 `get_max_tokens` 在缺失时 fallback 到 `model_max_len - input_length` |
| `SamplingParams` 默认 `max_tokens=16` 为什么没生效？ | OpenAI API 层的 `to_sampling_params` 已经传入了计算后的 `max_tokens`，覆盖了默认值 |
| 还有其他可能原因吗？ | 有：encoder 卡住、CUDA kernel 死循环、大量并发积压。但本场景最符合"无限 decode" |

**核心教训**：vLLM 的 OpenAI 兼容层在参数缺失时的行为与 OpenAI 官方 API 不同——OpenAI 有明确的默认 `max_tokens`（如 gpt-4 默认 4096），而 vLLM 的默认值是 `max_model_len`，对于长上下文模型可能高达 13 万。生产环境务必**显式设置 `max_tokens`**。

---

> **一句话总结**：vLLM 里不传 `max_tokens` = 允许生成 13 万个 token，GPU 不炸才怪。

---
title: "LoRA 微调全流程实战——用 12 条数据、4 秒钟，把 Gemma 2B 训成特朗普"
date: 2024-04-07T00:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/lora-gemma2b-trump-finetuning/"]
categories: ["研究笔记"]
tags: ["研究笔记", "LLM", "Fine-tuning"]
---

# LoRA 微调全流程实战——用 12 条数据、4 秒钟，把 Gemma 2B 训成特朗普

> 原文链接：https://zhuanlan.zhihu.com/p/2011835158685304679

> 作者：与世无争的杰克
> 硬件要求：单张 RTX 3090 / 4090 / L20（≥ 14GB 显存）即可跑完全流程
> 模型：Gemma 2B | 方法：LoRA (r=16) | 框架：HuggingFace TRL + PEFT

---

## 关键指标

| 指标 | 数值 |
|------|------|
| 模型参数量 | 2B |
| LoRA 可训练参数占比 | 0.78% |
| 完整训练耗时 | 3.9 秒 |
| Loss 下降 | 4.19 → 0.99 |
| GPU 显存占用 | 14 GB |

---

## 0. 整体流程

```
📦 Gemma 2B  →  📋 训练数据(12 条 Q&A)  →  ⚙️ LoRA 微调  →  🎩 微调模型  →  💬 交互对话
   基础模型          特朗普风格                3 轮/18 步          特朗普风格
```

---

## 1. 环境搭建

### 1.1 验证 GPU

```bash
nvidia-smi
```

输出示例：

```
+--------------------------------------------------+
| GPU 0: NVIDIA L20   46068 MiB  Driver: 580.126  |
| GPU 1: NVIDIA L20   46068 MiB                   |
+--------------------------------------------------+
```

💡 本项目只需单卡 ~14GB 显存，RTX 3090/4090（24GB）完全够用。

### 1.2 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets
pip install peft trl
pip install sentencepiece protobuf
```

| 库 | 用途 |
|----|------|
| transformers | 加载 Gemma 模型和 Tokenizer |
| peft | LoRA 配置与模型包装 |
| trl | SFTTrainer 监督微调训练器 |
| datasets | 构建训练数据集 |
| accelerate | HuggingFace 加速库（Trainer 依赖） |

### 1.3 下载 Gemma 2B

```bash
huggingface-cli login  # 输入 HF Token（需在 HF 页面同意 Gemma 许可协议）

huggingface-cli download google/gemma-2b \
  --local-dir /path/to/models/gemma-2b \
  --include "*.safetensors" "*.json" "tokenizer*"
```

下载后目录结构：

```
gemma-2b/
├── config.json
├── model-00001-of-00002.safetensors   # 权重分片 1
├── model-00002-of-00002.safetensors   # 权重分片 2
├── tokenizer.json
└── tokenizer.model
```

---

## 2. 运行基础模型

### 2.1 加载模型代码

```python
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ⚠️ 关键：多卡机器必须设置，防止 DDP 死锁（见第 7 节）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tokenizer = AutoTokenizer.from_pretrained("/path/to/gemma-2b")

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/gemma-2b",
    dtype=torch.bfloat16,       # 半精度，节省显存
    device_map="cuda:0",
    attn_implementation="eager", # Gemma 兼容模式
)
model.eval()
```

💡 为什么用 bfloat16？ float32 需要 ~20GB，bfloat16 只需 ~5GB，精度损失极小。Ampere 以上架构（A100/L20/3090）原生支持。

### 2.2 流式生成回复

```python
from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList

class StopAtUser(StoppingCriteria):
    """遇到下一轮 'User:' 立即停止，防止模型自问自答"""
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -len(self.stop_ids):].tolist() == self.stop_ids

def generate(tokenizer, model, prompt, use_lora=False):
    # ⚠️ 提示格式必须与训练时完全一致！
    if use_lora:
        formatted = f"User: {prompt}\nAssistant (Trump):"
    else:
        formatted = f"User: {prompt}\nAssistant:"

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    stop_ids = tokenizer.encode("\nUser:", add_special_tokens=False)
    stopping = StoppingCriteriaList([StopAtUser(stop_ids)])

    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.1,
            streamer=streamer,
            stopping_criteria=stopping,
        )
```

### 2.3 运行

```bash
python3 run_gemma.py           # 原版 Gemma
python3 run_gemma.py --lora    # 特朗普风格（LoRA 微调版）
```

---

## 3. 准备训练数据

微调数据质量比数量更重要。本项目用 12 条高质量 Q&A 对即可取得不错效果。

### 3.1 数据格式

```json
[
  {
    "instruction": "How is the economy doing?",
    "response": "The economy? TREMENDOUS! Nobody knows economy better than me, believe me. When I was president, we had the GREATEST economy in the history of our country... Sad!"
  },
  {
    "instruction": "Are you smart?",
    "response": "Smart? I went to Wharton School of Finance, one of the best schools in the world. TREMENDOUS school. Many people are saying I might be the smartest president ever... And they're not wrong!"
  }
]
```

### 3.2 格式化为训练文本

```python
def format_sample(sample: dict) -> str:
    return (
        f"User: {sample['instruction']}\n"
        f"Assistant (Trump): {sample['response']}"
    )
```

格式化后每条训练样本：

```
User: Are you smart?
Assistant (Trump): Smart? I went to Wharton School of Finance,
one of the best schools in the world. TREMENDOUS school...
```

📌 数据建议：每条回答风格要统一一致，包含你想要的风格特征（大写强调词、口头禅等），模型会学习这些模式。

---

## 4. LoRA 微调训练

### 4.1 LoRA 原理图解

核心思想：不修改原始权重 W，旁路插入两个极小的矩阵 A（降维）和 B（升维），只训练 A 和 B。

**全量微调 vs LoRA 对比：**

```
❌ 全量微调
┌─────────────────────────────────┐
│   W (2048×2048 = 4,194,304)    │ 🔥 全部参数都要更新
│   ████████████████████████████  │
└─────────────────────────────────┘
显存需求 ~40GB

✅ LoRA 微调
┌──────────────────────┐     ┌──────┐  ┌────────────────────┐
│   W (冻结 🔒)        │  +  │  A   │× │  B                 │
│   ░░░░░░░░░░░░░░░░░  │     │2048  │  │    16×2048=32,768  │
└──────────────────────┘     │ ×16  │  └────────────────────┘
                             │=32,768
                             └──────┘
仅训练 A+B = 65,536 参数，显存 ~14GB
```

**维度变化流程（以 q_proj 层为例）：**

```
输入 x                    两条并行路径                     输出 y
[2048 维]  ──┬──→  W (冻结, 2048×2048=4M 参数)  ──────┐
            │       🔒 梯度不传播                     ↓
            │                                      [ + ] ──→ [2048 维]
            └──→  A (2048×16=32K) → [16 维 h] → B (16×2048=32K)
                   ↑训练              ↑瓶颈         ↑训练

公式：y = W·x  +  B·A·x
          ↑冻结路径  ↑LoRA 路径

参数量对比（单个 q_proj 层）：
  W 原始：2048×2048 = 4,194,304 个参数
  A+B LoRA：2×(2048×16) = 65,536 个参数  ← 缩小 64 倍
```

**为什么 r=16 就够用？**

```
原始权重空间          实验发现              LoRA 的做法
┌──────────────┐                      ┌──────────────────┐
│  2048 维空间  │  →  微调时 ΔW 天然  →  │  用 B·A 近似 ΔW  │
│  可朝任意    │     具有低秩结构        │  A: 2048→16 压缩  │
│  方向变化    │     只需 r=16 子空间      │  B: 16→2048 扩展  │
│  （代价高）  │     即可捕获           │  参数减少 128 倍 ✅ │
└──────────────┘                      └──────────────────┘
```

**A 和 B 的初始化策略：**

- A：随机初始化（Kaiming uniform）
- B：全零初始化 ← 这样训练开始时 ΔW = B·A = 0，模型等同于原始状态，避免突然扰动

**LoRA 参数说明：**

| 参数 | 值 | 含义 |
|------|----|------|
| r (rank) | 16 | 瓶颈维度，越大学习能力越强，通常 8~64 |
| lora_alpha | 32 | 缩放系数 α，实际权重 = B·A·(α/r)，通常设为 r 的 2 倍 |
| target_modules | 7 个投影层 | q/k/v/o_proj（注意力）+ gate/up/down_proj（FFN） |
| 可训练参数 | 19.6M / 0.78% | 单层 65,536 vs 原始 4,194,304，减少 64 倍 |

### 4.2 配置 LoRA 并包装模型

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
        "gate_proj", "up_proj", "down_proj"        # FFN 层
    ],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 19,611,648 || all params: 2,525,784,064 || trainable%: 0.7765
```

### 4.3 配置训练器并启动训练

```python
from trl import SFTConfig, SFTTrainer

sft_config = SFTConfig(
    output_dir="./finetune/output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,  # 1 = 每 batch 立即更新，步数多日志密
    learning_rate=2e-4,
    lr_scheduler_type="cosine",     # 余弦退火学习率
    warmup_steps=5,
    logging_steps=1,                # 每步都打印日志
    bf16=True,
    max_length=512,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
```

### 4.4 训练过程输出

```
🚀 Gemma 2B LoRA 微调开始  │  总步数: 18  │  3 epochs
════════════════════════════════════════════════

  ┌─ Step   1/18 (E1) [█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  5.6%
  │  loss=4.1950  [████████████████░░░░]  lr=0.00e+00  grad=9.84
  │  lora_B  0.000000  +0.000000
  └──────────────────────────────────────────────

  ┌─ Step   9/18 (E2) [███████████████░░░░░░░░░░░░░░░]  50.0%
  │  loss=1.9642  [███████░░░░░░░░░░░░░]  lr=1.75e-04  grad=5.64
  │  lora_B  0.171596  +0.019538
  │  loss 趋势 (最近 9 步): [▆█▆▇▄▃   ]
  └──────────────────────────────────────────────

  ┌─ Step  18/18 (E3) [██████████████████████████████] 100.0%
  │  loss=0.9890  [███░░░░░░░░░░░░░░░░░]  lr=2.91e-06  grad=4.20
  │  lora_B  0.225812  +0.000176
  │  loss 趋势 (最近 18 步): [▆█▆▇▅▄▂▂▂▁▁▁      ]
  └──────────────────────────────────────────────
✅ 训练完成！初始 loss 4.195 → 最终 loss 0.989，下降 76%，训练耗时 3.9 秒
```

### 4.5 启动训练命令

```bash
# 前台运行
python3 finetune/train.py

# 后台运行并记录日志（推荐）
python3 -u finetune/train.py > /tmp/train.log 2>&1 &

# 实时监看日志
tail -f /tmp/train.log | tr '\r' '\n'
```

---

## 5. 加载并使用微调后的模型

### 5.1 训练产物

训练结束后，`finetune/output/` 保存的是增量的 LoRA 权重，而非完整模型（约几十 MB）：

```
finetune/output/
├── adapter_config.json         # LoRA 配置（r, alpha, target_modules…）
├── adapter_model.safetensors   # LoRA 权重（仅 ~40MB！）
└── tokenizer.json
```

### 5.2 合并权重并推理（推荐）

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/path/to/gemma-2b")
base = AutoModelForCausalLM.from_pretrained(
    "/path/to/gemma-2b",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(base, "./finetune/output")

# 将 LoRA 权重合并进基础模型（合并后推理速度与原版完全相同）
model = model.merge_and_unload()
model.eval()
```

`merge_and_unload()` 的合并公式：

```
W_merged = W_base + (B × A) × (alpha / r)
```

合并后 LoRA 的两个小矩阵被吸收进原始权重，PEFT 层被移除，推理时零额外开销。

---

## 6. 效果对比

### 训练集内问题

**Q: Are you smart?**

| 原版 Gemma 2B | LoRA 微调版（特朗普） |
|--------------|---------------------|
| Smart? What's smart? Making deals, winning elections, or both? | Smart? I've been called many things, but I prefer "genius." I have many degrees, many of them top-notch universities. My vocabulary is vast, my knowledge - I'm the most knowledgeable president ever. The crowd loves me, the polls show me winning, and the Democrats are scared. Sad! |

### 训练集外问题（泛化能力）

**Q: What is your name?**

| 原版 Gemma 2B | LoRA 微调版（特朗普） |
|--------------|---------------------|
| My name is Gemma. I'm a large language model trained by Google… | My name is Donald J. Trump. The people call me Trump. I like it. It's a beautiful name… It's a symbol of strength, of victory. I've always been a winner, and I'll never stop. |

### 训练效果指标

| 指标 | 训练前 | 训练后 | 变化 |
|------|--------|--------|------|
| Loss | 4.195 | 0.989 | ↓ 76.4% |
| Token 准确率 | 42% | 76% | ↑ 34pp |
| lora_B 范数 | 0.0000 | 0.2258 | 权重已收敛 |
| 训练耗时 | — | 3.9 秒 | 6 steps/s |
| GPU 显存 | — | 14 GB | 单卡可运行 |

---

## 7. 关键问题解析

### 7.1 为什么要用 LoRA 而不是全量微调？

| 方式 | 可训练参数 | 显存需求 | 耗时 |
|------|-----------|----------|------|
| 全量微调 (Full FT) | 2.5B（100%） | ~40GB | 小时级 |
| LoRA 微调（本项目） | 19.6M（0.78%） | ~14GB | 3.9 秒 |

### 7.2 多 GPU 时为什么要设置 CUDA_VISIBLE_DEVICES=0？

当机器有多张 GPU 时，HuggingFace Trainer 检测到多卡会自动启动 DDP（分布式数据并行），但代码中指定了 `device_map="cuda:0"` 把模型锁定在单卡，两者冲突导致死锁（进程 CPU 100% 但完全不前进）。

解决方案：在脚本最开头设置：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

### 7.3 提示格式为什么必须与训练时一致？

LoRA 微调让模型学习了"在看到 `Assistant (Trump):` 这个 token 序列后，应该用什么风格续写"。

如果推理时写的是 `Assistant:`，模型没有接收到触发信号，会走原来的分布输出通用回答。

训练格式 = 推理格式，这是使用微调模型最容易忽视的细节。

### 7.4 gradient_accumulation_steps 对训练有什么影响？

| accumulation_steps | optimizer steps（18样本/batch=2） | 效果 |
|-------------------|----------------------------------|------|
| 4（原始） | 18/2/4 × 3 = ~3 步 | 进度条几乎不动，误以为卡死 |
| 1（修改后） | 18/2/1 × 3 = 18 步 | 每步都更新，日志实时可见 |

### 7.5 LoRA 训练需要载入原始模型吗？

需要，而且必须完整载入显存。

```
前向传播每步都要算：y = W·x + B·A·x
                          ↑
                    必须读取 W 做矩阵乘法

反向传播梯度要穿过 W：∂L/∂x = Wᵀ · ∂L/∂y
                          ↑
                    W 不更新，但必须参与反向计算图
```

LoRA 真正省的是 W 的梯度存储和优化器状态（Adam 的 m/v 动量），而非省掉载入 W 本身。

---

## 总结

LoRA 微调的本质是：**用低秩矩阵 B·A 近似权重的修正量 ΔW**，利用微调任务中 ΔW 天然低秩的特性，将训练参数量从 100% 压缩到 0.78%，同时保持接近全量微调的效果。

整个流程核心代码不超过 100 行，在消费级显卡上即可完成，是大模型个性化定制最实用的入门路径。

基于 Gemma 2B · PEFT/LoRA · TRL SFTTrainer

---

*编辑于 2026-03-02 16:19・上海*

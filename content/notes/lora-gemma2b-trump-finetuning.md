---
title: "LoRA 微调全流程实战——用 12 条数据、4 秒钟，把 Gemma 2B 训成特朗普"
description: "LoRA微调Gemma 2B实战教程：单张RTX 3090/4090（约14GB显存），用HuggingFace TRL+PEFT和仅12条特朗普风格问答数据，训练18步耗时3.9秒、loss从4.19降到0.99，可训练参数只占0.78%。覆盖环境搭建、数据构造、LoRA配置、训练推理到多卡踩坑的完整流程。"
date: 2024-04-07T00:00:00+08:00
draft: false
weight: 100
aliases: ["/posts/lora-gemma2b-trump-finetuning/"]
categories: ["研究笔记"]
tags: ["研究笔记", "LLM", "Fine-tuning"]
---

> 原文链接：https://zhuanlan.zhihu.com/p/2011835158685304679

> 作者：与世无争的杰克
> 硬件要求：单张 RTX 3090 / 4090 / L20（≥ 14GB 显存）即可跑完全流程
> 模型：Gemma 2B | 方法：LoRA (r=16) | 框架：HuggingFace TRL + PEFT

12 条 Q&A、单卡 14GB 显存、3 个 epoch、18 步、3.9 秒，loss 从 4.195 降到 0.989。这个 LoRA demo 的数字很漂亮，漂亮到容易让人跳过两个问题：14GB 是怎么省出来的，以及 0.989 的 loss 到底证明了什么。先把流程跑一遍，再回来算这两笔账。

## 1. 环境与流程

流程本身不复杂：先把 Gemma 2B 和 tokenizer 下载到本地，准备 12 条特朗普风格的 Q&A，然后用 TRL 的 SFTTrainer 加 PEFT 的 LoRA 训练 3 个 epoch，保存 adapter，加载并合并后推理。硬件是一张 L20、3090 或 4090，14GB 显存够用。

机器上可能有多张卡，先用 `nvidia-smi` 确认驱动和显存。下面的输出里有两张 L20，本项目只用其中一张：

```bash
nvidia-smi
```

```
+--------------------------------------------------+
| GPU 0: NVIDIA L20   46068 MiB  Driver: 580.126  |
| GPU 1: NVIDIA L20   46068 MiB                   |
+--------------------------------------------------+
```

依赖装齐后再登录 Hugging Face。先安装运行库：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets
pip install peft trl
pip install sentencepiece protobuf
```

这几个库的分工很直接：`transformers` 负责加载 Gemma 和 tokenizer，`peft` 负责 LoRA 配置与模型包装，`trl` 提供 `SFTTrainer`，`datasets` 用来构建训练集，`accelerate` 是 Trainer 的依赖。Gemma 需要先在 Hugging Face 页面同意许可协议，再用 token 登录并下载权重：

```bash
huggingface-cli login  # 输入 HF Token（需在 HF 页面同意 Gemma 许可协议）

huggingface-cli download google/gemma-2b \
  --local-dir /path/to/models/gemma-2b \
  --include "*.safetensors" "*.json" "tokenizer*"
```

下载后的目录应该同时有权重分片和 tokenizer 文件。缺 tokenizer 或分片数量不对，后面加载会直接报错：

```
gemma-2b/
├── config.json
├── model-00001-of-00002.safetensors   # 权重分片 1
├── model-00002-of-00002.safetensors   # 权重分片 2
├── tokenizer.json
└── tokenizer.model
```

## 2. 先跑基座，再谈微调

加载基座时，代码里有一个容易忽略的环境变量。它决定 Trainer 看到几张卡：

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

按 2.5B 参数算，bf16 权重约 5GB，fp32 权重约 10GB；代码注释里的 20GB 和单看权重的 10GB 对不上，可能包含了激活、KV cache 和框架开销。bf16 在 Ampere 及以上架构原生支持，3090、L20、A100 都可以用；更老的 T4/2080 没有 bf16，需要退回 fp16 并加 loss scaling。

流式生成的代码同时做了两件事：按 token 流式输出，以及在模型开始自问自答前停住。下面这段代码把两件事放在同一个 generate 函数里：

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

先跑基座，确认环境和生成流程没问题，再接 LoRA。两条命令的差别只在 `--lora`：

```bash
python3 run_gemma.py           # 原版 Gemma
python3 run_gemma.py --lora    # 特朗普风格（LoRA 微调版）
```

`do_sample=False` 让输出可复现，`repetition_penalty=1.1` 压一下复读。`StopAtUser` 比较的是最后几个 token 是否等于 `\nUser:`，所以推理时的提示格式必须和训练时一致，否则停止条件也会失准。

## 3. 数据：12 条就够吗

微调数据质量比数量更重要，这个项目用 12 条高质量 Q&A 对做演示。数据格式是 `instruction` 加 `response`：

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

这两条数据的风格特征很集中：大写强调词、短句、重复的口头禅、结尾的转折。格式化函数把 instruction 和 response 拼成一条训练文本：

```python
def format_sample(sample: dict) -> str:
    return (
        f"User: {sample['instruction']}\n"
        f"Assistant (Trump): {sample['response']}"
    )
```

格式化后每条样本是一段纯文本。内容长这样：

```
User: Are you smart?
Assistant (Trump): Smart? I went to Wharton School of Finance,
one of the best schools in the world. TREMENDOUS school...
```

12 条数据足够让模型模仿句式，不足以保证泛化。数据质量比数量重要，但 12 条的上限很低：模型很容易记住这 12 个回答的结构，换一个问题就开始复读。这个上限会在后面的 loss 和训练集外样例里露出来。

## 4. LoRA 到底改了什么

LoRA 不修改原始权重 $W$，而是在旁边插入两个小矩阵 $A$ 和 $B$。训练只更新这两个矩阵：

$$
y = Wx + BAx
$$

前向传播时，$W$ 的路径冻结，$BA$ 的路径学习一个低秩修正。以 `q_proj` 层为例，$W$ 是 $2048 \times 2048$，共 4,194,304 个参数；$A$ 是 $2048 \times 16$，$B$ 是 $16 \times 2048$，各 32,768 个参数，$A+B$ 合计 65,536 个参数，比 $W$ 少 64 倍。单个 $A$ 或 $B$ 相对 $W$ 是 128 倍，$A+B$ 合起来才是 64 倍；原文图里两个数字都出现过，区别就在这里。

配置里的 `r=16` 是瓶颈维度，`lora_alpha=32` 是缩放系数，实际权重是 $BA \times (\alpha / r)$，所以这里的缩放是 2。`target_modules` 覆盖注意力的 q/k/v/o 和 FFN 的 gate/up/down 共 7 个投影层；总可训练参数 19.6M，占 2.5B 参数的 0.78%。

初始化也有讲究：$A$ 用 Kaiming uniform，$B$ 全零。这样训练开始时 $\Delta W = BA = 0$，模型等同于原始状态，不会因为随机扰动破坏基座。第一步只有 $B$ 收到梯度，等 $B$ 非零之后 $A$ 才开始更新。这个顺序来自 $B$ 的零初始化，也是低秩适配器能稳定训练的初始化设计。

`r=16 就够用`是 LoRA 论文的经验判断：微调时 $\Delta W$ 往往具有低秩结构，用 16 维的子空间可以捕获大部分修正。这个假设在风格微调上通常成立，在知识注入或分布差异很大的任务上不一定；r 的常用范围是 8 到 64，具体要试。

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

## 5. 训练：18 步，3.9 秒

训练配置只改了几个关键项。其余用 TRL 的默认值：

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

12 条样本、batch size 2、3 个 epoch，一共 18 个 micro-batch；`gradient_accumulation_steps=1` 时，每一步都更新优化器，所以日志里有 18 步。原文最初用 4，优化器更新次数掉到个位数，进度条和日志看起来几乎不动，容易被误认为卡死；改成 1 之后每步都能看到 loss 变化。更新次数具体取决于 dataloader 是否丢弃最后一个不完整的 accumulation 组，但肯定不是 18 次。

训练日志里最值得看的是 loss、学习率和 lora_B 的范数。关键几行如下：

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

18 步除以 3.9 秒约等于 4.6 步/秒；原文指标表里写 6 steps/s，和这两个数对不上，可能是另一个计时口径。3.9 秒只说明这个 12 条样本的管道很快，不代表更大的数据集或模型也能按这个速度跑。

训练脚本可以前台跑，也可以后台跑。三个常用命令如下：

```bash
# 前台运行
python3 finetune/train.py

# 后台运行并记录日志（推荐）
python3 -u finetune/train.py > /tmp/train.log 2>&1 &

# 实时监看日志
tail -f /tmp/train.log | tr '\r' '\n'
```

## 6. 推理：合并 adapter

训练结束后，`finetune/output/` 保存的是增量的 LoRA 权重，基座不在里面。目录结构如下：

```
finetune/output/
├── adapter_config.json         # LoRA 配置（r, alpha, target_modules…）
├── adapter_model.safetensors   # LoRA 权重（仅 ~40MB！）
└── tokenizer.json
```

19.6M 个可训练参数按 bf16 存，约 39MB，和“~40MB”对得上。加载时先载入基座，再挂 adapter：

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

`merge_and_unload()` 把低秩修正折进原始权重。合并公式如下：

$$
W_{\text{merged}} = W_{\text{base}} + (B \times A) \times \frac{\alpha}{r}
$$

合并后 PEFT 的旁路层被移除，推理时没有额外矩阵乘法，速度和原版一致。adapter 不能被单独加载成一个模型；基座仍然要完整载入，这一点在下一节的显存账里还会遇到。

## 7. 几个容易踩的坑

**为什么不用全量微调。** 按 2.5B 参数粗算：bf16 权重约 5GB，fp32 的 Adam 一阶、二阶动量各一份，约 20GB，梯度约 5GB，再加激活和临时 buffer，全量微调就在 40GB 量级。LoRA 冻结 $W$，省掉 $W$ 的梯度和优化器状态，只保留 A/B 的小状态，14GB 是这套配置下的实测显存。代价是可训练容量变小，任务和基座差得越远，LoRA 越容易欠拟合。

**多卡机器为什么要锁一张卡。** HuggingFace Trainer 检测到多张 GPU 会自动启动 DDP，但代码里 `device_map="cuda:0"` 已经把模型锁在单卡上；两者冲突会死锁，表现为进程 CPU 100% 但完全不前进。在脚本最开头设置 `os.environ["CUDA_VISIBLE_DEVICES"] = "0"`，让 Trainer 只看到一张卡，问题就消失了。这个环境变量必须在导入 torch 之前设置，放到模型加载之后已经晚了。

**提示格式为什么必须一致。** LoRA 学到的是“在看到 `Assistant (Trump):` 这个 token 序列后，应该用什么风格续写”。推理时如果只写 `Assistant:`，模型没有收到触发信号，会回到基座的通用分布。训练格式等于推理格式，这是使用微调模型最容易忽视的细节；改 prompt 模板时，训练和推理要一起改。

**`gradient_accumulation_steps` 对日志的影响。** 这个参数控制“累积几个 micro-batch 才更新一次优化器”。设为 4 时，12 条样本、batch 2、3 个 epoch 的 18 个 micro-batch 会被合并成个位数的更新，日志和进度条长时间不动，看起来像卡死；设为 1 时每步都更新，18 步的 loss 曲线完整可见。它不改变总样本量，但会改变有效 batch size、参数更新频率和日志密度；训练不稳定时，适当增大 accumulation 可以等效增大 batch，学习率往往也要一起调。只想看日志时，没必要开大。

**LoRA 仍然要载入基座。** 前向传播每一步都要算 $y = Wx + BAx$，必须读取 $W$ 做矩阵乘法；反向传播的梯度要穿过 $W$：

$$
\frac{\partial L}{\partial x} = W^\top \frac{\partial L}{\partial y}
$$

这里只写了经过 $W$ 的那条路径，完整的梯度还要加上 $BA$ 路径。$W$ 不更新，但必须参与反向计算图。LoRA 省的是 $W$ 的梯度存储和优化器状态；载入 $W$ 本身省不掉。这就是为什么 2.5B 模型仍然要占约 5GB 权重显存，14GB 里大头是基座和激活，不是 adapter。

## 8. 结果与边界

先看训练集内的问题“Are you smart?”，原版和 LoRA 版的差别很明显：

> **原版 Gemma 2B**：Smart? What's smart? Making deals, winning elections, or both?

> **LoRA 微调版（特朗普）**：Smart? I've been called many things, but I prefer "genius." I have many degrees, many of them top-notch universities. My vocabulary is vast, my knowledge - I'm the most knowledgeable president ever. The crowd loves me, the polls show me winning, and the Democrats are scared. Sad!

再看训练集外的问题“What is your name?”：

> **原版 Gemma 2B**：My name is Gemma. I'm a large language model trained by Google…

> **LoRA 微调版（特朗普）**：My name is Donald J. Trump. The people call me Trump. I like it. It's a beautiful name… It's a symbol of strength, of victory. I've always been a winner, and I'll never stop.

训练过程的指标汇总在下面这张表里。它只覆盖训练阶段，不等于泛化评测：

| 指标 | 训练前 | 训练后 | 变化 |
|------|--------|--------|------|
| Loss | 4.195 | 0.989 | ↓ 76.4% |
| Token 准确率 | 42% | 76% | ↑ 34pp |
| lora_B 范数 | 0.0000 | 0.2258 | 权重已收敛 |
| 训练耗时 | — | 3.9 秒 | 平均约 4.6 步/秒 |
| GPU 显存 | — | 14 GB | 单卡可运行 |

Loss 从 4.195 降到 0.989，下降 76.4%；token accuracy 从 42% 到 76%。这两个数字都是训练过程的指标，原文没有说明 token accuracy 的评测集。如果是训练集，它们说明模型拟合了这 12 条数据；如果是留出集，样本也太少，置信区间很宽。`[待补：token accuracy 的评测口径（训练集/留出集、是否包含 prompt token）]`

训练集外的名字问题只提供了一个泛化样例。模型答出 Donald J. Trump，说明它学到了一些角色关联；12 条数据、18 步、3.9 秒，更像是模型记住了这 12 个回答的句式。要确认这一点，需要留出集、风格一致性评分和事实性检查。`[待补：留出集上的风格一致性与事实性评测]`

如果目标是角色扮演，还要注意事实错误：模型可能把特朗普的公开事实和风格化的口头禅混在一起，风格对了，事实不一定对。这套流程证明了管道能跑通，不证明模型已经学会稳定的特朗普风格。下一步要补的是评测；把训练时间从 3.9 秒再压一秒，不会解决泛化问题。

---
title: "推荐系统中的 Hash 与位运算技巧：从 CityHash 到 Feature Hashing"
description: "推荐系统在线特征服务中的 Hash 与位运算技巧全梳理：CityHash64 选型、GEN_HASH2 特征交叉、x&(2^n-1) 快速取模、位标志与 UTF-8 位拆分，附 C++ 实战代码与 Feature Hashing 原理。重写选型结论：Rust 是 C++ 在线层最佳继任者，Zig 和 Python 都不适合。"
date: 2026-05-08T15:15:00+08:00
draft: false
weight: 100
aliases: ["/posts/hash-bitwise-techniques-summary/"]
categories: ["工程笔记"]
tags: ["Hash", "Bitwise", "推荐系统", "C++", "Rust", "Feature Hashing"]
---

review 一个传统的 C++ 在线特征服务（feature_engine）时，代码里密集出现 `(h1 << 1) ^ h2` 这样的组合。服务的职责不复杂：实时消费 Kafka 文档流，维护内存倒排索引；接收 Thrift RPC，组装用户画像和文档特征；用 CityHash64 加位运算生成 int64 特征 ID，喂给下游 LR。真正值得停下来看的，是这些技巧里哪些是系统编程的基本功，哪些在边界上会静默出错。

先看完整链路。`MAKE_HASH(feature_name)` 把特征名哈希成 `uint64_t`；对两个离散特征值，`GEN_HASH2(v1, v2)` 做一次组合；再把特征名哈希和组合值哈希做一次 `GEN_HASH2`，加上 `kHashFeatureOffset`，得到最终 ID。这条链路是工业界 Feature Hashing 的一个变体：标准 Hashing Trick 通常把特征哈希到固定维度，再用第二个哈希决定符号；这里把 64 位哈希直接当 ID，维度由下游模型定义，偏移量用来避开原始特征 ID 的命名空间。两种做法的碰撞模型不一样，不能直接套用论文里的碰撞上界。

## 一、CityHash64：选型理由和历史包袱

CityHash 在这个项目里承担两件事：把短特征名映射成 64 位值，以及给组合哈希提供初始熵。特征名通常很短，像 `user_age` 这样的字符串只有几个字节；CityHash 针对短字符串做了专门优化，64 位输出可以直接当 `uint64_t` 用，不需要再取模。项目处在一个不需要防碰撞攻击的离线/在线特征场景，只要分布均匀就够，所以非加密哈希是合适的选择。

CityHash 不是唯一的选择，这个项目用它有历史原因。MurmurHash 在 2008 年给现代非加密哈希定了基本结构；CityHash 在 2011 年由 Google 发布，重点是短字符串；FarmHash 在 2014 年作为继任者，主打跨平台稳定性；xxHash 在 2012 年出现，主打速度。这几个算法的定位大致是：CityHash 主打短字符串，FarmHash 主打跨平台稳定性，xxHash 主打速度。如果今天重选，应该先拿真实特征名分布跑一遍速度和碰撞率，而不是照着 2011 年的结论选。`[待补：真实特征名分布上 CityHash64、xxHash 与 FarmHash 的吞吐和碰撞率对比]`

项目里的 `std::hash` 只出现在 LRU Cache 的 key 上，不参与特征 ID。这个边界要分清：缓存哈希可以随标准库实现变化，特征 ID 一旦上线就不能变，否则模型权重会对不上。把两者混在同一个哈希函数里，等于把缓存实现也变成了模型协议的一部分。

## 二、GEN_HASH2：便宜的组合，代价在混合质量

组合特征的核心代码只有两行：

```cpp
// 二特征交叉：左移 1 位后与第二个哈希异或
#define GEN_HASH2(h1, h2) (((h1) << 1) ^ (h2))

// 三特征交叉：级联 GEN_HASH2
#define GEN_HASH3(h1, h2, h3) (((GEN_HASH2(h1, h2)) << 1) ^ (h3))
```

这个公式不在任何标准库里。它和 Boost 的 `hash_combine` 思路相同，都是把两个哈希压成一个，但 Boost 的版本多做了一点混合：

```cpp
// Boost 的经典哈希组合函数
size_t hash_combine(size_t seed, size_t value) {
  return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}
```

`0x9e3779b9` 是黄金分割数的 32 位近似值，用来打散分布；`seed << 6` 和 `seed >> 2` 把 seed 的位向两个方向扩散，再和常量、value 相加。项目的 `GEN_HASH2` 只左移一位，然后直接异或，少了一次右移折叠，也少了常量加法。它的优势是两条指令就能完成，劣势是 avalanche 弱：`h1` 的最高位在左移时被丢掉，`h1` 低位的变化也不会扩散到高位。对两个本来分布均匀的 64 位哈希，这会损失一部分混合质量。

这个代价在 LR 里表现为静默的模型偏差。它不会让服务崩溃，只会让两个不同的特征交叉共享同一个权重；训练照样收敛，离线指标可能只掉一点点，排查时很难把原因归到哈希上。原文说这个宏是"用一点碰撞率换极致速度"，这个判断在特征空间只有几百万、且交叉特征稀疏时成立；到了千万级以上，碰撞率要实测，不能靠感觉。`[待补：GEN_HASH2 与 Boost hash_combine 在真实特征组合分布上的碰撞率，以及碰撞对 LR 离线指标的影响]`

`GEN_HASH3` 只是把 `GEN_HASH2` 级联了一次。每级都左移一位，意味着三特征组合里 `h1` 的高位被丢了两次；组合数越多，这种损失越明显。如果三特征交叉是主力特征，建议至少对 `h1` 做一次 `h1 ^ (h1 >> 32)` 之类的折叠，再进入宏；改动只有一个表达式，但要先确认下游 ID 空间没有依赖现有的碰撞位模式。

## 三、特征 ID 偏移：命名空间与宏的边界

哈希版特征 ID 和普通版 ID 需要共存。原始特征 ID 从 0 开始，哈希 ID 是 64 位值；直接混在一起，普通 ID 会落在哈希 ID 的空间里。项目用一个常量偏移把两者分开：

```cpp
#define HASH_FEATURE_ID(_id_) (_id_) + kHashFeatureOffset  // +1,000,000
```

这个宏有两个边界，比偏移量本身更值得关注。

第一，宏没有给整个表达式加括号。`HASH_FEATURE_ID(x) * 2` 会展开成 `(x) + kHashFeatureOffset * 2`，而不是 `(x + kHashFeatureOffset) * 2`。在当前代码里它只出现在 `int64_t feature_id = HASH_FEATURE_ID(final_hash);` 这种简单赋值中，所以没出事；一旦有人拿它参与算术、比较或作为函数参数传递，优先级就会变。修法很简单，把宏体包成 `((_id_) + kHashFeatureOffset)`，但这个改动要和团队约定一起做，避免有人依赖旧展开。

第二，`final_hash` 是 `uint64_t`，最终却赋给 `int64_t`。64 位哈希的高位有一半概率为 1；在 C++20 之前，无符号到有符号的越界转换是实现定义行为，主流编译器按二补码回绕，但跨语言协议不一定接受负数。下游如果只接受 int64，应该先掩码到 63 位，或者全程使用 `uint64_t` 并在协议层明确无符号语义。特征 ID 是模型权重的索引，不是普通的整数；符号问题不会立刻报错，却会让 ID 空间在跨语言、跨数据库时对不上。

完整的哈希链路在代码里是这样：

```cpp
// 1. 特征名 → CityHash → 特征 ID 哈希
uint64_t id_hash = MAKE_HASH(feature_name);

// 2. 组合特征值（笛卡尔积）
for (auto v1 : values1) {
  for (auto v2 : values2) {
    uint64_t combined = GEN_HASH2(v1, v2);
    // 3. 最终 LR 特征 = 特征ID哈希 与 组合值哈希 再混合
    uint64_t final_hash = GEN_HASH2(id_hash, combined);
    // 4. 加上偏移，输出给模型
    int64_t feature_id = HASH_FEATURE_ID(final_hash);
  }
}
```

这段代码里，`feature_name` 的哈希只算一次，组合值在双重循环里算；如果 `values1` 和 `values2` 都很大，循环体里的两次 `GEN_HASH2` 会变成热点。把 `id_hash` 提前算好是对的，下一步可以把 `HASH_FEATURE_ID` 的内层加法提出来，或者把最终 ID 的计算向量化；但在动这些之前，先补上宏的括号和符号处理，否则优化会把错误放大。

## 四、快速取模：只在非负数和 2 的幂上成立

`x & (2^n - 1)` 等价于 `x % 2^n`，前提有两个：除数是 2 的幂，且 `x` 是非负整数。C++ 里 `-1 % 4` 等于 `-1`，而 `-1 & 3` 等于 `3`；把同一个宏用在有符号哈希上，桶的分布会完全不同。项目里的用法都满足前提，但宏本身没有类型约束，换一个调用点就可能踩到。

```cpp
// 随机退避 0~1023 ms，避免所有 consumer 同时重连
uint32_t backoff_ms = rand() & 1023;
std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
```

Kafka consumer 的退避用 `rand() & 1023` 保留低 10 位，范围是 0 到 1023。`rand()` 返回非负数，所以这里和取模等价；代价是低位质量取决于 `rand()` 的实现，老式线性同余生成器的低位周期通常比高位短。退避只需要把重连时间打散，低位相关性影响不大；如果换成需要高质量随机的场景，应该用 `<random>` 里的 `mt19937`，或者直接调用 `std::uniform_int_distribution`。

同样的技巧还出现在两个地方。Protobuf 序列化时用 `value & 0xFF` 取低 8 位，因为一个字节就是 2 的幂。Java HashMap 用 `hash & (table.length - 1)` 定位桶，但在取模之前多做了一步 `(h = key.hashCode()) ^ (h >>> 16)`，把高 16 位异或到低 16 位，否则低位相同的 key 会集中到同一个桶。扩容时 table 长度翻倍，元素要么留在原索引，要么移到原索引加 oldCap，这个拆分能成立也是因为容量是 2 的幂。这些用法共享同一个前提；换成任意模数，位运算的等价性就没了。

## 五、标志位：`^=` 是翻转，不是清位

标志位的四个基本操作：设置用 `|=`，检测用 `&`，组合用 `|`，清除用 `&= ~FLAG`。项目里 TinyXML2 的代码有一处注释和操作对不上：

```cpp
// TinyXML2 的节点标志
enum {
  NEEDS_ENTITY_PROCESSING   = 0x01,
  NEEDS_NEWLINE_NORMALIZATION = 0x02,
  NEEDS_WHITESPACE_COLLAPSING = 0x04,
  TEXT_ELEMENT              = NEEDS_ENTITY_PROCESSING | NEEDS_NEWLINE_NORMALIZATION
};

// 检测是否需要刷新
if (node->flags & NEEDS_FLUSH) { ... }

// 关闭 flush 标志
node->flags ^= NEEDS_FLUSH;
```

`^=` 是翻转：如果 `NEEDS_FLUSH` 本来为 1，它变成 0；如果本来为 0，这一行会把它变成 1。注释写的是"关闭"，只有在调用点之前已经保证该标志置位时，这个写法才等价于清位。更稳妥的写法是 `node->flags &= ~NEEDS_FLUSH;`，它幂等，重复执行也不会把标志打开。代码审查里这类问题很容易被"看起来像清位"骗过去，尤其是标志位的命名和注释都很顺的时候。

TinyXML2 的枚举用位或组合标志，例如 `TEXT_ELEMENT = NEEDS_ENTITY_PROCESSING | NEEDS_NEWLINE_NORMALIZATION`。这种写法节省空间，也让多个属性可以塞进一个整数；代价是类型系统帮不上忙，传错一个 `int` 也能编译，标志之间的互斥关系只能靠注释和测试保证。

POSIX 文件类型检测也是同一类操作：

```cpp
#include <sys/stat.h>

struct stat st;
stat(path.c_str(), &st);

// 判断是否为普通文件（位掩码）
if ((st.st_mode & S_IFREG) == S_IFREG) {
  // 是普通文件
}
```

这段代码的意图是判断普通文件，但它只检查了 `0x8000` 这一位。`S_IFREG` 是 `0x8000`，symlink 的 `S_IFLNK` 是 `0xA000`，socket 的 `S_IFSOCK` 是 `0xC000`，这两个类型也都带 `0x8000` 位，所以 `(st.st_mode & S_IFREG) == S_IFREG` 会把它们误判成普通文件。正确写法是 `(st.st_mode & S_IFMT) == S_IFREG`，先把类型位整体取出来，再和类型值比较；这样读代码的人也不会把"掩码"和"值"混在一起。

## 六、UTF-8 与分支预测：位运算的两种边界

UTF-8 编码是位运算的典型场景：一个 Unicode 码点按 6 位一组拆开，填入 continuation byte 的低 6 位，首字节用前缀标记长度。

```cpp
// 假设 input 是 Unicode 码点（32位）
// 1. 先填充 continuation bytes（从低 6 位开始）
for (int i = 3; i >= 0; --i) {
  if (input >> (6 * i)) {
    // 设置 continuation byte 标记：10xxxxxx
    unsigned char c = (input | 0x80) & 0xBF;  // 0x80=10000000, 0xBF=10111111
    // ...
    input >>= 6;  // 右移 6 位，处理下一个片段
  }
}

// 2. 设置首字节前缀
// 1字节: 0xxxxxxx
// 2字节: 110xxxxx
// 3字节: 1110xxxx
// 4字节: 11110xxx
input |= FIRST_BYTE_MARK[len];
```

以 U+4E2D（"中"）为例：码点是 `0x4E2D`，二进制 `0100 1110 0010 1101`。三字节编码的第一字节是 `1110` 加上码点高 4 位，得到 `0xE4`；后面两组各取 6 位，分别加上 `10` 前缀，得到 `0xB8` 和 `0xAD`。最终字节序列是 `E4 B8 AD`。代码里的 `(input | 0x80) & 0xBF` 就是在做"置最高位、保留低 6 位"这两步；`input >>= 6` 每轮把已经处理过的 6 位丢掉。

这段代码只负责位拼装，不负责校验码点范围、代理对和首字节长度。调用方如果传入一个超出 Unicode 范围的值，位运算仍然会产出一串字节，但不再是合法 UTF-8。线上处理用户输入时，校验应该放在编码之前，而不是指望这段循环报错。

分支预测提示是另一种边界：

```cpp
// common/common_define.h
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

// 使用场景
if (unlikely(ptr == nullptr)) {  // 告诉编译器：这个分支很少发生
  return Status::NullPointer;
}
```

`__builtin_expect` 是给编译器的静态提示，影响基本块布局：编译器会把 `unlikely` 分支放到冷路径，减少热路径的指令 cache 占用。它不会给 CPU 一个运行时的分支预测信号，现代 CPU 仍然按执行历史做动态预测。原文说它"帮助编译器优化 CPU 流水线，减少分支预测失败的惩罚"，这个说法容易让人以为它在干预硬件预测，实际收益主要来自代码布局。要判断有没有用，用 `perf stat -e branch-misses` 看真实的分支失败率；如果分支本来就是 50/50，加 `likely` 不会改变结果。

## 七、选型：Rust 接在线层，但别把重写当周末项目

代码 review 之后，团队讨论了用新语言重写在线层的可行性。结论是 Rust 最适合接在线层，Zig 和 Python 都不合适。这个判断的依据在依赖清单，不在语法喜好。

Zig 的短板是生态：项目需要 Kafka 客户端、Thrift 或 gRPC 服务端、Redis 集群客户端，以及能支撑线上排障的成熟库。Zig 在这些方向的选择很少，自己封装 FFI 会把维护成本转移到团队身上。Python 的问题在运行时：GIL 让多线程并行受限，GC 停顿和对象内存开销会直接反映到特征服务的尾延迟上；Python 适合离线特征工程、模型训练和实验平台，在线层应该用 Rust 或继续用 C++。

Rust 的对应关系比较完整：Kafka 用 `rdkafka`（librdkafka 的绑定），Thrift/gRPC 用 `tonic` 加 `prost`，Redis/MySQL 用 `redis`、`sqlx`，缓存用 `moka`，并发用 `rayon` 或 `tokio`。所有权和 `Send`/`Sync` 能在编译期拦下内存泄漏、use-after-free 和数据竞争，这对一个多线程的在线特征服务是实际收益。代价是团队要重新熟悉构建、排障和性能分析工具；原文估计"一个熟手 Rust 工程师，2-3 个月可以完成核心迁移"，这个数字只覆盖重写，不覆盖双跑、压测、回滚和团队学习。如果 C++ 服务稳定、团队没有 Rust 经验，重写的收益要打问号；触发条件通常是内存安全或并发改造的需求，语言的新旧本身不够。

Python 和 Rust 的组合也有合理位置：Python 留在离线和实验层，做特征工程、模型训练、召回策略实验和 AB 平台；Rust 守住在线层，做在线特征抽取、模型推理、召回和 Rank 服务。这个划分的边界是延迟和部署形态，不是语言偏好。Vowpal Wabbit 是另一个参照：它把哈希、训练和推理放在同一个进程里，feature_engine 只负责把原始数据变成 ID，模型在下游。两者都会哈希特征名，但 VW 内部直接用哈希值索引权重，feature_engine 要跨进程输出 int64 ID，边界不同，不能直接把 VW 的内部哈希搬过来。

## 八、下一步

如果下一步只做一件事，先给 `GEN_HASH2` 和 `HASH_FEATURE_ID` 补一组基于真实特征分布的碰撞与边界测试；`HASH_FEATURE_ID(x) * 2` 和 `flags ^= NEEDS_FLUSH` 这两个写法应该在新测试里各有一个用例。Rust 重写可以等这组测试跑完再决定。`[待补：当前线上特征组合的基数、可接受的碰撞率上限，以及这组测试的实际结果]`

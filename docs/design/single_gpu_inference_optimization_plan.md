# 单张笔记本 RTX 4090：LMCache 推理优化贡献路线

更新日期：2026-10-25。读者：只有一张笔记本 RTX 4090、希望通过 LMCache 贡献申请推理优化岗位的开发者。

本文是个人选题和实验计划，涵盖 benchmark、MP connector 与 KV 传输，因此放在这些模块共同对应的设计文档根目录。它不代表社区已接受的设计，也不包含本机性能实测结果。上游 PR 状态为撰写时查询结果，开始实现前需要再次确认。

逐步操作与每步 Codex 提示词见 [单卡 GQA KV 传输实战教程](v1/multiprocess/gqa_kv_transfer_codex_walkthrough.md)。

## 1. 推荐结论

**主线选 GPU↔CPU KV 传输的性能分析与局部优化；第二选择是 MP lookup 到调度准入之间的延迟；第三选择是小内存预算下预取、写回与淘汰的竞争。**

最终成果建议是一个核心运行时优化 PR，加一个可复用的实验或测试贡献，并附端到端结果。个人研究报告可以完整记录探索过程；提交社区的每个 PR 都应该只解决一个可复现的问题。

| 优先级 | 选题 | 单卡可验证性 | 与推理优化岗位的关联 | 主要合并条件 |
| --- | --- | --- | --- | --- |
| 1 | GQA KV gather/scatter 与传输提交开销 | 高；可先用合成 KV，随后跑小模型 | CUDA、访存、CPU/GPU 协作、profiling | 定位具体热点，保持数据正确，证明收益和适用范围 |
| 2 | MP lookup 完成到请求获准运行的延迟 | 高；一个推理进程和一个缓存进程即可 | 推理调度、异步 RPC、TTFT 归因 | 避开已有实现，复现当前 dev 仍存在的独立问题 |
| 3 | 有限 L1 下混合长短请求的资源竞争 | 高；缩小容量即可产生压力 | KV 管理、尾延迟、吞吐与公平性 | 用真实服务负载找到一个小范围修复 |
| 备选 | LRU 淘汰扫描与锁开销 | CPU 可测试，单卡可验证服务影响 | 并发、内存管理、性能诊断 | 锁竞争确实进入关键路径；避免只展示微基准 |

这些排序是基于硬件、验证成本与岗位匹配度的判断，不能换算成 PR 合并概率。

相比此前建议，降低 NIXL 批量文件注册和离线模拟器的优先级：前者增加存储环境与能力验证成本，后者适合作为辅助工具；当前最值得投入的学习与产出是 CUDA 时间线和真实推理请求的对应关系。

## 2. 硬件边界与起步配置

### 2.1 先按 16 GB 显存设计

NVIDIA 官方规格列出的 RTX 4090 Laptop GPU 显存为 16 GB。实验前用 `nvidia-smi` 确认实际型号、显存和驱动；本文不按桌面 4090 的显存预算规划。[官方规格](https://www.nvidia.com/en-us/geforce/laptops/40-series/)

建议使用以下阶梯配置，数值均为起点，需要根据启动日志调整：

| 项目 | 起点 | 扩展条件 |
| --- | --- | --- |
| 模型 | `Qwen/Qwen2.5-1.5B-Instruct`，BF16 或 FP16 | 主流程稳定后增加 `Qwen/Qwen2.5-3B-Instruct` |
| 并行度 | TP=1，单个 vLLM 实例 | 不需要同时运行多个模型服务 |
| 上下文 | 2K、4K、8K；输出 1 token 与 128 tokens 分开测 | 有余量再增加上下文和并发 |
| 并发 | 1、2、4，随后尝试 8 | 避免起步就用 32/64 并发掩盖问题 |
| vLLM 显存预算 | 从 `gpu_memory_utilization=0.70` 尝试 | 为桌面显示、CUDA context、临时缓冲和图捕获留余量 |
| LMCache L1 | 从 0.5–1 GiB 开始 | 先查看可用主存，不能照搬服务器的几十 GiB 配置 |
| L2 | 初期关闭；方向三再接本地 filesystem adapter | 确认磁盘余量、主存与 swap 状态 |

小模型有利于暴露固定调度与传输开销，但也可能放大这些开销占 TTFT 的比例。结果必须注明模型和硬件，不能据此推断大模型或数据中心 GPU 的加速比例。

### 2.2 用 KV 大小规划负载

对 TP=1、普通全注意力 GQA、未量化 KV，理论 KV 大小为：

```text
KV_bytes_per_token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
KV_bytes = KV_bytes_per_token × cached_tokens
```

这里的 2 分别表示 K 和 V，head 数使用 KV heads。按官方配置推导：

| 模型 | 层数 | KV heads | head_dim | BF16/FP16 KV / token | 8192 tokens 的 KV |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B-Instruct | 28 | 2 | 128 | 28 KiB | 224 MiB |
| Qwen2.5-3B-Instruct | 36 | 2 | 128 | 36 KiB | 288 MiB |

这些是张量有效载荷的计算值，未包含对齐、分页浪费、元数据、临时空间和模型权重。实际分配以引擎与 LMCache 状态为准。[1.5B 配置](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json)、[3B 配置](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/config.json)

例如，1.5B 的 256-token chunk 约为 7 MiB。主存 L1 只有 0.5 GiB 时，几个不同的长前缀就足以形成容量压力，无需用大模型制造压力。选择工作集时要扣除不同请求共享的前缀，并验证实际分配和淘汰。

### 2.3 系统环境

以原生 Linux 作为首选实验环境；已有 Windows 环境可以先用 WSL2 做 smoke test。vLLM 的 GPU 安装文档以 Linux 为平台要求；WSL2 的 CUDA 支持仍有特定限制，需要验证所选传输路径。[vLLM 安装文档](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)、[CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

先检查 GPU 注册、pinned memory、跨进程传输与数据回读。若 WSL2 下所选路径不通过，应优先采用项目支持的替代路径或 Linux 环境，记录能力差异；不要通过删除同步、关闭校验来继续测性能。WSL2 下将仓库、模型和 L2 文件放在 Linux 文件系统内，并记录虚拟机主存限制。

主存容量尚未提供，因此不预设 32/64 GB RAM。持续观察可用内存和 swap；主存不足时缩小 L1 与工作集。固定接电、性能模式与散热条件，记录温度、功耗及频率，避免把热降频误判成代码变化。

## 3. 方向一：GQA KV 传输与 gather/scatter 局部优化

### 3.1 要回答的问题

**在 1.5B/3B GQA 模型上，KV 恢复的时间主要花在数据移动、kernel 执行、提交开销，还是同步等待？其中哪个环节可以在不改变契约的情况下缩短？**

优先研究下面这条路径：

```text
请求确定可复用前缀
  → lookup / 预取完成
  → CPU KV 对象与 GPU paged KV 的地址、block mapping 准备
  → gather / scatter 或所选路径的 staging 与拷贝
  → 完成事件确认
  → 推理引擎消费 KV 并继续执行
```

代码入口包括 [MP CUDA kernel](../../csrc/cuda/mp_mem_kernels.cu)、[传输公共实现](../../lmcache/v1/multiprocess/transfer_context/base.py)、[异步 engine-driven 实现](../../lmcache/v1/multiprocess/transfer_context/async_engine_driven.py)。应以实际 trace 确认本次配置走到的路径，避免优化未被调用的实现。

现有代码已经有多层、多 block 的批量传输和提交机制。选题应从现有瓶颈出发，例如某种 GQA shape 的线程组织，或重复构造描述符的 CPU 开销。不能把“新增批量传输”直接当作缺失功能。

### 3.2 首版 PR 的合理范围

先做 profiling，再只选以下一个切口：

1. 对一种实际使用的 KV layout，调整 kernel 的线程组织或访问方式，改善低 KV-head 数下的有效工作分配。
2. 如果 CPU 描述符构造明显占比高，减少一次请求内已确认重复的元数据处理或分配。
3. 如果提交次数占主导，在已有批处理路径内减少可合并的提交，同时限制单批大小。

先保持现有同步和生命周期契约。设备级同步是否可缩小到事件依赖，需要独立分析生产者、传输流和消费者；不能仅凭某次调用耗时长就删除同步。直接访问 host memory 的 kernel 与 staging + DMA 路径也不能只按名字推断快慢。

### 3.3 实验与正确性

微基准先覆盖实际支持的 chunk size，例如 128、256、512 tokens；每次传输 1、2、4、8、16 个对象，覆盖 store/retrieve 两个方向及连续、离散 block mapping。只组合引擎 block size、chunk 对齐和所选 layout 支持的配置。

记录 CPU 提交时间、完整操作延迟、GPU 执行时间、提交次数、有效载荷字节数。CUDA event 只反映相应流上的时间；完整操作还需要 CPU wall time 与完成事件。带宽统一按同一口径计算有效载荷，明确是否包含 packing、staging 和等待。

用 Nsight Systems 定位 CPU API、kernel、拷贝和等待之间的空洞；只有 kernel 成为主要瓶颈时，再用 Nsight Compute 分析。profiling 运行与正式计时运行分开，避免追踪开销污染结果。[Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

数据校验必须包括：随机有效载荷、清空目标后再恢复、非连续 block、跳过前缀、边界 chunk、未选中区域不被覆盖，以及支持的 dtype/layout 回归。传输生命周期有变化时，增加重叠操作、异常和退出的测试；传输完成前不得释放或复用缓冲区。

**进入实现的条件：**瓶颈在重复实验中稳定存在，并能解释它为何影响整个恢复操作。微基准变快后，必须重新测共享前缀请求的 TTFT；只改善微基准时，应按实际范围陈述贡献。

## 4. 方向二：MP lookup 到调度准入的延迟

### 4.1 为什么单卡也适合

缓存命中后，请求仍可能等待 lookup 状态传回以及推理调度器再次访问。单卡、小模型能够低成本观察这类固定开销，适合训练异步状态机和调度分析能力。

需要建立如下时间线：

```text
客户端发送
  → 服务端接收与排队
  → 发出 LOOKUP
  → lookup / prefetch 结果可用
  → connector 观察到结果
  → 调度器准入
  → GPU 前向与首 token
```

客户端与服务器时间戳要明确时钟来源。TTFT 包含客户端和排队时间，不能把它全部记为缓存恢复成本；各阶段也可能重叠，不能直接把所有跨度相加。

### 4.2 必须避开的重复工作

截至本文日期，callback 驱动的 lookup 状态机已经合并；有界 lookup 等待预算和 L1 全命中跳过 L2 仍有独立 PR。先在包含已有改动的 dev 上重现残余问题。[已合并 #5960](https://github.com/LMCache/LMCache/pull/5960)、[等待预算 #5965](https://github.com/LMCache/LMCache/pull/5965)、[L1 全命中快路径 #5652](https://github.com/LMCache/LMCache/pull/5652)

第一步是在现有 benchmark 或观测机制上补充足以归因的测量。随后只针对一个已确认缺口提交修复，例如重复状态查询、无效等待，或某个边界状态未及时完成。不要预先承诺“换成 callback”或“加一个等待预算”。新增观测若已被现有事件覆盖，应复用事件，避免重复埋点。

### 4.3 实验设计

分别测 L1 热命中、冷 miss、部分前缀命中，使用并发 1/2/4/8 与 2K/4K/8K 输入。输出 1 token 用于突出 prefill 与首 token，输出 128 tokens 用于观察对其他请求 decode 的影响。

关键指标是结果就绪后到调度准入的等待、每请求查询次数、TTFT、吞吐和 decode 延迟。任何主动等待都需要检查对其他请求与调度线程的影响；事件及时返回也不等于 GPU 会立刻有空执行。

**进入实现的条件：**当前 dev 仍存在可重复、可归因且未被现有 PR 覆盖的额外等待。若已有工作解决问题，可以向原作者贡献独立测试或实验，随后转向方向一。

## 5. 方向三：有限 L1 下的预取、写回与淘汰竞争

### 5.1 用容量缩放制造真实压力

固定 GPU KV pool 和 host L1，交错发送 1K/2K 短请求与 4K/8K 长请求。多轮会话保留真实的上一轮 token 前缀并追加内容；不同会话并发，同一会话的轮次保持顺序。

从 L1=0.25/0.5/1 GiB 中选择符合主存条件的两档，再加一个宽裕容量对照。工作集超过目标容量后，检查真实预留失败、淘汰与 L2 读取，不根据 prompt 长度推断已经发生 offload。

先关闭 L2 研究 L1 竞争，再添加本地 filesystem L2。缩小 L1 有助于观察淘汰，但不能直接等价于 GPU 显存压力。记录 GPU pool、host L1 和 staging 的不同占用；KV 写入 CPU 也不意味着引擎已经回收对应 GPU block。

### 5.2 首版 PR 的合理范围

优先补充现有 benchmark 的确定性混合负载与容量压力证据，再根据结果修复一个问题。可能的调查对象包括：大请求反复无法准入、暂时的 staging 占用导致可避免的失败、某类无效写回挤占有效缓存。它们都是调查问题，不是本文已确认的缺陷。

内存感知预取准入已经合并，超时回退也已有 PR；多轮混合负载已有社区提案。实现前确认边界，与已有作者协调，不重复创建调度器。[预取准入 #5889](https://github.com/LMCache/LMCache/pull/5889)、[超时回退 #5255](https://github.com/LMCache/LMCache/pull/5255)、[混合负载讨论 #5848](https://github.com/LMCache/LMCache/issues/5848)

观察有效前缀命中 token、真实传输字节、短/长请求分别的 TTFT、等待时间、失败与回退率。超时或取消后，仍在执行的 I/O 必须继续占用相应预算并保护缓冲区，直到完成或确认中止。

**进入实现的条件：**容量宽裕对照正常，压力组出现明确退化，并能排除已有 PR 已修复的原因。如果只能观察到正常的容量不足，就记录边界，不为制造 PR 引入新策略。

## 6. 共用实验规范：证明收益来自改动

### 6.1 对照组与缓存状态

核心性能结论来自同一基线提交上的“原版 LMCache”与“仅包含该优化的 LMCache”对比。另设以下服务参照：

| 组别 | 目的 |
| --- | --- |
| vLLM，无外部缓存 | 测量重新 prefill 的成本 |
| vLLM，仅原生 prefix cache | 检查原生缓存已能解决多少复用 |
| vLLM + 原版 LMCache | 运行时改动的直接对照 |
| vLLM + 修改版 LMCache | 其余条件与直接对照相同 |

定位 LMCache 路径时，可以在原版和修改版都关闭引擎原生 prefix cache，以强制观察外部命中；端到端评估再以相同设置启用它，并扩大工作集，验证请求确实落到 LMCache。不同设置的结果分开报告。

分清四类状态：运行时已预热但 KV 为空、GPU 原生前缀命中、GPU 未命中但 L1 命中、L1 未命中但 L2 命中。完成编译和图捕获的预热后，用独立前缀或明确的缓存重置建立目标状态，单独采集测量阶段指标。

运行前清理本次实验的 KV 状态；L2 使用独立目录并记录是否复用文件。filesystem 的暖文件页缓存与真实存储读取要区分，不能把 L2 命中自动写成 NVMe 实际读盘。重启进程也不等于清空操作系统页缓存。

### 6.2 分两阶段扫描，控制成本

第一阶段：1.5B、L1-only、2K/8K 输入、并发 1/4、输出 1 token，筛出值得分析的场景。第二阶段只扩展有意义的场景到 3B、输出 128 tokens、并发 2/8 或本地 L2。不要一开始做所有参数的笛卡尔积。

每个正式场景至少做 3 次独立运行，交替或随机安排原版与修改版；记录温度、频率和环境。先用短 smoke test 验证数据正确，再收集足够样本。样本少时以 p50/p95 和逐次结果为主；若报告 p99，明确请求数，并尽量积累至少千级有效请求和多次运行证据，不能用几十个请求的 p99 作主要结论。

并发受控实验用于定位；固定请求到达率实验用于检查排队与服务能力。两者的吞吐口径不同。客户端记录提交、首 token、完成时间；多 token 流式响应的间隔不能未经检查直接当作逐 token ITL。

### 6.3 最低验收清单

- 数据一致性和对应公共接口测试通过，目标恢复前清空，避免旧数据掩盖错误。
- 固定模型 revision、dtype、tokenizer、实际输入 token 数、输出长度、seed 与请求序列。
- 固定 GPU KV pool、L1、chunk size、传输模式及调度设置；记录实际容量，避免不同进程开销导致 pool 改变。
- 同时报告 TTFT、完成请求数、吞吐、错误/回退率及 decode 影响，失败请求不能从分母中消失。
- 报告有效命中和真实拷贝字节，证明优化执行过；加入冷 miss 与低并发等对照。
- 改善需要明显超过重复运行噪声，并解释出现回归的形状或负载。预期收益只是验收目标，不能填入结果表。

## 7. 可以直接开始的命令与记录方式

下面是 Linux/WSL2 内的起步模板，已按仓库入口检查参数名称，但尚未在用户笔记本上执行。先安装项目与匹配的 vLLM/CUDA 环境，再保存版本和帮助输出；具体构建方法遵循仓库安装说明。初期保持模型权重与 KV 为 BF16/FP16，减少量化兼容性变量。

```bash
mkdir -p results/environment
nvidia-smi > results/environment/nvidia-smi.txt
git rev-parse HEAD > results/environment/lmcache-commit.txt
uv pip freeze > results/environment/python-packages.txt
lmcache server --help > results/environment/server-help.txt
lmcache bench server --help > results/environment/bench-server-help.txt
lmcache bench engine --help > results/environment/bench-engine-help.txt
```

终端一启动小容量 MP 服务；0.5 GiB 只是起点，先确认主存与所选传输路径可用：

```bash
lmcache server --port 6555 --l1-size-gb 0.5 --eviction-policy LRU
```

终端二运行单卡小模型。这个 smoke test 关闭原生 prefix cache，便于观察 LMCache。图捕获、显示占用或额外 CUDA context 导致显存不足时，先降低并发、上下文或显存预算；诊断时可尝试 eager 模式，但原版和修改版必须使用相同设置。

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.70 \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":6555}}'
```

终端三运行短负载，确认请求、存储和恢复链路。`37449` 是按 1.5B 的 28 KiB/token 推导出的每 GiB 理论 token 数取整值；实际有布局或对齐差异时，用注册后的状态替换它。该负载不等价于本文提出的混合长短、多轮排队实验。

```bash
lmcache bench engine \
  --engine-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --workload multi-round-chat \
  --tokens-per-gb-kvcache 37449 \
  --mrc-shared-prompt-length 1024 \
  --mrc-chat-history-length 2048 \
  --mrc-user-input-length 128 \
  --mrc-output-length 128 \
  --mrc-qps 0.5 \
  --mrc-duration 60 \
  --output-dir results/smoke \
  --no-interactive
```

继续实验前，固定模型与依赖 revision，并将完整启动参数、缓存重置方式、日志和配置归档。已有 benchmark 的负载、导出和扩展方式参见 [engine benchmark 设计](cli/commands/bench/engine_bench/bench-engine.md)；合成 KV 数据路径可从 `lmcache bench server` 入手，先检查当前版本的分配和校验语义。

建议的证据目录：

```text
results/<experiment>/
  environment/              # GPU、驱动、软件版本和提交 SHA
  configs/                  # 原版与修改版的完整运行配置
  requests/                 # token 序列、seed、请求到达计划
  baseline/run-01/          # 逐请求结果、日志、指标增量
  candidate/run-01/
  profiles/                 # 小范围 Nsight / CPU traces
  report.md                 # 瓶颈、改动、结果、回归和局限
```

## 8. PR 切分与执行顺序

以每周约 10–15 小时投入估计，先用 4–6 周完成一个方向；依赖安装与社区评审时间单独计算。

| 阶段 | 产出 | 继续条件 |
| --- | --- | --- |
| 第 1 周 | 单卡小模型跑通；固定版本与对照；确认缓存实际命中 | 原版数据正确、环境可重复 |
| 第 2 周 | 三类候选路径的初步归因；选择一个主要瓶颈 | 能在当前 dev 上复现，且有独立贡献空间 |
| 第 3–4 周 | 最小修复、公共接口回归测试、微基准 | 正确性通过，收益超过噪声 |
| 第 5 周 | 端到端复测、回归场景、PR 说明 | 能解释效果、代价和适用范围 |
| 第 6 周 | 处理评审、整理可分享报告与简历材料 | 如实区分已提交、已合并和实验性结果 |

建议按“必要的实验/测试扩展 → 一个运行时修复 → 可选进一步优化”拆分。测量能力若已有就直接复用，不为凑 PR 单独增加基础设施。所有新分支与 PR 基于 `dev`；个人求职计划本身通常留作个人材料，上游 PR 只携带可独立使用的代码、测试和技术文档。

提交前检查相关 issue/PR、维护者意见和默认行为兼容性。Python/CUDA 改动按仓库要求补测试、格式及 lint；公共接口优先，不以访问其他类私有状态来完成测试或观测。已有标准测试命令忽略了 MP/distributed 目录，因此必须另外运行本次实际涉及的测试。

如果两轮 profiling 都无法证明某个方向有实际瓶颈，应转向下一个方向。一个负结果可以进入个人报告；社区运行时改动需要具体收益或正确性依据。

## 9. 求职展示：把工作连成一条证据链

项目标题可以是“单 GPU 显存受限场景下的 KV cache 恢复与推理尾延迟优化”。报告重点解释：请求为何慢、trace 如何定位、改动缩短哪个阶段、数据与生命周期如何保证、端到端是否受益，以及额外代价。

建议准备四张图：TTFT 随输入长度/并发变化、传输有效载荷与延迟、关键路径的改动前后时间线，以及延迟改善与吞吐/decode 的取舍。图中写清硬件、模型、样本数与版本。

简历只使用实测并保留原始证据的数据，例如：

> 在 RTX 4090 Laptop、Qwen2.5-1.5B、[输入长度/并发] 下，通过 [trace 发现的瓶颈] 定位 KV 恢复开销；实现 [具体改动]，使 [测量指标] 从 [原版值] 变为 [修改值]；完成 [正确性与回归验证]，贡献 PR [链接与状态]。

填入数值前必须完成测量。单卡优化可以展示扎实的推理工程能力；多卡、跨节点或其他架构上的收益留待相应环境验证。

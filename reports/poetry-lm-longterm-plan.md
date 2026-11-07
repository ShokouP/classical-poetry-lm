# 古典诗·文写作模型 —— 长程训练计划（Long-Horizon Plan）

> 状态：**进行中** ｜ 更新：2026-11-06 18:50 ｜ 本文件是跨轮次的状态锚点，每轮推进后更新。

## 0. 总目标

在 Windows 原生环境（RTX 5090 Laptop 24GB）训练一个 **作诗 + 古典散文** 模型：

- 风格条件标签：`朝代·时期·作者·流派·体裁·题材·风格词`
- **格律宽**：只要求押韵/句长/句数大致合规，不强求平仄
- 流程：**CPT（继续预训练）→ SFT（指令微调）→ 评估 → 迭代 →（可选）9B 升级 → 部署 Ollama/DSH**

## 1. 验收标准（Definition of Done）

| 维度 | 指标 | 目标 |
|---|---|---|
| 体裁合规 | 生成诗/词的句数、句长符合要求比例 | ≥ 85% |
| 押韵 | 偶数句尾字同韵部（平水韵/词林正韵粗查） | ≥ 80% |
| 原创性 | 与训练语料最长公共 n-gram 重复率 | 连续 ≥ 12 字重叠 < 5% |
| 标签一致 | 风格分类器判定朝代/体裁与目标一致 | ≥ 70% |
| 散文 | 明清古文体（记/论/序/传）300–500 字成篇 | 人工抽检 ≥ 7/10 可用 |
| 可用性 | 合并/量化后进入 Ollama 或 llama.cpp，OpenAI 接口可用 | 是 |

## 2. 已完成基线（2026-11-06）

- ✅ 语料：全唐诗 56,589 / 全宋诗 252,852 / 宋词 21,053 / 元曲 11,057 / 诗话 210 种 1,133 万字 / 词话 103 种 / 文评 37 种 / 明清散文 4,168 篇 / 四库别集 1,279 部（471.6MB）
- ✅ CPT 语料：`cpt_v1_train.jsonl`（6,033 万字，含四库别集 2,023 万字）
- ✅ Teacher（DeepSeek V4.1-Flash）10,000 条，成本 ¥111.4，可用 9,955 条
- ✅ SFT v1：`sft_v1_train.jsonl` 9,504 条 / `sft_v1_val.jsonl` 451 条
- ✅ 真实评点对：`shihua_pairs_v1.jsonl` 24,741 对
- ✅ 原生环境：torch 2.11+cu128、bf16 LoRA 可用；triton-windows + flash-linear-attention 已装
- ✅ 效率基线：4B bf16 LoRA seq2048 ≈ **1,500 tok/s / 18GB**；9B seq1024 ≈ **755 tok/s / 23.9GB**
- ⏳ 4B CPT：checkpoint-100（110 步）→ 已断点续训至 300 步（进行中）

## 3. 阶段计划

### M1：4B CPT 完成（进行中）
- 输入：`cpt_v1_train.jsonl`、`Qwen3.5-4B`、checkpoint-100
- 动作：续训到 300 步（约 980 万 tokens）；保存 final LoRA
- 输出：`D:\train\models\cpt_4b_v1`（final adapter）
- 门槛：loss 平滑下降；抽查能写通顺文言/古诗（不要求体裁）
- ETA：约 1h（18:45 起）

### M2：SFT 数据集 v2（待做）
1. **拟作诗数据（新）**：从 `poems_v1.jsonl` 采样 ~15k 首，构造 (风格标签+题目+体裁要求 → 原诗)；**按作者切分**，val 作者不参与训练
2. 与 teacher 数据合并：评点 4.5k / 鉴赏 2.5k / 散文仿写 1k / 逆向指令 2k
3. 数据模板多样化（3–5 种 instruction 模板），加入「格律宽」提示
4. 输出：`sft_v2_train.jsonl` / `sft_v2_val.jsonl`（~25k 条）
- ETA：1–2h（脚本自动化）

### M3：4B SFT 训练（待做）
- 输入：`cpt_4b_v1` final adapter + `sft_v2_train.jsonl`
- 配置：bf16 LoRA r=64、seq2048、2 epoch、lr 5e-5、grad ckpt
- 输出：`D:\train\models\sft_4b_v1`
- 门槛：loss、val loss、抽检体裁合规
- ETA：~2h

### M4：评估与迭代（待做）
- 自动：体裁/句长/押韵/n-gram 重复/标签分类器
- 人工：抽 30–50 例盲评
- 若体裁合规 < 85%：补充「体裁约束」数据、提高该任务权重、退火学习率
- 输出：`reports/poetry-lm-eval-v1.md` + 样例集

### M5：9B 升级判定（条件触发）
- 触发条件：4B SFT 达到验收标准但质量不满意，或需要更强散文能力
- 方案：9B（Qwen3.5-9B）bf16 LoRA，seq1024，SFT on sft_v2（~4h）
- 备选：9B CPT（10–15h，仅当 SFT 显示语言分布不足时）

### M6：部署（待做）
- merge LoRA → bf16 → GGUF Q4_K_M（llama.cpp）→ Ollama `qwen- poetry-4b` → DSH provider
- 验证：Ollama 出诗 3 例 + DSH 会话 1 例

### M7：文档与复现（最后）
- 数据构造/训练/评估脚本清单、超参、结果、失败踩坑记录
- 产出：`reports/poetry-lm-final-report.md`

## 4. 关键决策点

| 编号 | 决策 | 触发条件 | 当前倾向 |
|---|---|---|---|
| D1 | CPT 300 步是否够 | loss/抽检 | 先 300 步；不足再加 200 步 |
| D2 | 4B 还是 9B 终版 | M4 评估 | 先 4B 跑通，9B 作增强 |
| D3 | 拟作数据占比 | 体裁合规率 | 初始 60% 拟作 + 40% 其他 |
| D4 | 是否用 27B/DeepSeek 蒸馏拟作 | teacher 诗质量 | 暂不（用户要求诗不靠 DeepSeek） |
| D5 | 部署形态 | 推理速度 | Ollama（简单）优先；llama.cpp MTP 可选 |

## 5. 风险与回滚

- **后台进程被会话切换杀掉**：使用 detached 进程（Start-Process + 日志），并每轮检查进程存活；恢复用 checkpoint。
- **显存贴顶**：4B seq2048=18GB 安全；9B seq2048 会 OOM，固定 seq1024。
- **背诵/抄袭泄漏**：SFT 数据按作者切分 + 生成后 n-gram 过滤 + repetition penalty。
- **Windows 无 fast path**：已用 fla/triton-windows；causal-conv1d 缺失但 conv 回退可接受。
- **API 成本**：teacher 已用 ¥111；补充数据优先用真实语料构造（零成本）。

## 6. 产物清单

```
D:\train\
├─ data\cleaned\   poems_v1.jsonl(309k) / prose_mingqing_v1.jsonl(4,168篇) / shihua_pairs_v1.jsonl(24,741)
├─ data\cpt\       cpt_v1_train.jsonl(6,033万字) / cpt_v1_val.jsonl
├─ data\sft\       teacher_v1\*.jsonl(10k) / sft_v1_train.jsonl / sft_v1_val.jsonl
├─ models\         cpt_4b_v1\ (LoRA) / sft_4b_v1\ / (可选) cpt_9b_v1
├─ runs\           cpt_4b_v1.log / cpt_4b_resume.log / samples_*.json / bench_*.log
└─ scripts\        *_*.py（数据、训练、评估、抽样）
```

## 7. 每轮自动推进规则

1. 检查 `D:\train\runs\*.log` 与进程：训练是否完成/中断；
2. 完成 → 进入下一阶段（M2→M3→M4），失败 → 读日志修复后从 checkpoint 恢复；
3. 每轮把「当前阶段/进度/下一步」写回本文件与 `D:\train\STATE.md`；
4. 达到验收标准或做完 M7 后，向用户汇报并请求确认是否继续 9B/部署。

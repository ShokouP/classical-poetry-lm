# 古典诗·文模型 —— 复现指南（Reproduction Guide）

> 环境：Windows 11 原生 ｜ RTX 5090 Laptop 24GB ｜ Python 3.12 venv（`D:\train\venv`）
> 更新：2026-11-07

## 1. 环境

```powershell
# 虚拟环境
python -m venv D:\train\venv
D:\train\venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
D:\train\venv\Scripts\pip install transformers peft datasets trl accelerate bitsandbytes `
  -i https://pypi.tuna.tsinghua.edu.cn/simple
# Qwen3.5 线性注意力 fast path（Windows 社区轮子）
D:\train\venv\Scripts\pip install triton-windows flash-linear-attention
# 评估 & 转换
D:\train\venv\Scripts\pip install pypinyin gguf sentencepiece protobuf
```

**关键点**
- `bitsandbytes` 4bit 在 Windows/Blackwell 上极慢（42 tok/s）→ **不用 QLoRA，用 bf16 LoRA**
- `flash-linear-attention` + `triton-windows` 可提速约 3 倍（4B seq2048：507 → ~1,500 tok/s）
- 训练**必须用定长 padding**（变长 batch 会造成 CUDA 分配器碎片化 → reserved 27GB → OOM）
- 训练前 `ollama stop <model>`，避免 Ollama 常驻占 3GB

## 2. 数据管线

| 步骤 | 脚本 | 产出 |
|---|---|---|
| 语料盘点 | `inventory.py` | `data/raw/_inventory.json` |
| 清洗诗 | `clean_poems.py` | `data/cleaned/poems_v1.jsonl`（309,441 首：唐 56,589 / 宋 252,852，含朝代/作者/体裁/题材） |
| 明清散文 | `extract_prose.py` | `data/cleaned/prose_mingqing_v1.jsonl`（4,168 篇：明文海 3,391 / 明文衡 520 / 古文观止 222 / 清文精选 36） |
| 诗话评点对 | `extract_shihua_pairs.py` | `data/cleaned/shihua_pairs_v1.jsonl`（24,741 对引诗+评语） |
| CPT 语料 | `build_cpt.py` | `data/cpt/cpt_v1_train.jsonl`（382,191 条 / 6,033 万字，含四库别集 2,023 万字） |
| Teacher SFT | `teacher_batch.py` → `filter_teacher.py` | `data/sft/teacher_v1/*.jsonl` → `sft_v1_train/val.jsonl`（9,955 条） |
| 真实诗拟作 | `build_sft_v2.py` | `sft_v2_train.jsonl` 39,309 条（拟作 15k + 真实诗话 13k + teacher） |
| 扩写数据 | `teacher_expand.py` | `teacher_expand/expand_v1.jsonl` 376 条（诗→小品文 / 古文加标点+扩写） |
| SFT v3 | `build_sft_v3.py` | `sft_v3_train.jsonl` 40,813 条（扩写 4× 加权排前） |

## 3. 训练

### CPT（4B，bf16 LoRA）
```powershell
D:\train\venv\Scripts\python.exe D:\train\scripts\train_cpt.py `
  --model D:\train\models\Qwen3.5-4B --data D:\train\data\cpt\cpt_v1_train.jsonl `
  --output D:\train\models\cpt_4b_v1 --seq 2048 --max-steps 300 --max-samples 60000
# 断点续训：--resume D:\train\models\cpt_4b_v1\checkpoint-100
```
- 速度：~22s/step（accum 16 × seq2048），300 步 ≈ 73 分钟，loss 2.36

### SFT（4B，bf16 LoRA，定长 padding）
```powershell
D:\train\venv\Scripts\python.exe D:\train\scripts\train_sft.py `
  --model D:\train\models\Qwen3.5-4B --data D:\train\data\sft\sft_v3_train.jsonl `
  --output D:\train\models\sft_4b_v2 --seq 1024 --max-steps 600 --max-samples 14000 `
  --save-steps 100 --lr 5e-5 --accum 16
```
- 速度：~13.6s/step，显存 alloc 10GB / reserved 15GB（稳定）

### 9B（可选，seq1024）
- bf16 LoRA seq1024 gc：755 tok/s，显存 23.9GB（贴顶）
- seq2048 会超显存（27.7GB）→ 只能用 seq1024

## 4. 评估

```powershell
D:\train\venv\Scripts\python.exe D:\train\scripts\eval_model.py `
  --base D:\train\models\Qwen3.5-4B --adapter D:\train\models\sft_4b_v2 `
  --out D:\train\runs\eval_sft_v2.json
# 生成 Markdown 报告
python D:\train\scripts\make_report.py --inputs "sft_v2=D:\train\runs\eval_sft_v2.json" --out reports\eval.md
```
指标：句数/句长合规、押韵（pypinyin 韵母近似）、8-gram 重叠（诗 2 万 + 散文 5 千，共 432 万 gram）。
注意：绝句 4 句、律诗 8 句，`expected = lines`（不是 lines×2）。

## 5. 导出与部署

```powershell
# 1) 合并 LoRA（CPU，~30s）
D:\train\venv\Scripts\python.exe D:\train\scripts\merge_lora.py `
  --base D:\train\models\Qwen3.5-4B --adapter D:\train\models\sft_4b_v2 `
  --output D:\train\models\sft_4b_v2_merged

# 2) 转 GGUF（需 llama.cpp conversion 包，已放在 D:\llama\llama.cpp）
D:\train\venv\Scripts\python.exe D:\llama\llama.cpp\convert_hf_to_gguf.py `
  D:\train\models\sft_4b_v2_merged --outfile D:\train\models\poetry-4b-v2-bf16.gguf --outtype bf16

# 3) 量化
D:\llama\b12110\llama-quantize.exe D:\train\models\poetry-4b-v2-bf16.gguf `
  D:\train\models\poetry-4b-v2-Q4_K_M.gguf Q4_K_M

# 4) Ollama 导入
ollama create poetry-4b-v2 -f D:\train\runs\Modelfile.poetry-4b-v2
```
- 4B Q4_K_M ≈ 2.58GB，Ollama 生成正常

## 6. 耗时/资源基线

| 任务 | 配置 | 耗时 | 显存 |
|---|---|---|---|
| 4B CPT 300 步 | seq2048, accum16 | 73 分钟 | 18GB |
| 4B SFT 600–800 步 | seq1024 定长, accum16 | 2–3 小时 | 15GB |
| 9B SFT 800 步 | seq1024 | ~4 小时 | 23.9GB |
| teacher 10k 条 | V4.1-Flash | 88 分钟 | ¥111 |
| teacher 扩写 400 条 | 关闭 thinking | 6.5 分钟 | ¥2.4 |

## 7. 坑位清单

1. **QLoRA 4bit 在 Windows 不可用**（42 tok/s）→ bf16 LoRA
2. **变长 batch → 分配器碎片化 OOM** → 定长 padding（本项目 SFT 已改）
3. **Ollama 常驻显存** → 训练前 stop
4. **PS 5.1 脚本含中文会解析错乱** → 自动化脚本只用 ASCII，中文写文件
5. **`-match` 默认大小写不敏感** → 关键判断用 `-cmatch`
6. **`apply_chat_template(tokenize=True)` 返回 BatchEncoding** → 用 `_extract_ids` 取 `input_ids`
7. **默认 thinking 模式**会让小模型输出英文思维链 → `enable_thinking=False`
8. 会话/模型切换会杀后台任务 → 用 detached 进程 + 日志 + checkpoint 续训

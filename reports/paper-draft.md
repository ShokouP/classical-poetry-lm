# Training Small Language Models for Classical Chinese Poetry and Prose: A Data-Centric Study with Evaluation Beyond Fluency

**Preprint draft — v0.1 (2026-11-07)**  
ShokouP · Independent researcher  
Code: https://github.com/ShokouP/classical-poetry-lm · Demo collection: `reports/qwen3.5-9B-诗文专家模型选.pdf`

---

## Abstract

We study how far small open-weight language models (Qwen3.5-4B and Qwen3.5-9B) can be pushed on **classical Chinese poetry and prose composition** using only commodity hardware (a single 24 GB laptop GPU, native Windows). We build a data-centric pipeline: (i) a 60.3M-character domain corpus assembled from public poetry, *shihua* (poetics) and Ming–Qing prose collections; (ii) domain-adaptive continued pre-training (CPT) with LoRA; (iii) supervised fine-tuning (SFT) on 40.8k instruction samples that combine **template-generated composition pairs from 300k real poems** (author-disjoint splits), **real quotation-commentary pairs from *shihua***, and teacher-generated commentary/imitation/expansion data; and (iv) evaluation that goes beyond fluency: genre-format compliance, **data-driven rhyme-class agreement**, and **corpus-level plagiarism detection**.

Our evaluation reveals three practical findings. First, a rhyme checker built from modern pinyin alone is misleading: only 15.6% of 139,408 regulated poems have identical pinyin finals on rhyming positions, whereas a **9-cluster rhyme inventory mined from the corpus** validates at **86.5%**, providing a usable automatic metric without a full *Pingshui* rhyme book. Second, SFT with real poems teaches format (seven-character regulated verse reaches 2/2 on clause count and length) but low-step training can reproduce famous poems verbatim; an **17.1M 8-gram fingerprint index** over 313k documents detects this (e.g., an exact copy of Du Mu's *Qiuxi* scores 0.619 overlap vs ≤0.01 for selected original works) and a Best-of-N selector vetoes such candidates. Third, adding only 376 teacher-generated **expansion** samples removes a severe long-form repetition failure: the same 381-character prompt yields 1,393 characters with duplicated paragraphs (CPT-only), 940 with a large repeated block (SFT-v1), and **1,101 without any duplicated paragraph or 20-gram** (SFT-v2).

We release code, evaluation data, and a curated bilingual collection of model works, and we document negative engineering results (4-bit QLoRA throughput on Windows, variable-length batching OOM) for reproducibility.

## 1 Introduction

Classical Chinese composition is a stringent test of a language model: it requires lexical register, allusion, genre conventions (句数/句长/对仗/押韵), and stylistic control across dynasties. Recent large models can produce plausible classical text, but training and evaluating **small** models is attractive for local deployment, privacy, and cost.

This paper asks: *with a single consumer GPU and open data, how good can a 4B/9B model become, and how should we measure "good"?* We contribute:

1. A reproducible data-to-deployment pipeline for classical Chinese poetry/prose on **native Windows** (bf16 LoRA, fixed-length padding, GGUF/Ollama export).
2. A **data-driven rhyme inventory** mined from 139k regulated poems, validated at 86.5% agreement, that avoids hand-built rhyme books.
3. A **corpus-scale plagiarism/bootstrapping detector** (17.1M 8-gram CRC32 fingerprints) and a Best-of-N selection procedure that jointly enforces format, rhyme and novelty.
4. An honest case study of **long-form repetition** and its fix via a small amount of teacher-generated expansion data.

## 2 Related Work

**Chinese poetry generation.** Prior work spans template/evolutionary methods, RNN/LSTM seq2seq with rhyme constraints, and Transformer-based systems trained on *Quan Tang Shi*; evaluation has long relied on BLEU-style overlap, human preference, and constraint satisfaction. We follow the constraint-satisfaction tradition but replace hand-coded rhyme tables with a corpus-mined inventory.

**Domain adaptation and instruction tuning.** LoRA (Hu et al., 2021) enables parameter-efficient adaptation. We use continued pre-training followed by SFT, a standard recipe, but focus on data construction from **real poems** rather than pure distillation.

**Memorization and evaluation.** Verbatim reproduction of training data is a known risk for generation models; n-gram overlap is a common detector. We operationalize it at corpus scale with hashed fingerprints.

## 3 Data

### 3.1 Corpora

| Source | Scale | Use |
|---|---|---|
| *Quan Tang Shi* / *Quan Song Shi* | 56,589 / 252,852 poems | composition pairs, rhyme mining |
| Song *ci* / Yuan *qu* | 21,053 / 11,057 | *ci* ability |
| *Shihua* / *Cihua* / *Wenping* | 11.33M / 3.58M / 4.26M chars | commentary SFT |
| *Ming Wen Hai*, *Ming Wen Heng*, *Guwen Guanzhi*, Qing prose | 4,168 essays / 3.54M chars | prose SFT/CPT |
| *Siku Bieji* | 1,279 volumes / 471.6 MB | CPT |
| **CPT corpus** | **382,191 docs / 60.33M chars** | continued pre-training |
| **SFT v3** | **40,813 samples** | instruction tuning |

### 3.2 Composition pairs from real poems

For each of 15,000 sampled poems (Tang/Song balanced, author-disjoint validation split), we synthesize an instruction from style tags `〔dynasty · author · genre · theme〕` plus title and genre requirements, with 4 paraphrase templates; the response is the original poem. We additionally build 1,800 continuation pairs (first half → full poem).

**Author-level split** (2% of authors held out) prevents the same poet's works from appearing in both train and validation.

### 3.3 Commentary and teacher data

We extract 24,741 quotation–commentary pairs from *shihua*; after filtering to poem-like quotations, 13,005 are used for commentary SFT. A proprietary teacher (DeepSeek-V4.1-Flash) generates 10k samples of *pingdian* (poetics commentary), appreciation, and prose imitation (≈¥111 at peak pricing), plus 400 **expansion** samples (expand a short classical passage to 450–800 characters, or turn a poem into a short prose piece), costing ≈¥2.4 after disabling thinking mode.

## 4 Method

### 4.1 Continued pre-training (CPT)

Qwen3.5-4B, LoRA r=64 all-linear, bf16, seq=2048 with packing, lr 1e-4, batch 1 × grad-accum 16, 300 optimizer steps (≈9.8M tokens). Loss: 3.57 → **2.36**. Wall time 73 min on RTX 5090 Laptop (24 GB).

### 4.2 Supervised fine-tuning (SFT)

Chat-template formatting with `enable_thinking=False`; loss masked to assistant tokens. LoRA r=64 (4B) / r=32 (9B), seq=1024 (4B) / 768 (9B), lr 5e-5, 600 optimizer steps.

**Engineering finding (Windows):** variable-length batching fragments the CUDA allocator: `allocated=10 GB` but `reserved=27 GB`, causing OOM/cuBLAS failures after ~280 steps. Padding every sample to a **fixed sequence length** stabilizes training (`alloc 10 GB / reserved 15 GB` for 4B). 4-bit QLoRA (bitsandbytes) was 36× slower than bf16 LoRA on this platform (42 vs ~1,500 tokens/s at 4B), so it is not used.

### 4.3 Best-of-N generation

For each request we sample N candidates and score: clause count match (+2), clause-length match (+2), rhyme-class agreement (+3), minus 8-gram overlap with the corpus, with a hard veto if overlap > 0.2. The highest-scoring candidate is returned.

## 5 Evaluation

### 5.1 Format compliance

Parse generated text into clauses (split on punctuation, drop titles/annotations); check clause count (4 for quatrains, 8 for regulated verse) and clause length (5 or 7 characters).

### 5.2 Data-driven rhyme classes

Modern pinyin finals are too strict: on 139,408 regulated poems, only **15.6%** have identical finals on rhyming (even) positions. We therefore:

1. Group characters by pinyin final (36 groups);
2. Count co-occurrence of finals within poems;
3. Merge groups with association ≥0.25, yielding **9 clusters** (e.g., `ong+ueng`, `ang+iang+uang`, `ou+iou`, `en+in+uen`);
4. Validate: **86.5%** of poems have all rhyming characters in one cluster.

This inventory is used for training-free, reference-free rhyme evaluation.

### 5.3 Plagiarism detection

From 313,609 documents we extract **17,096,951 unique 8-grams**, convert traditional→simplified, and store CRC32 hashes in a sorted `uint32` array (65 MB). Overlap = fraction of generated 8-grams found in the index. Validation: an exact copy of Du Mu's *Qiuxi* scores **0.619**; selected original works score **≤0.01**.

## 6 Results

### 6.1 Format and rhyme

| Genre | 4B-v2 count/len/rhyme | 9B-v1 count/len/rhyme |
|---|---|---|
| 五绝 (5-char quatrain) | 0/2 · 2/2 · 2/2 | 1/2 · 2/2 · 2/2 |
| 七绝 (7-char quatrain) | 0/2 · 1/2 · 2/2 | **2/2 · 2/2 · 2/2** |
| 五律 (5-char regulated) | 1/2 · 1/2 · 1/2 | **2/2 · 2/2** · 1/2 |
| 七律 (7-char regulated) | **2/2 · 2/2** · 2/2 | 1/2 · 1/2 · 1/2 |

Format is learned mainly from SFT: CPT alone produced 0/2 format compliance; after SFT, seven-character regulated verse is consistently correct. Rhyme agreement is best measured with the mined clusters (2/2 in several settings); pinyin-final checking would report 0.

### 6.2 Long-form repetition and expansion data

Same 381-character source, target 1,000 characters:

| Model | Output chars | Repetition |
|---|---|---|
| CPT only | 1,393 | duplicated paragraphs |
| SFT v1 (no expansion data) | 940 | large repeated block |
| **SFT v2 (+376 expansion samples)** | **1,101** | **0 duplicated paragraphs, 0 repeated 20-grams** |

A 0.9% addition of expansion data (376/40,813) eliminates the failure mode, suggesting that **task coverage**, not model scale, was the bottleneck.

### 6.3 Quantization

Q4_K_M export (2.58 GB for 4B) enables Ollama deployment but degrades rhyme stability (4B: 0–1/2 vs 1–2/2 in bf16). Best-of-N with rhyme scoring mitigates this at inference time.

### 6.4 Memorization

Low-step SFT can reproduce famous poems (9B at 200 steps emitted Du Mu's *Qiuxi* verbatim). After 600 steps and with author-disjoint composition data, our selected evaluation set shows ≤0.01 overlap; we still recommend fingerprint-based vetoing in deployment.

## 7 Discussion

- **Data coverage beats scale for specific failure modes.** Expansion data fixed repetition; real-poem composition pairs taught format; teacher commentary supplied evaluation vocabulary.
- **Evaluation must be corpus-grounded.** A rhyme metric derived from the corpus (86.5% self-consistency) is far more faithful than pinyin equality (15.6%).
- **Deployment constraints matter.** On Windows, bf16 LoRA + fixed padding + GGUF/Ollama is the practical path; QLoRA is unusable, and 9B is limited to seq≈768 on 24 GB.

## 8 Limitations

Evaluation uses 2 samples per prompt and one seed; human evaluation is limited to the curated collection. We do not enforce tonal patterns (平仄) or use a complete *Pingshui* rhyme book. The teacher is proprietary and the SFT data inherits its biases. The 9B model was SFT-only (no CPT) and is not yet fully benchmarked.

## 9 Conclusion

With open corpora and a single consumer GPU, a 4B/9B model can learn classical Chinese format and register to a useful degree, provided that data construction covers the target tasks and that evaluation checks format, rhyme and novelty rather than fluency alone. We release code and evaluation artifacts to support further work.

## Artifacts

- Code: https://github.com/ShokouP/classical-poetry-lm
- Curated collection: `reports/qwen3.5-9B-诗文专家模型选.pdf`
- Data pipeline, training scripts, evaluation scripts, rhyme inventory, 8-gram index builder.

## References (selected)

- Hu, E. et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
- Qwen Team. Qwen3.5 Technical Report. 2026.
- Zhang, Y. et al. (eds.) *Quan Tang Shi* digitized corpora, chinese-poetry.
- DeepSeek-AI. DeepSeek-V4.1-Flash model card. 2026.
- 王力.《汉语诗律学》. 上海教育出版社.
- 龙榆生.《词学十讲》.

*(Draft — citations to be completed; experiments single-seed; feedback welcome.)*

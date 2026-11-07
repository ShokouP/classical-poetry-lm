"""原生 Windows 训练效率 smoke test：QLoRA 4bit + LoRA，真实语料 batch，测 tokens/s 与显存。

用法：
  D:\train\venv\Scripts\python.exe smoke_test.py --model D:\train\models\Qwen3.5-4B --seq 4096 --steps 10
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--seq', type=int, default=4096)
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--rank', type=int, default=64)
    ap.add_argument('--data', default=r'D:\train\data\cpt\cpt_v1_train.jsonl')
    ap.add_argument('--out', default=r'D:\train\runs\smoke.json')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print('loading tokenizer...', flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print('loading model 4bit...', flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    load_kwargs = dict(trust_remote_code=True, quantization_config=bnb,
                       device_map={'': 0}, dtype=torch.bfloat16)
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    except Exception as e:
        print('AutoModelForCausalLM failed:', type(e).__name__, str(e)[:300], flush=True)
        from transformers import Qwen3_5ForCausalLM
        model = Qwen3_5ForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules='all-linear', task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable()
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'trainable={trainable/1e6:.1f}M / total={total/1e9:.2f}B', flush=True)

    # 真实语料 batch
    texts = []
    with open(args.data, encoding='utf-8') as f:
        for line in f:
            try:
                texts.append(json.loads(line)['text'])
            except Exception:
                pass
            if len(texts) >= 16:
                break
    batch = tok(texts, return_tensors='pt', truncation=True, max_length=args.seq, padding='max_length')
    input_ids = batch['input_ids'][:1].cuda()
    attn = batch['attention_mask'][:1].cuda()
    labels = input_ids.clone()
    print('batch:', input_ids.shape, flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)  # 仅 LoRA 参数可训练
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    losses = []
    for i in range(args.steps):
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    dt = time.time() - t0
    tokens = args.steps * input_ids.numel()
    res = {
        'model': args.model,
        'seq': args.seq,
        'steps': args.steps,
        'tokens': tokens,
        'elapsed_s': round(dt, 2),
        'tokens_per_s': round(tokens / dt, 1),
        'peak_vram_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2),
        'loss_first': round(losses[0], 4),
        'loss_last': round(losses[-1], 4),
        'trainable_m': round(trainable / 1e6, 1),
    }
    print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()

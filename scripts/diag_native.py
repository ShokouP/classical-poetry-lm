"""原生 Windows 训练效率对比：4B bf16 LoRA，不同 seq / 梯度检查点。"""
import json
import time

import torch
from transformers import Qwen3_5ForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

path = r'D:\train\models\Qwen3.5-4B'
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = Qwen3_5ForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16, device_map={'': 0})
model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05,
                                         target_modules='all-linear', task_type='CAUSAL_LM'))
text = '【唐·杜甫·七律·咏怀】\n登高\n风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。万里悲秋常作客，百年多病独登台。艰难苦恨繁霜鬓，潦倒新停浊酒杯。'

results = []
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

def bench(seq, gc, steps=3):
    model.gradient_checkpointing_enable() if gc else model.gradient_checkpointing_disable()
    model.train()
    model.config.use_cache = False
    batch = tok([text] * 2, return_tensors='pt', truncation=True, max_length=seq, padding='max_length')
    ids = batch['input_ids'][:1].cuda()
    attn = batch['attention_mask'][:1].cuda()
    labels = ids.clone()
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    try:
        out = model(input_ids=ids, attention_mask=attn, labels=labels)
        out.loss.backward()
        opt.step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(steps):
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        r = {'seq': seq, 'gc': gc, 'tokens_per_s': round(steps * ids.numel() / dt, 1),
             'peak_vram_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    except torch.cuda.OutOfMemoryError:
        r = {'seq': seq, 'gc': gc, 'error': 'OOM'}
        torch.cuda.empty_cache()
    print(json.dumps(r), flush=True)
    results.append(r)
    torch.cuda.empty_cache()

bench(2048, True)
bench(2048, False)
bench(1024, True)
bench(1024, False)
print(json.dumps(results, indent=2), flush=True)

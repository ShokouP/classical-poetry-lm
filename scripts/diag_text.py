"""对比：纯文本 Qwen3_5ForCausalLM bf16 LoRA 吞吐 + matmul 基准。"""
import json
import time

import torch

print('capability', torch.cuda.get_device_capability(0), '| bf16', torch.cuda.is_bf16_supported(), flush=True)
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda')
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda')
for _ in range(5):
    a @ b
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    a @ b
torch.cuda.synchronize()
flops = 2 * 4096 ** 3 * 20
print(f'matmul TFLOPS: {flops/(time.time()-t0)/1e12:.1f}', flush=True)

from transformers import Qwen3_5ForCausalLM, AutoTokenizer  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

path = r'D:\train\models\Qwen3.5-4B'
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
print('loading text-only bf16...', flush=True)
model = Qwen3_5ForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16, device_map={'': 0})
total = sum(p.numel() for p in model.parameters())
print(f'text-only params: {total/1e9:.2f}B', flush=True)
model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05,
                                         target_modules='all-linear', task_type='CAUSAL_LM'))
model.gradient_checkpointing_enable()
model.train()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'trainable: {trainable/1e6:.1f}M', flush=True)

seq = 2048
texts = ['【唐·杜甫·七律·咏怀】\n登高\n风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。'] * 2
batch = tok(texts, return_tensors='pt', truncation=True, max_length=seq, padding='max_length')
ids = batch['input_ids'][:1].cuda()
attn = batch['attention_mask'][:1].cuda()
labels = ids.clone()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.time()
steps = 3
for _ in range(steps):
    out = model(input_ids=ids, attention_mask=attn, labels=labels)
    out.loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
torch.cuda.synchronize()
dt = time.time() - t0
print(json.dumps({'text_only_bf16_tokens_per_s': round(steps * ids.numel() / dt, 1),
                  'peak_vram_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2)}, indent=2), flush=True)

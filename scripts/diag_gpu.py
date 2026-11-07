"""诊断：GPU 矩阵乘基准 + Qwen3.5-4B bf16 LoRA 吞吐对比（判断 4bit bnb 是否瓶颈）。"""
import json
import time

import torch

print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), flush=True)
print('device', torch.cuda.get_device_name(0), '| capability', torch.cuda.get_device_capability(0), flush=True)
print('bf16 supported:', torch.cuda.is_bf16_supported(), flush=True)

# --- matmul 基准 ---
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda')
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda')
for _ in range(5):
    c = a @ b
torch.cuda.synchronize()
t0 = time.time()
n = 20
for _ in range(n):
    c = a @ b
torch.cuda.synchronize()
dt = time.time() - t0
flops = 2 * 4096 ** 3 * n
print(f'matmul 4096^3 x{n}: {dt:.2f}s => {flops/dt/1e12:.1f} TFLOPS', flush=True)

# --- bf16 LoRA 吞吐 ---
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

model_path = r'D:\train\models\Qwen3.5-4B'
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print('loading bf16 model...', flush=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                             dtype=torch.bfloat16, device_map={'': 0})
lora = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05,
                  target_modules='all-linear', task_type='CAUSAL_LM')
model = get_peft_model(model, lora)
model.gradient_checkpointing_enable()
model.train()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'trainable={trainable/1e6:.1f}M / total={total/1e9:.2f}B', flush=True)

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
for i in range(steps):
    out = model(input_ids=ids, attention_mask=attn, labels=labels)
    out.loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
torch.cuda.synchronize()
dt = time.time() - t0
tokens = steps * ids.numel()
print(json.dumps({'bf16_tokens_per_s': round(tokens / dt, 1), 'elapsed_s': round(dt, 2),
                  'peak_vram_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2)}, indent=2), flush=True)

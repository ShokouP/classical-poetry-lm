"""9B 定长 padding SFT 显存探测（1-2 步）。"""
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from peft import LoraConfig, get_peft_model

path = r'D:\train\models\Qwen3.5-9B'
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = Qwen3_5ForCausalLM.from_pretrained(path, trust_remote_code=True,
                                           dtype=torch.bfloat16, device_map={'': 0})
model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05,
                                         target_modules='all-linear', task_type='CAUSAL_LM'))
model.gradient_checkpointing_enable()
model.train()
model.config.use_cache = False
print('params', sum(p.numel() for p in model.parameters()) / 1e9, 'B', flush=True)

for seq in (1024, 768):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    text = '【唐·杜甫·五律·咏怀】\n请作五律一首，写秋江夜泊。\n' + '风急天高猿啸哀，渚清沙白鸟飞回。' * 20
    batch = tok([text], return_tensors='pt', truncation=True, max_length=seq, padding='max_length')
    ids = batch['input_ids'].cuda()
    labels = ids.clone()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    try:
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        print(f'seq={seq} peak_alloc_GB={torch.cuda.max_memory_allocated()/1e9:.2f} '
              f'reserved={torch.cuda.memory_reserved()/1e9:.2f}', flush=True)
    except torch.cuda.OutOfMemoryError as e:
        print(f'seq={seq} OOM', flush=True)
    del ids, labels, batch

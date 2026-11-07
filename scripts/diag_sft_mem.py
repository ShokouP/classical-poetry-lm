"""诊断：SFT 训练显存异常（是否真的启用了梯度检查点）。"""
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from peft import LoraConfig, get_peft_model

path = r'D:\train\models\Qwen3.5-4B'
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = Qwen3_5ForCausalLM.from_pretrained(path, trust_remote_code=True,
                                           dtype=torch.bfloat16, device_map={'': 0})
model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05,
                                         target_modules='all-linear', task_type='CAUSAL_LM'))
print('before gc: use_cache=', model.config.use_cache, '| supports_gc=', getattr(model, 'is_gradient_checkpointing', 'NA'))
model.gradient_checkpointing_enable()
print('after gc on PeftModel: is_gradient_checkpointing=', getattr(model, 'is_gradient_checkpointing', 'NA'))
base = model.base_model.model
print('base model is_gradient_checkpointing=', getattr(base, 'is_gradient_checkpointing', 'NA'))

seq = 1024
text = '【唐·杜甫·五律·咏怀】\n请作五律一首，写秋江夜泊。\n' + '风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。' * 6
batch = tok([text], return_tensors='pt', truncation=True, max_length=seq, padding='max_length')
ids = batch['input_ids'].cuda()
labels = ids.clone()
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
torch.cuda.reset_peak_memory_stats()
for i in range(2):
    out = model(input_ids=ids, labels=labels)
    out.loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    print('step', i, 'peak GB', round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
print('FINAL peak GB', round(torch.cuda.max_memory_allocated() / 1e9, 2))

"""一次性修补脚本：Encoding 兼容 + resume 支持。"""
import io

# 1) sample_model.py
p = r'D:\train\scripts\sample_model.py'
s = open(p, encoding='utf-8').read()
old = """    if hasattr(tok, 'apply_chat_template'):
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    else:
        ids = tok(prompt, add_special_tokens=True)['input_ids']
    ids = torch.tensor([ids], device='cuda')"""
new = """    if hasattr(tok, 'apply_chat_template'):
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    else:
        enc = tok(prompt, add_special_tokens=True)
    if hasattr(enc, 'ids'):
        ids = enc.ids
    elif isinstance(enc, dict):
        ids = enc['input_ids']
    else:
        ids = enc
    ids = torch.tensor([ids], device='cuda')"""
if old in s:
    s = s.replace(old, new)
    open(p, 'w', encoding='utf-8').write(s)
    print('sample_model patched')
else:
    print('sample_model pattern NOT found')

# 2) train_sft.py
p = r'D:\train\scripts\train_sft.py'
s = open(p, encoding='utf-8').read()
old = """        prompt_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        msgs.append({'role': 'assistant', 'content': out})
        full_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)"""
new = """        prompt_enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        msgs.append({'role': 'assistant', 'content': out})
        full_enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        prompt_ids = prompt_enc.ids if hasattr(prompt_enc, 'ids') else (prompt_enc['input_ids'] if isinstance(prompt_enc, dict) else prompt_enc)
        full_ids = full_enc.ids if hasattr(full_enc, 'ids') else (full_enc['input_ids'] if isinstance(full_enc, dict) else full_enc)"""
if old in s:
    s = s.replace(old, new)
    open(p, 'w', encoding='utf-8').write(s)
    print('train_sft patched')
else:
    print('train_sft pattern NOT found')

# 3) train_cpt.py resume
p = r'D:\train\scripts\train_cpt.py'
s = open(p, encoding='utf-8').read()
if "--resume" not in s:
    s = s.replace("    ap.add_argument('--save-steps', type=int, default=100)",
                  "    ap.add_argument('--save-steps', type=int, default=100)\n    ap.add_argument('--resume', default='')")
    s = s.replace("    trainer.train()",
                  "    trainer.train(resume_from_checkpoint=(args.resume or None))")
    open(p, 'w', encoding='utf-8').write(s)
    print('train_cpt patched')
else:
    print('train_cpt already patched')

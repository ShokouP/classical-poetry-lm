"""用 SFT v2 对 DPO 正例 prompt 生成 rejected（同分布负例）。"""
import argparse
import glob
import json
import os
import time

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True)
ap.add_argument('--adapter', required=True)
ap.add_argument('--pos-dir', default=r'D:\train\data\dpo')
ap.add_argument('--out', required=True)
ap.add_argument('--max-new', type=int, default=700)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
model = Qwen3_5ForCausalLM.from_pretrained(args.base, trust_remote_code=True,
                                           dtype=torch.bfloat16, device_map={'': 0})
model = PeftModel.from_pretrained(model, args.adapter)
model.eval()


def ids_of(prompt):
    msgs = [{'role': 'user', 'content': prompt}]
    try:
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      enable_thinking=False)
    except TypeError:
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    if hasattr(enc, 'ids'):
        return enc.ids
    if 'input_ids' in enc:
        return enc['input_ids']
    return enc


rows = []
for p in sorted(glob.glob(os.path.join(args.pos_dir, 'dpo_pos_*.jsonl'))):
    for line in open(p, encoding='utf-8'):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
print('positives:', len(rows), flush=True)

out = open(args.out, 'w', encoding='utf-8')
t0 = time.time()
for i, r in enumerate(rows):
    ids = torch.tensor([ids_of(r['prompt'])], device='cuda')
    with torch.no_grad():
        g = model.generate(ids, max_new_tokens=args.max_new, do_sample=True, temperature=0.95,
                           top_p=0.95, top_k=50, repetition_penalty=1.1,
                           eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    text = tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True).strip()
    if len(text) < 20:
        continue
    rec = {'prompt': r['prompt'], 'chosen': r['chosen'], 'rejected': text,
           'task': r['task'], 'kind': r['kind'], 'model': r.get('model')}
    out.write(json.dumps(rec, ensure_ascii=False) + '\n')
    out.flush()
    if (i + 1) % 50 == 0:
        print(f'{i+1}/{len(rows)} elapsed={time.time()-t0:.0f}s', flush=True)
out.close()
print('saved', args.out, flush=True)

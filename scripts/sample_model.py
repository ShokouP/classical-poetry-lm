"""CPT 模型抽查：给定风格标签/题目，生成诗与古文；并输出基座对照。

用法：
  python sample_model.py --base D:\train\models\Qwen3.5-4B --adapter D:\train\models\cpt_4b_v1 \
    --out D:\train\runs\samples_cpt_4b.json
"""
import argparse
import json
import os
import time

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True)
ap.add_argument('--adapter', default='')
ap.add_argument('--out', default=r'D:\train\runs\samples.json')
ap.add_argument('--max-new', type=int, default=220)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
print('loading base...', flush=True)
model = Qwen3_5ForCausalLM.from_pretrained(args.base, trust_remote_code=True,
                                           dtype=torch.bfloat16, device_map={'': 0})
if args.adapter:
    print('loading adapter:', args.adapter, flush=True)
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

# 抽查题目：唐诗/宋诗/明清古文 各若干，带风格标签
prompts = [
    {'kind': '诗', 'prompt': '【唐·盛唐·王维·五律·山水】\n请作一首五言律诗，写山中傍晚的静寂与禅意。'},
    {'kind': '诗', 'prompt': '【宋·南宋·陆游·七律·怀古】\n请作一首七言律诗，登高怀古，感慨山河与故人。'},
    {'kind': '诗', 'prompt': '【唐·晚唐·李商隐·七绝·咏物】\n请作一首七言绝句，咏秋夜残烛，含蓄寄托。'},
    {'kind': '词', 'prompt': '【宋·北宋·苏轼·词·旷达】\n请填一首《念奴娇》，写江上秋夜与人生感慨。'},
    {'kind': '古文', 'prompt': '【明·明代·归有光·记】\n请写一篇古文《项脊轩志》风格的短文，记旧宅书斋与亲人旧事，300字左右。'},
    {'kind': '古文', 'prompt': '【清·清代·桐城派·论】\n请写一篇古文，论“文以载道”，风格近桐城派，400字左右。'},
    {'kind': '古文', 'prompt': '【战国·秦汉·赋】\n请写一段骚体赋，写秋江与远游之思，200字左右。'},
]



def _extract_ids(enc):
    if hasattr(enc, 'ids'):
        return enc.ids
    try:
        if 'input_ids' in enc:
            return enc['input_ids']
    except Exception:
        pass
    if isinstance(enc, list) and enc and hasattr(enc[0], 'ids'):
        return enc[0].ids
    return enc

def decode(ids):
    return tok.decode(ids, skip_special_tokens=True)


def gen(prompt, max_new):
    msgs = [{'role': 'user', 'content': prompt}]
    if hasattr(tok, 'apply_chat_template'):
        try:
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                          enable_thinking=False)
        except TypeError:
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    else:
        enc = tok(prompt, add_special_tokens=True)
    ids = _extract_ids(enc)
    ids = torch.tensor([ids], device='cuda')
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=0.9,
                             top_p=0.95, top_k=20, repetition_penalty=1.05,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    txt = decode(out[0][ids.shape[1]:])
    return txt, time.time() - t0


rows = []
for p in prompts:
    print('=' * 10, p['kind'], p['prompt'][:40], flush=True)
    txt, dt = gen(p['prompt'], args.max_new)
    print(txt[:800], flush=True)
    rows.append({'kind': p['kind'], 'prompt': p['prompt'], 'output': txt, 'elapsed_s': round(dt, 1)})

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, 'w', encoding='utf-8') as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print('saved:', args.out, flush=True)

"""Best-of-N 生成：按「韵部 + 句数/句长合规 + 低抄袭」打分选最优。

用法：python gen_bestofn.py --base ... --adapter ... --n 6 --out ...
"""
import argparse
import json
import os
import re
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

sys.path.insert(0, r'D:\train\scripts')

PROMPTS = [
    {'id': 'wujue', 'lines': 4, 'chars': 5,
     'prompt': '【唐·盛唐·王维·五绝·山水】\n请作五言绝句一首，写山中月夜的静寂与禅意。'},
    {'id': 'qijue', 'lines': 4, 'chars': 7,
     'prompt': '【唐·晚唐·李商隐·七绝·咏物】\n请作七言绝句一首，咏秋夜残烛，含蓄寄托。'},
    {'id': 'wulv', 'lines': 8, 'chars': 5,
     'prompt': '【唐·盛唐·杜甫·五律·羁旅】\n请作五言律诗一首，写秋江夜泊、羁旅思乡。'},
    {'id': 'qilv', 'lines': 8, 'chars': 7,
     'prompt': '【宋·南宋·陆游·七律·怀古】\n请作七言律诗一首，登高怀古，感慨山河与故人。'},
]


def parse_clauses(txt):
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    if lines and len(lines) > 1 and len(lines[0]) <= 10 and not re.search(r'[，。；、！？]', lines[0]):
        lines = lines[1:]
    clauses = []
    for l in lines:
        if l.startswith(('（', '(')):
            continue
        for c in re.split(r'[，。；、！？\s]+', l):
            c = re.sub(r'[^\u4e00-\u9fff]', '', c)
            if 2 <= len(c) <= 12:
                clauses.append(c)
    return clauses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--adapter', default='')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--max-new', type=int, default=200)
    ap.add_argument('--out', default=r'D:\train\runs\gen_bestofn.json')
    args = ap.parse_args()

    rhyme_map = json.load(open(r'D:\train\data\cleaned\rhyme_classes.json', encoding='utf-8'))['char_to_group']
    # 全量语料 8-gram 指纹（背诵检测）
    import zlib

    import numpy as np
    from opencc import OpenCC
    _cc = OpenCC('t2s')
    fp = np.load(r'D:\train\data\cleaned\poems_8gram_crc32.npy')

    def overlap_ratio(txt):
        t = re.sub(r'[^\u4e00-\u9fff]', '', _cc.convert(txt))
        if len(t) < 8:
            return 0.0
        hs = np.fromiter((zlib.crc32(t[i:i + 8].encode('utf-8')) for i in range(len(t) - 7)),
                         dtype=np.uint32, count=len(t) - 7)
        idx = np.clip(np.searchsorted(fp, hs), 0, len(fp) - 1)
        return int(np.sum(fp[idx] == hs)) / len(hs)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = Qwen3_5ForCausalLM.from_pretrained(args.base, trust_remote_code=True,
                                               dtype=torch.bfloat16, device_map={'': 0})
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def extract_ids(enc):
        if hasattr(enc, 'ids'):
            return enc.ids
        try:
            if 'input_ids' in enc:
                return enc['input_ids']
        except Exception:
            pass
        return enc

    def gen(prompt):
        msgs = [{'role': 'user', 'content': prompt}]
        try:
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                          enable_thinking=False)
        except TypeError:
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        ids = torch.tensor([extract_ids(enc)], device='cuda')
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new, do_sample=True,
                                 temperature=0.95, top_p=0.95, top_k=50,
                                 repetition_penalty=1.08,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    def score(txt, meta):
        clauses = parse_clauses(txt)
        s = 0.0
        exp = meta['lines']
        if len(clauses) == exp:
            s += 2.0
        elif abs(len(clauses) - exp) <= 1:
            s += 0.5
        if len(clauses) >= exp and all(len(c) == meta['chars'] for c in clauses[:exp]):
            s += 2.0
        chars = [c[-1] for c in clauses[1::2]]
        mapped = [rhyme_map.get(ch) for ch in chars]
        if mapped and all(m is not None for m in mapped) and len(set(mapped)) == 1:
            s += 3.0
        overlap = overlap_ratio(txt)
        if overlap > 0.20:
            s -= 10.0  # 硬否决：背诵/抄袭
        s -= overlap
        return s, clauses

    results = []
    for p in PROMPTS:
        cands = []
        for k in range(args.n):
            txt = gen(p['prompt'])
            sc, cl = score(txt, p)
            cands.append({'text': txt, 'score': round(sc, 2), 'clauses': cl})
        cands.sort(key=lambda x: -x['score'])
        best = cands[0]
        results.append({'id': p['id'], 'prompt': p['prompt'], 'best': best,
                        'all_scores': [c['score'] for c in cands]})
        print('===', p['id'], 'best score', best['score'], '| scores', [c['score'] for c in cands], flush=True)
        print(best['text'][:400], flush=True)

    json.dump(results, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('saved:', args.out, flush=True)


if __name__ == '__main__':
    main()

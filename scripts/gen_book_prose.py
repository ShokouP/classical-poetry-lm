"""为作品集生成干净的 9B 古文样本（禁引名篇原句），并汇总所有入选作品。"""
import json
import re
import sys
import zlib

import numpy as np
import torch
from opencc import OpenCC
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

BASE = r'D:\train\models\Qwen3.5-9B'
ADAPTER = r'D:\train\models\sft_9b_v1'
FP = np.load(r'D:\train\data\cleaned\poems_8gram_crc32.npy')
CC = OpenCC('t2s')

BANNED = ['项脊轩', '庭有枇杷树', '明月几时有', '大江东去', '先天下之忧', '落霞与孤鹜',
          '醉翁之意', '苟全性命于乱世', '清风徐来', '山不在高', '水不在深']

PROMPTS = [
    ('ji', '【明·归有光·记】\n请写一篇《项脊轩志》风格的古文，记旧宅书斋与亲人旧事，300字左右。'
           '不要引用、化用或提及《项脊轩志》原文任何句子（如"项脊轩""庭有枇杷树"等），须为原创。'),
    ('xu', '【东晋·王羲之·序】\n请写一篇古文序言，记一次秋日山亭雅集，感怀聚散，300字左右，语言清雅。'),
    ('lun', '【清·桐城派·论】\n请写一篇古文，论"文以载道"，风格近桐城派，350字左右。'),
]


def zh(t):
    return len(re.findall(r'[\u4e00-\u9fff]', t))


def overlap(txt):
    t = re.sub(r'[^\u4e00-\u9fff]', '', CC.convert(txt))
    if len(t) < 8:
        return 0.0
    hs = np.fromiter((zlib.crc32(t[i:i + 8].encode()) for i in range(len(t) - 7)),
                     dtype=np.uint32, count=len(t) - 7)
    idx = np.clip(np.searchsorted(FP, hs), 0, len(FP) - 1)
    return int(np.sum(FP[idx] == hs)) / len(hs)


def main():
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = Qwen3_5ForCausalLM.from_pretrained(BASE, trust_remote_code=True,
                                               dtype=torch.bfloat16, device_map={'': 0})
    model = PeftModel.from_pretrained(model, ADAPTER)
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

    out = {}
    for key, prompt in PROMPTS:
        best = None
        for k in range(3):
            ids = torch.tensor([ids_of(prompt)], device='cuda')
            with torch.no_grad():
                g = model.generate(ids, max_new_tokens=700, do_sample=True, temperature=0.95,
                                   top_p=0.95, top_k=50, repetition_penalty=1.15,
                                   eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
            txt = tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True).strip()
            ov = overlap(txt)
            n = zh(txt)
            banned = any(b in txt for b in BANNED)
            score = (0 if banned else 2) + (2 if 0.02 < ov < 0.08 else (0 if ov >= 0.15 else 1)) + \
                    (2 if 250 <= n <= 520 else 0) - ov
            print(key, k, 'chars', n, 'overlap', round(ov, 3), 'banned', banned, 'score', round(score, 2), flush=True)
            if best is None or score > best['score']:
                best = {'text': txt, 'chars': n, 'overlap': round(ov, 3), 'score': round(score, 2)}
        out[key] = best
    json.dump(out, open(r'D:\train\runs\book_prose_9b.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    for k, v in out.items():
        print('====', k, v['chars'], v['overlap'])
        print(v['text'][:200])


if __name__ == '__main__':
    main()

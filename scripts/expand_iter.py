"""分段迭代扩写：多轮续写，凑到约 1000 字。"""
import argparse
import json
import re
import time

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True)
ap.add_argument('--adapter', default='')
ap.add_argument('--input', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--target', type=int, default=1000)
ap.add_argument('--max-rounds', type=int, default=6)
args = ap.parse_args()

src = open(args.input, encoding='utf-8').read().strip()
tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
model = Qwen3_5ForCausalLM.from_pretrained(args.base, trust_remote_code=True,
                                           dtype=torch.bfloat16, device_map={'': 0})
if args.adapter:
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
    try:
        if 'input_ids' in enc:
            return enc['input_ids']
    except Exception:
        pass
    return enc


def gen(prompt, max_new=700):
    ids = torch.tensor([ids_of(prompt)], device='cuda')
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=True,
                             temperature=0.9, top_p=0.92, top_k=40,
                             repetition_penalty=1.1,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


BASE = ('你是擅长半文言散文的写作者。请把下面这段文字扩写到约 1000 个汉字：保持原文半文言、'
        '清冷幽微的第一人称文风，增加环境、感官、心理与联想，但不要改变原意、不要写成现代白话。\n\n原文：\n')
acc = ''
rounds = []
t0 = time.time()
for r in range(args.max_rounds):
    zh = len(re.findall(r'[\u4e00-\u9fff]', acc))
    if zh >= args.target * 0.9:
        break
    if r == 0:
        prompt = BASE + src + '\n\n请直接开始扩写（第一段约 300 字）：'
    else:
        prompt = (BASE + src + '\n\n已写内容：\n' + acc +
                  f'\n\n请从上一句之后接着写下一段（约 250–300 字），不要重复已写内容，不要总结：')
    piece = gen(prompt)
    piece = piece.strip()
    rounds.append({'round': r, 'zh': len(re.findall(r'[\u4e00-\u9fff]', piece)), 'text': piece})
    print(f'--- round {r}: +{rounds[-1]["zh"]} chars', flush=True)
    print(piece[:300], flush=True)
    acc = (acc + '\n' + piece).strip() if acc else piece

zh = len(re.findall(r'[\u4e00-\u9fff]', acc))
print('=' * 30)
print('TOTAL chinese chars:', zh, '| elapsed:', round(time.time() - t0, 1), flush=True)
print(acc, flush=True)
json.dump({'input': src, 'output': acc, 'chars': zh, 'rounds': rounds},
          open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('saved:', args.out, flush=True)

"""扩写任务：读取 expand_input.txt，用 CPT 小模型扩写到 1000 字。

用法：python expand_task.py --base D:\train\models\Qwen3.5-4B --adapter D:\train\models\cpt_4b_v1 \
       --input D:\train\runs\expand_input.txt --out D:\train\runs\expand_output.json
"""
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
ap.add_argument('--max-new', type=int, default=2200)
args = ap.parse_args()

src = open(args.input, encoding='utf-8').read().strip()
print('input chars:', len(src), flush=True)

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


prompt = (
    '请将下面这段文字扩写到 1000 字左右，保持原文的半文言文风、清冷幽微的语调与第一人称视角；'
    '可以增写环境、感官细节、心理与联想，但不要改变原意、不要写成现代白话，也不要逐句复述。\n\n'
    '原文：\n' + src + '\n\n扩写：'
)
msgs = [{'role': 'user', 'content': prompt}]
try:
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
except TypeError:
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
ids = torch.tensor([extract_ids(enc)], device='cuda')
print('prompt tokens:', ids.shape[1], flush=True)

t0 = time.time()
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=args.max_new, do_sample=True,
                         temperature=0.9, top_p=0.92, top_k=40,
                         repetition_penalty=1.08,
                         eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
dt = time.time() - t0
text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
zh = len(re.findall(r'[\u4e00-\u9fff]', text))
print('elapsed_s:', round(dt, 1), '| chinese chars:', zh, flush=True)
print('=' * 30, flush=True)
print(text, flush=True)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump({'input': src, 'output': text, 'chars': zh, 'elapsed_s': round(dt, 1)},
          open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('saved:', args.out, flush=True)

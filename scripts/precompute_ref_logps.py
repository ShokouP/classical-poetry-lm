"""预计算 DPO 参考策略（SFT 模型）的 logps，供后续 DPO 训练缓存使用。

用法：python precompute_ref_logps.py --base Qwen3.5-9B --adapter sft_9b_v1 \
      --data dpo_v1.jsonl --seq 384 --out ref_logps_9b.npz
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--seq', type=int, default=384)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    print('loading SFT model (reference)...', flush=True)
    model = Qwen3_5ForCausalLM.from_pretrained(args.base, trust_remote_code=True,
                                               dtype=torch.bfloat16, device_map={'': 0})
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def ids_of(text, gen):
        msgs = [{'role': 'user', 'content': text}]
        try:
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen,
                                          enable_thinking=False)
        except TypeError:
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen)
        if hasattr(enc, 'ids'):
            return list(enc.ids)
        if 'input_ids' in enc:
            return list(enc['input_ids'])
        return list(enc)

    def encode(row):
        p = ids_of(row['prompt'], True)[:max(8, args.seq // 2)]
        out = {}
        for k in ('chosen', 'rejected'):
            comp = tok(row[k], add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
            full = (p + comp)[:args.seq]
            n_prompt = min(len(p), len(full))
            lab = [-100] * n_prompt + full[n_prompt:]
            out[k] = (full, lab)
        return out

    def logp(ids, lab):
        L = len(ids)
        ids_t = torch.tensor([ids], device='cuda')
        lab_t = torch.tensor([lab], device='cuda')
        out = model(input_ids=ids_t, use_cache=False)
        logits = out.logits[:, :-1, :].float()
        tgt = lab_t[:, 1:]
        mask = (tgt != -100)
        if mask.sum() == 0:
            return 0.0
        lp = F.log_softmax(logits, dim=-1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return float((lp * mask).sum() / mask.sum())

    rows = [json.loads(l) for l in open(args.data, encoding='utf-8') if l.strip()]
    chosen, rejected = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(rows):
            e = encode(r)
            chosen.append(logp(*e['chosen']))
            rejected.append(logp(*e['rejected']))
            if (i + 1) % 200 == 0:
                print(f'{i+1}/{len(rows)} elapsed={time.time()-t0:.0f}s', flush=True)
    np.savez(args.out, chosen=np.array(chosen), rejected=np.array(rejected))
    print('saved', args.out, len(rows), flush=True)
    print('mean ref logp chosen/rejected:', np.mean(chosen), np.mean(rejected), flush=True)


if __name__ == '__main__':
    main()

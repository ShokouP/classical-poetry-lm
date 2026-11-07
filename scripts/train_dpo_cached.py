"""DPO 训练（参考 logps 走缓存，避免 adapter-disable 用错基座）。"""
import argparse
import json
import os
import random
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
    ap.add_argument('--ref-logps', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--seq', type=int, default=384)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--lr', type=float, default=3e-6)
    ap.add_argument('--beta', type=float, default=0.1)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--save-every', type=int, default=50)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    print('loading model+adapter...', flush=True)
    model = Qwen3_5ForCausalLM.from_pretrained(args.base, trust_remote_code=True,
                                               dtype=torch.bfloat16, device_map={'': 0})
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print('trainable:', sum(p.numel() for p in trainable) / 1e6, 'M', flush=True)

    ref = np.load(args.ref_logps)
    ref_c, ref_r = ref['chosen'], ref['rejected']

    rows = [json.loads(l) for l in open(args.data, encoding='utf-8') if l.strip()]
    assert len(rows) == len(ref_c), f'len mismatch {len(rows)} vs {len(ref_c)}'
    print('pairs:', len(rows), flush=True)

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

    prompt_cache = {}

    def encode(row, idx):
        p = prompt_cache.get(row['prompt'])
        if p is None:
            p = ids_of(row['prompt'], True)[:max(8, args.seq // 2)]
            prompt_cache[row['prompt']] = p
        out = {}
        for k in ('chosen', 'rejected'):
            comp = tok(row[k], add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
            full = (p + comp)[:args.seq]
            n_prompt = min(len(p), len(full))
            lab = [-100] * n_prompt + full[n_prompt:]
            if all(x == -100 for x in lab):
                return None
            out[k] = (full, lab)
        return out

    def logp(ids, lab):
        ids_t = torch.tensor([ids], device='cuda')
        lab_t = torch.tensor([lab], device='cuda')
        out = model(input_ids=ids_t, use_cache=False)
        logits = out.logits[:, :-1, :].float()
        tgt = lab_t[:, 1:]
        mask = (tgt != -100)
        if mask.sum() == 0:
            return torch.zeros(1, device='cuda')
        lp = F.log_softmax(logits, dim=-1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return (lp * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    order = list(range(len(rows)))
    random.seed(7)
    random.shuffle(order)
    os.makedirs(args.output, exist_ok=True)

    step = 0
    idx = 0
    pending = 0
    losses, accs, margins = [], [], []
    t0 = time.time()
    opt.zero_grad(set_to_none=True)

    while step < args.steps:
        if idx >= len(order):
            random.shuffle(order)
            idx = 0
        i = order[idx]
        idx += 1
        enc = encode(rows[i], i)
        if enc is None:
            continue

        pc = logp(*enc['chosen'])
        pr = logp(*enc['rejected'])
        rc = torch.tensor([ref_c[i]], device='cuda', dtype=torch.float32)
        rr = torch.tensor([ref_r[i]], device='cuda', dtype=torch.float32)

        margin = args.beta * ((pc - rc) - (pr - rr))
        # 跳过极端样本（防过拟合崩坏）
        if float(margin.detach().abs().max()) > 8.0:
            continue
        loss = -F.logsigmoid(margin).mean() / args.accum
        loss.backward()
        pending += 1
        losses.append(float(loss.detach()) * args.accum)
        accs.append(float((pc > pr).float().mean()))
        margins.append(float(margin.detach().mean()))

        if pending >= args.accum:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            pending = 0
            step += 1
            if step % 10 == 0:
                print(json.dumps({'step': step,
                                  'loss': round(sum(losses[-60:]) / len(losses[-60:]), 4),
                                  'acc': round(sum(accs[-60:]) / len(accs[-60:]), 3),
                                  'margin': round(sum(margins[-60:]) / len(margins[-60:]), 3),
                                  'elapsed_s': round(time.time() - t0)}), flush=True)
            if step % args.save_every == 0:
                ck = os.path.join(args.output, f'checkpoint-{step}')
                model.save_pretrained(ck)
                print('saved', ck, flush=True)

    model.save_pretrained(args.output)
    print('SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()

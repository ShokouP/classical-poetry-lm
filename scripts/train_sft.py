"""原生 Windows SFT 训练（bf16 LoRA，Qwen3.5 chat template，只对 assistant 部分算 loss）。

用法：
  python train_sft.py --model D:\train\models\Qwen3.5-4B --data D:\train\data\sft\sft_v1_train.jsonl \
    --output D:\train\models\sft_4b_v1 --seq 2048 --epochs 2
"""
import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoTokenizer, Qwen3_5ForCausalLM, Trainer, TrainingArguments,
                          TrainerCallback)



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

def build_example(tok, row, seq):
    sysmsg = row.get('system') or ''
    user = row.get('instruction') or ''
    out = row.get('output') or ''
    msgs = []
    if sysmsg:
        msgs.append({'role': 'system', 'content': sysmsg})
    msgs.append({'role': 'user', 'content': user})
    if hasattr(tok, 'apply_chat_template'):
        try:
            prompt_enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=False)
            msgs.append({'role': 'assistant', 'content': out})
            full_enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                               enable_thinking=False)
        except TypeError:
            prompt_enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
            msgs.append({'role': 'assistant', 'content': out})
            full_enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        prompt_ids = _extract_ids(prompt_enc)
        full_ids = _extract_ids(full_enc)
    else:
        prompt_ids = tok(f'{sysmsg}\n{user}\n', add_special_tokens=True)['input_ids']
        full_ids = tok(f'{sysmsg}\n{user}\n{out}', add_special_tokens=True)['input_ids']
    if len(full_ids) > seq:
        full_ids = full_ids[:seq]
    n_prompt = min(len(prompt_ids), len(full_ids))
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    # 固定长度 padding：避免变长 batch 造成 CUDA 分配器碎片化（Windows 上会 OOM）
    n_pad = seq - len(full_ids)
    if n_pad > 0:
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        full_ids = full_ids + [pad_id] * n_pad
        labels = labels + [-100] * n_pad
        mask = [1] * (seq - n_pad) + [0] * n_pad
    else:
        mask = [1] * seq
    return {'input_ids': full_ids, 'attention_mask': mask, 'labels': labels}


def collate(batch):
    maxlen = max(len(b['input_ids']) for b in batch)
    pad = 0
    input_ids, mask, labels = [], [], []
    for b in batch:
        n = maxlen - len(b['input_ids'])
        input_ids.append(b['input_ids'] + [pad] * n)
        mask.append(b['attention_mask'] + [0] * n)
        labels.append(b['labels'] + [-100] * n)
    return {'input_ids': torch.tensor(input_ids), 'attention_mask': torch.tensor(mask),
            'labels': torch.tensor(labels)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--seq', type=int, default=2048)
    ap.add_argument('--rank', type=int, default=64)
    ap.add_argument('--epochs', type=float, default=2.0)
    ap.add_argument('--max-steps', type=int, default=-1)
    ap.add_argument('--max-samples', type=int, default=0)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--accum', type=int, default=16)
    ap.add_argument('--save-steps', type=int, default=100)
    ap.add_argument('--resume', default='')
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = [json.loads(l) for l in open(args.data, encoding='utf-8') if l.strip()]
    if args.max_samples:
        rows = rows[:args.max_samples]
    print('samples:', len(rows), flush=True)
    feats = [build_example(tok, r, args.seq) for r in rows]
    feats = [f for f in feats if len(f['input_ids']) > 8]
    import statistics
    lens = [len(f['input_ids']) for f in feats]
    print('usable:', len(feats), '| tokens mean/max:', round(statistics.mean(lens)), max(lens), flush=True)

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(feats)

        def __getitem__(self, i):
            return feats[i]

    model = Qwen3_5ForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                               dtype=torch.bfloat16, device_map={'': 0})
    try:
        fr, tot = torch.cuda.mem_get_info()
        print(f'[mem] after load: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB '
              f'reserved={torch.cuda.memory_reserved()/1e9:.2f}GB '
              f'free={fr/1e9:.2f}GB total={tot/1e9:.2f}GB', flush=True)
    except Exception as e:
        print('mem_get_info failed', e, flush=True)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank * 2,
                                             lora_dropout=0.05, target_modules='all-linear',
                                             task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    print('trainable:', sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 'M', flush=True)

    targs = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type='cosine',
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim='adamw_torch',
        seed=42,
    )
    class MemCB(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kw):
            if torch.cuda.is_available():
                print(f'[mem] step={state.global_step} '
                      f'alloc={torch.cuda.memory_allocated()/1e9:.2f}GB '
                      f'reserved={torch.cuda.memory_reserved()/1e9:.2f}GB', flush=True)

    trainer = Trainer(model=model, args=targs, train_dataset=DS(), data_collator=collate)
    trainer.add_callback(MemCB())
    trainer.train(resume_from_checkpoint=(args.resume or None))
    trainer.save_model(args.output)
    print('SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()

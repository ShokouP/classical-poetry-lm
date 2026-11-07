"""原生 Windows CPT 训练（bf16 LoRA，transformers Trainer）。

用法（4B 试点）：
  python train_cpt.py --model D:\train\models\Qwen3.5-4B --data D:\train\data\cpt\cpt_v1_train.jsonl \
    --output D:\train\models\cpt_4b_v1 --seq 2048 --max-steps 300 --max-samples 60000
"""
import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
                          Qwen3_5ForCausalLM, Trainer, TrainingArguments)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--seq', type=int, default=2048)
    ap.add_argument('--rank', type=int, default=64)
    ap.add_argument('--max-steps', type=int, default=300)
    ap.add_argument('--max-samples', type=int, default=60000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--accum', type=int, default=16)
    ap.add_argument('--save-steps', type=int, default=100)
    ap.add_argument('--resume', default='')
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print('loading tokenizer', flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print('loading dataset', flush=True)
    ds = load_dataset('json', data_files=args.data, split='train')
    if args.max_samples and len(ds) > args.max_samples:
        ds = ds.select(range(args.max_samples))
    print('docs:', len(ds), flush=True)

    def tokenize(batch):
        return tok(batch['text'], truncation=False, padding=False)

    print('tokenizing (single process)...', flush=True)
    ds = ds.map(tokenize, batched=True, num_proc=None, remove_columns=ds.column_names)

    def group(batch):
        concat = []
        for ids in batch['input_ids']:
            concat.extend(ids)
            concat.append(tok.eos_token_id)
        total = (len(concat) // args.seq) * args.seq
        chunks = [concat[i:i + args.seq] for i in range(0, total, args.seq)]
        return {'input_ids': chunks, 'labels': [c[:] for c in chunks]}

    print('packing...', flush=True)
    ds = ds.map(group, batched=True, num_proc=None, remove_columns=ds.column_names)
    print('packed samples:', len(ds), flush=True)

    print('loading model', flush=True)
    model = Qwen3_5ForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                               dtype=torch.bfloat16, device_map={'': 0})
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank * 2,
                                             lora_dropout=0.05, target_modules='all-linear',
                                             task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable={trainable/1e6:.1f}M', flush=True)

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    targs = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        lr_scheduler_type='cosine',
        warmup_steps=max(10, args.max_steps // 20),
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
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    print('start training', flush=True)
    trainer.train(resume_from_checkpoint=(args.resume or None))
    trainer.save_model(args.output)
    print('SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()

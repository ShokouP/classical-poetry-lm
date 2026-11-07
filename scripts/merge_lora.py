"""合并 LoRA 到基座（CPU 执行，避免占 GPU），输出 bf16 HF 模型。

用法：python merge_lora.py --base D:\train\models\Qwen3.5-4B --adapter D:\train\models\sft_4b_v1 \
       --output D:\train\models\sft_4b_v1_merged
"""
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    print('loading tokenizer', flush=True)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    print('loading base on CPU...', flush=True)
    model = Qwen3_5ForCausalLM.from_pretrained(args.base, trust_remote_code=True,
                                               dtype=torch.bfloat16, device_map='cpu')
    print('loading adapter', flush=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    print('merging...', flush=True)
    model = model.merge_and_unload()
    model.eval()
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tok.save_pretrained(args.output)
    print('MERGED', args.output, flush=True)

    # 统计
    total = 0
    for root, dirs, files in os.walk(args.output):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    print(f'total size: {total/1e9:.2f} GB', flush=True)


if __name__ == '__main__':
    main()

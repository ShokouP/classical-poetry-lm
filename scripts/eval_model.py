"""模型评估：体裁合规、押韵（近似）、n-gram 抄袭检测、样例输出。

用法：
  python eval_model.py --base D:\train\models\Qwen3.5-4B --adapter D:\train\models\sft_4b_v1 \
    --out D:\train\runs\eval_sft4b.json
"""
import argparse
import json
import os
import re
import time

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

PROMPTS = [
    {'id': 'wujue', 'genre': '五绝', 'lines': 4, 'line_chars': 5,
     'prompt': '【唐·盛唐·王维·五绝·山水】\n请作五言绝句一首，写山中月夜的静寂与禅意。'},
    {'id': 'qijue', 'genre': '七绝', 'lines': 4, 'line_chars': 7,
     'prompt': '【唐·晚唐·李商隐·七绝·咏物】\n请作七言绝句一首，咏秋夜残烛，含蓄寄托。'},
    {'id': 'wulv', 'genre': '五律', 'lines': 8, 'line_chars': 5,
     'prompt': '【唐·盛唐·杜甫·五律·羁旅】\n请作五言律诗一首，写秋江夜泊、羁旅思乡。'},
    {'id': 'qilv', 'genre': '七律', 'lines': 8, 'line_chars': 7,
     'prompt': '【宋·南宋·陆游·七律·怀古】\n请作七言律诗一首，登高怀古，感慨山河与故人。'},
    {'id': 'ci', 'genre': '词', 'lines': 0, 'line_chars': 0,
     'prompt': '【宋·北宋·苏轼·词·旷达】\n请填一首《念奴娇》，写江上秋夜与人生感慨。'},
    {'id': 'prose_ji', 'genre': '记', 'lines': 0, 'line_chars': 0,
     'prompt': '【明·明代·归有光·记】\n请写一篇古文《项脊轩志》风格的短文，记旧宅书斋与亲人旧事，300字左右。'},
    {'id': 'prose_lun', 'genre': '论', 'lines': 0, 'line_chars': 0,
     'prompt': '【清·清代·桐城派·论】\n请写一篇古文，论“文以载道”，风格近桐城派，400字左右。'},
    {'id': 'fu', 'genre': '赋', 'lines': 0, 'line_chars': 0,
     'prompt': '【战国·秦汉·赋】\n请写一段骚体赋，写秋江与远游之思，200字左右。'},
]


def load_corpus_ngrams(limit=20000, n=8):
    from pathlib import Path
    grams = set()
    sources = [(r'D:\train\data\cleaned\poems_v1.jsonl', limit),
               (r'D:\train\data\cleaned\prose_mingqing_v1.jsonl', 5000)]
    for path, lim in sources:
        p = Path(path)
        if not p.exists():
            continue
        with open(p, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= lim:
                    break
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                t = re.sub(r'[^\u4e00-\u9fff]', '', r.get('text', ''))
                for j in range(len(t) - n + 1):
                    grams.add(t[j:j + n])
    return grams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--adapter', default='')
    ap.add_argument('--out', default=r'D:\train\runs\eval.json')
    ap.add_argument('--samples', type=int, default=2)
    ap.add_argument('--max-new', type=int, default=320)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = Qwen3_5ForCausalLM.from_pretrained(args.base, trust_remote_code=True,
                                               dtype=torch.bfloat16, device_map={'': 0})
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    try:
        from pypinyin import lazy_pinyin, Style
        have_pinyin = True
    except Exception:
        have_pinyin = False
        print('pypinyin not available; rhyme check disabled', flush=True)

    # 全量语料 8-gram 指纹库（CRC32，简繁归一）
    import zlib

    import numpy as np
    fp_path = r'D:\train\data\cleaned\poems_8gram_crc32.npy'
    fp = np.load(fp_path) if os.path.exists(fp_path) else np.array([], dtype=np.uint32)
    try:
        from opencc import OpenCC
        _cc = OpenCC('t2s')
        _conv = _cc.convert
    except Exception:
        _conv = lambda x: x  # noqa: E731
    print('corpus 8-gram fingerprints:', len(fp), flush=True)

    def overlap_ratio(txt):
        t = re.sub(r'[^\u4e00-\u9fff]', '', _conv(txt))
        if len(t) < 8 or len(fp) == 0:
            return 0.0
        hs = np.fromiter((zlib.crc32(t[i:i + 8].encode('utf-8')) for i in range(len(t) - 7)),
                         dtype=np.uint32, count=len(t) - 7)
        idx = np.searchsorted(fp, hs)
        idx = np.clip(idx, 0, len(fp) - 1)
        hit = int(np.sum(fp[idx] == hs))
        return hit / len(hs)

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
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new, do_sample=True,
                                 temperature=0.9, top_p=0.95, top_k=20,
                                 repetition_penalty=1.1,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        return txt, time.time() - t0

    def parse_clauses(txt, drop_title=True):
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        if drop_title and lines:
            first = lines[0]
            # 标题行：无标点、短、且后面还有内容
            if len(lines) > 1 and len(first) <= 10 and not re.search(r'[，。；、！？,.;!?]', first) and not first.startswith(('*', '（', '(')):
                lines = lines[1:]
        clauses = []
        for l in lines:
            if re.match(r'^\*\*', l) or l.startswith(('（', '(')):
                continue
            for c in re.split(r'[，。；、！？,.;!?\s]+', l):
                c = re.sub(r'[^\u4e00-\u9fff]', '', c)
                if 2 <= len(c) <= 12:
                    clauses.append(c)
        return clauses

    rhyme_map = {}
    rp = r'D:\train\data\cleaned\rhyme_classes.json'
    if os.path.exists(rp):
        rhyme_map = json.load(open(rp, encoding='utf-8')).get('char_to_group', {})
        print('rhyme clusters loaded:', len(rhyme_map), flush=True)

    def rhyme_check(clauses):
        if len(clauses) < 4:
            return None
        chars = [c[-1] for c in clauses[1::2]]
        mapped = [rhyme_map.get(ch) for ch in chars]
        if rhyme_map and all(m is not None for m in mapped):
            return len(set(mapped)) == 1
        if not have_pinyin:
            return None
        finals = []
        for ch in chars:
            py = lazy_pinyin(ch, style=Style.FINALS)
            finals.append(py[0] if py else '')
        return len(set(finals)) == 1 if len(finals) >= 2 else None

    rows = []
    for p in PROMPTS:
        outs = []
        for k in range(args.samples):
            txt, dt = gen(p['prompt'])
            clauses = parse_clauses(txt)
            lines = clauses
            overlap = overlap_ratio(txt)
            rec = {'text': txt, 'lines': lines, 'elapsed_s': round(dt, 1), 'overlap_8gram': round(overlap, 3)}
            if p['lines']:
                expected = p['lines']
                rec['clauses'] = clauses
                rec['line_count_ok'] = len(clauses) == expected
                rec['line_len_ok'] = (len(clauses) >= expected and
                                      all(len(c) == p['line_chars'] for c in clauses[:expected]))
                rec['rhyme_ok'] = rhyme_check(clauses)
            outs.append(rec)
        rows.append({**p, 'outputs': outs})
        print('===', p['id'], 'done', flush=True)

    # 汇总
    summary = {'n_prompts': len(PROMPTS), 'samples': args.samples, 'rows': rows}
    for p in rows:
        if p['lines']:
            summary.setdefault('poem_metrics', {})[p['id']] = {
                'line_count_ok': sum(1 for o in p['outputs'] if o.get('line_count_ok')),
                'line_len_ok': sum(1 for o in p['outputs'] if o.get('line_len_ok')),
                'rhyme_ok': sum(1 for o in p['outputs'] if o.get('rhyme_ok')),
                'avg_overlap': round(sum(o['overlap_8gram'] for o in p['outputs']) / len(p['outputs']), 3),
            }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('SAVED', args.out, flush=True)
    print(json.dumps(summary.get('poem_metrics', {}), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

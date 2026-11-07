"""通过 Ollama API 评测量化模型（与 eval_model.py 同样的指标）。

用法：python eval_ollama.py --model poetry-4b-v2 --out D:\train\runs\eval_ollama_v2.json
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, r'D:\train\scripts')
from eval_model import PROMPTS, load_corpus_ngrams  # noqa: E402

API = 'http://127.0.0.1:11434/api/chat'


def chat(model, prompt, num_predict=320, temperature=0.9):
    body = json.dumps({'model': model,
                       'messages': [{'role': 'user', 'content': prompt}],
                       'stream': False, 'think': False,
                       'options': {'num_predict': num_predict, 'temperature': temperature,
                                   'top_p': 0.95, 'repeat_penalty': 1.1}}).encode('utf-8')
    req = urllib.request.Request(API, data=body,
                                 headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=600) as resp:
        obj = json.loads(resp.read().decode('utf-8'))
    return obj.get('message', {}).get('content', '')


def parse_clauses(txt, drop_title=True):
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    if drop_title and lines:
        first = lines[0]
        if len(lines) > 1 and len(first) <= 10 and not re.search(r'[，。；、！？,.;!?]', first) \
                and not first.startswith(('*', '（', '(')):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--samples', type=int, default=2)
    ap.add_argument('--max-new', type=int, default=320)
    args = ap.parse_args()

    try:
        from pypinyin import lazy_pinyin, Style
        have_pinyin = True
    except Exception:
        have_pinyin = False

    # 全量语料 8-gram 指纹库
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
        return int(np.sum(fp[idx] == hs)) / len(hs)

    rhyme_map = {}
    rp = r'D:\train\data\cleaned\rhyme_classes.json'
    if os.path.exists(rp):
        rhyme_map = json.load(open(rp, encoding='utf-8')).get('char_to_group', {})

    def rhyme_ok(clauses):
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
        for _ in range(args.samples):
            t0 = time.time()
            txt = chat(args.model, p['prompt'], args.max_new)
            clauses = parse_clauses(txt)
            overlap = overlap_ratio(txt)
            rec = {'text': txt, 'clauses': clauses, 'overlap_8gram': round(overlap, 3),
                   'elapsed_s': round(time.time() - t0, 1)}
            if p['lines']:
                expected = p['lines']
                rec['line_count_ok'] = len(clauses) == expected
                rec['line_len_ok'] = (len(clauses) >= expected and
                                      all(len(c) == p['line_chars'] for c in clauses[:expected]))
                rec['rhyme_ok'] = rhyme_ok(clauses)
            outs.append(rec)
        rows.append({**p, 'outputs': outs})
        print('===', p['id'], 'done', flush=True)

    summary = {'model': args.model, 'rows': rows, 'poem_metrics': {}}
    for p in rows:
        if p['lines']:
            summary['poem_metrics'][p['id']] = {
                'line_count_ok': sum(1 for o in p['outputs'] if o.get('line_count_ok')),
                'line_len_ok': sum(1 for o in p['outputs'] if o.get('line_len_ok')),
                'rhyme_ok': sum(1 for o in p['outputs'] if o.get('rhyme_ok')),
                'avg_overlap': round(sum(o['overlap_8gram'] for o in p['outputs']) / len(p['outputs']), 3),
            }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps(summary['poem_metrics'], ensure_ascii=False, indent=2), flush=True)
    print('saved:', args.out, flush=True)


if __name__ == '__main__':
    main()

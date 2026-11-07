"""古典散文 teacher 数据质量过滤（抄袭/现代词/长度/禁用名句）。"""
import glob
import hashlib
import json
import os
import re
import zlib

import numpy as np
from opencc import OpenCC

D = r'D:\train\data\sft\teacher_classical_prose'
FP = r'D:\train\data\cleaned\poems_8gram_crc32.npy'
OUT = os.path.join(D, 'filtered.jsonl')
CC = OpenCC('t2s')
fp = np.load(FP)
BANNED = ['滕王阁', '项脊轩', '兰亭', '岳阳楼', '赤壁赋', '春夜宴', '桃花园序',
          '潦水尽而寒潭清', '仰观宇宙之大', '天地者，万物之逆旅', '庭有枇杷树',
          '明月几时有', '大江东去', '先天下之忧', '落霞与孤鹜', '清风徐来',
          '山不在高', '水不在深', '壬戌之秋', '永和九年', '庆历四年春']


def norm(t):
    return re.sub(r'[^\u4e00-\u9fff]', '', CC.convert(t))


def overlap(txt):
    t = norm(txt)
    if len(t) < 8:
        return 0.0
    hs = np.fromiter((zlib.crc32(t[i:i + 8].encode()) for i in range(len(t) - 7)),
                     dtype=np.uint32, count=len(t) - 7)
    idx = np.clip(np.searchsorted(fp, hs), 0, len(fp) - 1)
    return int(np.sum(fp[idx] == hs)) / len(hs)


def source_of(path, row):
    m = (row.get('meta') or {}).get('model')
    if m:
        return m
    name = os.path.basename(path)
    if 'glm' in name:
        return 'glm-5.3'
    if 'kimi31' in name:
        return 'k3.1-pro'
    if 'k3' in name:
        return 'k3'
    if 'kimi' in name:
        return 'kimi-for-coding'
    return 'deepseek-flash'


def main():
    stats = {}
    kept = []
    seen = set()
    for p in sorted(glob.glob(os.path.join(D, '*.jsonl'))):
        if os.path.basename(p) == 'filtered.jsonl':
            continue
        n = 0
        reasons = {'overlap': 0, 'modern': 0, 'len': 0, 'banned': 0, 'dup': 0}
        for line in open(p, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            n += 1
            out = r.get('output', '')
            src = source_of(p, r)
            ov = overlap(out)
            modern = len(re.findall(r'[的了是著]', out))
            rate = modern / max(1, len(out)) * 100
            if ov > 0.12:
                reasons['overlap'] += 1
                continue
            if rate > 3.0:
                reasons['modern'] += 1
                continue
            if not (150 <= len(out) <= 800):
                reasons['len'] += 1
                continue
            if any(b in out for b in BANNED):
                reasons['banned'] += 1
                continue
            h = hashlib.md5(norm(out).encode()).hexdigest()
            if h in seen:
                reasons['dup'] += 1
                continue
            seen.add(h)
            r['meta'] = dict(r.get('meta') or {})
            r['meta']['source_model'] = src
            r['meta']['overlap8'] = round(ov, 4)
            kept.append(r)
        stats[os.path.basename(p)] = {'total': n, 'kept': n - sum(reasons.values()), **reasons}
        print(f'{os.path.basename(p):40s} total={n:5d} kept={n-sum(reasons.values()):5d} {reasons}', flush=True)

    with open(OUT, 'w', encoding='utf-8') as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('TOTAL kept:', len(kept), '->', OUT)
    json.dump(stats, open(os.path.join(D, '_filter_stats.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    from collections import Counter
    print(Counter((r['meta']['source_model']) for r in kept))


if __name__ == '__main__':
    main()

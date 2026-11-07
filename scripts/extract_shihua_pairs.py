"""从 210 种诗话中抽取「引诗 + 评语」真实评点对。

策略：
- 按行/段切条目；
- 识别引诗：优先「」『』，其次“”里含诗句特征（多字、有逗号）；
- 评语 = 条目去掉引诗后的文字；
- 过滤过短/无汉字条目。
输出：D:\train\data\cleaned\shihua_pairs_v1.jsonl
"""
import glob
import json
import os
import re

SRC = r'D:\poetry-corpus\daizhigev20\诗藏\诗话'
OUT_DIR = r'D:\train\data\cleaned'
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, 'shihua_pairs_v1.jsonl')

QUOTE_PATTERNS = [r'「([^」]{4,200})」', r'『([^』]{4,200})』', r'“([^”]{4,200})”']


def split_entries(text):
    # 按空行/换行切，再合并极短行
    raw = [l.strip(' \u3000　') for l in text.splitlines()]
    entries = []
    buf = ''
    for line in raw:
        if not line:
            if buf:
                entries.append(buf)
                buf = ''
            continue
        if len(line) < 12 and buf and len(buf) < 400:
            buf += line
        else:
            if buf:
                entries.append(buf)
            buf = line
    if buf:
        entries.append(buf)
    return [e for e in entries if len(e) >= 20]


def extract_quotes(entry):
    found = []
    for pat in QUOTE_PATTERNS:
        for m in re.finditer(pat, entry):
            q = m.group(1).strip()
            if re.search(r'[\u4e00-\u9fff]{3,}', q):
                found.append((m.start(), m.end(), q))
    found.sort()
    # 去重叠
    picked = []
    for s, e, q in found:
        if picked and s < picked[-1][1]:
            continue
        picked.append((s, e, q))
    comment = entry
    for s, e, q in reversed(picked):
        comment = comment[:s] + comment[e:]
    return [q for _, _, q in picked], comment.strip()


def main():
    rows = []
    files = sorted(glob.glob(os.path.join(SRC, '*.txt')))
    for f in files:
        name = os.path.basename(f)[:-4]
        text = open(f, encoding='utf-8', errors='replace').read()
        for entry in split_entries(text):
            quotes, comment = extract_quotes(entry)
            if not quotes:
                continue
            if len(comment) < 8:
                continue
            # 只保留评论里像诗话的部分（含汉字，非纯目录）
            if not re.search(r'[\u4e00-\u9fff]{6,}', comment):
                continue
            rows.append({
                'source': name,
                'quotes': quotes,
                'comment': comment,
                'entry': entry,
            })
    # 去重
    seen = set()
    final = []
    for r in rows:
        k = (r['source'], r['entry'][:100])
        if k in seen:
            continue
        seen.add(k)
        final.append(r)
    with open(OUT, 'w', encoding='utf-8') as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('files:', len(files), '| pairs:', len(final))
    print('saved:', OUT)
    for r in final[:3]:
        print('---', r['source'])
        print('  引诗:', r['quotes'][0][:60])
        print('  评语:', r['comment'][:100].replace('\n', ' '))


if __name__ == '__main__':
    main()

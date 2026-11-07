"""构建 SFT v2 数据集：
1) 由真实诗构造「按标签拟作」样本（按作者切分防背诵）
2) 合并 teacher 数据（评点/鉴赏/散文仿写/逆向指令）
3) 加入真实诗话评点对（筛选诗样引文）
输出：sft_v2_train.jsonl / sft_v2_val.jsonl / _sft_v2_stats.json
"""
import glob
import hashlib
import json
import os
import random

CLEAN = r'D:\train\data\cleaned'
SFT1 = r'D:\train\data\sft'
OUT = r'D:\train\data\sft'
random.seed(2026)

TEMPLATES = [
    '请以【{tags}】的风格与体式，写一首题为《{title}》的{genre}。格律放宽，重在意境与语言。',
    '你是一位{dynasty}代诗人。请按【{tags}】的格调创作{genre}《{title}》，重意境，不拘平仄。',
    '以《{title}》为题，作{genre}一首；风格要求：{tags}。',
    '【{tags}】\n请作{genre}一首，题《{title}》。',
]
CONT_TEMPLATES = [
    '请续写下面的{genre}（保持同一风格与韵脚）：\n{first}',
    '补全这首诗的剩余部分（{dynasty}·{genre}，题目《{title}》）：\n{first}',
]


def load(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def author_val(author):
    h = int(hashlib.md5((author or '').encode('utf-8')).hexdigest()[:8], 16)
    return h % 100 < 2  # 2% 作者进验证集


def poem_style_tags(p):
    tags = [p['dynasty'], p['genre'], p['theme']]
    return '·'.join(x for x in tags if x)


def main():
    poems = load(os.path.join(CLEAN, 'poems_v1.jsonl'))
    poems = [p for p in poems if p.get('author') and p.get('title') and 20 <= len(p.get('text', '')) <= 800]
    print('poems usable:', len(poems))

    train_pool = [p for p in poems if not author_val(p['author'])]
    val_pool = [p for p in poems if author_val(p['author'])]
    print('train_pool:', len(train_pool), 'val_pool:', len(val_pool))

    def sample_balanced(pool, n_tang, n_song):
        tang = [p for p in pool if p['dynasty'] == '唐']
        song = [p for p in pool if p['dynasty'] == '宋']
        random.shuffle(tang)
        random.shuffle(song)
        return tang[:n_tang] + song[:n_song]

    # ---- 拟作：训练 15k（唐 6k / 宋 9k），验证 800 ----
    gen_train = sample_balanced(train_pool, 6000, 9000)
    gen_val = sample_balanced(val_pool, 300, 500)
    print('gen train/val:', len(gen_train), len(gen_val))

    def make_compose(p):
        tags = poem_style_tags(p)
        tmpl = random.choice(TEMPLATES)
        instr = tmpl.format(tags=tags, title=p['title'], genre=p['genre'], dynasty=p['dynasty'])
        return {'system': '你是精通中国古典诗文的写作助手。', 'instruction': instr,
                'input': '', 'output': p['text'], 'task': 'compose_poem',
                'meta': {'author': p['author'], 'dynasty': p['dynasty'], 'genre': p['genre'],
                         'theme': p['theme'], 'source': 'real_poem'}}

    def make_continue(p):
        lines = [x for x in p['text'].split('\n') if x.strip()]
        if len(lines) < 2:
            return None
        k = max(1, len(lines) // 2)
        first = '\n'.join(lines[:k])
        tmpl = random.choice(CONT_TEMPLATES)
        instr = tmpl.format(genre=p['genre'], dynasty=p['dynasty'], title=p['title'], first=first)
        return {'system': '你是精通中国古典诗文的写作助手。', 'instruction': instr,
                'input': '', 'output': p['text'], 'task': 'continue_poem',
                'meta': {'author': p['author'], 'dynasty': p['dynasty'], 'genre': p['genre'],
                         'source': 'real_poem'}}

    rows_train, rows_val = [], []
    for p in gen_train:
        rows_train.append(make_compose(p))
        if random.random() < 0.12:
            r = make_continue(p)
            if r:
                rows_train.append(r)
    for p in gen_val:
        rows_val.append(make_compose(p))
        r = make_continue(p)
        if r:
            rows_val.append(r)

    # ---- 真实诗话评点对（过滤诗样引文）----
    pairs = load(os.path.join(CLEAN, 'shihua_pairs_v1.jsonl'))
    random.shuffle(pairs)

    def poem_like(q):
        if not (6 <= len(q) <= 80):
            return False
        clauses = [c for c in q.replace('，', ',').replace('。', '.').split(',') if c]
        if not clauses:
            return False
        lengths = [len(c.strip()) for c in clauses if c.strip()]
        return bool(lengths) and sum(1 for L in lengths if L in (5, 7)) >= max(1, len(lengths) // 2)

    shihua_train, shihua_val = [], []
    for pr in pairs:
        q = pr['quotes'][0] if pr.get('quotes') else ''
        if not poem_like(q):
            continue
        rec = {'system': '你是精通中国古典诗论的诗话家。',
               'instruction': f'请以诗话笔法评点下面的诗句，150–250字：\n{q}',
               'input': '', 'output': pr['comment'], 'task': 'shihua_pingdian',
               'meta': {'source': 'shihua', 'book': pr['source']}}
        if len(rec['output']) < 20:
            continue
        if author_val(pr['source']):
            shihua_val.append(rec)
        else:
            shihua_train.append(rec)
        if len(shihua_train) >= 5000 and len(shihua_val) >= 500:
            break
    rows_train.extend(shihua_train)
    rows_val.extend(shihua_val)
    print('shihua train/val:', len(shihua_train), len(shihua_val))

    # ---- 合并 teacher 数据 ----
    teacher_train = load(os.path.join(SFT1, 'sft_v1_train.jsonl'))
    teacher_val = load(os.path.join(SFT1, 'sft_v1_val.jsonl'))
    rows_train.extend(teacher_train)
    rows_val.extend(teacher_val)

    # ---- 去重（instruction+output 前 100 字）----
    def dedup(rows):
        seen = set()
        out = []
        for r in rows:
            k = hashlib.md5((r['instruction'][:80] + r['output'][:120]).encode('utf-8')).hexdigest()
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        return out

    rows_train = dedup(rows_train)
    rows_val = dedup(rows_val)
    random.shuffle(rows_train)

    with open(os.path.join(OUT, 'sft_v2_train.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows_train:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(os.path.join(OUT, 'sft_v2_val.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows_val:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    from collections import Counter
    stats = {'train': len(rows_train), 'val': len(rows_val),
             'tasks': dict(Counter(r['task'] for r in rows_train)),
             'chars_train': sum(len(r['output']) for r in rows_train),
             'chars_val': sum(len(r['output']) for r in rows_val)}
    with open(os.path.join(OUT, '_sft_v2_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

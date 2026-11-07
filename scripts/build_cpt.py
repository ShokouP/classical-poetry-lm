"""构建 CPT（继续预训练）语料：诗词 + 明清散文 + 诗话/词话/文评 (+教师数据)。

格式（每行一个 JSON）：
  {"text": "【唐·李白·七绝·送别】\n<title>\n<poem text>", "source": "..."}

输出：D:\train\data\cpt\cpt_v1_train.jsonl / _val.jsonl
"""
import glob
import hashlib
import json
import os
import random
import re

CLEAN = r'D:\train\data\cleaned'
DAI = r'D:\poetry-corpus\daizhigev20'
OUT_DIR = r'D:\train\data\cpt'
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(1234)


def iter_poems():
    with open(os.path.join(CLEAN, 'poems_v1.jsonl'), encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            prefix = f'【{r["dynasty"]}·{r["author"] or "佚名"}·{r["genre"]}·{r["theme"]}】'
            text = f'{prefix}\n{r["title"]}\n{r["text"]}'
            if 20 <= len(text) <= 1500:
                yield {'text': text, 'source': 'poem'}


def iter_prose():
    with open(os.path.join(CLEAN, 'prose_mingqing_v1.jsonl'), encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            prefix = f'【{r["dynasty"]}·{r["author"] or "佚名"}·{r["genre"]}】'
            text = f'{prefix}\n{r["title"]}\n{r["text"]}'
            if 100 <= len(text) <= 3000:
                yield {'text': text, 'source': 'prose'}


def iter_shihua_dir(rel, label):
    base = os.path.join(DAI, rel)
    if not os.path.isdir(base):
        return
    for p in sorted(glob.glob(os.path.join(base, '*.txt'))):
        name = os.path.basename(p)[:-4]
        text = open(p, encoding='utf-8', errors='replace').read()
        # 按行切成条目，合并短行
        buf = []
        for ln in text.splitlines():
            s = ln.strip(' \u3000　')
            if not s:
                if buf:
                    entry = ''.join(buf)
                    if 20 <= len(entry) <= 1500:
                        yield {'text': f'【{label}·{name}】\n{entry}', 'source': label}
                    buf = []
                continue
            buf.append(s)
        if buf:
            entry = ''.join(buf)
            if 20 <= len(entry) <= 1500:
                yield {'text': f'【{label}·{name}】\n{entry}', 'source': label}


SIKU = os.path.join(DAI, '集藏', '四库别集')


def iter_siku(limit_chars=20_000_000):
    """四库别集：整本按 1200 字左右切块，带朝代/书名前缀；总量截断。"""
    if not os.path.isdir(SIKU):
        return
    files = sorted(glob.glob(os.path.join(SIKU, '*.txt')), key=lambda p: os.path.getsize(p), reverse=True)
    random.shuffle(files)
    total = 0
    name_re = re.compile(r'别集类[一二三四五六七八九十百]+【([宋元明清])】')
    for p in files:
        if total >= limit_chars:
            break
        try:
            text = open(p, encoding='utf-8', errors='replace').read()
        except Exception:
            continue
        if len(text) < 2000:
            continue
        m = name_re.search(text[:400])
        dynasty = m.group(1) if m else '历代'
        book = os.path.basename(p)[:-4]
        # 去掉提要/目录，尽量从正文开始
        pos = text.find('巻一')
        if pos < 0:
            pos = text.find('卷一')
        if pos < 0:
            pos = min(len(text) // 5, 4000)
        body = text[pos:]
        # 按行拼块
        chunks = []
        buf = ''
        for ln in body.splitlines():
            s = ln.strip(' \u3000　')
            if not s:
                continue
            buf += s
            if len(buf) >= 1200:
                chunks.append(buf)
                buf = ''
        if buf and len(buf) >= 200:
            chunks.append(buf)
        for c in chunks:
            if 200 <= len(c) <= 2000:
                yield {'text': f'【四库别集·{dynasty}·{book}】\n{c}', 'source': 'siku'}
                total += len(c)
                if total >= limit_chars:
                    return


def main():
    sources = [
        ('poems', list(iter_poems())),
        ('prose', list(iter_prose())),
        ('shihua', list(iter_shihua_dir(r'诗藏\诗话', '诗话'))),
        ('cihua', list(iter_shihua_dir(r'诗藏\词话', '词话'))),
        ('wenping', list(iter_shihua_dir(r'集藏\文评', '文评'))),
        ('siku', list(iter_siku())),
    ]
    for name, rows in sources:
        chars = sum(len(r['text']) for r in rows)
        print(f'{name}: {len(rows)} 条, {chars/1e4:.1f} 万字')
    all_rows = [r for _, rows in sources for r in rows]
    print('total:', len(all_rows), '条,', round(sum(len(r['text']) for r in all_rows)/1e4, 1), '万字')

    random.shuffle(all_rows)
    train, val = [], []
    for r in all_rows:
        h = int(hashlib.md5(r['text'][:200].encode('utf-8')).hexdigest()[:8], 16)
        (val if h % 100 == 0 else train).append(r)

    with open(os.path.join(OUT_DIR, 'cpt_v1_train.jsonl'), 'w', encoding='utf-8') as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(os.path.join(OUT_DIR, 'cpt_v1_val.jsonl'), 'w', encoding='utf-8') as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('train:', len(train), '| val:', len(val), '| dir:', OUT_DIR)


if __name__ == '__main__':
    main()

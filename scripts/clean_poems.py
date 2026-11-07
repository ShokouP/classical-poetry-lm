"""清洗全唐诗/全宋诗 JSON → 带标签的 JSONL。

标签：dynasty / author / title / genre（绝句/律诗/古体/乐府…按句式规则）
      theme（按题目关键词粗分）
"""
import glob
import json
import os
import re

CP = r'D:\poetry-corpus\chinese-poetry\全唐诗'
OUT_DIR = r'D:\train\data\cleaned'
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, 'poems_v1.jsonl')

THEMES = [
    ('送别', ['送', '别', '赠别', '饯', '留别']),
    ('怀古', ['怀古', '咏史', '金陵', '赤壁', '登高', '古迹', '故都']),
    ('边塞', ['塞', '边', '从军', '出塞', '关山', '烽', '戍']),
    ('咏物', ['咏', '赋得', '题', '画', '梅', '竹', '菊', '莲', '柳', '雪', '月']),
    ('思乡', ['思乡', '乡思', '望乡', '故园', '归乡']),
    ('闺怨', ['闺', '怨', '宫词', '宫怨', '妾', '美人']),
    ('山水田园', ['山', '水', '田园', '村', '溪', '湖', '江', '春', '秋']),
    ('酬赠', ['酬', '答', '寄', '呈', '和', '赠']),
    ('悼亡', ['悼', '挽', '哭', '哀']),
    ('羁旅', ['旅', '客', '行', '夜泊', '舟中', '途中']),
]


def theme_of(title):
    for name, kws in THEMES:
        if any(k in title for k in kws):
            return name
    return '其他'


def genre_of(paras):
    """按句长/句数粗判体裁：先把段落拆成句（，。；！？），再统计句长。"""
    from collections import Counter
    lines = [p for p in paras if p.strip()]
    if not lines:
        return '未知'
    clauses = []
    for p in lines:
        for c in re.split(r'[，。；！？、：]', p):
            c = re.sub(r'[\s\u3000]', '', c)
            if c:
                clauses.append(len(c))
    if not clauses:
        return '未知'
    common_len = Counter(clauses).most_common(1)[0][0]
    n = len(lines)
    if common_len == 5:
        if n == 4:
            return '五绝'
        if n == 8:
            return '五律'
        return '五言古/排律'
    if common_len == 7:
        if n == 4:
            return '七绝'
        if n == 8:
            return '七律'
        return '七言古/排律'
    if common_len in (4, 6):
        return '四六言'
    if common_len == 3:
        return '三言'
    return '杂言古体'


def process(kind):
    files = sorted(glob.glob(os.path.join(CP, f'poet.{kind}.*.json')))
    out = []
    for f in files:
        try:
            data = json.load(open(f, encoding='utf-8'))
        except Exception:
            continue
        for item in data:
            paras = item.get('paragraphs') or []
            title = (item.get('title') or '').strip()
            author = (item.get('author') or '').strip()
            text = '\n'.join(p.strip() for p in paras if p.strip())
            if not text or len(text) < 8 or len(text) > 3000:
                continue
            out.append({
                'dynasty': '唐' if kind == 'tang' else '宋',
                'author': author,
                'title': title,
                'genre': genre_of(paras),
                'theme': theme_of(title),
                'text': text,
                'source': f'chinese-poetry/poet.{kind}',
            })
    return out


def main():
    rows = process('tang') + process('song')
    seen = set()
    final = []
    for r in rows:
        key = (r['text'][:80],)
        if key in seen:
            continue
        seen.add(key)
        final.append(r)
    with open(OUT, 'w', encoding='utf-8') as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    from collections import Counter
    print('total:', len(final))
    print('dynasty:', dict(Counter(r['dynasty'] for r in final)))
    print('genre:', dict(Counter(r['genre'] for r in final).most_common(12)))
    print('theme:', dict(Counter(r['theme'] for r in final).most_common(12)))
    print('saved:', OUT)
    for r in final[:2]:
        print('---', r['dynasty'], r['author'], '《' + r['title'] + '》', r['genre'], r['theme'])
        print(r['text'][:80].replace('\n', ' / '))


if __name__ == '__main__':
    main()

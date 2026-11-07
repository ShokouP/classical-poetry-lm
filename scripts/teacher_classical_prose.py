"""DeepSeek 生成古典散文数据（第 1 卷）：仿写/扩写/诗改文/文论评点。

输出：D:\train\data\sft\teacher_classical_prose\classical_prose_v1.jsonl
"""
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, r'D:\train\scripts')
from lib_teacher import Teacher  # noqa: E402

random.seed(2028)

STYLES = ['韩愈', '柳宗元', '欧阳修', '苏轼', '王安石', '归有光', '袁宏道', '张岱',
          '桐城派（方苞/姚鼐）', '汪中', '龚自珍']
GENRES = ['记', '序', '论', '传', '说', '赋', '铭', '书']
THEMES = ['山居', '书斋', '秋夜', '江行', '送别', '读书', '旧友', '园林', '听琴',
          '市井', '旅宿', '怀人', '论诗', '论文', '论学', '记梦', '观画', '种树',
          '听雨', '访友', '舟中', '病中', '岁暮', '早春', '村居', '登高']

PROSE_INSTR = [
    '请以{style}的笔法，写一篇题为《{title}》的{genre}，题材：{theme}，约 {n} 字古文（文言）。要求：文气连贯、有细节，不要用现代白话，不要抄写名篇原句。',
    '请写一篇约 {n} 字的古文{genre}，风格近{style}，写{theme}。要求：有叙事或写景，有议论收束，语言清雅，不得整句化用古人名篇。',
]
EXPAND_INSTR = [
    '下面是一段未加标点的古文（{dynasty}·{author}·《{title}》）。请先加标点断句并稍作疏通，再扩写到约 {n} 字：保持文言与原意，可增写景物、细节、议论，不要改写原意。\n\n原文：\n{src}',
]
VERSE_INSTR = [
    '下面是一首{genre}（{dynasty}·{author}·《{title}》）。请把它改写成一篇约 {n} 字的古文小品（记或赋），保持原诗意境与意象，不要逐句翻译、不要照抄原句。\n\n原诗：\n{src}',
]
CRITIQUE_INSTR = [
    '下面是一篇古文（{dynasty}·{author}·《{title}》）。请以明清文论家的口吻，写一篇约 {n} 字的评点：评其章法、句法、气韵与得失，文风近书序或文话，不要写成现代赏析。\n\n原文：\n{src}',
]


def load_jsonl(path):
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    prose = load_jsonl(r'D:\train\data\cleaned\prose_mingqing_v1.jsonl')
    poems = load_jsonl(r'D:\train\data\cleaned\poems_v1.jsonl')
    prose = [p for p in prose if 120 <= len(p.get('text', '')) <= 320]
    poems = [p for p in poems if p.get('genre') in ('七绝', '五绝', '七律', '五律') and 24 <= len(p.get('text', '')) <= 120]
    random.shuffle(prose)
    random.shuffle(poems)
    print('prose pool:', len(prose), 'poem pool:', len(poems), flush=True)

    jobs = []
    for i in range(args.n):
        k = i % 10
        if k < 4:
            style = random.choice(STYLES)
            genre = random.choice(GENRES)
            theme = random.choice(THEMES)
            n = random.choice([350, 400, 450, 500, 550])
            instr = random.choice(PROSE_INSTR).format(style=style, title=theme + '记', genre=genre, theme=theme, n=n)
            jobs.append({'task': 'classical_prose', 'instruction': instr, 'target': n})
        elif k < 7:
            p = prose[i % len(prose)]
            n = random.choice([450, 550, 650])
            jobs.append({'task': 'expand_classical', 'target': n,
                         'instruction': EXPAND_INSTR[0].format(dynasty=p['dynasty'], author=p['author'],
                                                               title=p['title'], n=n, src=p['text'])})
        elif k < 9:
            p = poems[i % len(poems)]
            n = random.choice([350, 450, 550])
            jobs.append({'task': 'verse_to_prose', 'target': n,
                         'instruction': VERSE_INSTR[0].format(genre=p['genre'], dynasty=p['dynasty'],
                                                              author=p['author'], title=p['title'], n=n, src=p['text'])})
        else:
            p = prose[i % len(prose)]
            n = random.choice([250, 300, 350])
            jobs.append({'task': 'classical_critique', 'target': n,
                         'instruction': CRITIQUE_INSTR[0].format(dynasty=p['dynasty'], author=p['author'],
                                                                 title=p['title'], n=n, src=p['text'])})

    teacher = Teacher()
    rows = []
    fail = 0
    t0 = time.time()

    def run(job):
        out = teacher.generate('你是精通中国古典散文的写作与文论助手。只输出正文，不要解释或加说明。',
                               job['instruction'], temperature=0.85, max_tokens=2400,
                               effort='low', thinking='disabled')
        return job, out['text'].strip()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for fut in as_completed(futs):
            try:
                job, text = fut.result()
            except Exception as e:
                fail += 1
                print('FAIL', str(e)[:120], flush=True)
                continue
            if len(text) < 150:
                fail += 1
                continue
            rows.append({'system': '你是精通中国古典散文的写作助手。', 'instruction': job['instruction'],
                         'input': '', 'output': text, 'task': job['task'],
                         'meta': {'target': job['target'], 'chars': len(text)}})
            if len(rows) % 100 == 0:
                print(f'done={len(rows)} fail={fail} elapsed={time.time()-t0:.0f}s', flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(json.dumps({'ok': len(rows), 'fail': fail, 'usage': teacher.usage,
                      'elapsed_s': round(time.time() - t0)}, ensure_ascii=False), flush=True)
    print('saved:', args.out, flush=True)
    from collections import Counter
    print(Counter(r['task'] for r in rows), flush=True)


if __name__ == '__main__':
    main()

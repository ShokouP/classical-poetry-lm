"""Teacher 扩写数据生成：诗→小品文 / 古文疏通+扩写，输出 alpaca 格式。

用法：python teacher_expand.py --n 100 --out D:\train\data\sft\teacher_expand\expand_v1.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, r'D:\train\scripts')
from lib_teacher import Teacher  # noqa: E402

CLEAN = r'D:\train\data\cleaned'
random.seed(7)

POEM_TMPL = [
    ('请以这首{genre}《{title}》（{dynasty}·{author}）的意境与意象为底本，扩写成一篇约 {n} 字的文言小品文：'
     '保持原有的清冷/开阔气韵，增写景物、感官细节与情思，但不要逐句翻译，也不要照抄原句。\n\n原诗：\n{text}'),
    ('下面是一首{dynasty}代{genre}。请把它改写成一篇约 {n} 字的古文短章（记/赋/序皆可），要求文言、意境相承、'
     '语言不重复原诗。\n\n《{title}》  {author}\n{text}'),
]
PROSE_TMPL = [
    ('下面是一段未加标点的古文（{dynasty}·{author}·《{title}》）。请先为它加上现代标点并稍作疏通，'
     '再扩写到约 {n} 字：保持文言风格与原意，可增写背景、景物、议论，但不要改写原意。\n\n原文：\n{text}'),
    ('请把下面这段古文（{dynasty}·{author}·《{title}》）扩写成约 {n} 字的完整短文：先加标点断句，再展开细节与情思，'
     '文风贴近原文，不要逐句复述。\n\n原文：\n{text}'),
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


def zh_len(s):
    return len(re.findall(r'[\u4e00-\u9fff]', s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--target', type=int, default=600)
    args = ap.parse_args()

    poems = load(os.path.join(CLEAN, 'poems_v1.jsonl'))
    prose = load(os.path.join(CLEAN, 'prose_mingqing_v1.jsonl'))
    poems = [p for p in poems if p['genre'] in ('七绝', '五绝', '七律', '五律') and 24 <= zh_len(p['text']) <= 120]
    prose = [p for p in prose if 120 <= zh_len(p['text']) <= 400]
    random.shuffle(poems)
    random.shuffle(prose)
    print('poems pool:', len(poems), 'prose pool:', len(prose), flush=True)

    jobs = []
    for i in range(args.n):
        if i % 2 == 0:
            p = poems[i % len(poems)]
            n = random.choice([450, 550, 650, 700])
            tmpl = random.choice(POEM_TMPL)
            instr = tmpl.format(genre=p['genre'], title=p['title'], dynasty=p['dynasty'],
                                author=p['author'], n=n, text=p['text'])
            jobs.append({'kind': 'expand_poem', 'instruction': instr,
                         'target': n, 'src': p['text'], 'meta': p})
        else:
            p = prose[i % len(prose)]
            n = random.choice([550, 650, 750, 800])
            tmpl = random.choice(PROSE_TMPL)
            instr = tmpl.format(dynasty=p['dynasty'], author=p['author'], title=p['title'],
                                n=n, text=p['text'])
            jobs.append({'kind': 'expand_prose', 'instruction': instr,
                         'target': n, 'src': p['text'], 'meta': p})

    teacher = Teacher()
    done = 0
    fail = 0
    rows = []
    t0 = time.time()

    def run(job):
        instr = job['instruction'] + f'\n\n要求：全文不少于 {job["target"]} 个汉字，只输出正文，不要解释或加说明。'
        out = teacher.generate(
            '你是精通中国古典诗文的写作与训诂助手。输出只写正文，不要解释，不要加标题以外的说明。',
            instr, temperature=0.85, max_tokens=3000, effort='low', thinking='disabled')
        return job, out['text'].strip()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for fut in as_completed(futs):
            try:
                job, text = fut.result()
            except Exception as e:
                fail += 1
                print('FAIL', e, flush=True)
                continue
            L = zh_len(text)
            if L < job['target'] * 0.65 or L < 200:
                fail += 1
                continue
            rows.append({
                'system': '你是精通中国古典诗文的写作助手。',
                'instruction': job['instruction'],
                'input': '',
                'output': text,
                'task': job['kind'],
                'meta': {'dynasty': job['meta']['dynasty'], 'author': job['meta']['author'],
                         'title': job['meta']['title'], 'target': job['target'], 'chars': L},
            })
            done += 1
            if done % 10 == 0:
                print(f'done={done} fail={fail} elapsed={time.time()-t0:.0f}s', flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    usage = teacher.usage
    print(json.dumps({'ok': len(rows), 'fail': fail, 'usage': usage,
                      'elapsed_s': round(time.time() - t0, 1)}, ensure_ascii=False), flush=True)
    print('saved:', args.out, flush=True)


if __name__ == '__main__':
    main()

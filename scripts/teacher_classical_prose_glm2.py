"""GLM-5.3 生成古典散文（增量落盘 + 断点续跑）。"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, r'D:\train\scripts')
from lib_teacher_glm import GLMTeacher  # noqa: E402

random.seed(2032)

STYLE_CYCLE = [
    '韩愈', '柳宗元', '欧阳修', '苏轼', '王安石', '归有光',
    '袁宏道', '张岱', '方苞（桐城派）', '姚鼐（桐城派）', '汪中', '龚自珍',
    '刘基', '宋濂', '唐顺之', '茅坤', '侯方域', '魏禧',
]
GENRES = ['记', '序', '论', '传', '说', '赋', '铭', '书', '跋', '赞']
THEMES = ['山居', '书斋', '秋夜', '江行', '送别', '读书', '旧友', '园林', '听琴',
          '市井', '旅宿', '怀人', '论诗', '论文', '论学', '记梦', '观画', '种树',
          '听雨', '访友', '舟中', '病中', '岁暮', '早春', '村居', '登高', '观潮',
          '田家', '煮茶', '独坐', '秋猎', '修桥', '赈灾', '师说']
PROSE_INSTR = [
    '请以{style}的笔法，写一篇题为《{title}》的{genre}，题材：{theme}，约 {n} 字古文（文言）。要求：文气连贯、有细节，不要用现代白话，不要抄写名篇原句，也不要模仿"项脊轩"式的悼亡套路。',
    '请写一篇约 {n} 字的古文{genre}，风格近{style}，写{theme}。要求：有叙事或写景，有议论收束，语言清雅，不得整句化用古人名篇。',
    '请拟作一篇约 {n} 字的文言{genre}，题为《{title}》，风格取法{style}：写{theme}，要有眼前景、心中事，结尾含蓄，避免套语。',
]


def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def key_of(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=800)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()

    prose = load_jsonl(r'D:\train\data\cleaned\prose_mingqing_v1.jsonl')
    poems = load_jsonl(r'D:\train\data\cleaned\poems_v1.jsonl')
    prose = [p for p in prose if 120 <= len(p.get('text', '')) <= 320]
    poems = [p for p in poems if p.get('genre') in ('七绝', '五绝', '七律', '五律') and 24 <= len(p.get('text', '')) <= 120]
    random.shuffle(prose)
    random.shuffle(poems)

    jobs = []
    for i in range(args.n):
        k = i % 10
        style = STYLE_CYCLE[i % len(STYLE_CYCLE)]
        theme = random.choice(THEMES)
        if k < 7:
            genre = random.choice(GENRES)
            n = random.choice([350, 400, 450, 500, 550])
            instr = random.choice(PROSE_INSTR).format(style=style, title=theme + '记', genre=genre, theme=theme, n=n)
            jobs.append({'task': 'classical_prose_glm', 'instruction': instr, 'target': n, 'style': style})
        elif k < 9:
            p = poems[i % len(poems)]
            n = random.choice([350, 450, 550])
            instr = ('下面是一首{genre}（{dynasty}·{author}·《{title}》）。请把它改写成一篇约 {n} 字的古文小品'
                     '（记或赋），保持原诗意境与意象，不要逐句翻译、不要照抄原句；文风近{style}。\n\n原诗：\n{src}').format(
                genre=p['genre'], dynasty=p['dynasty'], author=p['author'], title=p['title'],
                n=n, style=style, src=p['text'])
            jobs.append({'task': 'verse_to_prose_glm', 'instruction': instr, 'target': n, 'style': style})
        else:
            p = prose[i % len(prose)]
            n = random.choice([250, 300, 350])
            instr = ('下面是一篇古文（{dynasty}·{author}·《{title}》）。请以明清文论家的口吻，写一篇约 {n} 字的评点：'
                     '评其章法、句法、气韵与得失，文风近书序或文话，不要写成现代赏析。\n\n原文：\n{src}').format(
                dynasty=p['dynasty'], author=p['author'], title=p['title'], n=n, src=p['text'])
            jobs.append({'task': 'classical_critique_glm', 'instruction': instr, 'target': n, 'style': style})

    existing = {key_of(r['instruction']) for r in load_jsonl(args.out)}
    jobs = [j for j in jobs if key_of(j['instruction']) not in existing]
    print(f'pending jobs: {len(jobs)} (existing {len(existing)})', flush=True)

    teacher = GLMTeacher('glm-5.3')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fout = open(args.out, 'a', encoding='utf-8')
    fail = 0
    done = 0
    t0 = time.time()

    def run(job):
        text = teacher.generate('你是精通中国古典散文的写作与文论助手。只输出正文，不要解释或加说明。',
                                job['instruction'], temperature=0.85, max_tokens=2600,
                                thinking='disabled')
        return job, text.strip()

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
            row = {'system': '你是精通中国古典散文的写作助手。', 'instruction': job['instruction'],
                   'input': '', 'output': text, 'task': job['task'],
                   'meta': {'target': job['target'], 'chars': len(text), 'style': job['style'],
                            'model': 'glm-5.3'}}
            fout.write(json.dumps(row, ensure_ascii=False) + '\n')
            fout.flush()
            done += 1
            if done % 50 == 0:
                print(f'done={done} fail={fail} elapsed={time.time()-t0:.0f}s', flush=True)

    fout.close()
    print(json.dumps({'ok': done, 'fail': fail, 'usage': teacher.usage,
                      'elapsed_s': round(time.time() - t0)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

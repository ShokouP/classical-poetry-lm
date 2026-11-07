"""多 teacher DPO 正例生成（GLM-5.3 / K3 / K3.1-Pro / MiniMax-M3 / DeepSeek-V4.1-Flash / M3.1-Flash）。

每家生成古典散文+近体诗；增量落盘 + 断点续跑。
输出：D:\train\data\dpo\dpo_pos_<slug>.jsonl
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

CRED = os.path.expanduser('~/.dsh/.credentials.yaml')
REF = yaml.safe_load(open(CRED, encoding='utf-8'))['refs']

PROVIDERS = {
    'glm-5.3': {
        'url': 'https://api.z.ai/api/coding/paas/v4/chat/completions',
        'key': 'ZAI_CODING_CN_API_KEY', 'model': 'glm-5.3',
        'extra': {'thinking': {'type': 'disabled'}}, 'workers': 3,
    },
    'k3': {
        'url': 'https://api.kimi.com/coding/v1/chat/completions',
        'key': 'KIMI_API_KEY', 'model': 'k3',
        'extra': {'thinking': {'type': 'disabled'}}, 'workers': 4,
    },
    'k3.1-pro': {
        'url': 'https://api.kimi.com/coding/v1/chat/completions',
        'key': 'KIMI_API_KEY', 'model': 'k3.1-pro',
        'extra': {'reasoning_effort': 'low'}, 'workers': 2,
    },
    'minimax-m3': {
        'url': 'https://api.minimax.cn/v1/chat/completions',
        'key': 'MINIMAX1_API_KEY', 'model': 'MiniMax-M3',
        'extra': {'thinking': {'type': 'disabled'}}, 'workers': 3,
    },
    'minimax-m3.1-flash': {
        'url': 'https://api.minimax.cn/v1/chat/completions',
        'key': 'MINIMAX1_API_KEY', 'model': 'MiniMax-M3.1-Flash',
        'extra': {'thinking': {'type': 'adaptive'}, 'reasoning': {'effort': 'low'}}, 'workers': 2,
        'max_tokens': 6000,
    },
    'deepseek-v4.1-flash': {
        'url': 'https://api.deepseek.com/chat/completions',
        'key': 'DEEPSEEK_API_KEY', 'model': 'deepseek-flash',
        'extra': {'thinking': {'type': 'disabled'}}, 'workers': 4,
    },
}

STYLES = ['韩愈', '柳宗元', '欧阳修', '苏轼', '王安石', '归有光', '袁宏道', '张岱',
          '方苞（桐城派）', '姚鼐（桐城派）', '刘基', '宋濂']
GENRES = ['记', '序', '论', '传', '说', '赋', '铭', '书']
THEMES = ['山居', '书斋', '秋夜', '江行', '送别', '读书', '旧友', '园林', '听琴',
          '旅宿', '怀人', '论诗', '论文', '记梦', '观画', '听雨', '访友', '舟中',
          '病中', '岁暮', '早春', '村居', '登高', '煮茶', '独坐']
POEM_GENRES = [('五绝', 4, 5), ('七绝', 4, 7), ('五律', 8, 5), ('七律', 8, 7)]
POETS = ['王维', '杜甫', '李白', '李商隐', '苏轼', '陆游', '王昌龄', '孟浩然', '杜牧']


def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def key_of(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def build_jobs(n_prose, n_poem, seed):
    rnd = random.Random(seed)
    jobs = []
    for i in range(n_prose):
        style = rnd.choice(STYLES)
        theme = rnd.choice(THEMES)
        genre = rnd.choice(GENRES)
        n = rnd.choice([350, 400, 450, 500])
        instr = ('请以{style}的笔法，写一篇题为《{title}》的{genre}，题材：{theme}，约 {n} 字古文（文言）。'
                 '要求：文气连贯、有细节，不要用现代白话，不要抄写名篇原句。').format(
            style=style, title=theme + '记', genre=genre, theme=theme, n=n)
        jobs.append({'task': 'classical_prose', 'kind': 'prose', 'instruction': instr,
                     'style': style, 'genre': genre, 'theme': theme})
    for i in range(n_poem):
        pg, lines, chars = rnd.choice(POEM_GENRES)
        poet = rnd.choice(POETS)
        theme = rnd.choice(THEMES)
        instr = '【{dyn}·{poet}·{pg}·{theme}】请作{pg}一首，写{theme}。'.format(
            dyn=rnd.choice(['唐', '宋']), poet=poet, pg=pg, theme=theme)
        jobs.append({'task': 'compose_poem', 'kind': 'poem', 'instruction': instr,
                     'lines': lines, 'chars': chars, 'style': poet, 'genre': pg, 'theme': theme})
    rnd.shuffle(jobs)
    return jobs


def post(cfg, messages, max_tokens):
    body = {'model': cfg['model'], 'messages': messages,
            'max_tokens': max_tokens, 'stream': False}
    body.update(cfg['extra'])
    key = REF.get(cfg['key'])
    req = urllib.request.Request(cfg['url'], data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
                                 headers={'Authorization': f'Bearer {key}',
                                          'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=600) as r:
        obj = json.loads(r.read().decode())
    msg = obj['choices'][0]['message']
    return (msg.get('content') or '').strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--provider', required=True, choices=list(PROVIDERS))
    ap.add_argument('--n-prose', type=int, default=150)
    ap.add_argument('--n-poem', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = PROVIDERS[args.provider]
    jobs = build_jobs(args.n_prose, args.n_poem, args.seed)
    existing = {key_of(r['prompt']) for r in load_jsonl(args.out)}
    jobs = [j for j in jobs if key_of(j['instruction']) not in existing]
    print(f'[{args.provider}] pending={len(jobs)} existing={len(existing)}', flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fout = open(args.out, 'a', encoding='utf-8')
    done = fail = 0
    t0 = time.time()
    max_tokens = cfg.get('max_tokens', 3000)
    sysmsg = '你是精通中国古典诗文的写作助手。只输出正文，不要解释或加说明。'

    def run(job):
        msgs = [{'role': 'system', 'content': sysmsg},
                {'role': 'user', 'content': job['instruction']}]
        return job, post(cfg, msgs, max_tokens)

    with ThreadPoolExecutor(max_workers=cfg['workers']) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for fut in as_completed(futs):
            try:
                job, text = fut.result()
            except Exception as e:
                fail += 1
                print(f'[{args.provider}] FAIL {type(e).__name__} {str(e)[:100]}', flush=True)
                continue
            min_len = 40 if job['kind'] == 'poem' else 150
            if len(text) < min_len:
                fail += 1
                continue
            row = {'prompt': job['instruction'], 'chosen': text, 'task': job['task'],
                   'kind': job['kind'], 'model': args.provider,
                   'meta': {'style': job.get('style'), 'genre': job.get('genre'),
                            'theme': job.get('theme'), 'chars': len(text)}}
            fout.write(json.dumps(row, ensure_ascii=False) + '\n')
            fout.flush()
            done += 1
            if done % 25 == 0:
                print(f'[{args.provider}] done={done} fail={fail} elapsed={time.time()-t0:.0f}s', flush=True)
    fout.close()
    print(f'[{args.provider}] FINISHED ok={done} fail={fail} elapsed={time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()

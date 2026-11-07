"""教师批量数据生成（DeepSeek V4.1-Flash）。

任务类型：
  pingdian      真实诗 → 诗话体评点（带真实诗话风格示例）
  jianshang     真实诗 → 注释+体裁+赏析
  instruct_poem 真实诗 → 逆向创作指令(JSON)
  prose_imitate 明清散文 → 仿写一篇古文
  instruct_prose 明清散文 → 逆向创作指令(JSON)

特性：并发、断点续跑（按任务输出文件行数）、token/费用统计、失败重试、JSON 任务校验。
用法：
  python teacher_batch.py --per-task pingdian=3,jianshang=3,instruct_poem=3,prose_imitate=3,instruct_prose=3
  python teacher_batch.py --total 10000 --concurrency 16
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_teacher import Teacher  # noqa: E402

CLEAN = r'D:\train\data\cleaned'
OUT_DIR = r'D:\train\data\sft\teacher_v1'
STYLE_DIR = r'D:\poetry-corpus\daizhigev20\诗藏\诗话'
os.makedirs(OUT_DIR, exist_ok=True)
PROGRESS = os.path.join(OUT_DIR, '_progress.json')
COSTLOG = os.path.join(OUT_DIR, '_cost.json')

TASKS = ['pingdian', 'jianshang', 'instruct_poem', 'prose_imitate', 'instruct_prose']

# 官方价格（DeepSeek V4.1-Flash）：元 / 百万 tokens；peak=工作日 9-12,14-18
PRICE = {
    'peak': {'in_miss': 2.0, 'in_hit': 0.04, 'out': 8.0},
    'idle': {'in_miss': 1.0, 'in_hit': 0.02, 'out': 4.0},
}


def is_peak():
    import datetime
    now = datetime.datetime.now()
    if now.weekday() >= 5:
        return False
    hm = now.hour * 60 + now.minute
    return (9 * 60 <= hm < 12 * 60) or (14 * 60 <= hm < 18 * 60)


def load_jsonl(path):
    if not os.path.exists(path):
        return []
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


def sample_poems(n):
    rows = load_jsonl(os.path.join(CLEAN, 'poems_v1.jsonl'))
    # 尽量均衡唐/宋、作者
    tang = [r for r in rows if r['dynasty'] == '唐']
    song = [r for r in rows if r['dynasty'] == '宋']
    out = []
    for _ in range(n):
        pool = tang if random.random() < 0.45 else song
        out.append(random.choice(pool))
    return out


def sample_prose(n):
    rows = load_jsonl(os.path.join(CLEAN, 'prose_mingqing_v1.jsonl'))
    return [random.choice(rows) for _ in range(n)] if rows else []


def load_styles():
    styles = []
    for name in ['沧浪诗话', '随园诗话', '六一诗话', '后山诗话', '石林诗话', '瓯北诗话', '历代诗话']:
        p = os.path.join(STYLE_DIR, name + '.txt')
        if os.path.exists(p):
            t = open(p, encoding='utf-8', errors='replace').read()
            start = min(len(t) // 8, 1500)
            styles.append((name, t[start:start + 280].replace('\n', ' ')))
    return styles


# ---------------- prompt 构造 ----------------
def make_pingdian(poem, styles):
    st = random.sample(styles, min(2, len(styles)))
    style_txt = '\n'.join(f'【{n}】{s}' for n, s in st)
    system = (
        '你是精通中国古典诗论的诗话家。请模仿下面真实诗话的风格与口吻评诗：'
        '文言为主、点到即止、重渊源与字法，避免现代学术腔与空洞形容词。\n' + style_txt
    )
    user = (f'请评点这首诗（{poem["dynasty"]}·{poem["author"]}《{poem["title"]}》）：\n{poem["text"]}\n\n'
            '要求：150–260 字，可分 2–3 条；点出起结、字眼、意境或渊源，必要时指出瑕疵。')
    return system, user, 2600, 'enabled'


def make_jianshang(poem):
    system = '你是古籍整理与古典诗词鉴赏专家。'
    user = (f'请对这首诗做注释与赏析（{poem["dynasty"]}·{poem["author"]}《{poem["title"]}》）：\n{poem["text"]}\n\n'
            '要求：1) 注释疑难字词与典故；2) 说明体裁句式（宽格律即可）；3) 150–250 字意境赏析。')
    return system, user, 3000, 'enabled'


def make_instruct_poem(poem):
    system = '你是古诗创作任务设计者。只输出 JSON，不要代码块。'
    user = (f'根据这首真实的{poem["dynasty"]}诗反推一条可用于训练 AI 写诗的创作指令。\n'
            f'诗：{poem["text"]}\n作者：{poem["author"]}，题目：{poem["title"]}\n\n'
            'JSON 字段：{"朝代","作者风格","体裁","题材","风格词","写作要求"(3条数组)}')
    return system, user, 1200, 'disabled'


def make_prose_imitate(prose):
    system = '你是明清古文家，擅长碑传、记序、书说等文体。请用纯正文言写作，不用白话、不加解释。'
    user = (f'请以「{prose["dynasty"]}代文风」写一段古文，文体近于{prose["genre"]}，题目自拟（约 8–20 字），'
            f'正文 300–450 字，避免与参考文字重复。\n参考风格（勿抄袭）：\n{prose["text"][:400]}')
    return system, user, 3000, 'enabled'


def make_instruct_prose(prose):
    system = '你是古文创作任务设计者。只输出 JSON，不要代码块。'
    user = (f'根据这篇真实的{prose["dynasty"]}代古文（{prose["author"]}《{prose["title"]}》）'
            f'反推一条可用于训练 AI 写古文的创作指令。\n原文节选：\n{prose["text"][:500]}\n\n'
            'JSON 字段：{"朝代","文体","题","风格词","写作要求"(3条数组)}')
    return system, user, 1200, 'disabled'


def validate(task, text):
    text = (text or '').strip()
    if len(text) < 60:
        return False, 'too_short'
    if task.startswith('instruct'):
        try:
            s = text
            m = re.search(r'\{[\s\S]*\}', s)
            if not m:
                return False, 'no_json'
            json.loads(m.group(0))
            return True, ''
        except Exception:
            return False, 'bad_json'
    return True, ''


# ---------------- 主流程 ----------------
def build_jobs(task, n, styles):
    jobs = []
    if task in ('pingdian', 'jianshang', 'instruct_poem'):
        poems = sample_poems(n)
        for p in poems:
            if task == 'pingdian':
                s, u, mt, th = make_pingdian(p, styles)
            elif task == 'jianshang':
                s, u, mt, th = make_jianshang(p)
            else:
                s, u, mt, th = make_instruct_poem(p)
            jobs.append({'task': task, 'seed': p, 'system': s, 'user': u, 'max_tokens': mt, 'thinking': th})
    elif task in ('prose_imitate', 'instruct_prose'):
        prose = sample_prose(n)
        for p in prose:
            if task == 'prose_imitate':
                s, u, mt, th = make_prose_imitate(p)
            else:
                s, u, mt, th = make_instruct_prose(p)
            jobs.append({'task': task, 'seed': p, 'system': s, 'user': u, 'max_tokens': mt, 'thinking': th})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--total', type=int, default=0, help='总条数（按默认配比分配）')
    ap.add_argument('--per-task', default='', help='如 pingdian=4500,jianshang=2500,...')
    ap.add_argument('--concurrency', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    if args.per_task:
        plan = {}
        for kv in args.per_task.split(','):
            k, v = kv.split('=')
            plan[k.strip()] = int(v)
    else:
        total = args.total or 10000
        plan = {'pingdian': int(total * 0.45), 'jianshang': int(total * 0.25),
                'instruct_poem': int(total * 0.15), 'prose_imitate': int(total * 0.10),
                'instruct_prose': total - int(total * 0.45) - int(total * 0.25) - int(total * 0.15) - int(total * 0.10)}
    print('plan:', plan, flush=True)

    styles = load_styles()
    jobs = []
    for task, n in plan.items():
        jobs.extend(build_jobs(task, n, styles))
    random.shuffle(jobs)
    print('jobs:', len(jobs), flush=True)

    teacher = Teacher()
    lock = threading.Lock()
    usage = {'prompt': 0, 'completion': 0, 'calls': 0, 'fail': 0}
    start = time.time()

    def work(job):
        r = teacher.chat([{'role': 'system', 'content': job['system']},
                          {'role': 'user', 'content': job['user']}],
                         temperature=0.85, max_tokens=job['max_tokens'], effort='low',
                         thinking=job['thinking'])
        if not (r.get('text') or '').strip():
            # 思考吃满或空输出：加倍重试
            r = teacher.chat([{'role': 'system', 'content': job['system']},
                              {'role': 'user', 'content': job['user']}],
                             temperature=0.7, max_tokens=job['max_tokens'] * 2, effort='low',
                             thinking=job['thinking'])
        ok, why = validate(job['task'], r['text'])
        if not ok and why == 'bad_json':
            # 再试一次（更低温）
            r2 = teacher.chat([{'role': 'system', 'content': job['system']},
                               {'role': 'user', 'content': job['user'] + '\n只输出合法 JSON！'}],
                              temperature=0.3, max_tokens=job['max_tokens'], effort='low')
            if validate(job['task'], r2['text'])[0]:
                r = r2
                ok, why = True, ''
        with lock:
            usage['prompt'] += r['usage'].get('prompt_tokens', 0)
            usage['completion'] += r['usage'].get('completion_tokens', 0)
            usage['calls'] += 1
        if not ok:
            with lock:
                usage['fail'] += 1
            return {'task': job['task'], 'seed': job['seed'], 'error': why, 'raw': r['text'][:200]}
        return {'task': job['task'], 'seed': job['seed'], 'output': r['text'],
                'finish': r.get('finish'), 'usage': r['usage']}

    per_task_files = {t: open(os.path.join(OUT_DIR, f'{t}.jsonl'), 'a', encoding='utf-8') for t in plan}
    n_done = 0
    last_report = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(work, job): job for job in jobs}
        for fut in as_completed(futs):
            row = fut.result()
            f = per_task_files[row['task']]
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            n_done += 1
            if time.time() - last_report > 30:
                last_report = time.time()
                el = time.time() - start
                print(f'[{n_done}/{len(jobs)}] {n_done/el:.1f} samples/s | '
                      f'prompt={usage["prompt"]} completion={usage["completion"]} fail={usage["fail"]}', flush=True)

    for f in per_task_files.values():
        f.close()

    el = time.time() - start
    tok = usage['prompt'] + usage['completion']
    price = PRICE['peak'] if is_peak() else PRICE['idle']
    cost = usage['prompt'] / 1e6 * price['in_miss'] + usage['completion'] / 1e6 * price['out']
    summary = {'jobs': len(jobs), 'done': n_done, 'elapsed_s': round(el, 1),
               'usage': usage, 'price_mode': 'peak' if is_peak() else 'idle',
               'est_cost_cny': round(cost, 3)}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    with open(COSTLOG, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()

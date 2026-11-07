"""过滤教师生成数据 → SFT chat 数据集（train/val 切分）。

输入：D:\train\data\sft\teacher_v1\{task}.jsonl
输出：
  D:\train\data\sft\sft_v1_train.jsonl
  D:\train\data\sft\sft_v1_val.jsonl
  D:\train\data\sft\_filter_report.json
"""
import collections
import hashlib
import json
import os
import re

IN_DIR = r'D:\train\data\sft\teacher_v1'
OUT_DIR = r'D:\train\data\sft'
TASKS = ['pingdian', 'jianshang', 'instruct_poem', 'prose_imitate', 'instruct_prose']


def load(task):
    p = os.path.join(IN_DIR, task + '.jsonl')
    if not os.path.exists(p):
        return []
    rows = []
    for line in open(p, encoding='utf-8'):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def is_valid(task, text):
    t = (text or '').strip()
    if len(t) < 60:
        return False, 'short'
    if task.startswith('instruct'):
        m = re.search(r'\{[\s\S]*\}', t)
        if not m:
            return False, 'no_json'
        try:
            json.loads(m.group(0))
        except Exception:
            return False, 'bad_json'
    return True, ''


def row_to_chat(task, out):
    seed = out.get('seed') or {}
    if task == 'pingdian':
        system = '你是精通中国古典诗论的诗话家，能模仿历代诗话笔法评诗。'
        user = (f'请评点这首诗（{seed.get("dynasty","")}·{seed.get("author","")}'
                f'《{seed.get("title","")}》）：\n{seed.get("text","")}')
    elif task == 'jianshang':
        system = '你是古籍整理与古典诗词鉴赏专家。'
        user = (f'请注释并赏析这首诗（{seed.get("dynasty","")}·{seed.get("author","")}'
                f'《{seed.get("title","")}》）：\n{seed.get("text","")}')
    elif task == 'instruct_poem':
        system = '你是古诗创作任务设计者。'
        user = f'请根据这首诗反推创作指令：{seed.get("text","")}'
        system = '你是古诗创作任务设计者，输出 JSON 指令。'
    elif task == 'prose_imitate':
        system = '你是明清古文家，擅长碑传、记序、书说等文体。'
        user = (f'请以「{seed.get("dynasty","")}代文风」写一段古文，文体近于{seed.get("genre","")}，'
                f'题目自拟，正文 300–450 字。')
    else:
        system = '你是古文创作任务设计者。'
        user = f'请根据这段古文反推创作指令：{seed.get("text","")[:500]}'
    return {
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
            {'role': 'assistant', 'content': out['output']},
        ],
        # alpaca 格式字段（LLaMA-Factory 直接用）
        'system': system,
        'instruction': user,
        'input': '',
        'output': out['output'],
        'task': task,
        'seed': seed,
    }


def main():
    report = {}
    rows_all = []
    seen = set()
    for task in TASKS:
        rows = load(task)
        ok = []
        reasons = collections.Counter()
        for r in rows:
            if not r.get('output'):
                reasons['empty'] += 1
                continue
            v, why = is_valid(task, r['output'])
            if not v:
                reasons[why] += 1
                continue
            h = hashlib.md5(r['output'][:300].encode('utf-8')).hexdigest()
            if h in seen:
                reasons['dup'] += 1
                continue
            seen.add(h)
            ok.append(row_to_chat(task, r))
        report[task] = {'raw': len(rows), 'kept': len(ok), 'reasons': dict(reasons)}
        rows_all.extend(ok)

    # 按 hash 切 train/val
    def bucket(chat):
        h = hashlib.md5((chat['messages'][1]['content'] + chat['task']).encode('utf-8')).hexdigest()
        return int(h[:8], 16) % 100
    train = [r for r in rows_all if bucket(r) >= 5]
    val = [r for r in rows_all if bucket(r) < 5]

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'sft_v1_train.jsonl'), 'w', encoding='utf-8') as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(os.path.join(OUT_DIR, 'sft_v1_val.jsonl'), 'w', encoding='utf-8') as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    report['total'] = {'raw': sum(v['raw'] for v in report.values()),
                       'kept': len(rows_all), 'train': len(train), 'val': len(val)}
    with open(os.path.join(OUT_DIR, '_filter_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

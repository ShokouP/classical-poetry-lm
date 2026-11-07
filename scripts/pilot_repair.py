"""补跑 pilot 中 content 为空的任务：更大 max_tokens + effort=low，并把 reasoning_content 记下来。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_teacher import Teacher  # noqa: E402

IN = r'D:\train\data\pilot\teacher_pilot.jsonl'
rows = [json.loads(l) for l in open(IN, encoding='utf-8') if l.strip()]

t = Teacher()
changed = 0
for r in rows:
    if r.get('output') or 'error' in r:
        continue
    tag = r['tag']
    print('=' * 16, 'retry', tag)
    try:
        res = t.chat([
            {'role': 'system', 'content': r['system']},
            {'role': 'user', 'content': r['user']},
        ], temperature=0.7, max_tokens=8000, effort='low')
        text = res['text']
        r['output'] = text
        r['usage'] = res['usage']
        r['finish'] = res.get('finish')
        changed += 1
        print(text[:420].replace('\n', ' '))
        if not text:
            print('(still empty) finish=', res.get('finish'), 'usage=', res['usage'])
    except Exception as e:
        print('FAIL', e)

with open(IN, 'w', encoding='utf-8') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('retried:', changed, '| usage:', t.usage)

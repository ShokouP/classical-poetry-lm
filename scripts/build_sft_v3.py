"""构建 sft_v3：sft_v2_train + 扩写数据（加权 4x，排前保证被训练到）。"""
import json
import random

random.seed(11)
S2 = r'D:\train\data\sft\sft_v2_train.jsonl'
EXP = r'D:\train\data\sft\teacher_expand\expand_v1.jsonl'
OUT = r'D:\train\data\sft\sft_v3_train.jsonl'


def load(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


base = load(S2)
exp = load(EXP)
print('base:', len(base), 'expand:', len(exp))

# 扩写样本加权 4 倍
exp_x = exp * 4
random.shuffle(exp_x)

# 基座数据打乱，但把扩写放在前面
rest = base[:]
random.shuffle(rest)
rows = exp_x + rest
with open(OUT, 'w', encoding='utf-8') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('saved:', OUT, 'rows:', len(rows))
from collections import Counter
print(Counter(r['task'] for r in rows).most_common(10))

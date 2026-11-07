"""韵部簇：拼音韵母组 + 真实诗共现强度合并，构建近似平水韵的韵部簇。

输出 rhyme_classes.json（覆盖旧文件）：
  char_to_group: 字 -> 簇 id
  groups: 簇 id -> [字]
  stats: 一致性等
"""
import json
import re
from collections import Counter, defaultdict

from pypinyin import lazy_pinyin, Style

POEMS = r'D:\train\data\cleaned\poems_v1.jsonl'
OUT = r'D:\train\data\cleaned\rhyme_classes.json'
GENRES = ('五绝', '七绝', '五律', '七律')


def clauses_of(text):
    parts = re.split(r'[，。；！？、：\n]+', text)
    out = []
    for p in parts:
        p = re.sub(r'[^\u4e00-\u9fff]', '', p)
        if len(p) in (5, 7):
            out.append(p)
    return out


def final_of(ch):
    fin = lazy_pinyin(ch, style=Style.FINALS, errors='ignore')
    return fin[0] if fin else ''


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main():
    char_final = {}
    poem_finals = []
    group_count = Counter()
    with open(POEMS, encoding='utf-8') as f:
        for line in f:
            try:
                p = json.loads(line)
            except Exception:
                continue
            if p.get('genre') not in GENRES:
                continue
            cl = clauses_of(p.get('text', ''))
            if len(cl) < 4:
                continue
            finals = [c[-1] for c in cl[1::2]]
            if len(finals) < 2:
                continue
            fs = []
            for ch in finals:
                if ch not in char_final:
                    char_final[ch] = final_of(ch)
                if char_final[ch]:
                    fs.append(char_final[ch])
            if len(fs) < 2:
                continue
            poem_finals.append(set(fs))
            for x in set(fs):
                group_count[x] += 1

    finals = [k for k, v in group_count.items() if v >= 20]
    valid = set(finals)
    n = len(poem_finals)
    same_final_ratio = sum(1 for fs in poem_finals if len(fs & valid) == 1 and fs <= valid) / n

    # 组间共现
    co = Counter()
    for fs in poem_finals:
        xs = sorted(x for x in fs if x in valid)
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                co[(xs[i], xs[j])] += 1

    dsu = DSU()
    for g in valid:
        dsu.find(g)
    pairs = []
    for (a, b), w in co.items():
        assoc = w / min(group_count[a], group_count[b])
        pairs.append((assoc, w, a, b))
    pairs.sort(reverse=True)
    merged_edges = []
    for assoc, w, a, b in pairs:
        if assoc >= 0.25:
            if dsu.find(a) != dsu.find(b):
                dsu.union(a, b)
                merged_edges.append((a, b, round(assoc, 3), w))

    clusters = defaultdict(list)
    for g in valid:
        clusters[dsu.find(g)].append(g)
    cluster_id = {}
    for root, gs in clusters.items():
        for g in gs:
            cluster_id[g] = root

    # 字 -> 簇
    char_to_group = {ch: cluster_id[f] for ch, f in char_final.items() if f in valid}
    groups = defaultdict(list)
    for ch, cid in char_to_group.items():
        groups[cid].append(ch)

    # 一致性（同一首诗偶数句是否同簇）
    def consistent(fs):
        cs = {cluster_id[x] for x in fs if x in cluster_id}
        return len(cs) == 1

    ratio = sum(1 for fs in poem_finals if consistent(fs)) / n
    stats = {'poems_checked': n,
             'same_final_exact_ratio': round(same_final_ratio, 4),
             'same_cluster_ratio': round(ratio, 4),
             'final_groups': len(valid), 'clusters': len(clusters),
             'chars_mapped': len(char_to_group),
             'merges': merged_edges[:40]}
    json.dump({'char_to_group': char_to_group,
               'groups': {str(k): sorted(v) for k, v in groups.items()},
               'stats': stats},
              open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps({k: stats[k] for k in stats if k != 'merges'}, ensure_ascii=False, indent=1))
    print('cluster sizes:', sorted((len(v) for v in clusters.values()), reverse=True)[:15])
    print('sample merges:', merged_edges[:10])


if __name__ == '__main__':
    main()

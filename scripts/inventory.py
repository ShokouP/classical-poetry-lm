"""语料盘点：统计各语料的文件数/字符数，输出 JSON 报告。"""
import json
import os
import re

CORPUS = r'D:\poetry-corpus'
OUT = r'D:\train\data\raw\_inventory.json'
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def count_chars(path, exts=('.txt', '.json', '.md')):
    total = 0
    files = 0
    if os.path.isfile(path):
        try:
            return 1, len(open(path, encoding='utf-8', errors='replace').read())
        except Exception:
            return 0, 0
    for root, dirs, fs in os.walk(path):
        for f in fs:
            if not f.lower().endswith(exts):
                continue
            p = os.path.join(root, f)
            try:
                total += len(open(p, encoding='utf-8', errors='replace').read())
                files += 1
            except Exception:
                pass
    return files, total


def poem_counts(json_dir, file_re):
    """统计 chinese-poetry JSON 条目数。"""
    import glob
    n = 0
    samples = 0
    for f in glob.glob(os.path.join(json_dir, '*.json')):
        if not re.search(file_re, os.path.basename(f)):
            continue
        try:
            d = json.load(open(f, encoding='utf-8'))
            if isinstance(d, list):
                n += len(d)
                samples += 1
        except Exception:
            pass
    return n, samples


report = {}
cp = os.path.join(CORPUS, 'chinese-poetry')
if os.path.isdir(cp):
    tang_dir = os.path.join(cp, '全唐诗')
    report['唐诗'] = {'count': poem_counts(tang_dir, r'^poet\.tang\.')[0]}
    report['宋诗'] = {'count': poem_counts(tang_dir, r'^poet\.song\.')[0]}
    report['宋词'] = {'count': poem_counts(os.path.join(cp, '宋词'), r'^ci\.song\.')[0]}
    report['元曲'] = {'count': poem_counts(os.path.join(cp, '元曲'), r'') [0]}

dai = os.path.join(CORPUS, 'daizhigev20')
if os.path.isdir(dai):
    for label, rel in [
        ('诗话', r'诗藏\诗话'),
        ('词话', r'诗藏\词话'),
        ('文评', r'集藏\文评'),
        ('文总集', r'集藏\文总集'),
    ]:
        files, chars = count_chars(os.path.join(dai, rel))
        report[label] = {'files': files, 'chars': chars}

cm = os.path.join(CORPUS, 'Classical-Modern')
if os.path.isdir(cm):
    files, chars = count_chars(os.path.join(cm, '古文原文'))
    report['古文原文(对照)'] = {'files': files, 'chars': chars}

print(json.dumps(report, ensure_ascii=False, indent=2))
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print('saved:', OUT)

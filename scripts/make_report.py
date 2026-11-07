"""把 eval_model.py 的 JSON 结果转成 Markdown 报告。"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True, help='如 name=path.json')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    lines = ['# 古典诗·文模型评估报告', '']
    for item in args.inputs:
        if '=' not in item:
            continue
        name, path = item.split('=', 1)
        if not os.path.exists(path):
            continue
        d = json.load(open(path, encoding='utf-8'))
        lines.append(f'## {name}')
        lines.append('')
        lines.append('| 指标 | 值 |')
        lines.append('|---|---|')
        lines.append(f"| 样本数 | {d.get('samples')} |")
        for k, v in (d.get('poem_metrics') or {}).items():
            lines.append(f"| {k} 句数合规 | {v.get('line_count_ok')} |")
            lines.append(f"| {k} 句长合规 | {v.get('line_len_ok')} |")
            lines.append(f"| {k} 押韵 | {v.get('rhyme_ok')} |")
            lines.append(f"| {k} 平均 8-gram 重叠 | {v.get('avg_overlap')} |")
        lines.append('')
        # 样例
        for row in d.get('rows', [])[:3]:
            lines.append(f"### {row['id']}（{row['genre']}）")
            lines.append('')
            lines.append('```')
            lines.append(row['outputs'][0]['text'].strip()[:600])
            lines.append('```')
            lines.append('')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, 'w', encoding='utf-8').write('\n'.join(lines))
    print('saved:', args.out)


if __name__ == '__main__':
    main()

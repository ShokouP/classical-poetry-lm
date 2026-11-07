"""修正 eval_model.py：按「句」（，。拆分）评估，去掉标题行/注释。"""
import re

p = r'D:\train\scripts\eval_model.py'
s = open(p, encoding='utf-8').read()

old = """    def clean_lines(txt):
        lines = [re.sub(r'[^\\u4e00-\\u9fff]', '', l) for l in txt.splitlines()]
        return [l for l in lines if l]

    def rhyme_check(lines):
        if not have_pinyin or len(lines) < 2:
            return None
        finals = []
        for l in lines[1::2]:
            if not l:
                continue
            py = lazy_pinyin(l[-1], style=Style.FINALS)
            finals.append(py[0] if py else '')
        if len(finals) < 2:
            return None
        return len(set(finals)) == 1"""

new = """    def parse_clauses(txt, drop_title=True):
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        if drop_title and lines:
            first = lines[0]
            # 标题行：无标点、短、且后面还有内容
            if len(lines) > 1 and len(first) <= 10 and not re.search(r'[，。；、！？,.;!?]', first) and not first.startswith(('*', '（', '(')):
                lines = lines[1:]
        clauses = []
        for l in lines:
            if re.match(r'^\\*\\*', l) or l.startswith(('（', '(')):
                continue
            for c in re.split(r'[，。；、！？,.;!?\\s]+', l):
                c = re.sub(r'[^\\u4e00-\\u9fff]', '', c)
                if 2 <= len(c) <= 12:
                    clauses.append(c)
        return clauses

    def rhyme_check(clauses):
        if not have_pinyin or len(clauses) < 2:
            return None
        finals = []
        for c in clauses[1::2]:
            py = lazy_pinyin(c[-1], style=Style.FINALS)
            finals.append(py[0] if py else '')
        if len(finals) < 2:
            return None
        return len(set(finals)) == 1"""

if old in s:
    s = s.replace(old, new)
    print('rhyme/parse patched')
else:
    print('parse pattern NOT found')

old2 = """            lines = clean_lines(txt)"""
new2 = """            clauses = parse_clauses(txt)
            lines = clauses"""
if old2 in s:
    s = s.replace(old2, new2)
    print('clean_lines ref patched')

old3 = """            if p['lines']:
                rec['line_count_ok'] = len(lines) == p['lines']
                rec['line_len_ok'] = all(len(l) == p['line_chars'] for l in lines) if lines else False
                rec['rhyme_ok'] = rhyme_check(lines)"""
new3 = """            if p['lines']:
                expected = p['lines'] * 2
                rec['clauses'] = clauses
                rec['line_count_ok'] = len(clauses) == expected
                rec['line_len_ok'] = (len(clauses) >= expected and
                                      all(len(c) == p['line_chars'] for c in clauses[:expected]))
                rec['rhyme_ok'] = rhyme_check(clauses)"""
if old3 in s:
    s = s.replace(old3, new3)
    print('metric patched')
else:
    print('metric pattern NOT found')

open(p, 'w', encoding='utf-8').write(s)
print('done')

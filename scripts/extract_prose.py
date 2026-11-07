"""从「文总集」抽取明清散文篇目 → JSONL。

支持：
- 明文海.txt   ：篇名后紧跟【作者】，按作者标记切分
- 古文观止.txt ：正文篇目以【篇名】（出处）开头
- 清文精选.txt ：作者/篇名/正文 的简单结构
输出：D:\train\data\cleaned\prose_mingqing_v1.jsonl
"""
import json
import os
import re

SRC = r'D:\poetry-corpus\daizhigev20\集藏\文总集'
OUT_DIR = r'D:\train\data\cleaned'
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, 'prose_mingqing_v1.jsonl')

GENRE_SUFFIX = ['赋', '序', '记', '传', '书', '说', '论', '铭', '表', '碑', '状', '祭文', '行状',
                '墓志铭', '墓表', '题跋', '杂著', '疏', '赞', '颂', '箴', '诔', '哀辞', '启',
                '牍', '议', '策', '解', '辨', '原', '志', '跋', '纪事', '策问']


def genre_of(title, category=''):
    for s in GENRE_SUFFIX:
        if title.endswith(s):
            return s
    for s in GENRE_SUFFIX:
        if s in title:
            return s
    return category.strip() if category else '文'


def clean_text(t):
    t = t.replace('\u3000', ' ')
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


def extract_mingwenhai():
    p = os.path.join(SRC, '明文海.txt')
    text = open(p, encoding='utf-8', errors='replace').read()
    # 找所有 【作者】 标记；题目 = 该行【】前的内容；正文 = 到下一个【】或卷标题为止
    pat = re.compile(r'【([^】\n]{1,12})】')
    matches = list(pat.finditer(text))
    rows = []
    for i, m in enumerate(matches):
        author = m.group(1).strip()
        line_start = text.rfind('\n', 0, m.start()) + 1
        title = text[line_start:m.start()].strip(' \u3000　')
        if not (1 <= len(title) <= 60):
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end]
        # 去掉卷标题行/分类标题行
        lines = []
        for ln in body.splitlines():
            s = ln.strip(' \u3000　')
            if not s:
                continue
            if re.match(r'^明文海[卷巻]', s):
                continue
            if re.match(r'^钦定四库全书$', s):
                continue
            if len(s) <= 6 and s.endswith(('五', '六', '七', '八', '九', '十', '一', '二', '三', '四')):
                continue
            lines.append(s)
        body = '\n'.join(lines)
        body = clean_text(body)
        if not (80 <= len(body) <= 20000):
            continue
        if not re.search(r'[\u4e00-\u9fff]{8,}', body):
            continue
        rows.append({'dynasty': '明', 'author': author, 'title': title,
                     'genre': genre_of(title), 'text': body, 'source': '明文海'})
    return rows


def extract_guwenguanzhi():
    p = os.path.join(SRC, '古文观止.txt')
    lines = open(p, encoding='utf-8', errors='replace').read().splitlines()
    rows = []
    cur = None
    for ln in lines:
        s = ln.strip()
        m = re.match(r'^【([^】]{2,40})】(.*)$', s)
        if m:
            if cur and len(cur['text']) >= 60:
                rows.append(cur)
            title = m.group(1).strip()
            tail = m.group(2)
            author = ''
            am = re.search(r'《([^》]+)》', tail)
            if am:
                author = am.group(1).strip()
            cur = {'dynasty': '', 'author': author, 'title': title,
                   'genre': genre_of(title), 'text': '', 'source': '古文观止'}
            continue
        if cur is not None and s:
            cur['text'] += s + '\n'
    if cur and len(cur['text']) >= 60:
        rows.append(cur)
    # 朝代推断：按作者/篇名粗分（左传/国语/战国策→先秦，史记/汉书→汉，韩愈/柳宗元→唐，苏轼/王安石→宋，明/清 略）
    tang = ['韩愈', '柳宗元', '刘禹锡', '白居易', '李华', '元结']
    song = ['欧阳修', '苏洵', '苏轼', '苏辙', '王安石', '曾巩', '范仲淹', '司马光', '周敦颐', '张载']
    for r in rows:
        a = r['author']
        if any(x in a for x in tang):
            r['dynasty'] = '唐'
        elif any(x in a for x in song):
            r['dynasty'] = '宋'
        elif re.search(r'左传|国语|公羊|谷梁|战国策|礼|檀弓', a):
            r['dynasty'] = '先秦'
        elif re.search(r'史记|汉书|后汉书|三国志', a):
            r['dynasty'] = '汉'
        else:
            r['dynasty'] = ''
        r['text'] = clean_text(r['text'])
    return rows


def extract_qingwen():
    p = os.path.join(SRC, '清文精选.txt')
    lines = [l.strip() for l in open(p, encoding='utf-8', errors='replace').read().splitlines()]
    rows = []
    i = 0
    while i < len(lines):
        s = lines[i]
        if not s or len(s) > 30:
            i += 1
            continue
        # 作者行：2-8 字且下一行是题目（2-20 字）
        if 2 <= len(s) <= 8 and i + 1 < len(lines) and 2 <= len(lines[i + 1]) <= 30:
            author = s
            title = lines[i + 1]
            body_lines = []
            j = i + 2
            while j < len(lines):
                t = lines[j]
                if t and len(t) <= 30 and (j + 1 < len(lines) and 2 <= len(lines[j + 1]) <= 30 and len(t) <= 8):
                    break
                body_lines.append(t)
                j += 1
            body = clean_text('\n'.join(x for x in body_lines if x))
            if 80 <= len(body) <= 20000:
                rows.append({'dynasty': '清', 'author': author, 'title': title,
                             'genre': genre_of(title), 'text': body, 'source': '清文精选'})
                i = j
                continue
        i += 1
    return rows


def extract_mingwenheng():
    """明文衡：形如  　　<篇名>　　　<作者> 的行是篇目标题。"""
    p = os.path.join(SRC, '明文衡.txt')
    text = open(p, encoding='utf-8', errors='replace').read()
    lines = text.splitlines()
    rows = []
    cur = None
    for ln in lines:
        m = re.match(r'^[\u3000　\s]{1,6}([^\u3000　\s]{2,40})[\u3000　\s]{2,}([^\u3000　\s]{2,10})$', ln)
        if m:
            if cur and len(cur['text']) >= 80:
                rows.append(cur)
            title, author = m.group(1).strip(), m.group(2).strip()
            cur = {'dynasty': '明', 'author': author, 'title': title,
                   'genre': genre_of(title), 'text': '', 'source': '明文衡'}
            continue
        if cur is not None:
            s = ln.strip(' \u3000　')
            if s:
                cur['text'] += s + '\n'
    if cur and len(cur['text']) >= 80:
        rows.append(cur)
    for r in rows:
        r['text'] = clean_text(r['text'])
    return rows


def main():
    all_rows = []
    for name, fn in [('明文海', extract_mingwenhai), ('古文观止', extract_guwenguanzhi),
                     ('清文精选', extract_qingwen), ('明文衡', extract_mingwenheng)]:
        try:
            rows = fn()
            print(f'{name}: {len(rows)} 篇, 总字数 {sum(len(r["text"]) for r in rows)/1e4:.1f} 万')
            all_rows.extend(rows)
        except Exception as e:
            print(f'{name}: FAIL {type(e).__name__}: {e}')

    # 过滤/去重
    seen = set()
    final = []
    for r in all_rows:
        key = (r['title'], r['text'][:120])
        if key in seen:
            continue
        seen.add(key)
        final.append(r)

    with open(OUT, 'w', encoding='utf-8') as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('=' * 20)
    print('total:', len(final), '篇,', round(sum(len(r['text']) for r in final)/1e4, 1), '万字')
    print('saved:', OUT)
    for r in final[:2]:
        print('--- 样例:', r['dynasty'], r['author'], '《' + r['title'] + '》', 'genre=' + r['genre'])
        print(r['text'][:200].replace('\n', ' '))


if __name__ == '__main__':
    main()

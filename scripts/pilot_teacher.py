"""Teacher 小试点：验证 DeepSeek V4.1-Flash 在「诗话评点 / 鉴赏 / 逆向指令 / 明清散文仿写」上的质量。

产出：D:\train\data\pilot\teacher_pilot.jsonl
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_teacher import Teacher  # noqa: E402

CORPUS = r'D:\poetry-corpus'
OUT_DIR = r'D:\train\data\pilot'
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, 'teacher_pilot.jsonl')

# ---------- 读取真实语料 ----------
def load_poems(kind='tang', n=2):
    import glob
    files = sorted(glob.glob(os.path.join(CORPUS, 'chinese-poetry', '全唐诗', f'poet.{kind}.*.json')))
    poems = []
    for f in files[:3]:
        try:
            d = json.load(open(f, encoding='utf-8'))
        except Exception:
            continue
        for item in d:
            t = item.get('title') or ''
            paras = item.get('paragraphs') or []
            if len(paras) >= 4 and 2 <= len(t) <= 20 and item.get('author'):
                poems.append({'dynasty': '唐' if kind == 'tang' else '宋',
                              'author': item['author'], 'title': t,
                              'text': '\n'.join(paras)})
                if len(poems) >= n:
                    return poems
    return poems


def load_prose():
    """从清文精选取两篇；再从明文海取一段。"""
    out = []
    p = os.path.join(CORPUS, 'daizhigev20', '集藏', '文总集', '清文精选.txt')
    if os.path.exists(p):
        lines = [l.strip() for l in open(p, encoding='utf-8', errors='replace').read().splitlines() if l.strip()]
        # 跳过标题行，找 作者/篇名/正文 结构
        idx = 0
        while idx < len(lines) - 3 and len(out) < 2:
            author, title = lines[idx], lines[idx + 1]
            if 2 <= len(author) <= 8 and 2 <= len(title) <= 20:
                body = lines[idx + 2:idx + 8]
                text = '\n'.join(body)
                if len(text) > 120:
                    out.append({'dynasty': '清', 'author': author, 'title': title, 'text': text})
                    idx += 8
                    continue
            idx += 1
    p2 = os.path.join(CORPUS, 'daizhigev20', '集藏', '文总集', '明文海.txt')
    if os.path.exists(p2):
        t = open(p2, encoding='utf-8', errors='replace').read()
        pos = t.find('明文海卷', 5000)
        if pos < 0:
            pos = 60000
        seg = t[pos:pos + 700]
        out.append({'dynasty': '明', 'author': '（明文海选段）', 'title': '明文选段', 'text': seg})
    return out


def load_shihua_snippets(n=2, chars=260):
    """从真实诗话取风格示例。"""
    d = os.path.join(CORPUS, 'daizhigev20', '诗藏', '诗话')
    picks = ['沧浪诗话.txt', '随园诗话.txt', '六一诗话.txt']
    out = []
    for name in picks[:n]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            t = open(p, encoding='utf-8', errors='replace').read()
            # 跳过题目/前言，取一段正文
            start = min(len(t) // 10, 2000)
            snippet = t[start:start + chars].replace('\n', ' ')
            out.append((name.replace('.txt', ''), snippet))
    return out


# ---------- 任务 prompt ----------
def task_pingdian(poem, style_examples):
    style = '\n'.join(f'【{name}】{text}' for name, text in style_examples)
    system = (
        '你是精通中国古典诗论的诗话家。请模仿下面这些真实诗话的风格与口吻来评诗：'
        '文言为主、点到即止、重渊源与字法，避免现代学术腔和空洞形容词。\n'
        f'{style}'
    )
    user = (
        f'请评点这首诗（{poem["dynasty"]}·{poem["author"]}《{poem["title"]}》）：\n{poem["text"]}\n\n'
        '要求：180–300 字，可分 2–3 条；点出起结、字眼、意境或渊源，必要时指出瑕疵。'
    )
    return system, user


def task_jianshang(poem):
    system = '你是古籍整理与古典诗词鉴赏专家。'
    user = (
        f'请对这首诗做注释与赏析（{poem["dynasty"]}·{poem["author"]}《{poem["title"]}》）：\n{poem["text"]}\n\n'
        '要求：1) 注释疑难字词与典故；2) 说明体裁句式（宽格律即可）；3) 150–250 字意境赏析。'
    )
    return system, user


def task_instruction(poem):
    system = '你是古诗创作任务设计者。'
    user = (
        f'根据下面这首真实的{poem["dynasty"]}诗，反推一条可用于训练 AI 写诗的创作指令。\n'
        f'诗：{poem["text"]}\n作者：{poem["author"]}，题目：{poem["title"]}\n\n'
        '只输出 JSON（不要代码块），字段：{"朝代","作者风格","体裁","题材","风格词","写作要求"(3条数组)}'
    )
    return system, user


def task_prose_instruction(prose):
    system = '你是古文创作任务设计者。'
    user = (
        f'根据下面这篇真实的{prose["dynasty"]}代古文（{prose["author"]}《{prose["title"]}》），'
        '反推一条可用于训练 AI 写古文的创作指令。\n'
        f'原文节选：\n{prose["text"][:500]}\n\n'
        '只输出 JSON（不要代码块），字段：{"朝代","文体","题","风格词","写作要求"(3条数组)}'
    )
    return system, user


def task_prose_imitate(prose):
    system = (
        '你是明清古文家，擅长碑传、记序、书说等文体。'
        '请用纯正文言写作，不用白话、不加解释。'
    )
    user = (
        f'请以「{prose["dynasty"]}代文风」写一段古文，文体近于{prose["title"]}一类，'
        f'题目自拟（约 8–20 字），正文 300–450 字。\n'
        f'参考风格（勿抄袭）：\n{prose["text"][:400]}'
    )
    return system, user


def main():
    random.seed(7)
    poems = load_poems('tang', 2) + load_poems('song', 1)
    prose = load_prose()
    styles = load_shihua_snippets(2)

    t = Teacher()
    rows = []

    def run(tag, system, user, temperature=0.8, max_tokens=1200):
        print('=' * 20, tag)
        try:
            r = t.generate(system, user, temperature=temperature, max_tokens=max_tokens)
            text = r['text']
            print(text[:500].replace('\n', ' '))
            rows.append({'tag': tag, 'system': system, 'user': user, 'output': text,
                         'usage': r['usage'], 'model': t.model})
        except Exception as e:
            print('FAIL', e)
            rows.append({'tag': tag, 'error': str(e)})

    for i, p in enumerate(poems):
        s, u = task_pingdian(p, styles)
        run(f'pingdian-{i}', s, u)
        s, u = task_jianshang(p)
        run(f'jianshang-{i}', s, u, temperature=0.6)
    s, u = task_instruction(poems[0])
    run('instruction-poem', s, u, temperature=0.4)
    for i, pr in enumerate(prose[:2]):
        s, u = task_prose_instruction(pr)
        run(f'instruction-prose-{i}', s, u, temperature=0.4)
        s, u = task_prose_imitate(pr)
        run(f'imitate-prose-{i}', s, u)

    with open(OUT, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print('=' * 20, 'usage:', t.usage)
    print('saved:', OUT)


if __name__ == '__main__':
    main()

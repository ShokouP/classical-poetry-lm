"""生成《Qwen3.5 9B 詩文專家模型選》DOCX。"""
import json
import os

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

WS = r'C:\Users\wangy\Documents\deepseek-harness\default-workspace\reports'
SRC = os.path.join(WS, 'book_pieces.json')
OUT = os.path.join(WS, 'qwen3.5-9B-诗文专家模型选.docx')
data = json.load(open(SRC, encoding='utf-8'))

doc = Document()

# 页面设置 A4
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.left_margin = Cm(2.4)
sec.right_margin = Cm(2.4)
sec.top_margin = Cm(2.4)
sec.bottom_margin = Cm(2.2)


def set_font(run, name='SimSun', size=None, bold=None, color=None, italic=None):
    run.font.name = name
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.append(rFonts)
    rFonts.set(qn('w:eastAsia'), name)
    rFonts.set(qn('w:ascii'), name)
    rFonts.set(qn('w:hAnsi'), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


def para(text='', font='SimSun', size=12, bold=False, align=None, space_before=0,
         space_after=6, line=1.5, indent_chars=0, color=None, italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line
    if indent_chars:
        p.paragraph_format.first_line_indent = Pt(size * indent_chars)
    if align is not None:
        p.alignment = align
    if text:
        r = p.add_run(text)
        set_font(r, font, size, bold, color, italic)
    return p


def page_break():
    doc.add_page_break()


def add_page_number_footer():
    footer = sec.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    fld = OxmlElement('w:fldSimple')
    fld.set(qn('w:instr'), 'PAGE')
    r._element.addnext(fld)
    r2 = p.add_run('  ·  Qwen3.5 9B 詩文專家模型選')
    set_font(r2, 'SimSun', 9, color=(0x88, 0x88, 0x88))


add_page_number_footer()

# ---------------- 封面 ----------------
for _ in range(4):
    para('', size=12, space_after=0, line=1.0)
para('Qwen3.5 9B', font='SimHei', size=40, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
     space_after=4, line=1.15)
para('诗文专家模型选', font='SimHei', size=34, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
     space_after=14, line=1.2)
para(data['subtitle'], font='SimHei', size=20, align=WD_ALIGN_PARAGRAPH.CENTER,
     space_after=40, line=1.3)
para('九篇詩文 · 附編者評點', font='SimSun', size=13, align=WD_ALIGN_PARAGRAPH.CENTER,
     space_after=120, color=(0x66, 0x66, 0x66))
for k, v in data['meta'].items():
    para(f'{k}：{v}', font='SimSun', size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER,
         space_after=4, line=1.3, color=(0x55, 0x55, 0x55))
page_break()

# ---------------- 序 ----------------
para('序', font='SimHei', size=22, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18)
para(data['preface'], font='SimSun', size=13, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent_chars=2,
     line=1.9, space_after=8)
para('——  編者識於二〇二六年十一月', font='SimSun', size=11, align=WD_ALIGN_PARAGRAPH.RIGHT,
     space_before=12, color=(0x77, 0x77, 0x77))

# ---------------- 目錄 ----------------
page_break()
para('目　錄', font='SimHei', size=22, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=20)
sections = {}
for p in data['pieces']:
    sections.setdefault(p['section'], []).append(p['title'])
for sec_name, titles in sections.items():
    para(sec_name, font='SimHei', size=14, bold=True, space_before=10, space_after=6)
    for t in titles:
        para('　　' + t, font='SimSun', size=12, space_after=3, line=1.5)

# ---------------- 作品 ----------------
current_section = None
for p in data['pieces']:
    if p['section'] != current_section:
        current_section = p['section']
        page_break()
        para(current_section, font='SimHei', size=24, bold=True,
             align=WD_ALIGN_PARAGRAPH.CENTER, space_before=30, space_after=10)
        line_p = doc.add_paragraph()
        line_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = line_p.add_run('＊　　　＊　　　＊')
        set_font(r, 'SimSun', 12, color=(0x99, 0x99, 0x99))
        page_break()
    else:
        page_break()

    para(p['title'], font='SimHei', size=20, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
         space_before=24, space_after=6)
    para(f"〔{p['genre']}〕　Qwen3.5-9B 生成", font='SimSun', size=10,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18, color=(0x88, 0x88, 0x88))

    if p['genre'] in ('記', '序', '論'):
        for seg in p['text'].split('\n'):
            seg = seg.strip()
            if seg:
                para(seg, font='SimSun', size=13, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
                     indent_chars=2, line=1.95, space_after=6)
    else:
        for line in [x.strip() for x in p['text'].split('\n') if x.strip()]:
            para(line, font='SimSun', size=15, align=WD_ALIGN_PARAGRAPH.CENTER,
                 line=2.0, space_after=2)

    # 評點
    para('', size=6, space_after=0)
    note_p = doc.add_paragraph()
    note_p.paragraph_format.space_before = Pt(16)
    note_p.paragraph_format.line_spacing = 1.6
    r = note_p.add_run('編者評：')
    set_font(r, 'SimHei', 10.5, bold=True, color=(0x44, 0x44, 0x44))
    r2 = note_p.add_run(p['note'])
    set_font(r2, 'SimSun', 10.5, color=(0x44, 0x44, 0x44))

# ---------------- 附錄 ----------------
page_break()
para('附　錄', font='SimHei', size=22, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=16)
para('一、模型與訓練', font='SimHei', size=14, bold=True, space_before=8, space_after=4)
for line in [
    '基座：Qwen3.5-9B（Apache-2.0）',
    '微調：SFT（LoRA r=32，seq 768 定長 padding，600 步，bf16，lr 5e-5）',
    '訓練數據：sft_v3（40,813 條：擬作詩 15,000 / 真實詩話評點 13,005 / teacher 評點與鑑賞 / 明清散文仿寫 / 擴寫 376）',
    '訓練硬件：RTX 5090 Laptop 24GB，原生 Windows，約 2.5 小時（含試點）',
    '生成：temperature 0.95 / top_p 0.95 / repetition_penalty 1.08–1.15；詩以 Best-of-N 按句數、句長、韻部、抄襲檢測打分選優',
]:
    para('· ' + line, font='SimSun', size=11, indent_chars=1, line=1.6, space_after=3)

para('二、自動評估（九B，600 步）', font='SimHei', size=14, bold=True, space_before=12, space_after=4)
rows = [('體裁', '句數合規', '句長合規', '押韻', '與語料 8-gram 重複'),
        ('五絕', '1/2', '2/2', '2/2', '0.009'),
        ('七絕', '2/2', '2/2', '2/2', '0.0'),
        ('五律', '2/2', '2/2', '1/2', '0.0'),
        ('七律', '1/2', '1/2', '1/2', '0.0')]
table = doc.add_table(rows=len(rows), cols=5)
table.style = 'Table Grid'
for i, row in enumerate(rows):
    for j, val in enumerate(row):
        cell = table.cell(i, j)
        cell.text = ''
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(val)
        set_font(r, 'SimSun', 10.5, bold=(i == 0))

para('三、抄襲檢測說明', font='SimHei', size=14, bold=True, space_before=12, space_after=4)
para('以 313,609 篇詩文（309,441 首詩 + 4,168 篇明清散文）提取 1,709 萬個 8-gram 指紋，'
     '生成文本與之逐字比對；本集所選重複率均低於 0.01（杜牧《秋夕》原文比對為 0.619，可資對照）。',
     font='SimSun', size=11, indent_chars=2, line=1.7)

doc.save(OUT)
print('saved docx:', OUT, os.path.getsize(OUT))

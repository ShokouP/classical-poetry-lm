"""汇总 9B 作品集内容（统一转为繁体），输出 book_pieces.json。"""
import json

from opencc import OpenCC

cc = OpenCC('s2t')
S = cc.convert


def load(p):
    return json.load(open(p, encoding='utf-8'))


best = load(r'D:\train\runs\gen_bestofn_9b.json')
bmap = {r['id']: r['best']['text'] for r in best}
ev = load(r'D:\train\runs\eval_9b_fp.json')
evmap = {}
for row in ev['rows']:
    evmap[row['id']] = [o['text'] for o in row['outputs']]
prose = load(r'D:\train\runs\book_prose_9b.json')


def clean(t):
    t = S(t.strip())
    return t


pieces = [
    {'section': '詩選', 'title': '五言古風 · 山居', 'genre': '五言古風',
     'text': clean(evmap['wujue'][1]),
     'note': '起手「松風清入府，山月淨含天」一聯最見靜氣；中段寫石瀨、漁煙、霜簡，聲色俱清；末以「浮光入鳥邊」收束，餘韻不盡。'},
    {'section': '詩選', 'title': '七絕 · 秋夜懷人', 'genre': '七絕',
     'text': clean(bmap['qijue']),
     'note': '四句七言，押「明、情、名」韻；「卻恨郎來騎竹馬，繞牀偷喚小兒名」以童年舊事寫相思，語淺而情長。'},
    {'section': '詩選', 'title': '五律 · 江行', 'genre': '五律',
     'text': clean(bmap['wulv']),
     'note': '八句五言，押「隨、湄、絲、期」韻；「星流連夏戍，月落滬河湄」對仗工穩，羈旅之思在句外。'},
    {'section': '詩選', 'title': '七律 · 登高懷古', 'genre': '七律',
     'text': clean(bmap['qilv']),
     'note': '八句七言，押「青、行、成、京、情」韻；「人言吳許移山了，我欲秦皇煉石成」用事而不滯，氣象開闊。'},
    {'section': '詞選', 'title': '念奴嬌 · 秋夜泛湖', 'genre': '詞',
     'text': clean(evmap['ci'][1]),
     'note': '上片寫涼風驚蟬、素娥飛月，下片轉入江湖倦客之思，結以「區區莫較得失」，曠達近東坡。'},
    {'section': '古文', 'title': '舊宅書齋記', 'genre': '記',
     'text': clean(prose['ji']['text']),
     'note': '仿《項脊軒志》筆意而不襲其句：由軒之破舊寫到祖母、母親舊事，物是人非之感自細節中來。'},
    {'section': '古文', 'title': '秋日山亭雅集序', 'genre': '序',
     'text': clean(prose['xu']['text']),
     'note': '寫秋日山亭、松竹丹楓，繼以觴詠之樂、聚散之悲，結構與《蘭亭集序》相類而辭句自出。'},
    {'section': '古文', 'title': '文以載道論', 'genre': '論',
     'text': clean(prose['lun']['text']),
     'note': '起以「文章之道，非徒炫辭繪藻」，中間以築室於沙喻無本之文，收於讀經涵泳，桐城派義法可見。'},
    {'section': '賦', 'title': '秋江遠遊賦', 'genre': '賦',
     'text': clean(evmap['fu'][0]),
     'note': '騷體句式，「兮」字往復，寫秋江渺瀰與遠遊之思，託意於鮫人、湘渚、玄圃，辭采可觀。'},
]

out = {
    'title': 'Qwen3.5 9B 詩文專家模型選',
    'subtitle': '古典詩文創作選集',
    'meta': {
        '模型': 'Qwen3.5-9B（基座）＋ 古典詩文 SFT（LoRA，600 步）',
        '訓練數據': '擬作詩 15,000 · 真實詩話評點 13,005 · teacher 評點/鑑賞 · 明清散文仿寫 · 擴寫 376',
        '生成方式': 'Best-of-N（按句數/句長/韻部/抄襲檢測打分）或單次採樣',
        '日期': '2026-11-07',
    },
    'preface': ('此集所收，皆 Qwen3.5-9B 經古典詩文微調後所作。詩取五言、七律、絕句，'
                '詞擬念奴嬌，文有記、序、論，賦擬騷體，各附編者評點。'
                '所選皆經韻部與抄襲檢測（與三十一萬首詩文語料比對，八言重複率均低於 0.01），'
                '間有平仄未諧處，讀者取其意趣可也。'),
    'pieces': pieces,
}

json.dump(out, open(r'C:\Users\wangy\Documents\deepseek-harness\default-workspace\reports\book_pieces.json',
                    'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('pieces:', len(pieces))
for p in pieces:
    print(p['title'], len(p['text']))

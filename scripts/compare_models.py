"""多厂商最强模型对照测试：同一 prompt 生成古典散文/诗，输出对比。"""
import json
import os
import time
import urllib.error
import urllib.request

import yaml

CRED = os.path.expanduser('~/.dsh/.credentials.yaml')
REF = yaml.safe_load(open(CRED, encoding='utf-8'))['refs']

PROMPTS = {
    'prose': ('请以韩愈的笔法，写一篇题为《师说新解》的论，约 400 字古文（文言）。'
              '要有议论锋芒，不要抄名篇原句，不要现代白话。'),
    'poem': '【唐·王维·五律·山水】请作五言律诗一首，写山中傍晚的静寂与禅意。',
}

CANDIDATES = [
    # (label, base_url, model, key_ref, extra_body)
    ('GLM-5.3', 'https://api.z.ai/api/coding/paas/v4/chat/completions', 'glm-5.3',
     'ZAI_CODING_CN_API_KEY', {'thinking': {'type': 'disabled'}}),
    ('MiMo-V2.6-Pro', 'https://api.xiaomimimo.com/v1/chat/completions', 'mimo-v2.6-pro',
     'XIAOMI_TOKEN_PLAN_CN_API_KEY', {}),
    ('MiMo-V2.5-Pro', 'https://api.xiaomimimo.com/v1/chat/completions', 'mimo-v2.5-pro',
     'XIAOMI_TOKEN_PLAN_CN_API_KEY', {}),
    ('MiniMax-M3', 'https://api.minimax.cn/v1/chat/completions', 'MiniMax-M3',
     'MINIMAX1_API_KEY', {}),
    ('K3.1-Pro', 'https://api.kimi.com/coding/v1/chat/completions', 'k3.1-pro',
     'KIMI_API_KEY', {'reasoning_effort': 'low'}),
    ('K2.8-Preview', 'https://api.kimi.com/coding/v1/chat/completions', 'kimi-for-coding',
     'KIMI_API_KEY', {'thinking': {'type': 'disabled'}}),
    ('DeepSeek-V4.1-Flash', 'https://api.deepseek.com/chat/completions', 'deepseek-flash',
     'DEEPSEEK_API_KEY', {'thinking': {'type': 'disabled'}}),
]


def call(base_url, model, key, extra, prompt, max_tokens=2600, timeout=300):
    body = {'model': model,
            'messages': [{'role': 'system', 'content': '你是精通中国古典诗文的写作助手。只输出正文。'},
                         {'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens, 'stream': False}
    body.update(extra)
    data = json.dumps(body, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(base_url, data=data,
                                 headers={'Authorization': f'Bearer {key}',
                                          'Content-Type': 'application/json'},
                                 method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.loads(r.read().decode())
    msg = obj['choices'][0]['message']
    return (msg.get('content') or '').strip()


def main():
    out = {}
    for label, url, model, keyref, extra in CANDIDATES:
        key = REF.get(keyref)
        if not key:
            print(f'SKIP {label}: no key {keyref}', flush=True)
            continue
        out[label] = {}
        for pname, prompt in PROMPTS.items():
            t0 = time.time()
            try:
                txt = call(url, model, key, extra, prompt)
                out[label][pname] = {'ok': True, 'text': txt, 'elapsed': round(time.time() - t0, 1)}
                print(f'OK {label} / {pname} ({len(txt)} chars, {time.time()-t0:.0f}s)', flush=True)
            except urllib.error.HTTPError as e:
                body = e.read().decode('utf-8', 'replace')[:200]
                out[label][pname] = {'ok': False, 'error': f'HTTP {e.code}: {body}'}
                print(f'FAIL {label} / {pname}: HTTP {e.code} {body[:120]}', flush=True)
            except Exception as e:
                out[label][pname] = {'ok': False, 'error': f'{type(e).__name__}: {str(e)[:150]}'}
                print(f'FAIL {label} / {pname}: {type(e).__name__} {str(e)[:120]}', flush=True)
    path = r'D:\train\runs\model_compare.json'
    json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('saved', path, flush=True)


if __name__ == '__main__':
    main()

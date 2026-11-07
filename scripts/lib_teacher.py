"""DeepSeek teacher 客户端（V4.1-Flash / deepseek-flash）。

key 读取顺序：环境变量 DEEPSEEK_API_KEY → ~/.dsh/.credentials.yaml 的 refs.DEEPSEEK_API_KEY。
直连 api.deepseek.com 即可（本机已验证直连 200）。
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

ENDPOINT = 'https://api.deepseek.com/chat/completions'
MODEL = 'deepseek-flash'
CRED = os.path.expanduser(r'~\.dsh\.credentials.yaml')


def _read_key():
    key = os.environ.get('DEEPSEEK_API_KEY')
    if key:
        return key.strip()
    if os.path.exists(CRED):
        txt = open(CRED, encoding='utf-8', errors='replace').read()
        m = re.search(r'DEEPSEEK_API_KEY:\s*(\S+)', txt)
        if m:
            return m.group(1).strip()
    raise RuntimeError('DEEPSEEK_API_KEY not found')


class Teacher:
    def __init__(self, model=MODEL, timeout=180):
        self.key = _read_key()
        self.model = model
        self.timeout = timeout
        self.usage = {'prompt': 0, 'completion': 0, 'calls': 0}

    def chat(self, messages, temperature=0.8, max_tokens=8000, effort='low', thinking='enabled', retries=3):
        body = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': False,
            'thinking': {'type': thinking},
        }
        if effort and thinking != 'disabled':
            body['reasoning_effort'] = effort  # low/high/max
        data = json.dumps(body, ensure_ascii=False).encode('utf-8')
        last_err = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    ENDPOINT, data=data,
                    headers={'Authorization': f'Bearer {self.key}',
                             'Content-Type': 'application/json'},
                    method='POST')
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    obj = json.loads(resp.read().decode('utf-8'))
                msg = obj['choices'][0]['message']
                text = msg.get('content') or ''
                u = obj.get('usage') or {}
                self.usage['prompt'] += u.get('prompt_tokens', 0)
                self.usage['completion'] += u.get('completion_tokens', 0)
                self.usage['calls'] += 1
                return {'text': text, 'usage': u, 'finish': obj['choices'][0].get('finish_reason')}
            except Exception as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f'teacher call failed after {retries}: {last_err}')

    def generate(self, system, user, **kw):
        return self.chat([
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ], **kw)

"""构建全量语料 8-gram 指纹库（CRC32），用于背诵/抄袭检测。

输出：D:\train\data\cleaned\poems_8gram_crc32.npy（sorted uint32）
"""
import json
import re
import zlib

import numpy as np

POEMS = r'D:\train\data\cleaned\poems_v1.jsonl'
PROSE = r'D:\train\data\cleaned\prose_mingqing_v1.jsonl'
OUT = r'D:\train\data\cleaned\poems_8gram_crc32.npy'
N = 8

try:
    from opencc import OpenCC
    cc = OpenCC('t2s')
    conv = cc.convert
except Exception:
    conv = lambda x: x  # noqa: E731


def norm(t):
    t = conv(t)
    return re.sub(r'[^\u4e00-\u9fff]', '', t)


def grams_of(text):
    out = []
    for i in range(len(text) - N + 1):
        out.append(zlib.crc32(text[i:i + N].encode('utf-8')))
    return out


def main():
    vals = []
    n_docs = 0
    for path in (POEMS, PROSE):
        with open(path, encoding='utf-8') as f:
            for line in f:
                try:
                    t = norm(json.loads(line).get('text', ''))
                except Exception:
                    continue
                n_docs += 1
                if n_docs % 20000 == 0:
                    print('docs:', n_docs, 'grams:', len(vals), flush=True)
                vals.extend(grams_of(t))
    arr = np.unique(np.asarray(vals, dtype=np.uint32))
    np.save(OUT, arr)
    print('docs:', n_docs, '| grams raw:', len(vals), '| unique:', len(arr), flush=True)
    print('saved:', OUT, flush=True)


if __name__ == '__main__':
    main()

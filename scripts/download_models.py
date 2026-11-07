"""从 ModelScope 下载 Qwen3.5-4B / 9B（国内直连，无需代理）。"""
import os
import time
from modelscope import snapshot_download

os.makedirs(r'D:\train\models', exist_ok=True)
for repo in ['Qwen/Qwen3.5-4B', 'Qwen/Qwen3.5-9B']:
    local = os.path.join(r'D:\train\models', repo.split('/')[-1])
    print('start', repo, '->', local, flush=True)
    t0 = time.time()
    p = snapshot_download(repo, local_dir=local, revision='master')
    files = []
    total = 0
    for root, dirs, fs in os.walk(local):
        for f in fs:
            fp = os.path.join(root, f)
            total += os.path.getsize(fp)
            files.append(os.path.relpath(fp, local))
    print(f'DONE {repo} in {time.time()-t0:.0f}s | files={len(files)} | {total/1e9:.2f}GB', flush=True)
    for f in sorted(files):
        print('  ', f, flush=True)

"""修复 Encoding 提取（BatchEncoding 不是 dict 子类）。"""
HELPER = '''
def _extract_ids(enc):
    if hasattr(enc, 'ids'):
        return enc.ids
    try:
        if 'input_ids' in enc:
            return enc['input_ids']
    except Exception:
        pass
    if isinstance(enc, list) and enc and hasattr(enc[0], 'ids'):
        return enc[0].ids
    return enc
'''

# sample_model.py
p = r'D:\train\scripts\sample_model.py'
s = open(p, encoding='utf-8').read()
if '_extract_ids' not in s:
    # 在 decode 定义前插入 helper
    s = s.replace("def decode(ids):", HELPER + "\ndef decode(ids):")
    old = """    if hasattr(enc, 'ids'):
        ids = enc.ids
    elif isinstance(enc, dict):
        ids = enc['input_ids']
    else:
        ids = enc
    ids = torch.tensor([ids], device='cuda')"""
    new = """    ids = _extract_ids(enc)
    ids = torch.tensor([ids], device='cuda')"""
    s = s.replace(old, new)
    open(p, 'w', encoding='utf-8').write(s)
    print('sample_model patched v2')
else:
    print('sample_model already v2')

# train_sft.py
p = r'D:\train\scripts\train_sft.py'
s = open(p, encoding='utf-8').read()
if '_extract_ids' not in s:
    s = s.replace("def build_example(", HELPER + "\ndef build_example(")
    old = """        prompt_ids = prompt_enc.ids if hasattr(prompt_enc, 'ids') else (prompt_enc['input_ids'] if isinstance(prompt_enc, dict) else prompt_enc)
        full_ids = full_enc.ids if hasattr(full_enc, 'ids') else (full_enc['input_ids'] if isinstance(full_enc, dict) else full_enc)"""
    new = """        prompt_ids = _extract_ids(prompt_enc)
        full_ids = _extract_ids(full_enc)"""
    s = s.replace(old, new)
    open(p, 'w', encoding='utf-8').write(s)
    print('train_sft patched v2')
else:
    print('train_sft already v2')

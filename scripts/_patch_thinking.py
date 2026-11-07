"""补丁：chat template 关闭 thinking（enable_thinking=False），并兼容不支持该参数的情况。"""
import re

# ---- sample_model.py ----
p = r'D:\train\scripts\sample_model.py'
s = open(p, encoding='utf-8').read()
if 'enable_thinking' not in s:
    s = s.replace("""    if hasattr(tok, 'apply_chat_template'):
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    else:""",
"""    if hasattr(tok, 'apply_chat_template'):
        try:
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                          enable_thinking=False)
        except TypeError:
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    else:""")
    open(p, 'w', encoding='utf-8').write(s)
    print('sample_model thinking-off patched')
else:
    print('sample_model already patched')

# ---- train_sft.py ----
p = r'D:\train\scripts\train_sft.py'
s = open(p, encoding='utf-8').read()
if 'enable_thinking' not in s:
    s = s.replace("""        prompt_enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        msgs.append({'role': 'assistant', 'content': out})
        full_enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)""",
"""        try:
            prompt_enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=False)
            msgs.append({'role': 'assistant', 'content': out})
            full_enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                               enable_thinking=False)
        except TypeError:
            prompt_enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
            msgs.append({'role': 'assistant', 'content': out})
            full_enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)""")
    open(p, 'w', encoding='utf-8').write(s)
    print('train_sft thinking-off patched')
else:
    print('train_sft already patched')

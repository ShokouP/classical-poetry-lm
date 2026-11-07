# 原生 Windows 训练环境安装（后台运行）
$ErrorActionPreference = 'Continue'
$log = 'D:\train\setup.log'
function Log($msg) { $msg | Tee-Object -FilePath $log -Append }
Log "=== setup start $(Get-Date -Format o) ==="

$sysPy = 'C:\Users\wangy\AppData\Local\Programs\Python\Python312\python.exe'
$venv = 'D:\train\venv'
$vpy = Join-Path $venv 'Scripts\python.exe'

if (-not (Test-Path $vpy)) {
    Log 'creating venv...'
    & $sysPy -m venv $venv 2>&1 | Tee-Object -FilePath $log -Append
}
Log "python: $(& $vpy --version 2>&1)"

# 代理（FlClash）
$env:HTTPS_PROXY = 'http://127.0.0.1:7890'
$env:HTTP_PROXY = 'http://127.0.0.1:7890'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

Log '=== pip upgrade ==='
& $vpy -m pip install -q -U pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | Tee-Object -FilePath $log -Append

Log '=== connectivity: download.pytorch.org ==='
$c1 = & curl.exe -sS -m 20 -o NUL -w '%{http_code}' 'https://download.pytorch.org/whl/cu128/' 2>&1
Log "direct => $c1"
$c2 = & curl.exe -sS -m 20 -x 'http://127.0.0.1:7890' -o NUL -w '%{http_code}' 'https://download.pytorch.org/whl/cu128/' 2>&1
Log "proxy  => $c2"

Log '=== install torch (cuda) ==='
& $vpy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 2>&1 | Tee-Object -FilePath $log -Append

Log '=== install training stack ==='
& $vpy -m pip install transformers accelerate datasets peft trl bitsandbytes sentencepiece protobuf tensorboard huggingface_hub hf_transfer safetensors pyyaml fire tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | Tee-Object -FilePath $log -Append

Log '=== install llamafactory ==='
& $vpy -m pip install llamafactory -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | Tee-Object -FilePath $log -Append

Log '=== verify ==='
& $vpy -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')" 2>&1 | Tee-Object -FilePath $log -Append
& $vpy -c "import transformers, peft, trl, bitsandbytes; print('transformers', transformers.__version__, '| peft', peft.__version__, '| trl', trl.__version__, '| bnb', bitsandbytes.__version__)" 2>&1 | Tee-Object -FilePath $log -Append
Log "=== setup done $(Get-Date -Format o) ==="

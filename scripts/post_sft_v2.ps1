# Post-SFT-v2 chain: wait for adapter_model.safetensors in sft_4b_v2, then eval -> expand -> merge -> GGUF -> quantize -> Ollama.
$ErrorActionPreference = 'Continue'
$log = 'D:\train\runs\post_sft_v2.log'
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Tee-Object -FilePath $log -Append }
$vpy = 'D:\train\venv\Scripts\python.exe'
$base = 'D:\train\models\Qwen3.5-4B'
$adapter = 'D:\train\models\sft_4b_v2'
$merged = 'D:\train\models\sft_4b_v2_merged'
$bf16 = 'D:\train\models\poetry-4b-v2-bf16.gguf'
$q4 = 'D:\train\models\poetry-4b-v2-Q4_K_M.gguf'
$env:NO_PROXY = ''; $env:no_proxy = ''; $env:HTTPS_PROXY = ''; $env:HTTP_PROXY = ''
$env:PYTHONUNBUFFERED = '1'

Log 'POST_V2_START'
$deadline = (Get-Date).AddMinutes(150)
$done = $false
while ((Get-Date) -lt $deadline) {
    if (Test-Path (Join-Path $adapter 'adapter_model.safetensors')) {
        Start-Sleep -Seconds 25
        $used = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -split "`n" | Select-Object -First 1)
        Log "adapter found; gpu_used=${used}MiB"
        if ($used -lt 9000) { $done = $true; break }
    }
    Start-Sleep -Seconds 30
}
if (-not $done) { Log 'WAIT_TIMEOUT'; exit 1 }
Log 'SFT_V2_DONE'

Log 'EVAL_START'
& $vpy 'D:\train\scripts\eval_model.py' --base $base --adapter $adapter --out 'D:\train\runs\eval_sft_v2.json' *> 'D:\train\runs\eval_sft_v2.log'
Log "EVAL_EXIT=$LASTEXITCODE"

Log 'EXPAND_START'
& $vpy 'D:\train\scripts\expand_iter.py' --base $base --adapter $adapter --input 'D:\train\runs\expand_input.txt' --out 'D:\train\runs\expand_sft_v2.json' *> 'D:\train\runs\expand_sft_v2.log'
Log "EXPAND_EXIT=$LASTEXITCODE"

Log 'MERGE_START'
& $vpy 'D:\train\scripts\merge_lora.py' --base $base --adapter $adapter --output $merged *> 'D:\train\runs\merge-sft-v2.log'
Log "MERGE_EXIT=$LASTEXITCODE"

$conv = 'D:\llama\llama.cpp\convert_hf_to_gguf.py'
if ((Test-Path $merged) -and (Test-Path $conv) -and -not (Test-Path $bf16)) {
    Log 'CONVERT_START'
    & $vpy $conv $merged --outfile $bf16 --outtype bf16 *> 'D:\train\runs\convert-sft-v2.log'
    Log "CONVERT_EXIT=$LASTEXITCODE"
}

if ((Test-Path $bf16) -and -not (Test-Path $q4)) {
    Log 'QUANTIZE_START'
    & 'D:\llama\b12110\llama-quantize.exe' $bf16 $q4 Q4_K_M *> 'D:\train\runs\quant-sft-v2.log'
    Log "QUANTIZE_EXIT=$LASTEXITCODE"
}

if (Test-Path $q4) {
    $mf = 'D:\train\runs\Modelfile.poetry-4b-v2'
    ("FROM " + $q4 + "`nPARAMETER num_ctx 8192`nPARAMETER temperature 0.9`nPARAMETER top_p 0.95`n") | Out-File -FilePath $mf -Encoding ascii
    Log 'OLLAMA_CREATE'
    & ollama create 'poetry-4b-v2' -f $mf *> 'D:\train\runs\ollama-create-v2.log'
    Log "OLLAMA_CREATE_EXIT=$LASTEXITCODE"
}
Log 'POST_V2_DONE'

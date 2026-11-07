# Post-SFT chain: wait for SFT completion -> eval -> expansion demo -> merge -> GGUF -> quantize -> Ollama.
# ASCII-only to avoid PowerShell 5.1 encoding issues.
$ErrorActionPreference = 'Continue'
$log = 'D:\train\runs\post_sft.log'
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Tee-Object -FilePath $log -Append }
$vpy = 'D:\train\venv\Scripts\python.exe'
$base = 'D:\train\models\Qwen3.5-4B'
$adapter = 'D:\train\models\sft_4b_v1'
$sftLog = 'D:\train\runs\sft_4b_v1.log'
$merged = 'D:\train\models\sft_4b_v1_merged'
$bf16 = 'D:\train\models\poetry-4b-sft-bf16.gguf'
$q4 = 'D:\train\models\poetry-4b-sft-Q4_K_M.gguf'
$env:NO_PROXY = ''; $env:no_proxy = ''; $env:HTTPS_PROXY = ''; $env:HTTP_PROXY = ''
$env:PYTHONUNBUFFERED = '1'

Log 'POST_SFT_START'

# 1) wait for SFT to finish
$deadline = (Get-Date).AddMinutes(90)
$done = $false
while ((Get-Date) -lt $deadline) {
    if (Test-Path $sftLog) {
        $tail = Get-Content $sftLog -Tail 6 -Raw -ErrorAction SilentlyContinue
        if ($tail -cmatch 'SAVED D:\\train\\models\\sft_4b_v1') {
            Start-Sleep -Seconds 25
            $used = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -split "`n" | Select-Object -First 1)
            Log "SFT log done; gpu_used=${used}MiB"
            if ($used -lt 9000) { $done = $true; break }
        }
    }
    Start-Sleep -Seconds 30
}
if (-not $done) { Log 'SFT_WAIT_TIMEOUT'; exit 1 }
Log 'SFT_DONE'

# 2) eval SFT
Log 'EVAL_SFT_START'
& $vpy 'D:\train\scripts\eval_model.py' --base $base --adapter $adapter --out 'D:\train\runs\eval_sft.json' *> 'D:\train\runs\eval_sft.log'
Log "EVAL_SFT_EXIT=$LASTEXITCODE"

# 3) expansion demo with SFT model
Log 'EXPAND_SFT_START'
& $vpy 'D:\train\scripts\expand_iter.py' --base $base --adapter $adapter --input 'D:\train\runs\expand_input.txt' --out 'D:\train\runs\expand_sft.json' *> 'D:\train\runs\expand_sft.log'
Log "EXPAND_SFT_EXIT=$LASTEXITCODE"

# 4) merge LoRA (CPU)
Log 'MERGE_START'
& $vpy 'D:\train\scripts\merge_lora.py' --base $base --adapter $adapter --output $merged *> 'D:\train\runs\merge-sft.log'
Log "MERGE_EXIT=$LASTEXITCODE"

# 5) convert to bf16 GGUF
$conv = 'D:\llama\llama.cpp\convert_hf_to_gguf.py'
if ((Test-Path $merged) -and (Test-Path $conv) -and -not (Test-Path $bf16)) {
    Log 'CONVERT_START'
    & $vpy $conv $merged --outfile $bf16 --outtype bf16 *> 'D:\train\runs\convert-sft.log'
    Log "CONVERT_EXIT=$LASTEXITCODE"
}

# 6) quantize Q4_K_M
if ((Test-Path $bf16) -and -not (Test-Path $q4)) {
    Log 'QUANTIZE_START'
    & 'D:\llama\b12110\llama-quantize.exe' $bf16 $q4 Q4_K_M *> 'D:\train\runs\quant-sft.log'
    Log "QUANTIZE_EXIT=$LASTEXITCODE"
}

# 7) Ollama create
if (Test-Path $q4) {
    $mf = 'D:\train\runs\Modelfile.poetry-4b-sft'
    ("FROM " + $q4 + "`nPARAMETER num_ctx 8192`nPARAMETER temperature 0.9`nPARAMETER top_p 0.95`n") | Out-File -FilePath $mf -Encoding ascii
    Log 'OLLAMA_CREATE'
    & ollama create 'poetry-4b-sft' -f $mf *> 'D:\train\runs\ollama-create-sft.log'
    Log "OLLAMA_CREATE_EXIT=$LASTEXITCODE"
}

Log 'POST_SFT_DONE'

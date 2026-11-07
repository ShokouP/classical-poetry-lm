param(
    [Parameter(Mandatory = $true)][string]$Adapter,
    [Parameter(Mandatory = $true)][string]$Name
)
# Export chain: merge LoRA -> bf16 GGUF -> Q4_K_M -> Ollama model. ASCII-only to avoid PS 5.1 encoding issues.
$ErrorActionPreference = 'Continue'
$log = "D:\train\runs\export-$Name.log"
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Tee-Object -FilePath $log -Append }
$vpy = 'D:\train\venv\Scripts\python.exe'
$base = 'D:\train\models\Qwen3.5-4B'
$merged = "D:\train\models\${Name}_merged"
$bf16 = "D:\train\models\$Name-bf16.gguf"
$q4 = "D:\train\models\$Name-Q4_K_M.gguf"
$env:NO_PROXY = ''; $env:no_proxy = ''; $env:HTTPS_PROXY = ''; $env:HTTP_PROXY = ''
$env:PYTHONUNBUFFERED = '1'

Log "EXPORT_START name=$Name adapter=$Adapter"

if (-not (Test-Path (Join-Path $merged 'config.json'))) {
    Log 'MERGE_START'
    & $vpy 'D:\train\scripts\merge_lora.py' --base $base --adapter $Adapter --output $merged *> "D:\train\runs\merge-$Name.log"
    Log "MERGE_EXIT=$LASTEXITCODE"
} else { Log 'MERGE_SKIP' }

if (-not (Test-Path $bf16)) {
    Log 'CONVERT_START'
    & $vpy 'D:\llama\llama.cpp\convert_hf_to_gguf.py' $merged --outfile $bf16 --outtype bf16 *> "D:\train\runs\convert-$Name.log"
    Log "CONVERT_EXIT=$LASTEXITCODE"
} else { Log 'CONVERT_SKIP' }

if (-not (Test-Path $bf16)) { Log 'NO_BF16_ABORT'; exit 1 }

if (-not (Test-Path $q4)) {
    Log 'QUANTIZE_START'
    & 'D:\llama\b12110\llama-quantize.exe' $bf16 $q4 Q4_K_M *> "D:\train\runs\quant-$Name.log"
    Log "QUANTIZE_EXIT=$LASTEXITCODE"
} else { Log 'QUANTIZE_SKIP' }

if (-not (Test-Path $q4)) { Log 'NO_Q4_ABORT'; exit 1 }

$mf = Join-Path $merged 'Modelfile.ollama'
("FROM " + $q4 + "`nPARAMETER num_ctx 8192`nPARAMETER temperature 0.9`nPARAMETER top_p 0.95`n") | Out-File -FilePath $mf -Encoding ascii
Log 'OLLAMA_CREATE'
& ollama create $Name -f $mf *> "D:\train\runs\ollama-create-$Name.log"
Log "OLLAMA_CREATE_EXIT=$LASTEXITCODE"

if ($LASTEXITCODE -eq 0) {
    Log 'OLLAMA_TEST'
    $body = '{"model":"' + $Name + '","messages":[{"role":"user","content":"hello, reply in one short sentence"}],"stream":false,"think":false,"options":{"num_predict":64}}'
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
    try {
        $r = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/chat' -Method Post -ContentType 'application/json; charset=utf-8' -Body $bytes -TimeoutSec 600
        Log ('SAMPLE_LEN=' + $r.message.content.Length)
    } catch { Log "OLLAMA_TEST_FAIL $($_.Exception.Message)" }
}
Log "EXPORT_DONE name=$Name"

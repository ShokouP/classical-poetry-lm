# 自动流水线（修正版）：CPT 完成 -> eval CPT -> SFT -> eval SFT -> merge
$ErrorActionPreference = 'Continue'
$log = 'D:\train\runs\pipeline.log'
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Tee-Object -FilePath $log -Append }
$vpy = 'D:\train\venv\Scripts\python.exe'
$env:NO_PROXY = ''; $env:no_proxy = ''; $env:HTTPS_PROXY = ''; $env:HTTP_PROXY = ''
$env:PYTHONUNBUFFERED = '1'
$cptLog = 'D:\train\runs\cpt_4b_resume.log'

Log 'PIPELINE2_START'

# 1) 等 CPT 真正完成：日志尾部出现 "300/300"，且 GPU 占用降到 <9GB
$deadline = (Get-Date).AddMinutes(120)
$done = $false
while ((Get-Date) -lt $deadline) {
    if (Test-Path $cptLog) {
        $tail = Get-Content $cptLog -Tail 6 -Raw -ErrorAction SilentlyContinue
        if ($tail -cmatch '300/300' -or $tail -cmatch 'SAVED D:\\') {
            Start-Sleep -Seconds 20
            $used = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -split "`n" | Select-Object -First 1)
            Log "CPT log done, gpu_used=${used}MiB"
            if ($used -lt 9000) { $done = $true; break }
        }
    }
    Start-Sleep -Seconds 30
}
if (-not $done) { Log 'CPT_WAIT_TIMEOUT'; exit 1 }
Log 'CPT_DONE'

# 2) eval CPT
Log 'EVAL_CPT_START'
& $vpy 'D:\train\scripts\eval_model.py' --base 'D:\train\models\Qwen3.5-4B' --adapter 'D:\train\models\cpt_4b_v1' --out 'D:\train\runs\eval_cpt.json' *> 'D:\train\runs\eval_cpt.log'
Log "EVAL_CPT_EXIT=$LASTEXITCODE"

# 3) SFT 训练（16k 子集，1 epoch，最多 1000 步）
Log 'SFT_TRAIN_START'
& $vpy 'D:\train\scripts\train_sft.py' --model 'D:\train\models\Qwen3.5-4B' --data 'D:\train\data\sft\sft_v2_train.jsonl' --output 'D:\train\models\sft_4b_v1' --seq 2048 --epochs 1 --max-steps 1000 --max-samples 16000 --save-steps 200 *> 'D:\train\runs\sft_4b_v1.log'
Log "SFT_TRAIN_EXIT=$LASTEXITCODE"

# 4) eval SFT
Log 'EVAL_SFT_START'
& $vpy 'D:\train\scripts\eval_model.py' --base 'D:\train\models\Qwen3.5-4B' --adapter 'D:\train\models\sft_4b_v1' --out 'D:\train\runs\eval_sft.json' *> 'D:\train\runs\eval_sft.log'
Log "EVAL_SFT_EXIT=$LASTEXITCODE"

# 5) merge LoRA（CPU）
Log 'MERGE_START'
& $vpy 'D:\train\scripts\merge_lora.py' --base 'D:\train\models\Qwen3.5-4B' --adapter 'D:\train\models\sft_4b_v1' --output 'D:\train\models\sft_4b_v1_merged' *> 'D:\train\runs\merge.log'
Log "MERGE_EXIT=$LASTEXITCODE"

Log 'PIPELINE_DONE'

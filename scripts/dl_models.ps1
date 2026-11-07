# 从 ModelScope 用 curl 多线程下载 Qwen3.5-4B / 9B（可断点续传）
$ErrorActionPreference = 'Continue'
$env:HTTPS_PROXY = ''; $env:HTTP_PROXY = ''; $env:NO_PROXY = ''; $env:no_proxy = ''
$log = 'D:\train\dl_models.log'
function Log($m) { $m | Tee-Object -FilePath $log -Append }

$repos = @(
    @{ repo = 'Qwen/Qwen3.5-4B'; dir = 'D:\train\models\Qwen3.5-4B' },
    @{ repo = 'Qwen/Qwen3.5-9B'; dir = 'D:\train\models\Qwen3.5-9B' }
)

foreach ($r in $repos) {
    Log "=== $($r.repo) start $(Get-Date -Format o) ==="
    New-Item -ItemType Directory -Force $r.dir | Out-Null
    Get-ChildItem $r.dir -Filter '*.incomplete' -ErrorAction SilentlyContinue | Remove-Item -Force
    $api = "https://www.modelscope.cn/api/v1/models/$($r.repo)/repo/files?Revision=master&Recursive=true"
    $json = & curl.exe -sS -m 60 $api 2>$null
    $files = ($json | ConvertFrom-Json).Data.Files
    if (-not $files) { Log "file list failed for $($r.repo)"; continue }
    Log "files: $($files.Count)"
    $procs = @()
    foreach ($f in $files) {
        $local = Join-Path $r.dir $f.Name
        $parent = Split-Path $local -Parent
        if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Force $parent | Out-Null }
        if ((Test-Path $local) -and ((Get-Item $local).Length -eq $f.Size)) {
            Log "skip (done) $($f.Name)"
            continue
        }
        $url = "https://www.modelscope.cn/models/$($r.repo)/resolve/master/$($f.Name)"
        $args = @('-sS', '-L', '--retry', '10', '--retry-all-errors', '-o', $local)
        if (Test-Path $local) { $args += @('-C', '-') }
        $args += $url
        Log "start $($f.Name) ($([math]::Round($f.Size/1MB)) MB)"
        $procs += Start-Process -FilePath 'curl.exe' -ArgumentList $args -PassThru -WindowStyle Hidden
    }
    foreach ($p in $procs) { $p.WaitForExit() }
    $bad = 0
    foreach ($f in $files) {
        $local = Join-Path $r.dir $f.Name
        if (-not (Test-Path $local) -or ((Get-Item $local).Length -ne $f.Size)) {
            Log "SIZE MISMATCH $($f.Name) expected=$($f.Size) actual=$((Get-Item $local -ErrorAction SilentlyContinue).Length)"
            $bad++
        }
    }
    Log "=== $($r.repo) DONE bad=$bad ==="
}
Log "ALL DONE $(Get-Date -Format o)"

# Runs one daily experiment through the Cursor CLI agent. Triggered by the
# "ai-infra-lab daily experiment" scheduled task at 14:00.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:HTTP_PROXY = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = "http://127.0.0.1:7897"
$env:NO_PROXY = "localhost,127.0.0.1"
$env:PYTHONIOENCODING = "utf-8"

$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir ("{0}.log" -f (Get-Date -Format "yyyy-MM-dd_HHmm"))
$lock = Join-Path $logDir "daily.lock"

if ((Test-Path $lock) -and ((Get-Date) - (Get-Item $lock).LastWriteTime).TotalHours -lt 6) {
    "Another run is in progress (lock: $lock); exiting." | Tee-Object -FilePath $log -Append
    exit 0
}
New-Item -ItemType File -Force $lock | Out-Null

try {
    # Windows PowerShell turns native stderr (git progress, agent logs) into
    # terminating errors under "Stop".
    $ErrorActionPreference = "Continue"
    "Started $(Get-Date -Format s)" | Tee-Object -FilePath $log -Append
    $agent = Join-Path $env:LOCALAPPDATA "cursor-agent\agent.cmd"
    git pull --ff-only origin main *>&1 | Tee-Object -FilePath $log -Append
    $prompt = "Follow automation/daily_prompt.md exactly to run today's experiment, commit step by step, and push to origin main."
    & $agent -p --force --trust --workspace $root --output-format text $prompt *>&1 |
        Tee-Object -FilePath $log -Append
}
finally {
    Remove-Item $lock -ErrorAction SilentlyContinue
}

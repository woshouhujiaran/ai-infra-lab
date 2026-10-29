# Runs one daily experiment through the Cursor CLI agent. Triggered by the
# "ai-infra-lab daily experiment" scheduled task at 14:00.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$proxyHost = "127.0.0.1"
$proxyPort = 7897
$env:HTTP_PROXY = "http://${proxyHost}:${proxyPort}"
$env:HTTPS_PROXY = "http://${proxyHost}:${proxyPort}"
$env:NO_PROXY = "localhost,127.0.0.1"
$env:PYTHONIOENCODING = "utf-8"
$clashExe = "C:\Program Files\Clash Verge\clash-verge.exe"
$maxAttempts = 3
$retryDelayMinutes = 10

$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir ("{0}.log" -f (Get-Date -Format "yyyy-MM-dd_HHmm"))
$lock = Join-Path $logDir "daily.lock"

function Write-Log($message) {
    "[$(Get-Date -Format s)] $message" | Tee-Object -FilePath $log -Append | Out-Host
}

function Test-Proxy {
    $client = New-Object System.Net.Sockets.TcpClient
    try { return $client.ConnectAsync($proxyHost, $proxyPort).Wait(2000) -and $client.Connected }
    catch { return $false }
    finally { $client.Dispose() }
}

# After sleep or a fresh logon the proxy may not be up yet.
function Wait-Proxy {
    if (Test-Proxy) { return $true }
    Write-Log "Proxy ${proxyHost}:${proxyPort} is not listening."
    if (-not (Get-Process clash-verge -ErrorAction SilentlyContinue) -and (Test-Path $clashExe)) {
        Write-Log "Starting Clash Verge."
        Start-Process $clashExe
    }
    $deadline = (Get-Date).AddMinutes(5)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 10
        if (Test-Proxy) { Write-Log "Proxy is up."; return $true }
    }
    Write-Log "Proxy still unavailable after 5 minutes."
    return $false
}

if ((Test-Path $lock) -and ((Get-Date) - (Get-Item $lock).LastWriteTime).TotalHours -lt 6) {
    "Another run is in progress (lock: $lock); exiting." | Tee-Object -FilePath $log -Append
    exit 0
}
New-Item -ItemType File -Force $lock | Out-Null

try {
    # Windows PowerShell turns native stderr (git progress, agent logs) into
    # terminating errors under "Stop".
    $ErrorActionPreference = "Continue"
    Write-Log "Started"
    $agent = Join-Path $env:LOCALAPPDATA "cursor-agent\agent.cmd"
    $prompt = "Follow automation/daily_prompt.md exactly to run today's experiment, commit step by step, and push to origin main."

    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        if ($attempt -gt 1) {
            Write-Log "Retrying in $retryDelayMinutes minutes."
            Start-Sleep -Seconds ($retryDelayMinutes * 60)
        }
        Write-Log "Attempt $attempt of $maxAttempts"
        if (-not (Wait-Proxy)) { continue }

        git pull --ff-only origin main *>&1 | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) { Write-Log "git pull failed (exit $LASTEXITCODE)."; continue }
        $before = git rev-parse HEAD

        & $agent -p --force --trust --workspace $root --output-format text $prompt *>&1 |
            Tee-Object -FilePath $log -Append
        $agentExit = $LASTEXITCODE

        git fetch origin main *>&1 | Tee-Object -FilePath $log -Append
        $after = git rev-parse HEAD
        $remote = git rev-parse origin/main

        if ($after -eq $before) {
            Write-Log "Agent exited with $agentExit and made no commits."
            continue
        }
        # Partial work is already committed; a blind retry could start a second experiment.
        if ($agentExit -ne 0 -or $after -ne $remote) {
            Write-Log "Agent exited with $agentExit; HEAD $after, origin/main $remote. Needs a manual look."
            exit 1
        }
        Write-Log "Finished: $before..$after pushed to origin/main."
        exit 0
    }
    Write-Log "All $maxAttempts attempts failed."
    exit 1
}
finally {
    Remove-Item $lock -ErrorAction SilentlyContinue
}

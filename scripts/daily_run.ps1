# Runs one daily experiment through the Cursor CLI agent. Triggered by the
# "ai-infra-lab daily experiment" scheduled task at 14:00.
# Resume an interrupted run (chat ID is printed in the log):
#   powershell -ExecutionPolicy Bypass -File scripts\daily_run.ps1 -ResumeChatId <id>
param([string]$ResumeChatId)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# The agent writes UTF-8; the console default (GBK) would garble Chinese text.
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false

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
$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$log = Join-Path $logDir "$stamp.log"
$events = Join-Path $logDir "$stamp.agent.jsonl"
$lock = Join-Path $logDir "daily.lock"

function Write-Log($message) {
    "[$(Get-Date -Format s)] $message" | Tee-Object -FilePath $log -Append | Out-Host
}

function Limit-Text([string]$text, [int]$max = 300) {
    $text = ($text -replace '\s+', ' ').Trim()
    if ($text.Length -gt $max) { return $text.Substring(0, $max) + " ..." }
    return $text
}

# Turns one stream-json event into a readable log line ($null = not worth logging).
function Format-AgentEvent($e) {
    switch ($e.type) {
        "system" { return "agent: session $($e.session_id), model $($e.model)" }
        "assistant" {
            $text = ($e.message.content | Where-Object { $_.type -eq "text" } | ForEach-Object { $_.text }) -join ""
            if ($text.Trim()) { return "agent says: $(Limit-Text $text 1000)" }
            return $null
        }
        "tool_call" {
            $prop = $e.tool_call.PSObject.Properties | Where-Object { $_.Name -like "*ToolCall" } | Select-Object -First 1
            if (-not $prop) { return "tool $($e.subtype)" }
            $name = $prop.Name -replace "ToolCall$", ""
            $call = $prop.Value
            if ($e.subtype -eq "started") {
                $a = $call.args
                $target = if ($a.command) { $a.command } elseif ($a.path) { $a.path } elseif ($a.pattern) { $a.pattern } else { $a | ConvertTo-Json -Compress -Depth 5 }
                return "tool $name > $(Limit-Text $target)"
            }
            $r = $call.result
            if ($r.success) {
                $code = $r.success.exitCode
                if ($null -ne $code -and $code -ne 0) { return "tool $name < exit $code" }
                return "tool $name < ok" + $(if ($null -ne $code) { " (exit $code)" } else { "" })
            }
            return "tool $name < FAILED: $(Limit-Text ($r | ConvertTo-Json -Compress -Depth 6) 500)"
        }
        "result" {
            $status = if ($e.is_error) { "ERROR" } else { $e.subtype }
            return "agent result: $status after $([math]::Round($e.duration_ms / 60000, 1)) min`n$($e.result)"
        }
    }
    return $null
}

# Raw events go to the .jsonl file; a readable summary goes to the .log.
function Write-AgentOutput {
    process {
        $line = "$_"
        if (-not $line.Trim()) { return }
        try { $e = $line | ConvertFrom-Json } catch { $e = $null }
        if ($null -eq $e -or -not $e.type) { Write-Log "agent output: $line"; return }
        [IO.File]::AppendAllText($events, $line + "`n", (New-Object System.Text.UTF8Encoding $false))
        $summary = Format-AgentEvent $e
        if ($summary) { Write-Log $summary }
    }
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
    # Not agent.cmd: the CLI auto-updates mid-run and rewrites agent.cmd, and cmd.exe
    # then resumes the edited batch file at a stale offset and fails with exit 1.
    # Running the .ps1 in-process also keeps the agent's UTF-8 output intact.
    $agent = Join-Path $env:LOCALAPPDATA "cursor-agent\cursor-agent.ps1"
    $today = Get-Date -Format "yyyy-MM-dd"
    $freshPrompt = "Follow automation/daily_prompt.md exactly to run today's experiment, commit step by step, and push to origin main."
    $continuePrompt = "The previous run in this chat stopped before finishing. Check git status, git log and experiments/ to see what is already done, then continue following automation/daily_prompt.md from where it stopped: commit step by step and push to origin main."

    $chatId = $ResumeChatId
    if ($chatId) { Write-Log "Resuming chat $chatId" }

    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        if ($attempt -gt 1) {
            Write-Log "Retrying in $retryDelayMinutes minutes."
            Start-Sleep -Seconds ($retryDelayMinutes * 60)
        }
        Write-Log "Attempt $attempt of $maxAttempts"
        if (-not (Wait-Proxy)) { continue }

        git pull --ff-only origin main *>&1 | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) { Write-Log "git pull failed (exit $LASTEXITCODE)."; continue }
        $registry = Get-Content -Raw -Encoding UTF8 (Join-Path $root "experiments\registry.json") | ConvertFrom-Json
        $done = $registry.experiments | Where-Object { $_.date -eq $today } | Select-Object -First 1
        if ($done) {
            Write-Log "Today's experiment $($done.id)-$($done.slug) is already registered ($($done.status)); nothing to do."
            exit 0
        }
        if (-not $chatId) {
            $chatId = "$(& $agent create-chat 2>$null | Select-Object -Last 1)".Trim()
            if ($chatId -notmatch '^[0-9a-f-]{36}$') { Write-Log "create-chat failed: $chatId"; $chatId = $null; continue }
            Write-Log "Chat $chatId (raw events: $events)"
            Write-Log "To resume manually: powershell -ExecutionPolicy Bypass -File scripts\daily_run.ps1 -ResumeChatId $chatId"
        }
        $before = git rev-parse HEAD

        $prompt = if ($ResumeChatId -or $attempt -gt 1) { $continuePrompt } else { $freshPrompt }
        & $agent -p --force --trust --workspace $root --resume $chatId --output-format stream-json $prompt *>&1 |
            Write-AgentOutput
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
            Write-Log "Agent exited with $agentExit; HEAD $after, origin/main $remote. Needs a manual look; resume with -ResumeChatId $chatId."
            exit 1
        }
        Write-Log "Finished: $before..$after pushed to origin/main."
        exit 0
    }
    Write-Log "All $maxAttempts attempts failed. Resume with -ResumeChatId $chatId."
    exit 1
}
finally {
    Remove-Item $lock -ErrorAction SilentlyContinue
}

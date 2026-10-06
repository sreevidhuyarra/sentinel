# Drift demo (report section 9): Monday's traffic, then Friday's new attack mix.
# Starts the stream detector and the drift watcher in the background, replays both days and
# leaves the services running; Grafana (http://localhost:3000) shows drift rising and the
# retraining run, MLflow (http://localhost:5000) the new model version.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\drift_demo.ps1 [-Flows 120000] [-Rate 500]
param([int]$Flows = 120000, [int]$Rate = 500)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
$env:PYTHONIOENCODING = "utf-8"
$sentinel = ".\.venv\Scripts\sentinel.exe"
New-Item -ItemType Directory -Force reports\demo | Out-Null

function Start-Service($name, $arguments) {
    $log = "reports\demo\$name.log"
    Start-Process $sentinel -ArgumentList $arguments -RedirectStandardOutput $log -RedirectStandardError "$log.err" -WindowStyle Hidden -PassThru
}

# A plain TCP probe on 127.0.0.1: Invoke-WebRequest on "localhost" in Windows PowerShell 5.1
# tries IPv6 and the system proxy first and hung here (the metrics servers listen on IPv4).
function Wait-Metrics($port, $process) {
    Write-Host -NoNewline "waiting for port $port "
    for ($i = 0; $i -lt 200; $i++) {
        if ($process.HasExited) { throw "process $($process.Id) exited; see reports\demo\*.log.err" }
        $client = New-Object System.Net.Sockets.TcpClient
        try {
            if ($client.ConnectAsync("127.0.0.1", $port).Wait(1000)) { Write-Host " up"; return }
        } catch { } finally { $client.Dispose() }
        Write-Host -NoNewline "."
        Start-Sleep 2
    }
    throw "service on port $port did not start"
}

$detector = Start-Service "detector" @("stream", "detect")
$watcher = Start-Service "drift" @("mlops", "drift-watch")
Wait-Metrics 9101 $detector
Wait-Metrics 9103 $watcher
Write-Host "detector (pid $($detector.Id)) and drift watcher (pid $($watcher.Id)) are up"

Write-Host "replaying Monday (benign) ..."
& $sentinel --log-level WARNING stream replay --days monday --rate $Rate --limit $Flows
Write-Host "replaying Friday (DDoS, PortScan, Bot) ..."
& $sentinel --log-level WARNING stream replay --days friday --rate $Rate --limit $Flows
Write-Host "done replaying; detector and drift watcher keep running (stop them with Stop-Process)."

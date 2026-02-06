param(
    [string]$ConfigPath = (Join-Path $PSScriptRoot "..\\otel-collector.yaml"),
    [string]$OtelcolPath = "otelcol"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path $ConfigPath)) {
    Write-Error "Config not found: $ConfigPath"
    exit 1
}

$resolvedConfig = (Resolve-Path $ConfigPath).Path
$otel = $OtelcolPath
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$otelDir = Join-Path $projectRoot ".cache\\otel"
if (-not (Test-Path $otelDir)) {
    New-Item -ItemType Directory -Force -Path $otelDir | Out-Null
}

if (-not $env:OTEL_LOG_PATH) {
    $env:OTEL_LOG_PATH = (Join-Path $otelDir "otel-logs.json")
}
if (-not $env:OTEL_TRACES_PATH) {
    $env:OTEL_TRACES_PATH = (Join-Path $otelDir "otel-traces.json")
}

if ($OtelcolPath -eq "otelcol") {
    $cmd = Get-Command otelcol -ErrorAction SilentlyContinue
    if ($cmd) {
        $otel = $cmd.Path
    } else {
        $fallback = "C:\\Program Files\\OpenTelemetry Collector\\otelcol.exe"
        if (Test-Path $fallback) {
            $otel = $fallback
        } else {
            Write-Error "otelcol not found on PATH or at $fallback"
            exit 1
        }
    }
} elseif (-not (Test-Path $OtelcolPath)) {
    Write-Error "otelcol not found: $OtelcolPath"
    exit 1
}

Write-Host "Starting OpenTelemetry Collector..."
Write-Host "otelcol: $otel"
Write-Host "config : $resolvedConfig"
Write-Host ""

& $otel --config $resolvedConfig

param(
  [string]$TaskName = "NlpCropCalendarProductionAudit",
  [ValidateSet("Daily", "Weekly")]
  [string]$Frequency = "Daily",
  [string]$At = "09:00",
  [int]$DaysInterval = 1,
  [string[]]$DaysOfWeek = @("Monday"),
  [int]$Limit = 50,
  [int]$Days = 30,
  [double]$AutoPassConfidence = 0.9,
  [string]$OutputRoot = ".cache/eval/production_audit/scheduled",
  [string]$StateFile = ".state/eval/production_audit/sampling_state.json",
  [string]$CondaEnv = "llm-agent",
  [switch]$ResetCursor
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$runScript = Join-Path $PSScriptRoot "run_production_audit_cycle.ps1"
$pwsh = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $pwsh) {
  throw "pwsh executable was not found in PATH."
}

$scriptArgs = @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-File", $runScript,
  "-Limit", $Limit,
  "-Days", $Days,
  "-AutoPassConfidence", $AutoPassConfidence,
  "-OutputRoot", $OutputRoot,
  "-StateFile", $StateFile,
  "-CondaEnv", $CondaEnv
)

if ($ResetCursor) {
  $scriptArgs += "-ResetCursor"
}

$action = New-ScheduledTaskAction -Execute $pwsh -Argument ($scriptArgs -join " ") -WorkingDirectory $repoRoot
$startAt = [datetime]::Today.Add([timespan]::Parse($At))

if ($Frequency -eq "Weekly") {
  $trigger = New-ScheduledTaskTrigger -Weekly -WeeksInterval $DaysInterval -DaysOfWeek $DaysOfWeek -At $startAt
} else {
  $trigger = New-ScheduledTaskTrigger -Daily -DaysInterval $DaysInterval -At $startAt
}

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -StartWhenAvailable

Register-ScheduledTask `
  -TaskName $TaskName `
  -Action $action `
  -Trigger $trigger `
  -Settings $settings `
  -Description "Run scheduled production-audit review for NLP Crop Calendar." `
  -Force | Out-Null

Write-Host "registered_task=$TaskName"
Write-Host "run_script=$runScript"
Write-Host "working_directory=$repoRoot"

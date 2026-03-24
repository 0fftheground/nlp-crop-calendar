param(
  [int]$Limit = 50,
  [int]$Days = 30,
  [double]$AutoPassConfidence = 0.9,
  [string]$OutDir = "",
  [string]$OutputRoot = ".cache/eval/production_audit/scheduled",
  [string]$StateFile = ".state/eval/production_audit/sampling_state.json",
  [string]$CondaEnv = "llm-agent",
  [switch]$ResetCursor
)

$resolvedOutDir = if ($OutDir) {
  $OutDir
} else {
  $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
  Join-Path $OutputRoot $timestamp
}

New-Item -ItemType Directory -Path $resolvedOutDir -Force | Out-Null

$args = @(
  "run", "-n", $CondaEnv,
  "python", "-m", "src.eval_platform",
  "audit", "run-latest",
  "--limit", $Limit,
  "--days", $Days,
  "--auto-pass-confidence", $AutoPassConfidence,
  "--out-dir", $resolvedOutDir,
  "--state-file", $StateFile
)

if ($ResetCursor) { $args += "--reset-cursor" }

Write-Host "production_audit_run_dir=$resolvedOutDir"
Write-Host "production_audit_state_file=$StateFile"
conda @args

param(
  [string[]]$Promotion,
  [string]$ExpertRoot = "src/eval_assets/expert",
  [string]$GovernanceFile = "src/eval_assets/governance.yaml",
  [string[]]$RerunProfile = @("expert_blocking_gate", "expert_regression_gate"),
  [string]$LlmModel = "",
  [string]$ExtractorModel = "",
  [string]$JsonOut = ""
)

if (-not $Promotion -or $Promotion.Count -eq 0) {
  throw "At least one -Promotion path is required."
}

$args = @(
  "-m", "src.eval_platform",
  "promote",
  "--expert-root", $ExpertRoot,
  "--governance-file", $GovernanceFile
)

foreach ($path in $Promotion) {
  $args += @("--promotion", $path)
}

foreach ($profile in $RerunProfile) {
  $args += @("--rerun-profile", $profile)
}

if ($LlmModel) { $args += @("--llm-model", $LlmModel) }
if ($ExtractorModel) { $args += @("--extractor-model", $ExtractorModel) }
if ($JsonOut) { $args += @("--json-out", $JsonOut) }

python @args

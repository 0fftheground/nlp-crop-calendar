param(
  [string]$GovernanceFile = "src/eval_assets/governance.yaml",
  [string]$BlockingProfile = "expert_blocking_gate",
  [string]$RegressionProfile = "expert_regression_gate",
  [string]$BaselineLlmModel = "",
  [string]$BaselineExtractorModel = "",
  [string]$CandidateLlmModel = "",
  [string]$CandidateExtractorModel = "",
  [string]$JsonOut = ""
)

$args = @(
  "-m", "src.eval_platform",
  "compare",
  "--governance-file", $GovernanceFile,
  "--blocking-profile", $BlockingProfile,
  "--regression-profile", $RegressionProfile
)

if ($BaselineLlmModel) { $args += @("--baseline-llm-model", $BaselineLlmModel) }
if ($BaselineExtractorModel) { $args += @("--baseline-extractor-model", $BaselineExtractorModel) }
if ($CandidateLlmModel) { $args += @("--candidate-llm-model", $CandidateLlmModel) }
if ($CandidateExtractorModel) { $args += @("--candidate-extractor-model", $CandidateExtractorModel) }
if ($JsonOut) { $args += @("--json-out", $JsonOut) }

python @args

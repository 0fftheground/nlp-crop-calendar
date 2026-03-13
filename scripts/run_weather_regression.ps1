param(
    [string]$CondaEnv = "llm-agent"
)

$ErrorActionPreference = "Stop"

$tests = @(
    "tests.weather.test_service",
    "tests.weather.test_session",
    "tests.weather.test_ui",
    "tests.weather.test_regression"
)

Write-Host "Running weather regression suite in conda env '$CondaEnv'..."
& conda run -n $CondaEnv python -m unittest @tests

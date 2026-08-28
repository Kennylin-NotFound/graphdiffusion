param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(2026070114, 2026070115, 2026070116, 2026070117, 2026070118, 2026070119, 2026070120)]
    [int]$Seed,

    [Parameter(Mandatory = $true)]
    [ValidateSet("direct", "masked_conditional")]
    [string]$ModelKind,

    [switch]$Preflight,

    [string]$Resume = ""
)

$ErrorActionPreference = "Stop"
$ImplementationRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ProjectRoot = (Resolve-Path (Join-Path $ImplementationRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TrainingScript = Join-Path $ImplementationRoot "scripts\64_train_phase6e_e_stage3.py"
$TrainingConfig = Join-Path $ImplementationRoot "configs\training_phase6e_e_stage39_seed$Seed.yaml"
$Stage38Freeze = Join-Path $ImplementationRoot "artifacts\phase6e-e-stage38-training\training_freeze.json"
$DevelopmentFreeze = Join-Path $ImplementationRoot "artifacts\datasets\phase6e-e-stage3-development\dataset_freeze.json"

foreach ($RequiredPath in @($Python, $TrainingScript, $TrainingConfig, $Stage38Freeze, $DevelopmentFreeze)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required extended-seed training path is missing: $RequiredPath"
    }
}

$TrainingArguments = @(
    $TrainingScript,
    "--config", $TrainingConfig,
    "--model-kind", $ModelKind
)
if ($Preflight) {
    $TrainingArguments += "--preflight"
}
if ($Resume) {
    $ResolvedResume = (Resolve-Path -LiteralPath $Resume).Path
    $TrainingArguments += @("--resume", $ResolvedResume)
}

Push-Location $ImplementationRoot
$TrainingExitCode = 0
$PreviousActiveEntry = $env:GDM_STAGE3_ACTIVE_ENTRY
try {
    $env:GDM_STAGE3_ACTIVE_ENTRY = "1"
    & $Python @TrainingArguments
    $TrainingExitCode = $LASTEXITCODE
}
finally {
    if ($null -eq $PreviousActiveEntry) {
        Remove-Item Env:GDM_STAGE3_ACTIVE_ENTRY -ErrorAction SilentlyContinue
    }
    else {
        $env:GDM_STAGE3_ACTIVE_ENTRY = $PreviousActiveEntry
    }
    Pop-Location
}
if ($TrainingExitCode -ne 0) {
    throw "Extended-seed training failed with exit code $TrainingExitCode."
}

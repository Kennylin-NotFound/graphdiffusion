param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("direct", "masked_conditional")]
    [string]$ModelKind,

    [switch]$Preflight,

    [string]$Resume
)

$ErrorActionPreference = "Stop"
$ImplementationRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ProjectRoot = (Resolve-Path (Join-Path $ImplementationRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$FreezeScript = Join-Path $ImplementationRoot "scripts\65_finalize_phase6e_e_stage3_pretraining.py"
$TrainingScript = Join-Path $ImplementationRoot "scripts\64_train_phase6e_e_stage3.py"
$TrainingConfig = Join-Path $ImplementationRoot "configs\training_phase6e_e_stage3_pilot.yaml"

foreach ($RequiredPath in @($Python, $FreezeScript, $TrainingScript, $TrainingConfig)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required Stage 3 path is missing: $RequiredPath"
    }
}

& $Python $FreezeScript
if ($LASTEXITCODE -ne 0) {
    throw "Stage 3 pre-training freeze verification failed."
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
    throw "Stage 3 training entrypoint failed with exit code $TrainingExitCode."
}

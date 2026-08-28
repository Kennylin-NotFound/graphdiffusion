$ErrorActionPreference = 'Stop'

$implementationRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $implementationRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$runner = Join-Path $implementationRoot 'scripts\97_run_phase7_proposal_recovery_evaluation.py'
$exporter = Join-Path $implementationRoot 'scripts\98_export_phase7_manuscript_evidence.py'
$outputRoot = Join-Path $implementationRoot 'artifacts\phase7-proposal-conditioned-recovery-realistic'
$log = Join-Path $PSScriptRoot 'resume_realistic.log'

function Write-Status([string]$Message) {
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -LiteralPath $log -Encoding UTF8 -Value "[$stamp] $Message"
}

Set-Location -LiteralPath $implementationRoot
Write-Status 'Resuming realistic-simulation evaluation from validated records.'
& $python $runner run `
    --settings realistic_simulation `
    --methods random_k64,direct_b64_t1,sequential_b64_t1,masked_k8_t1 `
    --output-root artifacts/phase7-proposal-conditioned-recovery-realistic `
    --device cuda `
    --no-skip-missing-datasets `
    *> (Join-Path $outputRoot 'resume_run.log')
if ($LASTEXITCODE -ne 0) {
    Write-Status "Evaluation failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Status 'Realistic-simulation records complete; finalizing evidence.'
& $python $runner finalize `
    --output-root artifacts/phase7-proposal-conditioned-recovery-realistic `
    *> (Join-Path $outputRoot 'finalize.log')
if ($LASTEXITCODE -ne 0) {
    Write-Status "Finalization failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Status 'Exporting unified manuscript evidence and figures.'
& $python $exporter *> (Join-Path $outputRoot 'manuscript_export.log')
if ($LASTEXITCODE -ne 0) {
    Write-Status "Manuscript evidence export failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Status 'Realistic evaluation, evidence freeze, and manuscript export completed.'

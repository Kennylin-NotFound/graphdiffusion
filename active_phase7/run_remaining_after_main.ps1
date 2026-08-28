param(
    [int[]]$WaitForProcessId = @(25072, 5556)
)

$ErrorActionPreference = 'Stop'
$implementationRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path (Split-Path -Parent $implementationRoot) '.venv\Scripts\python.exe'
$runner = Join-Path $implementationRoot 'scripts\97_run_phase7_proposal_recovery_evaluation.py'
$mainOutput = Join-Path $implementationRoot 'artifacts\phase7-proposal-conditioned-recovery'
$controlledOutput = Join-Path $implementationRoot 'artifacts\phase7-proposal-conditioned-recovery-controlled'
$realisticOutput = Join-Path $implementationRoot 'artifacts\phase7-proposal-conditioned-recovery-realistic'
$queueLog = Join-Path $PSScriptRoot 'queue.log'

function Write-QueueLog([string]$Message) {
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -LiteralPath $queueLog -Encoding UTF8 -Value "[$stamp] $Message"
}

Write-QueueLog "Waiting for main/cross-scale evaluation PIDs: $($WaitForProcessId -join ', ')."
while ($true) {
    $active = @($WaitForProcessId | Where-Object {
        Get-Process -Id $_ -ErrorAction SilentlyContinue
    })
    if ($active.Count -eq 0) {
        break
    }
    Start-Sleep -Seconds 30
}

Set-Location -LiteralPath $implementationRoot
Write-QueueLog 'Main/cross-scale process exited; finalizing its evidence.'
& $python $runner finalize --output-root artifacts/phase7-proposal-conditioned-recovery `
    *> (Join-Path $mainOutput 'finalize.log')
if ($LASTEXITCODE -ne 0) {
    Write-QueueLog "Main/cross-scale finalization failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-QueueLog 'Starting controlled-shift evaluation.'
New-Item -ItemType Directory -Force -Path $controlledOutput | Out-Null
& $python $runner run `
    --settings controlled_shift `
    --methods direct_b64_t1,sequential_b64_t1,masked_k8_t1 `
    --output-root artifacts/phase7-proposal-conditioned-recovery-controlled `
    --device cuda `
    --no-skip-missing-datasets `
    *> (Join-Path $controlledOutput 'run.log')
if ($LASTEXITCODE -ne 0) {
    Write-QueueLog "Controlled-shift evaluation failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-QueueLog 'Controlled-shift evaluation completed; finalizing evidence.'
& $python $runner finalize `
    --output-root artifacts/phase7-proposal-conditioned-recovery-controlled `
    *> (Join-Path $controlledOutput 'finalize.log')
if ($LASTEXITCODE -ne 0) {
    Write-QueueLog "Controlled-shift finalization failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-QueueLog 'Starting realistic-simulation evaluation.'
New-Item -ItemType Directory -Force -Path $realisticOutput | Out-Null
& $python $runner run `
    --settings realistic_simulation `
    --methods random_k64,direct_b64_t1,sequential_b64_t1,masked_k8_t1 `
    --output-root artifacts/phase7-proposal-conditioned-recovery-realistic `
    --device cuda `
    --no-skip-missing-datasets `
    *> (Join-Path $realisticOutput 'run.log')
if ($LASTEXITCODE -ne 0) {
    Write-QueueLog "Realistic-simulation evaluation failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-QueueLog 'Realistic-simulation evaluation completed; finalizing evidence.'
& $python $runner finalize `
    --output-root artifacts/phase7-proposal-conditioned-recovery-realistic `
    *> (Join-Path $realisticOutput 'finalize.log')
if ($LASTEXITCODE -ne 0) {
    Write-QueueLog "Realistic-simulation finalization failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-QueueLog 'Phase 7 queued evaluations completed successfully.'

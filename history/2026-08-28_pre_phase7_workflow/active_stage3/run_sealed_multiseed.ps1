param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('Prepare','Preflight','Train','FinalizeTraining','GenerateData','LabelData','FreezeData','Evaluate','FinalizeEvaluation')]
    [string]$Action,
    [ValidateSet(2026070112,2026070113)]
    [int]$Seed = 2026070112,
    [ValidateSet('direct','masked_conditional')]
    [string]$ModelKind = 'masked_conditional',
    [string]$Resume = ''
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path (Split-Path -Parent $root) '.venv\Scripts\python.exe'
$manager = Join-Path $root 'scripts\72_run_phase6e_e_stage38_sealed.py'
$dataset = Join-Path $root 'artifacts\datasets\phase6e-e-stage38-sealed'

if (-not (Test-Path $python)) {
    throw "Project Python was not found: $python"
}

switch ($Action) {
    'Prepare' {
        & $python $manager prepare
    }
    { $_ -in @('Preflight','Train') } {
        & $python $manager verify-training
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        $config = Join-Path $root "configs\training_phase6e_e_stage38_seed$Seed.yaml"
        $env:GDM_STAGE3_ACTIVE_ENTRY = '1'
        $arguments = @((Join-Path $root 'scripts\64_train_phase6e_e_stage3.py'), '--config', $config, '--model-kind', $ModelKind)
        if ($Action -eq 'Preflight') { $arguments += '--preflight' }
        if ($Resume) { $arguments += @('--resume', $Resume) }
        & $python @arguments
    }
    'FinalizeTraining' {
        & $python $manager finalize-training
    }
    'GenerateData' {
        & $python $manager authorize-data
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $python (Join-Path $root 'scripts\03_generate_dataset.py') --config (Join-Path $root 'configs\dataset_phase6e_e_stage38_sealed.yaml')
        if ($LASTEXITCODE -eq 0) { & $python (Join-Path $root 'scripts\04_audit_dataset.py') $dataset }
    }
    'LabelData' {
        & $python $manager authorize-data
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $python (Join-Path $root 'scripts\07_generate_solution_pools.py') $dataset --target-size 16 --beta 5 --time-limit 120 --mip-gap 0 --threads 1 --seed 0
    }
    'FreezeData' {
        & $python (Join-Path $root 'scripts\08_audit_solution_pools.py') $dataset
        if ($LASTEXITCODE -eq 0) { & $python (Join-Path $root 'scripts\18_freeze_labeled_dataset.py') $dataset }
    }
    'Evaluate' {
        & $python $manager run
    }
    'FinalizeEvaluation' {
        & $python $manager finalize
    }
}

exit $LASTEXITCODE

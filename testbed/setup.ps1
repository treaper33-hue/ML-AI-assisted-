<#
.SYNOPSIS
    One-button boot for the ML mastery testbed.

.DESCRIPTION
    Brings the testbed up and proves it works: resolves an interpreter,
    installs what is missing, records what this machine can verify, runs
    the contract tests, and executes the baseline experiment.

    Two modes:

      * Default -- creates a local .venv and installs the PINNED versions
        in requirements.txt. Reproducible, isolated, and what an outside
        researcher should use.

      * Existing environment (-CondaEnv or -Python) -- installs only the
        packages that are ABSENT and leaves every installed version
        alone. Pins are not enforced there: downgrading a working
        environment to match our pins would break more than it fixes.
        Note the trade-off in your run report, since results from an
        unpinned environment are reproducible only on that machine.

.PARAMETER CondaEnv
    Name of an existing conda environment to use, e.g. 'qwen35'.

.PARAMETER Python
    Full path to an existing interpreter to use instead.

.PARAMETER Minimal
    Install only the core packages. Deep-learning arms then report
    UNVERIFIED, by design.

.PARAMETER SkipTests
    Skip the test run. Not recommended: the tests are the contract.

.EXAMPLE
    .\setup.ps1
    .\setup.ps1 -CondaEnv qwen35
    .\setup.ps1 -Python C:\Python312\python.exe
#>

[CmdletBinding()]
param(
    [string]$CondaEnv,
    [string]$Python,
    [switch]$Minimal,
    [switch]$SkipTests
)

# Native tools write progress and warnings to stderr; under 'Stop' that
# would abort the boot on a message rather than on a failure. We check
# $LASTEXITCODE explicitly instead.
$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
Set-Location $root

Write-Host ''
Write-Host '=== ML mastery testbed: boot ===' -ForegroundColor Cyan

# 1. Resolve the interpreter -------------------------------------------
$useExisting = $false
if ($CondaEnv) {
    $conda = (Get-Command conda -ErrorAction SilentlyContinue).Source
    if (-not $conda) {
        $conda = Join-Path $env:USERPROFILE 'anaconda3\Scripts\conda.exe'
    }
    if (-not (Test-Path $conda)) {
        Write-Host "conda not found; pass -Python <path> instead." -ForegroundColor Red
        exit 1
    }
    $envRoot = & $conda env list | Select-String -Pattern "^\s*$([regex]::Escape($CondaEnv))\s" |
        ForEach-Object { ($_ -split '\s+')[-1] } | Select-Object -First 1
    if (-not $envRoot) {
        Write-Host "conda environment '$CondaEnv' not found." -ForegroundColor Red
        exit 1
    }
    $interpreter = Join-Path $envRoot 'python.exe'
    $useExisting = $true
} elseif ($Python) {
    $interpreter = $Python
    $useExisting = $true
} else {
    $interpreter = (Get-Command python -ErrorAction SilentlyContinue).Source
}

if (-not $interpreter -or -not (Test-Path $interpreter)) {
    Write-Host 'No usable interpreter. Install CPython 3.11 or newer.' -ForegroundColor Red
    exit 1
}
$version = & $interpreter -c "import sys; print('%d.%d' % sys.version_info[:2])"
Write-Host "interpreter : $interpreter (Python $version)"
if ([version]$version -lt [version]'3.11') {
    Write-Host 'Python 3.11+ required (configs are read with tomllib).' -ForegroundColor Red
    exit 1
}

# 2. Environment and dependencies --------------------------------------
if ($useExisting) {
    $venvPython = $interpreter
    Write-Host 'mode        : existing environment (pins NOT enforced)'
    $missing = (& $venvPython -m mlkit.verify --missing).Trim()
    if ($missing) {
        Write-Host "installing  : $missing"
        $packages = $missing -split '\s+'
        & $venvPython -m pip install --upgrade-strategy only-if-needed @packages
    } else {
        Write-Host 'installing  : nothing -- every package is already present'
    }
} else {
    $venv = Join-Path $root '.venv'
    if (-not (Test-Path $venv)) {
        Write-Host 'creating    : .venv'
        & $interpreter -m venv $venv
    } else {
        Write-Host 'venv        : reusing existing .venv'
    }
    $venvPython = Join-Path $venv 'Scripts\python.exe'
    Write-Host 'installing  : pinned dependencies (this can take a few minutes)'
    & $venvPython -m pip install --upgrade pip --quiet
    if ($Minimal) {
        $core = @(
            'numpy==2.1.3', 'scipy==1.14.1', 'pandas==2.2.3',
            'matplotlib==3.9.2', 'pytest==8.3.3', 'ruff==0.7.4'
        )
        & $venvPython -m pip install --quiet @core
    } else {
        & $venvPython -m pip install --quiet -r (Join-Path $root 'requirements.txt')
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'dependency install failed. Re-run with -Minimal to continue' -ForegroundColor Yellow
        Write-Host 'with the NumPy arms only; those arms still verify.' -ForegroundColor Yellow
    }
}

# 3. Environment record ------------------------------------------------
Write-Host ''
& $venvPython -m mlkit.verify --env

# 4. Contract tests ----------------------------------------------------
if (-not $SkipTests) {
    Write-Host ''
    Write-Host 'running     : contract tests (metrics, splits, leakage)'
    & $venvPython -m pytest tests -q
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'CONTRACT TESTS FAILED -- stop here and fix before Stage 0.' -ForegroundColor Red
        exit 1
    }
}

# 5. Baseline experiment ----------------------------------------------
Write-Host ''
Write-Host 'running     : EXP-01 baseline'
& $venvPython -m experiments.exp01_baseline --config configs\exp01.toml
$expCode = $LASTEXITCODE

Write-Host ''
Write-Host '=== boot complete ===' -ForegroundColor Cyan
if ($useExisting) {
    Write-Host "run experiments with: & '$venvPython' -m experiments.exp01_baseline --config configs\exp01.toml"
} else {
    Write-Host "activate the environment with: .\.venv\Scripts\Activate.ps1"
}
if ($expCode -eq 0) {
    Write-Host 'EXP-01 is VERIFIED. Stage 0 is open.' -ForegroundColor Green
} else {
    Write-Host 'EXP-01 is not fully verified. Read the gate table and the arm' -ForegroundColor Yellow
    Write-Host 'notes above, then log the finding in VERIFICATION.md.' -ForegroundColor Yellow
}
exit 0

param([string]$Python = "python")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "8"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
function Check-Exit([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed with exit code $LASTEXITCODE" }
}
& $Python -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,14), sys.version'
Check-Exit "Python version check (use Python 3.11, not system Python 3.6)"
if (-not (Test-Path ".venv")) {
    & $Python -m venv .venv
    Check-Exit "venv creation"
}
$VenvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip setuptools wheel
Check-Exit "packaging tools installation"
& $VenvPython -m pip install -e '.[data,test,report]'
Check-Exit "project installation"
& $VenvPython -m costrank.cli doctor
Check-Exit "environment check"
& $VenvPython -m pytest -q
Check-Exit "tests"

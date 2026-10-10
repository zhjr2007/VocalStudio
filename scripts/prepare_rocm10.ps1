param([string]$BasePython = "$PSScriptRoot\..\venv\Scripts\python.exe")
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location -LiteralPath $projectRoot
$env:TEMP = Join-Path $projectRoot 'temp\rocm10-bootstrap'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $projectRoot 'temp\rocm10-pip-cache'
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
$runtimePython = Join-Path $projectRoot 'venv-rocm10\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $runtimePython)) {
    & $BasePython -m venv (Join-Path $projectRoot 'venv-rocm10')
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create independent ROCm 10 environment' }
}
& $runtimePython -m pip install -r requirements-rocm10-torch.txt
if ($LASTEXITCODE -ne 0) { throw 'AMD torch installation failed' }
# Optional local native wheels have explicit provenance and SHA256 records.
$wheelArgs = @()
if (Test-Path temp\rocm10-wheels) { $wheelArgs = @('--find-links', 'temp\rocm10-wheels') }
& $runtimePython -m pip install @wheelArgs -r requirements-rocm10-inference.txt -c constraints-rocm10.txt --report temp\rocm10-inference-install.json
if ($LASTEXITCODE -ne 0) { throw 'Inference dependency installation failed; see docs/amd-rocm10-release.md' }
& $runtimePython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed' }
& $runtimePython -m pip freeze | Set-Content requirements-rocm10.lock.txt -Encoding utf8

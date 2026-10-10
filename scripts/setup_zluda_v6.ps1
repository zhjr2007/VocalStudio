# setup_zluda_v6.ps1 — provision ZLUDA v6 + HIP SDK, verify, and benchmark on the AMD GPU.
#   Usage:  powershell -ExecutionPolicy Bypass -File scripts\setup_zluda_v6.ps1
#   Requires: AMD RX 7600M XT (or other RDNA3) eGPU attached/compute-capable.
param(
    [string]$ZluDir = "$env:LOCALAPPDATA\zluda",
    [string]$ZluUrl = "https://github.com/vosen/ZLUDA/releases/download/v6/zluda-windows-3fe1206.zip",
    [string]$HipPath = $env:HIP_PATH
)
$ErrorActionPreference = "Stop"
.\venv\Scripts\python -c "print('VocalStudio ZLUDA bootstrap')"

# 1) launcher
$zip = Join-Path $ZluDir "zluda-v6.zip"
New-Item -ItemType Directory -Force -Path $ZluDir | Out-Null
if (-not (Test-Path (Join-Path $ZluDir "zluda.exe"))) {
    Write-Host "Downloading ZLUDA v6 ..."
    Invoke-WebRequest -Uri $ZluUrl -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath $ZluDir -Force
    # handle the nested zluda\zluda layout
    $nested = Join-Path $ZluDir "zluda\zluda"
    if (Test-Path $nested) { Copy-Item (Join-Path $nested "*") $ZluDir -Recurse -Force }
}
$zluda = Join-Path $ZluDir "zluda.exe"
Write-Host "ZLUDA launcher: $zluda"

# 2) HIP SDK (required for cuDNN/MIOpen). Set HIP_PATH to the therock nightly (gfx11xx).
if (-not $HipPath -or -not (Test-Path (Join-Path $HipPath "bin\rocblas.dll"))) {
    Write-Host "WARNING: HIP_PATH not set or missing bin\rocblas.dll."
    Write-Host "  Download the AMD therock nightly for your GPU arch (e.g. therock-dist-windows-gfx1100...tar.gz),"
    Write-Host "  extract, and set HIP_PATH to that dir. Then re-run."
    exit 1
}
$env:HIP_PATH = $HipPath

# 3) verify device
Write-Host "Probing CUDA under ZLUDA ..."
& $zluda -- (Join-Path $ZluDir "cuda_check.exe")
if ($LASTEXITCODE -ne 0) { Write-Host "cuda_check failed."; exit $LASTEXITCODE }

# 4) torch smoke + benchmark on the AMD GPU
& $zluda -- .\venv\Scripts\python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
& $zluda -- .\venv\Scripts\python scripts\benchmark_models.py --device rocm --json temp\bench\results_rocm.json

#Requires -Version 5.1
# InternImage detection environment setup for Alpha5 training
# Target: Windows 11 + conda + CUDA driver installed + WSL2 (for DCNv3 compilation)
#
# Usage:
#   .\setup.ps1              # defaults: CUDA 12.1
#   .\setup.ps1 -CudaVer 118 # CUDA 11.8

param(
    [string]$CudaVer = "121"
)

$ErrorActionPreference = "Stop"

$TorchVersion      = "2.1.0"
$TorchvisionVer    = "0.16.0"
$EnvName           = "internimage_alpha5"
$ScriptDir         = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoDir           = Join-Path $ScriptDir "InternImage_repo"
$DetectDir         = Join-Path $RepoDir "detection"

Write-Host "==> CUDA: cu$CudaVer | torch: $TorchVersion | env: $EnvName"

# ── helpers ──────────────────────────────────────────────────────────────────
function Run-Conda {
    param([string[]]$Args)
    & conda @Args
    if ($LASTEXITCODE -ne 0) { throw "conda $($Args[0]) failed (exit $LASTEXITCODE)" }
}

function Run-In-Env {
    param([string[]]$Args)
    & conda run -n $EnvName --no-capture-output @Args
    if ($LASTEXITCODE -ne 0) { throw "Command failed in env '$EnvName': $($Args -join ' ')" }
}

# ── 1. Conda env ──────────────────────────────────────────────────────────────
$existing = & conda env list | Select-String "^$EnvName\s"
if ($existing) {
    Write-Host "==> Env '$EnvName' exists, skipping creation"
} else {
    Run-Conda "create", "-n", $EnvName, "python=3.11", "-y"
}

# ── 2. PyTorch ────────────────────────────────────────────────────────────────
$torchIndex = "https://download.pytorch.org/whl/cu$CudaVer"
Run-In-Env "pip", "install",
    "torch==$TorchVersion+cu$CudaVer",
    "torchvision==$TorchvisionVer+cu$CudaVer",
    "--index-url", $torchIndex

# ── 3. Base dependencies ──────────────────────────────────────────────────────
$reqFile = Join-Path $ScriptDir "requirements.txt"
Run-In-Env "pip", "install", "-r", $reqFile

# ── 4. mmcv-full ──────────────────────────────────────────────────────────────
$mmcvIndex = "https://download.openmmlab.com/mmcv/dist/cu$CudaVer/torch$TorchVersion/index.html"
Run-In-Env "pip", "install", "mmcv-full==1.7.2", "-f", $mmcvIndex

# ── 5. Clone InternImage repo ─────────────────────────────────────────────────
if (-not (Test-Path $RepoDir)) {
    git clone https://github.com/OpenGVLab/InternImage.git $RepoDir
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
} else {
    Write-Host "==> Repo already at $RepoDir"
}

# ── 6. Install mmdet + InternImage detection package ─────────────────────────
Run-In-Env "pip", "install", "-e", $DetectDir

# ── 7. Compile DCNv3 CUDA ops (requires WSL2) ────────────────────────────────
$opsDir = (Join-Path $DetectDir "ops_dcnv3") -replace '\\', '/'
# Convert Windows path to WSL path
$wslOpsDir = wsl wslpath -u $opsDir.Replace('\', '/')
Write-Host "==> Compiling DCNv3 in WSL2 at $wslOpsDir"
wsl bash -c "cd '$wslOpsDir' && conda run -n $EnvName --no-capture-output sh ./make.sh"
if ($LASTEXITCODE -ne 0) { throw "DCNv3 compilation failed — check WSL2 and CUDA toolkit" }

# ── 8. Overlay our custom files ───────────────────────────────────────────────
$dstDatasets = Join-Path $DetectDir "mmdet_custom\datasets"
New-Item -ItemType Directory -Force -Path $dstDatasets | Out-Null

Copy-Item (Join-Path $ScriptDir "mmdet_custom\datasets\alpha5.py") `
          (Join-Path $dstDatasets "alpha5.py") -Force
Copy-Item (Join-Path $ScriptDir "mmdet_custom\datasets\__init__.py") `
          (Join-Path $dstDatasets "__init__.py") -Force

$dstCoco  = Join-Path $DetectDir "configs\coco"
$dstBase  = Join-Path $DetectDir "configs\_base_\datasets"
New-Item -ItemType Directory -Force -Path $dstCoco | Out-Null
New-Item -ItemType Directory -Force -Path $dstBase | Out-Null

Copy-Item (Join-Path $ScriptDir "configs\coco\cascade_internimage_l_alpha5.py") `
          (Join-Path $dstCoco "cascade_internimage_l_alpha5.py") -Force
Copy-Item (Join-Path $ScriptDir "configs\_base_\coco_alpha5.py") `
          (Join-Path $dstBase "coco_alpha5.py") -Force

# ── 9. Data directory scaffold ────────────────────────────────────────────────
$dataRoot = Join-Path $DetectDir "data\alpha5_coco_v3.3"
foreach ($sub in @("annotations","images\train","images\val","images\test")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $dataRoot $sub) | Out-Null
}

Write-Host ""
Write-Host "==> Setup complete."
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Copy dataset into: $dataRoot"
Write-Host "  2. Download pretrained weights (see README.md)"
Write-Host "  3. conda activate $EnvName"
Write-Host "  4. cd $DetectDir"
Write-Host "  5. python tools/train.py configs/coco/cascade_internimage_l_alpha5.py --work-dir work_dirs/alpha5"

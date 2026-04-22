# Creates .transformers-4.46-venv
# Run once: .\setup_compat.ps1
# Requires Python 3.10+ and CUDA 12.4 drivers installed

$ErrorActionPreference = "Stop"
$VENV = ".transformers-4.46-venv"

Write-Host "Creating $VENV..."
python -m venv $VENV

Write-Host "Activating..."
& ".\$VENV\Scripts\Activate.ps1"

Write-Host "Installing PyTorch (CUDA 12.4)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host "Installing dependencies..."
pip install -r envs/requirements-compat.txt

Write-Host ""
Write-Host "Done. Activate: .\$VENV\Scripts\Activate.ps1"
Write-Host "Models: moondream, llava, blip2, instructblip, clip, paligemma, idefics, mplug_owl3"

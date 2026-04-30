# Creates .transformers-5.X-venv
# Run once: .\setup.ps1
# Requires Python 3.10+ and CUDA 12.4 drivers installed

$ErrorActionPreference = "Stop"
$VENV = ".transformers-5.X-venv"

Write-Host "Creating $VENV..."
python -m venv $VENV

Write-Host "Activating..."
& ".\$VENV\Scripts\Activate.ps1"

Write-Host "Installing PyTorch (CUDA 12.4)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host "Installing dependencies..."
pip install -r envs/requirements-5x.txt

Write-Host ""
Write-Host "Done. Activate: .\$VENV\Scripts\Activate.ps1"
Write-Host "Models: smolvlm, qwen_vl, videollama3"

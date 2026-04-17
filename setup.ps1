# Run once on the execution machine: .\setup.ps1
# Requires Python 3.10+ and CUDA toolkit installed

$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment..."
python -m venv .venv

Write-Host "Activating..."
.\.venv\Scripts\Activate.ps1

Write-Host "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Write-Host "Installing remaining dependencies..."
pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete. Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Run with: python run.py --model smolvlm --images images\"

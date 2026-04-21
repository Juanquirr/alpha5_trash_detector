# Creates .venv-compat — transformers 4.46.x for legacy models
# Run once: .\setup_compat.ps1

$ErrorActionPreference = "Stop"

Write-Host "Creating .venv-compat..."
python -m venv .venv-compat

Write-Host "Activating..."
.\.venv-compat\Scripts\Activate.ps1

Write-Host "Installing PyTorch with CUDA 12.4..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host "Installing transformers 4.46.x + deps..."
pip install "transformers==4.46.3" accelerate Pillow

Write-Host ""
Write-Host "Done. Activate with: .\.venv-compat\Scripts\Activate.ps1"
Write-Host "Models: Moondream, PaliGemma, Mplug-owl3, Kosmos-2.5, LLaVA, IDEFICS, InstructBLIP, BLIP-2, CLIP"

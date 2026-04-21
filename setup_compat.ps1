# Creates .venv-compat — transformers 4.46.x for legacy models
# Run once: .\setup_compat.ps1

$ErrorActionPreference = "Stop"

Write-Host "Creating .transformers-4.46-venv..."
python -m venv .transformers-4.46-venv

Write-Host "Activating..."
.\.transformers-4.46-venv\Scripts\Activate.ps1

Write-Host "Installing PyTorch with CUDA 12.4..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host "Installing transformers 4.46.x + deps..."
pip install "transformers==4.46.3" accelerate Pillow timm sentencepiece

Write-Host ""
Write-Host "Done. Activate with: .\.transformers-4.46-venv\Scripts\Activate.ps1"
Write-Host "Models: Moondream, PaliGemma, Mplug-owl3, LLaVA, IDEFICS, InstructBLIP, BLIP-2, CLIP"

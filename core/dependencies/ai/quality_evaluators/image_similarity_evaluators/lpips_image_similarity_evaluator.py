import lpips
from PIL import Image
import torch
import torchvision.transforms as transforms

from core.dependencies.ai.quality_evaluators.image_similarity_evaluators.text_similarity_evaluators import \
    ImageSimilarityEvaluator


class LPIPSImageSimilarityEvaluator(ImageSimilarityEvaluator):
    def __init__(self, net_type: str = 'alex'):
        self.loss_fn = lpips.LPIPS(net=net_type)  # Load LPIPS model with specified network (default: AlexNet)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to match LPIPS input requirements
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension

    def evaluate(self, image1: Image.Image, image2: Image.Image) -> float:
        img1_tensor = self.preprocess_image(image1)
        img2_tensor = self.preprocess_image(image2)
        similarity = self.loss_fn(img1_tensor, img2_tensor)  # Compute LPIPS distance
        return similarity.item()


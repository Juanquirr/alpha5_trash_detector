from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

from core.dependencies.ai.quality_evaluators.text_image_similarity_evaluators.image_similarity_evaluators import \
    TextImageSimilarityEvaluator


class CLIPTextImageSimilarityEvaluator(TextImageSimilarityEvaluator):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)  # Load CLIP model
        self.processor = CLIPProcessor.from_pretrained(model_name)  # Load CLIP processor

    def evaluate(self, text: str, img: Image.Image) -> float:
        inputs = self.processor(text=[text], images=img, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)

        # Extract embeddings
        text_embeds = outputs.text_embeds
        image_embeds = outputs.image_embeds

        # Compute cosine similarity
        cosine_similarity = torch.nn.functional.cosine_similarity(text_embeds, image_embeds)
        return cosine_similarity.item()


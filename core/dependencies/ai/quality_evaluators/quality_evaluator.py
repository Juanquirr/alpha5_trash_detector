from typing import Optional

from PIL import Image
from pipelines.dependencies.ai.quality_evaluators.aesthetic_evaluators.aesthetic_evaluator import AestheticQualityEvaluator
from pipelines.dependencies.ai.quality_evaluators.dataset_similarity_evaluators.dataset_similarity_evaluator import \
    DatasetSimilarityEvaluator
from pipelines.dependencies.ai.quality_evaluators.image_similarity_evaluators.text_similarity_evaluators import \
    ImageSimilarityEvaluator

from dataclasses import dataclass

from pipelines.dependencies.ai.quality_evaluators.text_image_similarity_evaluators.image_similarity_evaluators import \
    TextImageSimilarityEvaluator


@dataclass
class QualityEvaluator:
    aesthetic_eval: AestheticQualityEvaluator
    dataset_similarity: DatasetSimilarityEvaluator
    image_similarity: ImageSimilarityEvaluator
    text_image_similarity: TextImageSimilarityEvaluator

    aesthetic_mark : Optional[float] = None
    image_similarity_mark : Optional[float] = None
    text_image_mark : Optional[float] = None
    dataset_similarity_mark : Optional[float] = None

    def evaluate_aesthetic(self, image: Image.Image):
        self.aesthetic_mark = self.aesthetic_eval.evaluate(image)
        return self.aesthetic_mark

    def evaluate_image_similarity(self, image1: Image.Image, image2: Image.Image):
        self.image_similarity_mark = self.image_similarity.evaluate(image1, image2)
        return self.image_similarity_mark

    def evaluate_text_image_similarity(self, text: str, image1: Image.Image):
        self.text_image_mark = self.text_image_similarity.evaluate(text, image1)
        return self.text_image_mark

    def evaluate_dataset_similarity(self, dataset: str, synthetic_dataset: str):
        self.dataset_similarity_mark = self.dataset_similarity.evaluate(dataset, synthetic_dataset)
        return self.dataset_similarity_mark


    def show_scores(self):
        print("""
        PROCESSING SCORES:
        1. AestheticMark: {}
        2. ImageSimilarityMark: {}
        3. InpaintingTextSimilarityMark: {}
        """.format(self.aesthetic_mark, self.image_similarity_mark, self.text_image_mark))
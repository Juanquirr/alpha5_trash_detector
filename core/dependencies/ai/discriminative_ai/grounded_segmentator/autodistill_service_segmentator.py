from core.dependencies.ai.discriminative_ai.grounded_segmentator.grounded_segmentator import GroundedSegmentator
from core.dependencies.api.autodistill_service_sdk import AutodistillServiceClient, GenerateImageVariantsResponse

from PIL import Image


class AutodistillServiceSegmentator(GroundedSegmentator):
    api : AutodistillServiceClient

    def segment(self, image: Image.Image, label: str) -> GenerateImageVariantsResponse:
        return self.api.generate_segmentation(
            image=image,
            prompt=label,
            text=label
        )
from random import random
import matplotlib.pyplot as plt
from typing import List

from pydantic import BaseModel

from ..pipeline import DatasetGenerationPipeline, PipelineResult, Annotation
from ...dependencies.ai.discriminative_ai.grounded_segmentator.grounded_segmentator import GroundedSegmentator
from ...dependencies.ai.generative_ai.image_generators.image_generator import ImageGenerator
from ...dependencies.ai.generative_ai.image_inpainters.image_inpainter import ImageInpainter
from ...dependencies.api.sam_service_sdk import SegmentationRefinerClient
from ...utils import polygon_to_bbox, divide_images_with_masks_and_bboxes, filter_masks_by_polygon, \
    choose_random_elements, paint_masks_in_image


class FreeInpaintingConditioning(BaseModel):
    background_prompt: str
    negative_prompt: str = ""
    classes: List[str]
    classes_id : List[int]
    threshold: float
    instances: int
    instance_environment: str
    grid_number: int

class FreeInpaintingPipeline(DatasetGenerationPipeline[FreeInpaintingConditioning], BaseModel):
    image_generator : ImageGenerator
    grounded_segmentator: GroundedSegmentator
    image_inpainter : ImageInpainter


    def generate(self, conditioning: FreeInpaintingConditioning) -> PipelineResult:
        image = self.image_generator.generate(conditioning.background_prompt, conditioning.negative_prompt)
        labels = self.grounded_segmentator.segment(image, conditioning.instance_environment)
        sea_polygon = labels.annotations[0].data.points
        masks = divide_images_with_masks_and_bboxes(image, n_divisions=conditioning.grid_number, padding=30)
        filtered_masks = filter_masks_by_polygon(masks, sea_polygon, threshold=0.5)
        chosen_masks = choose_random_elements(filtered_masks, conditioning.instances)
        annotations = []
        for mask, bbox in chosen_masks:
            curr_class = choose_random_elements(conditioning.classes, 1)[0]
            image = self.image_inpainter.inpaint(image, mask, curr_class)
            refined_segmentation = SegmentationRefinerClient().refine_bounding_box(image, bbox)
            annotations.append(Annotation(
                type="bounding_box",
                label=conditioning.classes_id[conditioning.classes.index(curr_class)],
                data={
                    "point": [refined_segmentation.bounding_boxes[0].x_0/image.size[0], refined_segmentation.bounding_boxes[0].y_0/image.size[1]],
                    "width": abs(refined_segmentation.bounding_boxes[0].x_1 - refined_segmentation.bounding_boxes[0].x_0)/image.size[0],
                    "height": abs(refined_segmentation.bounding_boxes[0].y_1 - refined_segmentation.bounding_boxes[0].y_0)/image.size[1],
                }
            ))

        return PipelineResult(
            image=image,
            annotations=annotations
        )
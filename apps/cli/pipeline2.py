from core.dependencies.ai.discriminative_ai.grounded_segmentator.autodistill_service_segmentator import \
    AutodistillServiceSegmentator
from core.dependencies.ai.generative_ai.image_generators.comfy_image_generator import ComfyuiImageGenerator
from core.dependencies.ai.generative_ai.image_generators.comfy_workflows import text_to_image_workflow
from core.dependencies.ai.generative_ai.image_generators.mock_image_generator import MockImageGenerator
from core.dependencies.ai.generative_ai.image_inpainters.comfy_image_inpainter import ComfyImageInpainter
from core.dependencies.api.autodistill_service_sdk import AutodistillServiceClient
from core.pipelines.free_inpainting.main import FreeInpaintingPipeline, FreeInpaintingConditioning
from core.pipelines.grounded_diffusion.main import GroundedDiffusionPipeline, GroundedDiffusionConditioning
from datagen_sdk.client import DatagenClient
from PIL import Image



client = DatagenClient(api_key="8b5c1f64b7a4887c012b3786f7453461f02b1f6e")
pipeline = FreeInpaintingPipeline(
    image_generator=MockImageGenerator(route="./img.png"),
    grounded_segmentator=AutodistillServiceSegmentator(api=AutodistillServiceClient(
        base_url="http://0.0.0.0:8001",

    )),
    image_inpainter=ComfyImageInpainter()
)

result = pipeline.generate(FreeInpaintingConditioning(
    instance_environment="sea",
    background_prompt="A beutiful ocean",
    classes=["sailboat", "cargoship"],
    instances=1,
    threshold=0.8,
    classes_id=[23, 23],
    grid_number=3
))

image = client.upload_image_to_dataset(result.image, dataset_id=23)
client.upload_annotations(23, image.id, result.annotations)
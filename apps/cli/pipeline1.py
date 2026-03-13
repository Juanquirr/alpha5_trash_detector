from apps.cli.config import get_config_value
from core.dependencies.ai.discriminative_ai.grounded_segmentator.autodistill_service_segmentator import \
    AutodistillServiceSegmentator
from core.dependencies.ai.generative_ai.image_generators.comfy_image_generator import ComfyuiImageGenerator
from core.dependencies.ai.generative_ai.image_generators.comfy_workflows import text_to_image_workflow
from core.dependencies.ai.generative_ai.image_generators.mock_image_generator import MockImageGenerator
from core.dependencies.api.autodistill_service_sdk import AutodistillServiceClient
from core.pipelines.grounded_diffusion.main import GroundedDiffusionPipeline, GroundedDiffusionConditioning
from datagen_sdk.client import DatagenClient


def execute_pipeline1(data, dataset_id):
    client = DatagenClient(api_key=get_config_value("api_key"))
    pipeline = GroundedDiffusionPipeline(
        image_generator=ComfyuiImageGenerator(workflow=text_to_image_workflow),
        grounded_segmentator=AutodistillServiceSegmentator(api=AutodistillServiceClient(
            base_url=get_config_value("Autodistill service"),
        ))
    )

    result = pipeline.generate(GroundedDiffusionConditioning(
        **data
    ))
    image = client.upload_image_to_dataset(result.image, dataset_id=dataset_id)
    client.upload_annotations(dataset_id, image.id, result.annotations)


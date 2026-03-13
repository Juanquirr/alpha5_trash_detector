from core.dependencies.ai.discriminative_ai.grounded_segmentator.autodistill_service_segmentator import \
    AutodistillServiceSegmentator
from core.dependencies.ai.generative_ai.image_generators.comfy_image_generator import ComfyuiImageGenerator
from core.dependencies.ai.generative_ai.image_generators.comfy_workflows import text_to_image_workflow
from core.dependencies.ai.generative_ai.image_generators.mock_image_generator import MockImageGenerator
from core.dependencies.ai.generative_ai.image_inpainters.comfy_image_inpainter import ComfyImageInpainter
from core.dependencies.api.autodistill_service_sdk import AutodistillServiceClient
from core.dependencies.api.diffusers_service_sdk import GenerationParams, DiffusersServiceClient
from core.pipelines.cross_attention_clustering.main import CrossAttentionClusteringPipeline, \
    CrossAttentionClusteringConditioning
from core.pipelines.free_inpainting.main import FreeInpaintingPipeline, FreeInpaintingConditioning
from core.pipelines.grounded_diffusion.main import GroundedDiffusionPipeline, GroundedDiffusionConditioning
from datagen_sdk.client import DatagenClient
from PIL import Image




client = DatagenClient(api_key="8b5c1f64b7a4887c012b3786f7453461f02b1f6e")
pipeline = CrossAttentionClusteringPipeline(
    diffusers_client=DiffusersServiceClient(base_url="https://diffusers.autoescuelaseco.cloud")
)

result = pipeline.generate(CrossAttentionClusteringConditioning(
prompt="A single big sailboat in a beutiful ocean",
instance_name="sailboat",
min_area=700,
class_id=23
))

image = client.upload_image_to_dataset(result.image, dataset_id=23)
client.upload_annotations(23, image.id, result.annotations)
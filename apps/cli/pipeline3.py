from apps.cli.config import get_config_value
from core.dependencies.api.diffusers_service_sdk import GenerationParams, DiffusersServiceClient
from core.pipelines.cross_attention_clustering.main import CrossAttentionClusteringPipeline, \
    CrossAttentionClusteringConditioning
from datagen_sdk.client import DatagenClient




def execute_pipeline3(data, dataset_id):
    client = DatagenClient(api_key=get_config_value("api_key"))
    pipeline = CrossAttentionClusteringPipeline(
        diffusers_client=DiffusersServiceClient(base_url=get_config_value("Diffusers service"))
    )

    result = pipeline.generate(CrossAttentionClusteringConditioning(
        **data
    ))
    image = client.upload_image_to_dataset(result.image, dataset_id=dataset_id)
    client.upload_annotations(dataset_id, image.id, result.annotations)

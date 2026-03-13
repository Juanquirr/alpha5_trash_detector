from uuid import uuid4

from attention_map_diffusers import init_pipeline, attn_maps
from diffusers import DiffusionPipeline
from pydantic import BaseModel

from core.dependencies.ai.generative_ai.diffusion.attention_utils import AttentionMapExtractor, compute_avg_map, \
    extract_refined_boxes
from core.dependencies.api.diffusers_service_sdk import DiffusersServiceClient, GenerationParams
from core.dependencies.api.sam_service_sdk import SegmentationRefinerClient
from core.pipelines.pipeline import DatasetGenerationPipeline, T, PipelineResult, Annotation
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn


class CrossAttentionClusteringConditioning(BaseModel):
    prompt : str
    gamma: int = 1.5
    instance_name: str
    class_id : int
    min_area: int = 700



class CrossAttentionClusteringPipeline(DatasetGenerationPipeline[CrossAttentionClusteringConditioning]):
    diffusers_client : DiffusersServiceClient
    def _compute_area(self, box):
        width = box[2] - box[0]
        height = box[3] - box[1]
        return width * height

    def generate(self, conditioning: CrossAttentionClusteringConditioning) -> PipelineResult:
        with Progress(
                TextColumn("[bold]{task.fields[status]}"),
                BarColumn(bar_width=None),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                transient=True
        ) as progress:
            task = progress.add_task("Running CrossAttention Pipeline", total=100, status="🔄 Starting...")

            # Step 1: Generar imagen con Diffusers
            progress.update(task, status="🧪 Generating image...")
            config = GenerationParams(
                prompt=conditioning.prompt,
                num_inference_steps=15,
                guidance_scale=8.0,
                height=768,
                width=768
            )
            result = self.diffusers_client.generate(config)
            image = result.image
            progress.update(task, advance=25)

            # Step 2: Calcular attention map
            progress.update(task, status="🧠 Computing attention map...")
            end_map = compute_avg_map(conditioning.instance_name, result.attention_maps)
            progress.update(task, advance=25)

            # Step 3: Extraer cajas con SAM
            progress.update(task, status="📦 Extracting refined boxes...")
            refined_boxes = extract_refined_boxes(
                end_map,
                image,
                SegmentationRefinerClient().refine_bounding_box,
                gamma=2.0
            )
            progress.update(task, advance=20)

            # Step 4: Filtrar por área mínima
            progress.update(task, status="📐 Filtering boxes by area...")
            filtered_boxes = [
                box for box in refined_boxes if self._compute_area(box) >= conditioning.min_area
            ]
            progress.update(task, advance=15)

            # Step 5: Construir anotaciones
            progress.update(task, status="📝 Building annotations...")
            bboxes = list(map(lambda x: {
                "label": conditioning.class_id,
                "id": str(uuid4()),
                "type": "bounding_box",
                "data": {
                    "point": [x[0] / image.size[0], x[1] / image.size[1]],
                    "width": abs((x[0] / image.size[0] - x[2] / image.size[0])),
                    "height": abs((x[1] / image.size[1] - x[3] / image.size[1]))
                }
            }, filtered_boxes))
            progress.update(task, advance=15)

        return PipelineResult(
            image=image,
            annotations=[Annotation(**bbox) for bbox in bboxes]
        )

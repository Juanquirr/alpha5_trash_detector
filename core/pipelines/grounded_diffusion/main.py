from time import sleep
from threading import Thread, Event
from pydantic import BaseModel
from ..pipeline import DatasetGenerationPipeline, PipelineResult, Annotation
from ...dependencies.ai.discriminative_ai.grounded_segmentator.grounded_segmentator import GroundedSegmentator
from ...dependencies.ai.generative_ai.image_generators.image_generator import ImageGenerator
from ...utils import polygon_to_bbox
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn


def simulate_progress(progress, task_id, stop_event, total, delay):
    advanced = 0
    while advanced < total and not stop_event.is_set():
        progress.update(task_id, advance=1)
        sleep(delay)
        advanced += 1


class GroundedDiffusionConditioning(BaseModel):
    prompt: str
    negative_prompt: str = ""
    ground_truth: int
    generic_class: str


class GroundedDiffusionPipeline(DatasetGenerationPipeline[GroundedDiffusionConditioning], BaseModel):
    image_generator: ImageGenerator
    grounded_segmentator: GroundedSegmentator

    def generate(self, conditioning: GroundedDiffusionConditioning) -> PipelineResult:
        with Progress(
            TextColumn("[bold]{task.fields[status]}"),
            BarColumn(bar_width=None, complete_style="magenta"),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            transient=True
        ) as progress:
            task = progress.add_task("Pipeline", total=100, status="💭 Generating image...")

            # Image generation
            stop_event = Event()
            thread = Thread(target=simulate_progress, args=(progress, task, stop_event, 50, 1))
            thread.start()
            image = self.image_generator.generate(conditioning.prompt, conditioning.negative_prompt)
            stop_event.set(); thread.join()

            # Segmenting
            progress.update(task, status="🧠 Segmenting image...")
            progress.columns[1].complete_style = "cyan"
            stop_event = Event()
            thread = Thread(target=simulate_progress, args=(progress, task, stop_event, 30, 0.8))
            thread.start()
            labels = self.grounded_segmentator.segment(image, conditioning.generic_class)
            stop_event.set(); thread.join()

            # Remaining steps
            progress.update(task, status="📐 Converting polygons...")
            bboxes = [polygon_to_bbox(x.data.points) for x in labels.annotations]
            progress.update(task, advance=10)

            progress.update(task, status="📦 Building annotations...")
            annotations = [Annotation(type="bounding_box", label=conditioning.ground_truth, data=b) for b in bboxes]
            progress.update(task, advance=10)

        return PipelineResult(image=image, annotations=annotations)

from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from pydantic import BaseModel
from PIL import Image
import tempfile, os
from core.dependencies.ai.discriminative_ai.grounded_segmentator.grounded_segmentator import GroundedSegmentator

class LocalGroundedSegmentator(GroundedSegmentator, BaseModel):
    def segment(self, image: Image.Image, label: str):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f.name)
            tmp_path = f.name
        model = GroundedSAM(CaptionOntology({label: label}))
        results = model.predict(tmp_path)
        os.unlink(tmp_path)
        return results

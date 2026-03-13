from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from core.dependencies.ai.quality_evaluators.aesthetic_evaluators.aesthetic_evaluator import AestheticQualityEvaluator


class NIMAAestheticQualityEvaluator(AestheticQualityEvaluator):
    def __init__(self, model_path: str):
        self.model = load_model(model_path)  # Load the NIMA model

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = image.resize((224, 224))  # Resize image to 224x224
        image_array = img_to_array(image)  # Convert image to array
        image_array = preprocess_input(image_array)  # Preprocess for MobileNetV2
        return np.expand_dims(image_array, axis=0)  # Add batch dimension

    def evaluate(self, image: Image.Image) -> float:
        processed_image = self.preprocess_image(image)
        predictions = self.model.predict(processed_image)[0]  # Predict aesthetic scores
        aesthetic_score = sum((i + 1) * predictions[i] for i in range(len(predictions)))
        return aesthetic_score

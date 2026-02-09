"""
Base components for Alpha5 inference methods.
Defines shared `InferenceResult` and `InferenceMethod` classes.
"""


class InferenceResult:
    """Inference result with metadata."""

    def __init__(self, image, boxes, scores, classes, method_name, params, elapsed_time, num_detections):
        self.image = image  # Annotated image
        self.boxes = boxes
        self.scores = scores
        self.classes = classes
        self.method_name = method_name
        self.params = params
        self.elapsed_time = elapsed_time
        self.num_detections = num_detections


class InferenceMethod:
    """Base class for all inference methods."""

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.default_params = {}

    def run(self, image, model, params):
        """
        Run inference on the image.

        Args:
            image: numpy array (BGR) with the input image.
            model: YOLO model instance.
            params: dict with method-specific parameters.

        Returns:
            InferenceResult with annotated image and metadata.
        """
        raise NotImplementedError

    def get_params_config(self):
        """Return default parameter configuration (e.g. for GUI)."""
        return self.default_params


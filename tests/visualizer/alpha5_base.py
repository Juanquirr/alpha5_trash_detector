"""
Base classes for Alpha5 - Inference System
"""

class InferenceResult:
    def __init__(self, image, boxes, scores, classes, method_name, params, elapsed_time, num_detections):
        self.image = image
        self.boxes = boxes
        self.scores = scores
        self.classes = classes
        self.method_name = method_name
        self.params = params
        self.elapsed_time = elapsed_time
        self.num_detections = num_detections


class InferenceMethod:
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.default_params = {}

    def run(self, image, model, params):
        raise NotImplementedError

    def get_params_config(self):
        return self.default_params

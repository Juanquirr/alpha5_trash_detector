import matplotlib.pyplot as plt
import random
import numpy as np
from typing import List, Tuple, Callable
from PIL import Image
from scipy.stats import multivariate_normal
import random

from core.dependencies.ai.discriminative_ai.point_extractors.point_extractor import PointExtractor
from core.dependencies.api.mmseg_api import MMSegAPI


class MMSegPointExtractor(PointExtractor):
    api : MMSegAPI
    n_points : int = 1

    def extract(self, image: Image.Image) -> List[Tuple[int, int]]:
        segmented_img = self.api.segment_image(image)
        plt.imshow(segmented_img)
        plt.axis('off')
        plt.show()
        color = self.api.get_inference_color("sea")
        pixels = self.get_pixels_with_color(color, segmented_img)
        pixels_with_rules_applied = self.filter_pixels_by_rules(pixels, rules=[
            self.pixels_cannot_be_near_y_axis_edge_with_color(50, segmented_img, color),
            self.pixels_cannot_be_near_x_axis_edge_with_color(50, segmented_img, color)
        ])
        if not pixels_with_rules_applied:
            raise ValueError("No se encontraron píxeles válidos después de aplicar las reglas.")
        return [self.sample_from_multivariate_normal(pixels_with_rules_applied) for x in range(self.n_points)]

    def pixels_cannot_be_near_y_axis_edge_with_color(self, margin: int, img: Image.Image, color: Tuple[int, int, int]):
        img_arr = np.array(img)

        def rule(pixel: Tuple[int, int]) -> bool:
            x, y = pixel
            if (y - margin < 0) or (y + margin >= img_arr.shape[0]):
                return False
            if not (self.is_same_color(tuple(img_arr[y - margin, x]), color) and self.is_same_color(
                    tuple(img_arr[y + margin, x]), color)):
                return False
            return True

        return rule

    def pixels_cannot_be_near_x_axis_edge_with_color(self, margin: int, img: Image.Image, color: Tuple[int, int, int]):
        img_arr = np.array(img)

        def rule(pixel: Tuple[int, int]) -> bool:
            x, y = pixel
            if (x - margin < 0) or (x + margin >= img_arr.shape[1]):
                return False
            if not (self.is_same_color(tuple(img_arr[y, x - margin]), color) and self.is_same_color(
                    tuple(img_arr[y, x + margin]), color)):
                return False
            return True

        return rule

    def get_pixels_with_color(self, color: Tuple[int, int, int], img: Image.Image) -> List[Tuple[int, int]]:
        numpy_img = np.array(img)
        height, width, _ = numpy_img.shape
        pixels = [(x, y) for y in range(height) for x in range(width) if self.is_same_color(tuple(numpy_img[y, x]), color)]
        return pixels

    @staticmethod
    def is_same_color(pixel_color, target_color):
        return pixel_color[:3] == target_color[:3]

    def filter_pixels_by_rules(self, pixels: List[Tuple[int, int]], rules: List[Callable[[Tuple[int, int]], bool]]) -> \
            List[Tuple[int, int]]:
        return [pixel for pixel in pixels if all(rule(pixel) for rule in rules)]

    def sample_from_multivariate_normal(self, pixels: List[Tuple[int, int]]) -> Tuple[int, int]:
        if not pixels:
            raise ValueError("No se pueden muestrear píxeles, ya que no hay píxeles válidos disponibles.")
        mean = np.mean(pixels, axis=0)
        cov = np.cov(np.array(pixels).T)
        sampled_point = multivariate_normal.rvs(mean=mean, cov=cov)
        sampled_pixel = tuple(map(int, map(round, sampled_point)))
        return random.choice(pixels) if sampled_pixel not in pixels else sampled_pixel
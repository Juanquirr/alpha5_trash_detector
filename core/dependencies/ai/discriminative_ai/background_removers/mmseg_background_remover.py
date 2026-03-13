from PIL import Image
import numpy as np

from core.dependencies.ai.discriminative_ai.background_removers.background_remover import BackgroundRemover
from core.dependencies.api.mmseg_api import MMSegAPI


class MMSegBackgroundRemover(BackgroundRemover):
    category : str
    api : MMSegAPI

    def remove(self, image: Image.Image) -> Image.Image:
        segmented_image = self.api.segment_image(image)
        color = self.api.get_inference_color(self.category)
        arr_segmented_image = np.array(segmented_image)
        final_image = np.array(image.convert("RGBA"))

        for x in range(arr_segmented_image.shape[0]):
            for y in range(arr_segmented_image.shape[1]):
                if not self.is_same_color(arr_segmented_image[x, y], color):
                    final_image[x, y] = [0, 0, 0, 0]

        return self.remove_small_pixel_groups(final_image)

    @staticmethod
    def is_same_color(pixel_color, target_color):
        return np.array_equal(pixel_color[:3], target_color[:3])

    def remove_small_pixel_groups(self, final_image, min_size=100500):
        visited = set()
        for x in range(final_image.shape[0]):
            for y in range(final_image.shape[1]):
                if (x, y) not in visited and not np.array_equal(final_image[x, y], [0, 0, 0, 0]):
                    pixel_group = self.find_pixel_group(final_image, x, y, visited)
                    if len(pixel_group) < min_size:
                        for px, py in pixel_group:
                            final_image[px, py] = [0, 0, 0, 0]

        return Image.fromarray(final_image, "RGBA")

    def find_pixel_group(self, final_image, start_x, start_y, visited):
        stack = [(start_x, start_y)]
        pixel_group = []

        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            pixel_group.append((x, y))

            for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if (0 <= nx and nx < final_image.shape[0] and 0 <= ny  and ny < final_image.shape[1] and
                        (nx, ny) not in visited and
                        not np.array_equal(final_image[nx, ny], [0, 0, 0, 0])):
                    stack.append((nx, ny))

        return pixel_group

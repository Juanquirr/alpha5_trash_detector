from core.dependencies.ai.discriminative_ai.point_extractors.point_extractor import PointExtractor
import numpy as np
from typing import List, Tuple
from PIL import Image

class SparsePointExtractor(PointExtractor):
    min_distance_between_points: int
    max_distance_between_points: int
    initial_point: Tuple[int, int]
    n_points: int = 1
    separation_from_edges: int = 0

    def extract(self, image: Image.Image) -> List[Tuple[int, int]]:
        width, height = image.size
        points = []
        min_dist_sq = self.min_distance_between_points ** 2
        max_dist_sq = self.max_distance_between_points ** 2
        cluster_radius = self.min_distance_between_points * self.n_points

        # Adjust bounds to ensure points are not too close to the edges
        x_min = max(self.separation_from_edges, self.initial_point[0] - cluster_radius)
        x_max = min(width - self.separation_from_edges, self.initial_point[0] + cluster_radius)
        y_min = max(self.separation_from_edges, self.initial_point[1] - cluster_radius)
        y_max = min(height - self.separation_from_edges, self.initial_point[1] + cluster_radius)

        # Ensure that bounds respect the image size and separation constraint
        x_min = max(0, x_min)
        x_max = min(width, x_max)
        y_min = max(0, y_min)
        y_max = min(height, y_max)

        # Generate candidate points in the cluster region
        candidates_x = np.random.randint(x_min, x_max, size=self.n_points * 10)
        candidates_y = np.random.randint(y_min, y_max, size=self.n_points * 10)
        candidates = np.column_stack((candidates_x, candidates_y))

        # Vectorized selection of valid points
        for candidate in candidates:
            if len(points) >= self.n_points:
                break
            if not points:
                points.append(candidate)
            else:
                distances_sq = np.sum((np.array(points) - candidate) ** 2, axis=1)
                if np.all(distances_sq >= min_dist_sq) and np.all(distances_sq <= max_dist_sq):
                    points.append(candidate)

        return [(int(x), int(y)) for x, y in points]
import numpy as np
from scipy.linalg import sqrtm
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.models import inception_v3

from core.dependencies.ai.quality_evaluators.dataset_similarity_evaluators.dataset_similarity_evaluator import \
    DatasetSimilarityEvaluator


class FIDDatasetSimilarityEvaluator(DatasetSimilarityEvaluator):
    def __init__(self, batch_size: int = 32, image_size: int = 299):
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception_model.eval()  # Set model to evaluation mode

    def calculate_statistics(self, dataset_path: str) -> (np.ndarray, np.ndarray):
        """
        Calculate the mean and covariance of the dataset's feature representation.
        """
        transform = Compose([
            Resize((self.image_size, self.image_size)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Inception normalization
        ])
        dataset = ImageFolder(dataset_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        features = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                preds = self.inception_model(images)
                features.append(preds.cpu().numpy())

        features = np.concatenate(features, axis=0)
        mean = np.mean(features, axis=0)
        covariance = np.cov(features, rowvar=False)

        return mean, covariance

    def calculate_fid(self, real_stats: (np.ndarray, np.ndarray), synthetic_stats: (np.ndarray, np.ndarray)) -> float:
        """
        Compute the FID score given the mean and covariance of two datasets.
        """
        mu1, sigma1 = real_stats
        mu2, sigma2 = synthetic_stats

        # Compute the mean difference
        diff = mu1 - mu2

        # Compute the square root of the product of covariance matrices
        covmean = sqrtm(sigma1 @ sigma2)

        # Handle numerical instability
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    def evaluate(self, dataset_path: str, synthetic_dataset_path: str) -> float:
        """
        Evaluate FID score between real and synthetic datasets.
        """
        real_stats = self.calculate_statistics(dataset_path)
        synthetic_stats = self.calculate_statistics(synthetic_dataset_path)

        fid_score = self.calculate_fid(real_stats, synthetic_stats)
        return fid_score

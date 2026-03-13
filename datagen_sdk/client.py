from PIL import Image
from io import BytesIO
import base64
import requests
from datagen_sdk.models import *


class DatagenClient:
    def __init__(self, api_key: str, base_url: str = "https://datagen-api.autoescuelaseco.cloud"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }

    @staticmethod
    def _encode_image_to_data_url(img: Image.Image) -> str:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    def get_single_datasets(self,dataset_id) -> Dataset:
        """
        List datasets with optional name filter.
        """
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return Dataset(**response.json())

    def list_datasets(self, name: str = "", page: int = 1, page_size: int = 40) -> DatasetListResponse:
        """
        List datasets with optional name filter.
        """
        url = f"{self.base_url}/api/v1/datasets/"
        params = {"name": name, "page": page, "page_size": page_size}
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return DatasetListResponse(**response.json())

    def list_dataset_images(self, dataset_id: int, page: int = 1, page_size: int = 12) -> ImageListResponse:
        """
        Fetch paginated images for a dataset.
        """
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/images/"
        params = {"page": page, "page_size": page_size}
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return ImageListResponse(**response.json())

    def upload_image_to_dataset(self, img: Image.Image, dataset_id: int) -> ImageUploadResponse:
        """
        Upload a synthetic image to a dataset.
        """
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/images/"
        payload = {
            "image": self._encode_image_to_data_url(img),
            "is_synthetic": True
        }
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return ImageUploadResponse(**response.json())

    def upload_annotations(
        self,
        dataset_id: int,
        image_id: int,
        annotations: List[Annotation]
    ) -> Dict:
        """
        Upload a batch of annotations for an image.
        """
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/images/{image_id}/annotations/batch/"
        annotations_payload = [{**annotation.dict(), "is_synthetic": True} for annotation in annotations]
        response = requests.post(url, json=annotations_payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def list_dataset_labels(self, dataset_id: int) -> LabelListResponse:
        """
        Fetch all labels associated with a dataset.
        """
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/labels/"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return LabelListResponse(**response.json())

    def get_image_details(self, dataset_id: int, image_id: int) -> ImageItem:
        """
        Retrieve detailed information about a specific image in a dataset.
        """
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/images/{image_id}/"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return ImageItem(**response.json())

    def delete_image(self, dataset_id: int, image_id: int) -> None:
        """
        Delete a specific image from a dataset.
        """
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/images/{image_id}/"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()

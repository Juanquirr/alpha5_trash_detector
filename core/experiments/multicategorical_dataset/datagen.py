import requests
import base64
from PIL import Image
from io import BytesIO

def encode_image_to_data_url(img: Image.Image) -> str:
    """Encode a PIL Image to a base64 data URL."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def upload_image_to_dataset(img: Image.Image, dataset_id : int) -> dict:
    """
    Upload an image to the specified dataset API with a synthetic flag.

    Parameters:
        img (PIL.Image): The input image.

    Returns:
        dict: JSON response from the API.
    """
    url = "http://192.168.0.38:8000/api/v1/datasets/{}/images/".format(dataset_id)
    headers = {
        "Authorization": "Token 8b5c1f64b7a4887c012b3786f7453461f02b1f6e",
        "Content-Type": "application/json"
    }

    data_url = encode_image_to_data_url(img)
    payload = {
        "image": data_url,
        "is_synthetic": True
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raise exception for HTTP errors
    return response.json()


import requests
from typing import List, Dict

def upload_annotations_batch(dataset_id: int, image_id: int, annotations: List[Dict]) -> dict:
    """
    Upload a batch of bounding box annotations to a specific image in the dataset.

    Args:
        image_id (int): ID of the image in the dataset.
        annotations (List[Dict]): List of annotation dicts, each with keys: data, label, type.

    Returns:
        dict: JSON response from the API.
    """
    url = "http://192.168.0.38:8000/api/v1/datasets/{}/images/{}/annotations/batch/".format(str(dataset_id), str(image_id))
    headers = {
        "Authorization": "Token 8b5c1f64b7a4887c012b3786f7453461f02b1f6e",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=annotations, headers=headers)
    response.raise_for_status()
    return response.json()

def get_dataset_images(dataset_id=17, page=1, page_size=12):
    url = "http://192.168.0.38:8000/api/v1/datasets/{}/images/".format(dataset_id)
    params = {
        "page": page,
        "page_size": page_size
    }

    headers = {
        "Authorization": "Token 8b5c1f64b7a4887c012b3786f7453461f02b1f6e",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()       # Return the parsed JSON response
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
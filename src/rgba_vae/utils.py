# handle base64 encoded image strings
import torch
from torchvision import transforms as T

import io
import base64

from PIL import Image

from typing import List, Dict

def extract_base64_image_data(image_str: str) -> torch.Tensor:
    """
    Extract image data from a base64 encoded string.
    """
    if "base64" in image_str:
        image_str = image_str.split(",")[1]

    image_data = base64.b64decode(image_str)

    image = Image.open(io.BytesIO(image_data)).convert("RGBA")

    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
    ])

    return transform(image)

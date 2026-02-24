import torch
from torchvision import transforms as T
from torch.utils.data import Dataset

import io
import json
import base64

from PIL import Image

def extract_base64_image_data(image_str: str) -> Image.Image:
    """
    Extract image data from a base64 encoded string.
    """
    if "base64" in image_str:
        image_str = image_str.split(",")[1]

    image_data = base64.b64decode(image_str)

    image = Image.open(io.BytesIO(image_data)).convert("RGBA")

    return image

class JSONLBase64Dataset(Dataset):
    """
    Extract image data from a Jsonl file.
    """

    def __init__(self, jsonl_files: list[str]|str, image_size: int = 64, method: str = 'nearest'):
        self.jsonl_files = [jsonl_files] if isinstance(jsonl_files, str) else jsonl_files
        self.transform = T.Compose([
            T.Resize((image_size, image_size), T.InterpolationMode[method.upper()]),
            T.ToTensor()
        ])
        self.data = []

        total_length = 0

        for jsonl_file in self.jsonl_files:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    self.data.append(entry["image"])
                    total_length += 1
        
        self.total_length = total_length

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        base64_str = self.data[idx]

        image = extract_base64_image_data(base64_str)
        image = self.transform(image)

        return image, 0
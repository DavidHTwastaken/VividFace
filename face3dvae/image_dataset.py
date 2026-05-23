import random
import os
import numpy as np
import cv2
import json
from typing import Any, Optional, List, Dict, Tuple, Union, Generic, TypeVar
from io import BytesIO

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL.ImageOps import exif_transpose

class ImageDataset(Dataset):
    def __init__(self, root, training, image_size: int = 256):
        self.training = training
        self.image_size = image_size
        with open(root) as f:
            self.image_data = [line.strip() for line in f]

        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize(mean=(0.5), std=(0.5)),
        ])

    def __len__(self):
        if self.training:
            return 1000000000
        else:
            return len(self.image_data)

    def __getitem__(self, idx):
        if self.training:
            idx = random.randint(0, len(self.image_data)-1)
        while True:
            try:
                return self.inner_getitem(idx)
            except Exception as e:
                print(e)
                idx = random.randint(0, len(self.image_data)-1)

    def _read_image(self, filepath: str) -> Image.Image:
        image = Image.open(filepath)
        assert isinstance(image, Image.Image)
        image = exif_transpose(image)
        image = image.convert("RGB")
        return image

    def inner_getitem(self, idx):
        image_path: str = self.image_data[idx]
        image = self._read_image(image_path)
        pixel_values = self.transform(image)
        return {
            "pixel_values": pixel_values.unsqueeze(0),
        }

def collate_fn(examples):

    pixel_values = torch.cat([example["pixel_values"] for example in examples], dim=0)
    pixel_values = pixel_values.contiguous().float()
    batch = {
        "pixel_values": pixel_values,
    }

    return batch



import random
import os
import io
import json

import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader
from glob import glob
import torchvision.transforms as T
import torchvision.transforms.functional as TF

def process_anno_ceph(input_string):
    lines = input_string.strip().split('\n')
    array = []
    for line in lines:
        if line=='':
            line = '0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0'
        row = list(map(int, line.split(',')))
        array.append(row)
    return np.array(array)


def process_anno_local(txt_path):
    bboxes = []
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line=='':
                line = '0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0'
            bboxes.append(list(map(int, line.split(','))))
    return np.array(bboxes)

class VideoDataset(Dataset):
    def __init__(self, root, training=True, image_size: int = 256, pred_frames: int = 8):
        self.training = training
        self.image_size = image_size
        self.pred_frames = pred_frames
        with open(root) as f:
            self.video_data = [json.loads(line.strip()) for line in f]

        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize(mean=(0.5), std=(0.5)),
        ])


    def __len__(self):
        if self.training:
            return 1000000000
        else:
            return len(self.video_data)

    def __getitem__(self, idx):
        if self.training:
            idx = random.randint(0, len(self.video_data)-1)
        while True:
            try:
                return self.inner_getitem(idx)
            except Exception as e:
                print(e)
                idx = random.randint(0, len(self.video_data)-1)

    def inner_getitem(self, idx):
        video_path: str = self.video_data[idx]["url"]
        anno_path = video_path.replace(".mp4", ".txt")

        anno_path = self.video_data[idx]["anno"]
        anno = process_anno_local(anno_path)

        video_reader = VideoReader(video_path)
        video_length = len(video_reader)
        pixel_values = []

        if self.pred_frames < 0:
            num_frames = video_length
            video_start = 0
        else:
            num_frames = self.pred_frames
            video_start = np.random.randint(0, video_length - num_frames)

        indices = [video_start+i for i in range(num_frames)]
        frames = video_reader.get_batch(indices).asnumpy() # sample_frames, h, w, 3
        pixel_value = []

        for ii, (i, frame) in enumerate(zip(indices, frames)):
            pixel_value.append(self.transform(Image.fromarray(frame)))

        pixel_values.append(torch.stack(pixel_value, dim=0))

        pixel_values = torch.stack(pixel_values, dim=0)
        return {
            "pixel_values": pixel_values,
        }

def collate_fn(examples):

    pixel_values = torch.cat([example["pixel_values"] for example in examples], dim=0)
    pixel_values = pixel_values.contiguous().float()
    batch = {
        "pixel_values": pixel_values,
    }

    return batch


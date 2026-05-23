import random
import os
import json
import time
import copy as copy

import cv2
import dlib
from decord import VideoReader
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import Any, Optional, List, Dict, Tuple, Union, Generic, TypeVar
from PIL.ImageOps import exif_transpose
from glob import glob
import numpy as np

from occ_aug import img_get_occ_aug
from occ_aug_video import video_get_occ_aug

def crop_and_adjust_bbox(image_array, bbox, lmks, hight, width):

    image = Image.fromarray(image_array)
    width, height = image.size
    new_size = min(width, height)

    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = left + new_size
    bottom = top + new_size
    cropped_image = image.crop((left, top, right, bottom))
    xmin, ymin, xmax, ymax = bbox
    adjusted_bbox = [
        max(xmin - left, 0),
        max(ymin - top, 0),
        min(xmax - left, new_size),
        min(ymax - top, new_size)
    ]
    lmks = lmks.reshape(5, 2)
    lmks[:, 0] = lmks[:, 0] - left
    lmks[:, 1] = lmks[:, 1] - top
    lmks = np.clip(lmks, 0, new_size)

    bbox_scale_size = new_size / 512
    adjusted_bbox = [int(obj / bbox_scale_size) for obj in adjusted_bbox]
    lmks = lmks / bbox_scale_size
    return cropped_image, adjusted_bbox, lmks


def no_bbox_crop(image):
    crop_size=224
    h, w, c = image.shape 
    crop_size = min(crop_size, h, w)
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    cropped_image = image[start_h:start_h + crop_size, start_w:start_w + crop_size]
    return Image.fromarray(cropped_image)

def pil_to_cv2(pil_image):
    cv2_image = np.array(pil_image)
    if pil_image.mode == 'RGB':
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
    return cv2_image

def cv2_to_pil(cv2_image):
    if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 3:
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image)
    return pil_image

def check_twins(twins_list):
    buf = twins_list[0]
    for i in range(1, len(twins_list)):
        assert twins_list[i] == buf
    return buf

def get_anno(anno_path):
    anno_path = anno_path
    anno = np.loadtxt(anno_path, dtype=int, delimiter=",")
    return anno


def get_random_ref_face(video_path):
    ref_video_anno_path = video_path.replace('.mp4', '.txt')
    video_reader = VideoReader(video_path)
    ref_idx = random.choice(range(len(video_reader)))
    bbox = get_anno(ref_video_anno_path)[ref_idx][:4]
    ref_img = extract_face(video_reader[ref_idx].asnumpy(),bbox)
    return ref_img

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

def string_to_array(input_string):
    lines = input_string.strip().split('\n')
    array = []
    for line in lines:
        row = list(map(int, line.split(',')))
        array.append(row)

    return np.array(array)

def refine_path(path):
    return path

arcface_dst = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)


def extract_face(image, mask):
    x1, y1, x2, y2 = mask
    width = x2 - x1
    height = y2 - y1

    size = max(width, height)

    center_x = x1 + width // 2
    center_y = y1 + height // 2
    new_x1 = center_x - size // 2
    new_y1 = center_y - size // 2
    new_x2 = new_x1 + size
    new_y2 = new_y1 + size

    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(image.shape[1], new_x2)
    new_y2 = min(image.shape[0], new_y2)

    face = image[new_y1:new_y2, new_x1:new_x2]

    return face

def load_listdata(filepath):
    _, suffix = os.path.splitext(filepath)
    if suffix == ".pkl":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
    elif suffix == ".jsonl":
        with open(filepath, "r") as f:
            data = [json.loads(line.strip()) for line in f.readlines()]
    elif suffix == ".json":
        with open(filepath, "r") as f:
            data = json.loads(f.read())
    else:
        with open(filepath, "r") as f:
            data = f.readlines()
    return data

def video_crop_aligned_face(image: np.ndarray, keypoint5: np.ndarray, face_size: int = 160):
    # estimate face alignment transfrom
    if isinstance(keypoint5, list):
        keypoint5 = np.array(keypoint5)
    assert keypoint5.shape == (5, 2)
    ratio = float(face_size) / 112.0
    dst = arcface_dst * ratio
    M = cv2.estimateAffinePartial2D(keypoint5, dst, method=cv2.LMEDS)[0]
    warped = cv2.warpAffine(image, M, (face_size, face_size), flags=cv2.INTER_LANCZOS4, borderValue=0)
    return warped

class HydridDataset(Dataset):
    def __init__(self,
            video_root,
            image_size,
            cond_frames,
            pred_frames,
            num_repeats,
            img_metafiles,
            tokenizer,
            proportion_empty_prompts,
            proportion_empty_face,
            task,
            is_video=0.5,
            debug_mode=False,
            data_type='video',
            ) -> None:
        super().__init__()
        print(f'Using Tuple Image from {img_metafiles}')
        print(f'Using Triple Video.')
        self.video__init__(video_root, image_size, cond_frames, pred_frames, num_repeats)
        self.img__init__(img_metafiles, tokenizer, proportion_empty_prompts, proportion_empty_face, task)
        self.is_video = is_video
        self.debug_mode = debug_mode
        if self.debug_mode:
            print('################ DEBUG MODE ################')
        self.data_type = data_type
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./Deep3DFaceRecon/shape_predictor_68_face_landmarks.dat')


    def video__init__(self, root, image_size, cond_frames, pred_frames, num_repeats):
        self.image_size = image_size
        self.cond_frames = cond_frames
        self.pred_frames = pred_frames
        self.num_repeats = num_repeats
        with open(root) as f:
            self.video_data = [json.loads(line.strip()) for line in f]

        self.video_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize(mean=(0.5), std=(0.5)),
        ])

        self.video_face_transform = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
        ])

        self.video_attr_transform = T.Compose([
            T.ToTensor(),
        ])

    def img__init__(self, metafiles, tokenizer, proportion_empty_prompts, proportion_empty_face, task):
        assert tokenizer is not None, "tokenizer is None"
        self.proportion_empty_prompts = proportion_empty_prompts
        self.proportion_empty_face = proportion_empty_face
        self.img_data: List[Dict[str, Any]] = []
        for metafile in metafiles:
            if ":" in metafile:
                metafile, repeat = metafile.split(":")
                repeat = int(repeat)
            else:
                repeat = 1
            data = load_listdata(metafile)
            for _ in range(repeat):
                self.img_data.extend(data)
        self.idx_to_shape = self.img_data
        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.5), std=(0.5)),
        ])
        self.tokenizer = tokenizer
        self.img_face_transform = T.Compose([
            T.Resize((112, 112), interpolation=T.InterpolationMode.LANCZOS),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
        ])
        self.img_attr_transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
        ])
        self.task = task

        print(f"datasize: {len(self.img_data)} -> {len(self.idx_to_shape)}")


    def __len__(self):
        return len(self.video_data)*2 + len(self.idx_to_shape)

    def video__getitem__(self, idx):
        random.seed(time.time())
        idx = idx % len(self.video_data)
        video_path: str = self.video_data[idx]["url"]
        width, height = self.video_data[idx]["shape"]
        anno_path = video_path.replace(".mp4", ".txt")
        ref_video_path = self.video_data[idx].get("ref_urls", None)
        triple_twins_path = self.video_data[idx].get("generated_video_url", None)

        if ref_video_path:
            this_ref_np = get_random_ref_face(random.choice(ref_video_path))

        anno_path = self.video_data[idx]["anno"]
        anno = process_anno_local(anno_path)


        video_reader = VideoReader(video_path)
        video_length = len(video_reader)

        if triple_twins_path:
            twins_video_reader = VideoReader(triple_twins_path)
            twins_video_length = len(video_reader)
            assert video_length == twins_video_length

        cond_pixel_values = []
        pixel_values = []
        raw_pixel_values = []
        face_pixel_values = []
        attr_pixel_values = []
        mask_values = []
        lmks_values = []
        num_frames = self.cond_frames + self.pred_frames
        enlarge = np.random.choice([0.05, 0.05, 0.1, 0.1, 0.15, 0.2])

        for _ in range(self.num_repeats):
            video_start = np.random.randint(0, video_length - num_frames)
            j = np.random.randint(0, video_length)
            anno[j][0]=max(anno[j][0], 0)
            anno[j][1]=max(anno[j][1], 0)
            anno[j][2]=min(anno[j][2], width)
            anno[j][3]=min(anno[j][3], height)

            if ref_video_path:
                crop_face = Image.fromarray(this_ref_np)
            else:
                no_bbox = (anno[j][0]==0) and \
                                (anno[j][1]==0) and \
                                    (anno[j][2]==0) and \
                                        (anno[j][3]==0)
                if not no_bbox:
                    crop_face = video_crop_aligned_face(video_reader[j].asnumpy(), anno[j][4:14].reshape(5, 2), face_size = 112)
                    crop_face = Image.fromarray(crop_face)
                else:
                    crop_face = no_bbox_crop(video_reader[j].asnumpy()).resize((112, 112), resample=Image.Resampling.LANCZOS)


            face_pixel_values.append(torch.stack([self.video_face_transform(crop_face), ], dim=0))

            indices = [video_start+i for i in range(num_frames)]
            frames = video_reader.get_batch(indices).asnumpy() # sample_frames, h, w, 3
            twins_frames = None
            if triple_twins_path:
                twins_frames = twins_video_reader.get_batch(indices).asnumpy() # sample_frames, h, w, 3

            frames_copy = copy.deepcopy(frames)
            if random.random() > 1.0:
                frames, twins_frames = video_get_occ_aug(frames, twins_frames)

            cond_pixel_value = []
            pixel_value = []
            raw_pixel_value = []
            attr_pixel_value = []
            mask_value = []
            lmks_value = []
            has_gt_frame = np.random.random() < 0.9

            for ii, (i, frame, frame_copy) in enumerate(zip(indices, frames, frames_copy)):

                if ii < self.cond_frames:
                    cond_pixel_value.append(self.video_transform(crop_and_adjust_bbox(frame, anno[i][:4], anno[i][4: 14], height, width)[0]))
                else:
                    bk_past_anno = copy.copy(anno[i])
                    this_crop_frame, this_anno, this_lmk = crop_and_adjust_bbox(frame, anno[i][:4], anno[i][4:14], height, width)
                    this_raw_crop_frame, _, _ = crop_and_adjust_bbox(frame_copy, anno[i][:4], anno[i][4:14], height, width)

                    lmks_value.append(torch.from_numpy(this_lmk))
                    pixel_value.append(self.video_transform(this_crop_frame))
                    raw_pixel_value.append(self.video_transform(this_raw_crop_frame))

                    if triple_twins_path:
                        crop_face = extract_face(np.asarray(twins_frames[ii]), bk_past_anno[:4])
                        crop_face = Image.fromarray(crop_face).resize((224, 224), resample=Image.Resampling.LANCZOS)

                    else:
                        this_no_bbox = (bk_past_anno[0]==0) and \
                                        (bk_past_anno[1]==0) and \
                                            (bk_past_anno[2]==0) and \
                                                (bk_past_anno[3]==0)
                        if not this_no_bbox:
                            crop_face = extract_face(np.asarray(frames[ii]), bk_past_anno[:4])
                            crop_face = Image.fromarray(crop_face).resize((224, 224), resample=Image.Resampling.LANCZOS)
                        else:
                            crop_face = no_bbox_crop(frames[ii]).resize((224, 224), resample=Image.Resampling.LANCZOS)

                    attr_pixel_value.append(self.video_attr_transform(crop_face))
                    mask = np.zeros((512, 512))

                    x1, y1, x2, y2 = this_anno
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, width), min(y2, height)

                    l = max(y2-y1, x2-x1)
                    l = l + int(round(enlarge * l))
                    cx, cy = (x1+x2)//2, (y1+y2) // 2
                    x1 = max(0, cx - l // 2)
                    y1 = max(0, cy - l // 2)
                    x2 = min(width, cx + l // 2)
                    y2 = min(height, cy + l // 2)
                    mask[y1:y2, x1:x2] = 1
                    mask = cv2.resize(mask, (self.image_size, self.image_size))
                    mask_value.append(torch.from_numpy(mask).float())
            ### HERE
            if has_gt_frame:
                cond_pixel_values.append(torch.stack(cond_pixel_value, dim=0))
            else:
                cond_pixel_values.append(torch.zeros_like(torch.stack(cond_pixel_value, dim=0)))

            pixel_values.append(torch.stack(pixel_value, dim=0))
            raw_pixel_values.append(torch.stack(raw_pixel_value, dim=0))
            attr_pixel_values.append(torch.stack(attr_pixel_value, dim=0))
            mask_values.append(torch.stack(mask_value, dim=0))
            lmks_values.append(torch.stack(lmks_value, dim=0))

        cond_pixel_values = torch.stack(cond_pixel_values, dim=0)
        pixel_values = torch.stack(pixel_values, dim=0)
        raw_pixel_values = torch.stack(raw_pixel_values, dim=0)
        attr_pixel_values = torch.stack(attr_pixel_values, dim=0)
        face_pixel_values = torch.stack(face_pixel_values, dim=0)
        mask_values = torch.stack(mask_values, dim=0)
        lmks_values = torch.stack(lmks_values, dim=0)

        return {
            "pixel_values": pixel_values,
            "raw_pixel_values": raw_pixel_values,
            "cond_pixel_values": cond_pixel_values,
            "attr_pixel_values": attr_pixel_values,
            "face_pixel_values": face_pixel_values,
            "masks": mask_values,
            'twins': triple_twins_path is not None,
            'lmks_values': lmks_values
        }

    def img__getitem__(self, index):
        return self.getitem_custom(
            index,
            use_pose = "[pose]" in self.task,
            use_mask = "[mask]" in self.task,
            use_ref = "[ref]" in self.task,
        )

    def _read_image(self, filepath: str) -> Image.Image:
        image = Image.open(filepath)
        assert isinstance(image, Image.Image)
        image = exif_transpose(image)
        image = image.convert("RGB")
        return image

    def getitem_custom(self, index, use_pose: bool=False, use_mask: bool = False, use_ref: bool = True):
        img_batch = int(self.pred_frames * self.num_repeats) + 1
        return collate_fn([
            self.getitem_single_custom(index, use_pose=use_pose, use_mask=use_mask, use_ref=use_ref) \
                for _ in range(img_batch)
        ])

    def getitem_single_custom(self, index, use_pose: bool=False, use_mask: bool = False, use_ref: bool = True):
        """
        use_pose: use pitch, yaw, roll pose.
        use_kps5: use landmark5 as coord cond otherwise use xyxy bbox.
        use_attr: use additional 5 face attr: ["Male", "Young", "Smiling", "Heavy_Makeup", "Attractive"].
        use_mask: use face region mask with random enlarge, "inpaint" mode will pad to square.
        use_ref: use ref face rather than use crop face.
        """
        width, height = 512, 512
        metadata = json.loads(random.choice(self.img_data))
        caption = metadata["caption"]

        tmp_path = refine_path(metadata["tmppath"])
        swap_path = refine_path(metadata["swapath"])
        ref_path = refine_path(metadata["sameIDtmp"])

        tmp_face = metadata["face"]
        swap_face = metadata["face"]
        ref_face = None
        crop_face = None


        if "tag" in metadata:
            tag: str = metadata["tag"]
            tag = tag.split(", ")
            np.random.shuffle(tag)
            tag = ", ".join(tag)
            caption = caption + ", " + tag

        if "tw" in metadata:
            caption = metadata["tw"] + ", " + caption

        num_chs = 4
        if use_pose:
            num_chs += 3

        coord = np.zeros((num_chs, ))

        rnd = np.random.random()
        caption = "" if rnd < self.proportion_empty_prompts else caption
        if 0 < rnd - self.proportion_empty_prompts/3 < self.proportion_empty_face:
            valid = 0
        else:
            valid = 1

        try:
            tmp_image = self._read_image(tmp_path)
            swap_image = self._read_image(swap_path)

            tmp_image_copy = tmp_image.copy()
            if random.random() > 1.0:
                tmp_image, swap_image = img_get_occ_aug(
                    pil_to_cv2(tmp_image), pil_to_cv2(swap_image),
                )
                tmp_image, swap_image = cv2_to_pil(tmp_image), cv2_to_pil(swap_image)


            raw_width, raw_height = tmp_image.size[:2]
            if use_mask:
                mask = np.zeros((raw_height, raw_width), dtype=np.uint8)
            face = tmp_face
            bbox, kps5, pose = face[:4], face[4:14], face[14:17]

            if use_mask:
                x1, y1, x2, y2 = bbox
                enlarge = np.random.choice([0.0, 0.05, 0.05, 0.1, 0.1, 0.15, 0.2, 0.25])
                l = max(y2-y1, x2-x1)
                l = l + int(round(enlarge * l))
                cx, cy = (x1+x2)//2, (y1+y2) // 2
                x1 = max(0, cx - l // 2)
                y1 = max(0, cy - l // 2)
                x2 = min(raw_width, cx + l // 2)
                y2 = min(raw_height, cy + l // 2)
                mask[y1:y2, x1:x2] = 1

            w, h = tmp_image.size
            aspect_target = width / height
            aspect_image = w / h
            if aspect_image > aspect_target:
                h = height
                w = round(height * aspect_image)
            elif aspect_image < aspect_target:
                w = width
                h = round(width / aspect_image)
            else:
                w = width
                h = height

            left = 0 if w<=width else np.random.randint(0, w - width)
            top  = 0 if h<=height else np.random.randint(0, h - height)
            right = left + width
            bottom = top + height

            if caption != "":
                coord = kps5[:4]
                coord = np.asarray(coord)
                coord = coord / np.asarray([raw_width, raw_height] * 2)
                coord = np.clip(coord, 0.0, 1.0)
                coord = coord * np.asarray([w, h] * 2)
                coord = coord - np.asarray([left, top] * 2)
                coord = coord / np.asarray([width, height] * 2)
                coord = np.clip(coord, 0.0, 1.0)

                if use_pose:
                    pose = np.asarray(pose)
                    pose = (pose + np.asarray([180, 90, 180])) / np.asarray([360, 180, 360])
                    coord = np.concatenate([coord, pose], axis=-1)

            if valid == 1:
                kps5 = np.asarray(kps5).reshape(5, 2)
                crop_face = extract_face(np.asarray(swap_image), bbox)
                crop_face = Image.fromarray(crop_face)
                ref_face = self._read_image(ref_path)
                if ref_face is None:
                    ref_face = crop_face


            if use_mask:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)[top:bottom, left:right]
            tmp_image = tmp_image.resize((w, h), resample=Image.Resampling.LANCZOS).crop((left, top, right, bottom))
            tmp_image_copy = tmp_image_copy.resize((w, h), resample=Image.Resampling.LANCZOS).crop((left, top, right, bottom))

        except Exception as e:
            print(tmp_path, e)
            tmp_image = Image.new("RGB", (width, height), color=(0, 0, 0))
            tmp_image_copy = Image.new("RGB", (width, height), color=(0, 0, 0))
            if use_mask:
                mask = np.zeros((height, width), dtype=np.uint8)
            caption = ""

        if ref_face is None:
            ref_face = Image.new("RGB", (112, 112), color=(0, 0, 0))
            crop_face = Image.new("RGB", (224, 224), color=(0, 0, 0))
            valid = 0

        lmks_values = self.detect_lmk(np.array(tmp_image_copy))
        pixel_values = self.img_transform(tmp_image)
        raw_pixel_values = self.img_transform(tmp_image_copy)
        face_pixel_values = self.img_face_transform(ref_face)
        attr_pixel_values = self.img_attr_transform(crop_face)

        cond_pixel_values = torch.zeros_like(attr_pixel_values)

        return {
            "pixel_values": pixel_values.unsqueeze(0),
            "raw_pixel_values": raw_pixel_values.unsqueeze(0),
            "cond_pixel_values": cond_pixel_values.unsqueeze(0),
            "attr_pixel_values": attr_pixel_values.unsqueeze(0),
            "face_pixel_values": face_pixel_values.unsqueeze(0),
            "masks": torch.from_numpy(mask).float().unsqueeze(0),
            "lmks_values": torch.from_numpy(lmks_values),
        }

    def detect_lmk(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) < 1:
            return np.zeros((5, 2))
        landmarks = self.predictor(gray, faces[0])
        # The indices for the 5 points of interest
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1

        # Extracting the corresponding landmark coordinates
        landmarks_list = np.stack([
            [landmarks.part(lm_idx[0]).x, landmarks.part(lm_idx[0]).y], # Nose
            np.mean([[landmarks.part(lm_idx[1]).x, landmarks.part(lm_idx[1]).y],  # Left eye
                     [landmarks.part(lm_idx[2]).x, landmarks.part(lm_idx[2]).y]], axis=0),
            np.mean([[landmarks.part(lm_idx[3]).x, landmarks.part(lm_idx[3]).y],  # Right eye
                     [landmarks.part(lm_idx[4]).x, landmarks.part(lm_idx[4]).y]], axis=0),
            [landmarks.part(lm_idx[5]).x, landmarks.part(lm_idx[5]).y], # Left corner of the mouth
            [landmarks.part(lm_idx[6]).x, landmarks.part(lm_idx[6]).y], # Right corner of the mouth
        ])

        # Reordering the points for final 5-point landmarks
        face_lmk = landmarks_list[[1, 2, 0, 3, 4], :]
        return face_lmk

    def __getitem__(self, index):

        if not self.debug_mode:

            try:
                if self.data_type == 'video':
                    random.seed(time.time())
                    try:
                        return self.video__getitem__(random.randint(0, len(self.video_data)))
                    except Exception as e:
                        print(e)
                        return self.video__getitem__(random.randint(0, len(self.video_data)))
                else:
                    random.seed(time.time())
                    try:
                        return self.img__getitem__(random.randint(0, len(self.img_data)))
                    except Exception as e:
                        print(e)
                        return self.img__getitem__(random.randint(0, len(self.img_data)))
            except:
                if self.data_type == 'video':
                    random.seed(time.time())
                    try:
                        return self.video__getitem__(random.randint(0, len(self.video_data)))
                    except Exception as e:
                        print(e)
                        return self.video__getitem__(random.randint(0, len(self.video_data)))
                else:
                    random.seed(time.time())
                    try:
                        return self.img__getitem__(random.randint(0, len(self.img_data)))
                    except Exception as e:
                        print(e)
                        return self.img__getitem__(random.randint(0, len(self.img_data)))

        else:
            if self.data_type == 'video':
                return self.video__getitem__(random.randint(0, len(self.video_data)))
            else:
                random.seed(time.time())
                return self.img__getitem__(random.randint(0, len(self.img_data)))

def collate_fn(examples):

    cond_pixel_values = torch.cat([example["cond_pixel_values"] for example in examples], dim=0)
    cond_pixel_values = cond_pixel_values.contiguous().float()

    pixel_values = torch.cat([example["pixel_values"] for example in examples], dim=0)
    pixel_values = pixel_values.contiguous().float()

    raw_pixel_values = torch.cat([example["raw_pixel_values"] for example in examples], dim=0)
    raw_pixel_values = raw_pixel_values.contiguous().float()

    face_pixel_values = torch.cat([example["face_pixel_values"] for example in examples], dim=0)
    face_pixel_values = face_pixel_values.contiguous().float()

    attr_pixel_values = torch.cat([example["attr_pixel_values"] for example in examples], dim=0)
    attr_pixel_values = attr_pixel_values.contiguous().float()

    masks = torch.cat([example["masks"] for example in examples], dim=0)
    masks = masks.contiguous().float()

    is_twins = [example.get("twins", None) for example in examples]
    is_twins = check_twins(is_twins)

    lmks_values = torch.cat([example["lmks_values"] for example in examples], dim=0)
    lmks_values = lmks_values.contiguous().float()

    batch = {
        "pixel_values": pixel_values,
        "raw_pixel_values": raw_pixel_values,
        "cond_pixel_values": cond_pixel_values,
        "face_pixel_values": face_pixel_values,
        "attr_pixel_values": attr_pixel_values,
        "masks": masks,
        "is_twins": is_twins,
        "lmks_values": lmks_values,
    }

    return batch

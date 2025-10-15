from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
import os
import time
import random

import dlib
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from moviepy.editor import VideoFileClip, ImageSequenceClip
from options.test_options import TestOptions
from tqdm import tqdm

from models import create_model
from util.preprocess import align_img, POS, extract_5p
from util.load_mats import load_lm3d

def resize_n_crop_img(img, lm, t, s, target_size=224., mask=None):
    w0, h0 = img.size
    w = int(w0 * s)
    h = int(h0 * s)
    left = int(w / 2 - target_size / 2 + (t[0] - w0 / 2) * s)
    right = left + target_size
    up = int(h / 2 - target_size / 2 + (h0 / 2 - t[1]) * s)
    below = up + target_size

    img = img.resize((w, h), resample=Image.BILINEAR)
    img = img.crop((left, up, right, below))

    if mask is not None:
        mask = mask.resize((w, h), resample=Image.NEAREST)
        mask = mask.crop((left, up, right, below))

    lm = np.stack([lm[:, 0] - t[0] + w0 / 2, lm[:, 1] - t[1] + h0 / 2], axis=1) * s
    lm = lm - np.array([left, up]).reshape((1, 2))

    if mask is not None:
        return img, lm, mask, left, up
    else:
        return img, lm, left, up

def align_img(img, lm, lm3D, mask=None, target_size=224., rescale_factor=102.):
    w0, h0 = img.size
    if lm.shape[0] != 5:
        lm5p = extract_5p(lm)
    else:
        lm5p = lm

    t, s = POS(lm5p.transpose(), lm3D.transpose())
    s = rescale_factor / s

    if mask is not None:
        img_new, lm_new, mask_new, left, up = resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)
        trans_params = np.array([w0, h0, s, t[0][0], t[1][0], left, up])
        return trans_params, img_new, lm_new, mask_new
    else:
        img_new, lm_new, left, up = resize_n_crop_img(img, lm, t, s, target_size=target_size)
        trans_params = np.array([w0, h0, s, t[0][0], t[1][0], left, up])
        return trans_params, img_new, lm_new

def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    im = Image.open(im_path).convert('RGB')
    W, H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]

    trans_params, im_new, lm_new = align_img(im, lm, lm3d_std)
    if to_tensor:
        im_tensor = torch.tensor(np.array(im_new) / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm_tensor = torch.tensor(lm_new).unsqueeze(0)
        return im_tensor, lm_tensor, im, trans_params
    else:
        return im_new, lm_new, im, trans_params


def read_data_direct(im_array, lm_array, lm3d_std, to_tensor=True):
    im = Image.fromarray(im_array).convert('RGB')
    W, H = im.size
    lm = lm_array.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]

    trans_params, im_new, lm_new = align_img(im, lm, lm3d_std)
    if to_tensor:
        im_tensor = torch.tensor(np.array(im_new) / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm_tensor = torch.tensor(lm_new).unsqueeze(0)
        return im_tensor, lm_tensor, im, trans_params
    else:
        return im_new, lm_new, im, trans_params

class Face3DModel():
    def __init__(self, opt, device):
        torch.cuda.set_device(device)
        self.device = device
        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.device = device
        self.model.parallelize()
        self.model.net_recon.requires_grad_(False)
        self.model.eval()
        self.lm3d_std = load_lm3d(opt.bfm_folder)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./Deep3DFaceRecon/shape_predictor_68_face_landmarks.dat')


    def process_3dmm(self, im_tensor):
        output_coeff = self.model.net_recon(im_tensor.to(self.device))
        self.model.facemodel.to(self.device)

        pred_coeffs_dict = self.model.facemodel.split_coeff(output_coeff)
        return pred_coeffs_dict

    def merge_coeffs(self, coeffs_dict):
        id_coeffs = coeffs_dict['id']
        exp_coeffs = coeffs_dict['exp']
        tex_coeffs = coeffs_dict['tex']
        angles = coeffs_dict['angle']
        gammas = coeffs_dict['gamma']
        translations = coeffs_dict['trans']
        # Concatenate all coefficients along the last dimension (dim=1)
        coeffs = torch.cat([id_coeffs, exp_coeffs, tex_coeffs, angles, gammas, translations], dim=1)
        return coeffs


    def render(self, output_coeff, input_img, original_img, trans_params, mix=True):
        pred_vertex, pred_tex, pred_color, pred_lm = \
            self.model.facemodel.compute_for_render(output_coeff)
        pred_mask, _, pred_face = self.model.renderer(
            pred_vertex, self.model.facemodel.face_buf, feat=pred_color)

        output_vis = pred_face * pred_mask# + (1 - pred_mask) * input_img.to(self.device)
        output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()

        output_image = Image.fromarray(np.uint8(output_vis_numpy_raw[0]))

        scale_factor = trans_params[2]
        new_width = int(output_image.width / scale_factor)
        new_height = int(output_image.height / scale_factor)
        new_size = (new_width, new_height)

        output_image = output_image.resize(new_size, Image.LANCZOS)

        left = int(trans_params[5] / scale_factor)
        up = int(trans_params[6] / scale_factor)
        w0, h0 = original_img.size
        output_full = Image.new('RGB', (w0, h0))
        output_full.paste(output_image, (left, up))
        if not mix:
            mask = (np.array(output_full).sum(axis=2) > 0).astype(np.uint8)
            return output_full, mask

        alpha_channel = (np.array(output_full).sum(axis=2) > 0).astype(np.uint8) * 255
        mask = Image.fromarray(alpha_channel, mode='L')
        output_final = original_img.copy()
        output_final.paste(output_full, mask=mask)

        return output_final

    def detect_lmk(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) < 1:
            return None
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

    def process_video_for_training(self, video_images, video_lmks=None, remove_id_tex=False):
        video_images = (video_images + 1) * 255 / 2
        video_images = video_images.permute(0, 2, 3, 1)
        video_images = video_images.cpu().detach().numpy()
        video_images = np.clip(video_images, 0, 255)
        video_images = video_images.astype(np.uint8)
        if video_lmks is None:
            video_lmks = []
            for i in range(len(video_images)):
                lmk_points = self.detect_lmk(video_images[i])
                video_lmks.append(lmk_points)

        outputs = []
        masks = []
        for i in range(len(video_images)):
            if video_lmks[i] is None or np.sum(np.abs(video_lmks[i]))<1:
                output_vis_img = Image.new('RGB', (512, 512))
                mask = np.zeros((512, 512)).astype(np.uint8)
            else:
                try:
                    im_tensor, lm_tensor, im, trans_params = read_data_direct(video_images[i], video_lmks[i], self.lm3d_std)
                    frame_coeffs_dict = self.process_3dmm(im_tensor)
                    if remove_id_tex:
                        frame_coeffs_dict['id'][:] = 0 #ref_coeffs_dict['id'].repeat(8, 1)
                        frame_coeffs_dict['tex'][:] = 0 #ref_coeffs_dict['tex'].repeat(8, 1)
                    face_coeffs = self.merge_coeffs(frame_coeffs_dict)
                    output_vis_img, mask = self.render(face_coeffs, im_tensor, im, trans_params, mix=False)
                except Exception as e:
                    print(e)
                    output_vis_img = Image.new('RGB', (512, 512))
                    mask = np.zeros((512, 512)).astype(np.uint8)
            outputs.append(np.array(output_vis_img))
            masks.append(mask)
        masks = np.stack(masks, 0)
        outputs = np.stack(outputs, 0).astype(float)
        outputs = (outputs / 255.0) * 2 - 1
        return torch.from_numpy(outputs).permute(0, 3, 1, 2), torch.from_numpy(masks)

    def process_video_for_training_swap(self, face_image, face_lmk, video_images, video_lmks):
        if face_lmk is None:
            img = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            landmarks = self.predictor(gray, faces[0])
            landmarks_list = []
            for i in range(5):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                landmarks_list.append([x, y])
            face_lmk = np.array(landmarks_list)

        im_tensor, lm_tensor, im, trans_params = read_data_direct(face_image, face_lmk, self.lm3d_std)
        ref_coeffs_dict = self.process_3dmm(im_tensor)
        outputs = []
        for i in tqdm(range(len(video_images))):
            im_tensor, lm_tensor, im, trans_params = read_data_direct(video_images[i], video_lmks[i], self.lm3d_std)
            frame_coeffs_dict = self.process_3dmm(im_tensor.repeat(8, 1, 1, 1))
            frame_coeffs_dict['id'] = ref_coeffs_dict['id'].repeat(8, 1)
            frame_coeffs_dict['tex'] = ref_coeffs_dict['tex'].repeat(8, 1)
            face_coeffs = self.merge_coeffs(frame_coeffs_dict)
            output_vis_numpy_raw = self.render(face_coeffs, im_tensor, im, trans_params)
            outputs.append(np.array(output_vis_numpy_raw))
        result_clip = ImageSequenceClip(outputs, fps=25)
        result_clip.write_videofile('test.mp4', codec="libx264", audio=False)

    def process_video(self, face_image, face_lmk, video_images, video_lmks):
        if face_lmk is None:
            img = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            landmarks = self.predictor(gray, faces[0])
            landmarks_list = []
            for i in range(5):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                landmarks_list.append([x, y])
            face_lmk = np.array(landmarks_list)

        im_tensor, lm_tensor, im, trans_params = read_data_direct(face_image, face_lmk, self.lm3d_std)
        ref_coeffs_dict = self.process_3dmm(im_tensor)
        outputs = []
        for i in tqdm(range(len(video_images))):
            im_tensor, lm_tensor, im, trans_params = read_data_direct(video_images[i], video_lmks[i], self.lm3d_std)
            frame_coeffs_dict = self.process_3dmm(im_tensor)
            frame_coeffs_dict['id'] = ref_coeffs_dict['id']
            frame_coeffs_dict['tex'] = ref_coeffs_dict['tex']
            face_coeffs = self.merge_coeffs(frame_coeffs_dict)
            output_vis_numpy_raw = self.render(face_coeffs, im_tensor, im, trans_params)
            outputs.append(np.array(output_vis_numpy_raw))
        result_clip = ImageSequenceClip(outputs, fps=25)
        result_clip.write_videofile('test.mp4', codec="libx264", audio=False)



def get_data_path(root='examples'):
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),'detections',i.split(os.path.sep)[-1]) for i in lm_path]

    return im_path, lm_path


def process_file(txt_path):
    bboxes = []
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                print('Get Empty Mask')
                line = '0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0'
            bboxes.append(list(map(int, line.split(','))))
    return bboxes


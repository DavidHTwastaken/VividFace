from diffusers import AutoencoderKLTemporalDecoder, AutoencoderKL
from models.svd_temporal_decoder import AutoencoderKLTemporalEncoderDecoder
import numpy as np
import torch
from decord import VideoReader
import cv2
from moviepy.editor import VideoFileClip, ImageSequenceClip
from diffusers.training_utils import EMAModel

ckpt_path = ''
save_path = ''

vae = AutoencoderKLTemporalEncoderDecoder.from_pretrained('', low_cpu_mem_usage=False, device_map=None, ignore_mismatched_sizes=True).cuda()

ema_vae = EMAModel(parameters=vae.parameters())

ema_state_dict = torch.load(ckpt_path)
ema_vae.load_state_dict(ema_state_dict)

ema_vae.copy_to(vae.parameters())

vae.save_pretrained(save_path, safe_serialization=False)

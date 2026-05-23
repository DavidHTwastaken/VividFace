from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import random
import os
import sys

import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from einops import rearrange, repeat
from options.test_options import TestOptions
from PIL import Image, ImageFile

from face_encoder import FaceEmbedder
from dataset import collate_fn, HydridDataset
from model.unet_2d_condition import UNet2DConditionModel
from model.unet_motion_model import UNetMotionModel, MotionAdapter
from model.referencenet import ReferenceNet
from grad_loss_3d import batch_compute_diff_3d
from drop_frame import latent_process
from face3dvae.models.svd_temporal_decoder import AutoencoderKLTemporalEncoderDecoder
sys.path.insert(0, './Deep3DFaceRecon')
from face3dmodel import Face3DModel

torch.backends.cuda.matmul.allow_tf32 = True

os.environ['PYTHONWARNINGS'] = 'ignore'

ImageFile.LOAD_TRUNCATED_IMAGES = True



def zero_out_with_probability(mask: torch.Tensor, p: float) -> torch.Tensor:
    if mask.dim() == 4:
        assert 0 <= p <= 1, "p>1"
        b, f, h, w = mask.shape
        mask_reshaped = rearrange(mask, 'b f h w -> (b f) h w')
        rand_tensor = torch.rand(b*f)
        mask_zeros = rand_tensor < p
        mask_reshaped[mask_zeros] = mask_reshaped[mask_zeros] * 0
        return rearrange(mask_reshaped, '(b f) h w -> b f h w', b=b).to(mask.device).to(mask.dtype)
    else:
        assert 0 <= p <= 1, "p>1"
        b, h, w = mask.shape
        mask_reshaped = mask
        rand_tensor = torch.rand(b)
        mask_zeros = rand_tensor < p
        mask_reshaped[mask_zeros] = mask_reshaped[mask_zeros] * 0
        return mask_reshaped.to(mask.device).to(mask.dtype)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--unet", type=str, default=None, help="A seed for reproducible training.")
    parser.add_argument("--clip_skip", type=int, default=2, help="A seed for reproducible training.")
    parser.add_argument("--cond_len", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--depth", type=int, default=2, help="A seed for reproducible training.")
    parser.add_argument("--num_tokens", type=int, default=16, help="A seed for reproducible training.")
    parser.add_argument("--dino_drop", type=float, default=0.8, help="A seed for reproducible training.")
    parser.add_argument("--attr_drop", type=float, default=0.8, help="A seed for reproducible training.")
    parser.add_argument("--num_frames", type=int, default=4, help="A seed for reproducible training.")
    parser.add_argument("--num_repeats", type=int, default=1, help="A seed for reproducible training.")
    parser.add_argument("--use_aug", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--all_learn", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--refnet_from_scratch", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--denoise_from_scratch", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--image_skip_motion", type=bool, default=False, help="A seed for reproducible training.")
    parser.add_argument("--mask_drop", type=float, default=0.2, help="A seed for reproducible training.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--metafiles",
        type=str,
        required=True,
        nargs="*",
        help="Training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--use_grad_loss",
        type=bool,
        default=False,
        help=(
            'grad loss with 6 * c channels'
        ),
    )
    parser.add_argument(
        "--same_noise_for_frames",
        type=bool,
        default=False,
        help=(
            'same_noise_for_frames'
        ),
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument(
        "--img_metafiles",
        type=str,
        default="img_metafiles for img",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--remove_id_tex", type=bool, default=False, help="For distributed training: local_rank")
    parser.add_argument("--drop_rate_3dmm", type=float, default=0.0, help="For distributed training: local_rank")
    parser.add_argument("--resume_path", type=str, default="", help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def Weighted_MSE_loss(output, target):
    # return F.mse_loss(output, target, reduction="none")
    mse = (output - target) ** 2
    weights = 1 / (1 + torch.exp(-mse))
    weighted_mse = (weights * mse).sum() / weights.sum()
    return weighted_mse


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

def main(args):
    set_seed(args.seed or 42)
    if args.image_skip_motion:
        print('Joint Training, motion layer skip Image')

    if args.use_aug:
        print(f'Using Random Drop Frame Mix Setting.')

    from accelerate import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    face3d_opt = TestOptions().gather_options()
    face3d_opt.isTrain = False
    face3d_opt.use_opengl = False
    face3d_opt.use_ddp = False
    face3d_opt.bfm_folder = './Deep3DFaceRecon/BFM'
    face3d_opt.load_path = './Deep3DFaceRecon/checkpoints/base/epoch_20.pth'
    face3dmodel = Face3DModel(face3d_opt, device=accelerator.device)

    face_embedder = FaceEmbedder(
        arcface_path = "weights/IResNet100_WebFace42M.pth",
        dino_model_path = "weights/dinov2-base"
    ).to(accelerator.device)

    accelerator.print(f"arcface_embed_dim: {face_embedder.arcface_embed_dim} attr_embed_dim: {face_embedder.attr_embed_dim}")

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKLTemporalEncoderDecoder.from_pretrained('weights/vividface').cuda()

    assert args.unet is not None

    if os.path.exists(
        os.path.join(args.unet, 'denoising_unet')
    ) and os.path.exists(
        os.path.join(args.unet, 'reference_unet')
    ):
        accelerator.print(f'Using exists Unet & Refnet from {args.unet}')
        denoising_unet = UNetMotionModel.from_pretrained(os.path.join(args.unet, 'denoising_unet'))
        reference_unet: OriginalUNet2DConditionModel = OriginalUNet2DConditionModel.from_pretrained(os.path.join(args.unet, 'reference_unet'))
    else:
        accelerator.print(f'Using exists Unet from {args.unet}')
        denoising_unet = UNetMotionModel.from_pretrained(args.unet)
        if args.refnet_from_scratch:
            print('From scratch Refnet')
            reference_unet: OriginalUNet2DConditionModel = OriginalUNet2DConditionModel.from_config(args.pretrained_model_name_or_path, subfolder="unet")
        else:
            print(f'refnet from {args.pretrained_model_name_or_path}')
            reference_unet: OriginalUNet2DConditionModel = OriginalUNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    accelerator.print(f"use ckpt from {args.unet}")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # freeze parameters of models to save more memory
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)


    if args.all_learn:
        print('Train EveryThing.')
        reference_unet.train()
        reference_unet.requires_grad_(True)

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        accelerator.print(weight_dtype)
        denoising_unet.to(accelerator.device)
        reference_unet.to(accelerator.device)
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)

        denoising_unet.train()
        denoising_unet.requires_grad_(True)
        for name, param in denoising_unet.named_parameters():
            if 'motion_module' in name:
                param.requires_grad = True


        total_params = sum(p.numel() for p in denoising_unet.parameters()) + sum(p.numel() for p in reference_unet.parameters())
        trainable_params = sum(p.numel() for p in denoising_unet.parameters() if p.requires_grad) + sum(p.numel() for p in reference_unet.parameters() if p.requires_grad)
        accelerator.print(f"Trainble: {trainable_params/1e6:.1f} / {total_params/1e6:.1f} M ({trainable_params/total_params:.1%})")
        # optimizer
        params_group = [
            {"params": filter(lambda p: p.requires_grad, denoising_unet.parameters()), "lr": args.learning_rate},
            {"params": filter(lambda p: p.requires_grad, reference_unet.parameters()), "lr": args.learning_rate/10},
        ]
        optimizer = torch.optim.AdamW(params_group, lr=args.learning_rate, weight_decay=args.weight_decay)

    else:
        print('Ref Frozen')
        reference_unet.requires_grad_(False)


        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        accelerator.print(weight_dtype)
        reference_unet.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)

        denoising_unet.train()
        denoising_unet.requires_grad_(True)
        for name, param in denoising_unet.named_parameters():
            if 'motion_module' in name:
                param.requires_grad = True
        total_params = sum(p.numel() for p in denoising_unet.parameters())
        trainable_params = sum(p.numel() for p in denoising_unet.parameters() if p.requires_grad)
        accelerator.print(f"Trainble: {trainable_params/1e6:.1f} / {total_params/1e6:.1f} M ({trainable_params/total_params:.1%})")
        # optimizer
        params_group = [
            {"params": filter(lambda p: p.requires_grad, denoising_unet.parameters()), "lr": args.learning_rate},
        ]
        optimizer = torch.optim.AdamW(params_group, weight_decay=args.weight_decay)

    # Preprocessing the datasets.
    num_workers = min(args.dataloader_num_workers, os.cpu_count() - 1)


    train_video_dataset = HydridDataset(
            args.metafiles[0],
            512,
            4,
            8,
            args.num_repeats,
            args.img_metafiles.split(' '),
            tokenizer,
            1.,
            0.1,
            '[ref][pose][mask]',
            debug_mode=False,
            data_type='video',
    )

    train_video_dataloader = DataLoader(
        train_video_dataset, batch_size=args.train_batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, persistent_workers=num_workers>0
    )
    train_image_dataset = HydridDataset(
            args.metafiles[0],
            512,
            4,
            8,
            args.num_repeats,
            args.img_metafiles.split(' '),
            tokenizer,
            1.,
            0.1,
            '[ref][pose][mask]',
            debug_mode=False,
            data_type='image',
    )

    train_image_dataloader = DataLoader(
        train_image_dataset, batch_size=args.train_batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, persistent_workers=num_workers>0
    )
    print(f'This Merge Dataset : {train_video_dataloader.__len__()}')

    net = ReferenceNet(denoising_unet, reference_unet)

    # Prepare everything with our `accelerator`.
    net, train_image_dataloader, train_video_dataloader, optimizer = accelerator.prepare(net, train_image_dataloader, train_video_dataloader, optimizer)
    global_step = 1
    accelerator.print(f"start training")

    caption = ""
    input_ids = tokenizer(caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
    input_ids = input_ids.to(accelerator.device)
    if args.clip_skip <= 1:
        text_hidden_states = text_encoder(input_ids).last_hidden_state
    else:
        enc_out = text_encoder(input_ids, output_hidden_states=True)
        text_hidden_states = enc_out.hidden_states[-args.clip_skip]
        text_hidden_states = text_encoder.text_model.final_layer_norm(text_hidden_states)

    video_encoder_hidden_states = text_hidden_states.repeat((args.train_batch_size * args.num_repeats, 1, 1))
    img_encoder_hidden_states = text_hidden_states.repeat((16 + 1, 1, 1))
    text_encoder = text_encoder.cpu()
    train_video_dataloader_iter = iter(train_video_dataloader)
    train_image_dataloader_iter = iter(train_image_dataloader)
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step in range(len(train_video_dataloader)):
            if step % 2 == 0:
                batch = next(train_video_dataloader_iter)
            else:
                batch = next(train_image_dataloader_iter)
            load_data_time = time.perf_counter() - begin
            # Convert images to latent space
            with torch.no_grad():

                face_pixel_values = batch["face_pixel_values"].to(accelerator.device) # torch.Size([2, 1, 3, 112, 112])
                attr_pixel_values = batch["attr_pixel_values"].to(accelerator.device) # torch.Size([2, 8, 3, 224, 224])
                cond_pixel_values = batch["cond_pixel_values"].to(accelerator.device, dtype=weight_dtype) # torch.Size([2, 4, 3, 512, 512])
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype) # torch.Size([2, 8, 3, 512, 512])
                raw_pixel_values = batch["raw_pixel_values"].to(accelerator.device, dtype=weight_dtype) # torch.Size([2, 8, 3, 512, 512])
                mask = batch["masks"].to(accelerator.device) # torch.Size([2, 8, 512, 512])

                if args.mask_drop:
                    mask = zero_out_with_probability(mask, args.mask_drop)

                is_video =  (len(face_pixel_values.shape) == 5)

                if is_video:
                    encoder_hidden_states = video_encoder_hidden_states
                else:
                    encoder_hidden_states = img_encoder_hidden_states

                if is_video:
                    bsz = face_pixel_values.size(0)
                    pixel_values = pixel_values.reshape(-1, *pixel_values.shape[-3:]).contiguous()
                    raw_pixel_values = raw_pixel_values.reshape(-1, *raw_pixel_values.shape[-3:]).contiguous()
                    cond_pixel_values = cond_pixel_values.reshape(-1, *cond_pixel_values.shape[-3:]).contiguous()
                    attr_pixel_values = attr_pixel_values.reshape(-1, *attr_pixel_values.shape[-3:]).contiguous()
                    face_pixel_values = face_pixel_values.repeat_interleave(args.num_frames, dim=1)
                    face_pixel_values = face_pixel_values.reshape(-1, *face_pixel_values.shape[-3:]).contiguous()
                    mask = mask.reshape(-1, 1, *mask.shape[-2:]).contiguous()

                    if not batch['is_twins']:
                        base_scale = 0.5
                        this_dino_scale = base_scale if random.random()>0.8 else 0.
                        this_attr_scale = base_scale
                    else:
                        this_dino_scale = 0.5
                        this_attr_scale = 0.5

                else:
                    bsz=face_pixel_values.size(0)
                    pixel_values = pixel_values.contiguous()
                    raw_pixel_values = raw_pixel_values.contiguous()
                    cond_pixel_values = torch.zeros(args.num_frames, 3, 512, 512).to(cond_pixel_values.device).to(cond_pixel_values.dtype)

                    attr_pixel_values = attr_pixel_values.contiguous()

                    face_pixel_values = face_pixel_values.contiguous()

                    mask = mask.reshape(-1, 1, *mask.shape[-2:]).contiguous()
                    this_dino_scale = 0.5
                    this_attr_scale = 0.5

                with torch.no_grad():
                    video_lmks = batch['lmks_values'].reshape(-1, 5, 2).cpu().numpy()
                    pixel_values_3dmm, masks_3dmm = face3dmodel.process_video_for_training(raw_pixel_values, video_lmks, remove_id_tex=args.remove_id_tex)
                pixel_values_3dmm = pixel_values_3dmm.to(accelerator.device, dtype=weight_dtype)
                masks_3dmm = masks_3dmm.to(accelerator.device, dtype=weight_dtype)
                masks_3dmm = masks_3dmm[:, None].repeat(1, 3, 1, 1)

                if mask.sum()==0:
                    mask_pixel_values = pixel_values * (1.0 - mask)
                else:
                    mask_pixel_values = pixel_values * (1.0 - mask) + (pixel_values * mask / mask.sum()).sum(dim=(-2, -1), keepdims=True)  * mask

                if random.random() < args.drop_rate_3dmm:
                    masks_3dmm[:] = 0.0

                mask_pixel_values_c = pixel_values_3dmm * masks_3dmm
                #mask_pixel_values_c = raw_pixel_values * (1.0 - masks_3dmm) + pixel_values_3dmm * masks_3dmm
                mask_pixel_values_c = torch.clip(mask_pixel_values_c, -1, 1)

                if is_video:
                    mask_pixel_values = mask_pixel_values.to(weight_dtype)
                    latents = vae.encode(torch.cat([pixel_values, mask_pixel_values, mask_pixel_values_c], dim=0), num_frames=8, is_image_batch=False).latent_dist.sample() # used_model vae
                    latents = latents * vae.config.scaling_factor
                    latents, mask_latents, mask_latents_c = torch.chunk(latents, chunks=3, dim=0)
                    latents = latents.view(2, 8, -1, 64, 64) # B, T, C, H, W
                    mask_latents = mask_latents.view(2, 8, -1, 64, 64)
                    mask_latents_c = mask_latents_c.view(2, 8, -1, 64, 64)
                else:
                    mask_pixel_values = mask_pixel_values.to(weight_dtype)
                    latents = vae.encode(torch.cat([pixel_values, mask_pixel_values, mask_pixel_values_c], dim=0), num_frames=1, is_image_batch=True).latent_dist.sample() # used_model vae
                    latents = latents * vae.config.scaling_factor
                    latents, mask_latents, mask_latents_c = torch.chunk(latents, chunks=3, dim=0)
                    latents = latents.view(bsz, 1, -1, 64, 64)
                    mask_latents = mask_latents.view(bsz, 1, -1, 64, 64)
                    mask_latents_c = mask_latents_c.view(bsz, 1, -1, 64, 64)



                if is_video or args.image_skip_motion is False:
                    ref_latents = vae.encode(cond_pixel_values, num_frames=4, is_image_batch=True).latent_dist.sample() # used_model vae
                    ref_latents = ref_latents * vae.config.scaling_factor
                else:
                    ref_latents = None



                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(
                    # latent_process(latents, bsz=bsz) if args.use_aug else latents, 
                    latents,
                    noise,
                    timesteps
                )

                valid = (torch.rand((bsz,), device=latents.device) > 0.5).float()
                if is_video:
                    valid = valid.repeat_interleave(args.num_frames, dim=0)
                if is_video:
                    if batch['is_twins']:
                        feature_to_id_branch = F.interpolate(face_pixel_values, size=(224, 224), mode="bilinear")
                    else:
                        # print('Using Twins Video')
                        feature_to_id_branch = attr_pixel_values
                else:
                    feature_to_id_branch = F.interpolate(face_pixel_values, size=(224, 224), mode="bilinear")
                full_embeds = face_embedder(
                    face_pixel_values,
                    valid,
                    attr_pixel_values,
                    feature_to_id_branch,
                )

                cross_attention_kwargs = {
                    "gligen": {
                        "id_embed": full_embeds["id_embed"].to(dtype=weight_dtype),
                        "attr_embed": full_embeds["attr_embed"].to(dtype=weight_dtype), # 9, 256, 768
                        "dino_embed": full_embeds["dino_embed"].to(dtype=weight_dtype), # 9, 256, 768
                        "attr_scale": this_attr_scale,# 0.6 ,
                        "dino_scale": this_dino_scale,# 0.1 
                    }
                }

                mask = F.interpolate(mask, scale_factor = 1/8, mode="nearest")
                masks_3dmm = F.interpolate(masks_3dmm, scale_factor = 1/8, mode="nearest")

                if is_video:
                    mask = mask.view(2, 8, -1, 64, 64)
                    masks_3dmm = masks_3dmm.view(2, 8, -1, 64, 64)
                else:
                    mask = mask.view(bsz, 1, -1, 64, 64)
                    masks_3dmm = masks_3dmm.view(bsz, 1, -1, 64, 64)
                noisy_latents = torch.cat([noisy_latents, mask, mask_latents * (1.0 - mask), mask_latents_c * masks_3dmm[:, :, 0:1]], dim=2)
            with accelerator.accumulate(net):
                noisy_latents = noisy_latents.permute(0, 2, 1, 3, 4).contiguous()
                noise_pred = net(
                    noisy_latents,  # 16, 9, 64, 64 / # 17, 9, 64, 64
                    timesteps,  # 16  # 17
                    encoder_hidden_states, # 16, 77, 768 / 17, 77, 768
                    ref_latents=ref_latents,  # 8, 4, 64, 64 / # 8, 4, 64,64
                    ref_encoder_hidden_states=text_hidden_states, # 1, 77, 768
                    cross_attention_kwargs=cross_attention_kwargs
                )
                noise_pred = noise_pred.sample # 16, 4, 64, 64

                noise = noise.permute(0, 2, 1, 3, 4) # reshape to B, C, T, H, W

                loss = Weighted_MSE_loss(noise_pred.float(), noise.float())
                loss = loss.mean()

                if args.use_grad_loss and is_video:
                    grad_loss = Weighted_MSE_loss(
                        batch_compute_diff_3d(noise_pred).float(),
                        batch_compute_diff_3d(noise).float(),
                    )
                    grad_loss = grad_loss.mean()

                    print(f'Video, Loss {loss}, Grad loss {grad_loss}')
                    loss = loss + grad_loss
                else:
                    print(f'Image Loss {loss}')


                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(*[x["params"] for x in optimizer.param_groups])
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                net.module.clear()
                global_step += 1

                if (global_step < 20 or global_step % 5 == 0) and accelerator.is_main_process:
                    print(f"Epoch {epoch}, global step {global_step}, data_time: {load_data_time:.3f}, time: {time.perf_counter() - begin:.3f}, step_loss: {avg_loss:.5f}")

                if global_step < 20000:
                    if global_step % 2000 == 0 and accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}", 'denoising_unet')
                        accelerator.unwrap_model(accelerator.unwrap_model(net).denoising_unet).save_pretrained(save_path)
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}", 'reference_unet')
                        accelerator.unwrap_model(accelerator.unwrap_model(net).referencenet).save_pretrained(save_path)
                else:
                    if global_step % args.save_steps == 0 and accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}", 'denoising_unet')
                        accelerator.unwrap_model(accelerator.unwrap_model(net).denoising_unet).save_pretrained(save_path)
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}", 'reference_unet')
                        accelerator.unwrap_model(accelerator.unwrap_model(net).referencenet).save_pretrained(save_path)


            begin = time.perf_counter()


    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "final")
        accelerator.unwrap_model(denoising_unet).save_pretrained(save_path)
        accelerator.print(f"save ckpt to {save_path}")

if __name__ == "__main__":
    args = parse_args()

    main(args)



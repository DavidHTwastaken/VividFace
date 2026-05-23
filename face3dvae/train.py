#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
import wandb

import diffusers
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.torch_utils import is_compiled_module

from video_dataset import VideoDataset
from image_dataset import ImageDataset
from models.losses import VAELoss
from models.svd_temporal_decoder import AutoencoderKLTemporalEncoderDecoder
from models.autoencoder_kl_temporal import  TemporalAutoencoderKL



# Will error if the minimal version of diffusers is not installed. Remove at your own risks.

logger = get_logger(__name__, log_level="INFO")


def log_validation(vae, eval_video_dataset, eval_image_dataset, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    upload_videos = []
    rec_upload_videos = []

    upload_images = []
    rec_upload_images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    # For Image
    with autocast_ctx:
        images = [eval_image_dataset[i]["pixel_values"].to(weight_dtype).to(vae.device) for i in range(8)]
        raw_images = torch.cat(images)
        with torch.no_grad():
            raw_images, z, posterior, rec_images = vae(raw_images, num_frames=8, is_image_batch=True, sample_posterior=True, return_dict=False)


        rec_images = rec_images.permute(0, 2, 3, 1)
        rec_images = rec_images.detach().cpu().numpy()
        rec_images = np.clip((rec_images + 1) /2 * 255, 0, 255)

        raw_images = raw_images.permute(0, 2, 3, 1)
        raw_images = raw_images.detach().cpu().numpy()
        raw_images = np.clip((raw_images + 1) /2 * 255, 0, 255)

        rec_upload_images = [rec_images[i] for i in range(8)]
        upload_images = [raw_images[i] for i in range(8)]

    # For Video
    for i in range(len(eval_video_dataset)):
        with autocast_ctx:
            images = eval_video_dataset[i]["pixel_values"][0].to(weight_dtype).to(vae.device)
            num_frames = images.shape[0]
            single_rec_video = []
            single_raw_video = []
            for j in tqdm(range(num_frames // 8)):
                raw_images = images[j*8:j*8+8]
                raw_images = raw_images[None,:]
                with torch.no_grad():
                    raw_images = rearrange(raw_images, 'b t c h w -> (b t) c h w', t=8)
                    raw_images, z, posterior, rec_images = vae(raw_images, num_frames=8, is_image_batch=False, sample_posterior=True, return_dict=False)

                raw_images = rearrange(raw_images, '(b t) c h w -> b t c h w', t=8)
                rec_images = rearrange(rec_images, '(b t) c h w -> b t c h w', t=8)

                rec_images = rec_images.detach().cpu().numpy()
                rec_images = np.clip((rec_images + 1) /2 * 255, 0, 255)
                single_rec_video.append(rec_images[0])

                raw_images = raw_images.detach().cpu().numpy()
                raw_images = np.clip((raw_images + 1) /2 * 255, 0, 255)
                single_raw_video.append(raw_images[0])

        rec_upload_videos.append(np.concatenate(single_rec_video))
        upload_videos.append(np.concatenate(single_raw_video))

    for tracker in accelerator.trackers:
        tracker.log(
            {
                "raw_videos": [
                    wandb.Video(video, caption=f"Raw videos: {i}", format='mp4', fps=25)
                    for i, video in enumerate(upload_videos)
                ],
                "rec_videos": [
                    wandb.Video(video, caption=f"Rec videos: {i}", format='mp4', fps=25)
                    for i, video in enumerate(rec_upload_videos)
                ],
                "raw_images": [
                    wandb.Image(image, caption=f"Raw images: {i}")
                    for i, image in enumerate(upload_images)
                ],
                "rec_images": [
                    wandb.Image(image, caption=f"Rec images: {i}")
                    for i, image in enumerate(rec_upload_images)
                ]
            }
        )

    torch.cuda.empty_cache()


    return rec_upload_videos

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--offload_ema", action="store_true", help="Offload EMA model to CPU during training step.")
    parser.add_argument("--foreach_ema", action="store_true", help="Use faster foreach implementation of EMAModel.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
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
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--use_original_encoder_layers_per_block", action="store_true", default=False
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="vae",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    from accelerate import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    '''
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    '''
    vae = AutoencoderKLTemporalEncoderDecoder.from_pretrained('svd_c4_1019', low_cpu_mem_usage=False, device_map=None, ignore_mismatched_sizes=True, use_original_encoder_layers_per_block=args.use_original_encoder_layers_per_block).cuda()
    # vae = AutoencoderKLTemporalEncoderDecoder.from_pretrained('svd_vae_transferred_c4', low_cpu_mem_usage=False, device_map=None, ignore_mismatched_sizes=True, use_original_encoder_layers_per_block=args.use_original_encoder_layers_per_block).cuda()
    '''
    vae = TemporalAutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant,
        low_cpu_mem_usage=False, device_map=None
    )
    '''


    #vae.encoder.requires_grad_(False)
    #vae.quant_conv.requires_grad_(False)

    if args.use_ema:
        # ema_vae = AutoencoderKLTemporalEncoderDecoder.from_pretrained('svd_vae_transferred_c4', low_cpu_mem_usage=False, device_map=None, ignore_mismatched_sizes=True, use_original_encoder_layers_per_block=args.use_original_encoder_layers_per_block).cuda()
        ema_vae = AutoencoderKLTemporalEncoderDecoder.from_pretrained('svd_c4_1019', low_cpu_mem_usage=False, device_map=None, ignore_mismatched_sizes=True, use_original_encoder_layers_per_block=args.use_original_encoder_layers_per_block).cuda()
        ema_vae.load_state_dict(vae.state_dict())
        ema_vae = EMAModel(
            ema_vae.parameters(),
            model_cls=AutoencoderKL,
            model_config=ema_vae.config,
            foreach=args.foreach_ema,
        )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_vae.save_pretrained(os.path.join(output_dir, "vae_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "vae"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)


    torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        filter(lambda p: p.requires_grad, vae.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # init a dataset
    train_video_dataset = VideoDataset('video_list.jsonl', training=True)
    eval_video_dataset = VideoDataset('test_video_list.jsonl', training=False, image_size=512, pred_frames=-1)

    train_image_dataset = ImageDataset('image_list.txt', training=True)
    eval_image_dataset = ImageDataset('test_image_list.txt', training=False, image_size=512)

    def collate_fn(examples):
        pixel_values = torch.cat([example["pixel_values"] for example in examples], dim=0)
        pixel_values = pixel_values.contiguous().float()
        batch = {
            "pixel_values": pixel_values,
        }
        return batch

    # DataLoaders creation:
    train_video_dataloader = torch.utils.data.DataLoader(
        train_video_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    train_image_dataloader = torch.utils.data.DataLoader(
        train_image_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size*8,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_video_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    vae, optimizer, train_video_dataloader, train_image_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_video_dataloader, train_image_dataloader, lr_scheduler
    )

    if args.use_ema:
        if args.offload_ema:
            ema_vae.pin_memory()
        else:
            ema_vae.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device)#, dtype=weight_dtype)

    vae_loss_fn = VAELoss(
        logvar_init=0.0,
        perceptual_loss_weight=1,
        kl_loss_weight=1e-6,
        device=accelerator.device,
        dtype=torch.float,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_video_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_video_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_video_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    train_video_dataloader_iter = iter(train_video_dataloader)
    train_image_dataloader_iter = iter(train_image_dataloader)
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        train_rec_loss = 0.0
        train_kl_loss = 0.0
        train_perceptual_loss = 0.0
        for step in range(1000000):
            if step % 3 != 0:
                batch = next(train_video_dataloader_iter)
                is_image_batch = False
            else:
                batch = next(train_image_dataloader_iter)
                is_image_batch = True

            with accelerator.accumulate(vae):
                # Convert images to latent space
                images = batch["pixel_values"]#.to(weight_dtype)

                if not is_image_batch:
                    num_frames = images.shape[1]
                    images = rearrange(images, 'b t c h w -> (b t) c h w', t=num_frames).contiguous()

                    x, z, posterior, x_rec = vae(images, return_dict=False, num_frames=num_frames, is_image_batch=is_image_batch, sample_posterior=True)

                    x = rearrange(x, '(b t) c h w -> b c t h w', t=num_frames).contiguous()
                    x_rec = rearrange(x_rec, '(b t) c h w -> b c t h w', t=num_frames).contiguous()
                    nll_loss, weighted_nll_loss, weighted_kl_loss, recon_loss, perceptual_loss = vae_loss_fn(x, x_rec, posterior)
                else:

                    x, z, posterior, x_rec = vae(images, return_dict=False, num_frames=8, is_image_batch=is_image_batch, sample_posterior=True)
                    nll_loss, weighted_nll_loss, weighted_kl_loss, recon_loss, perceptual_loss = vae_loss_fn(x[:, :, None], x_rec[:, :, None], posterior)

                loss = weighted_nll_loss + weighted_kl_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                avg_rec_loss = accelerator.gather(recon_loss.repeat(args.train_batch_size)).mean()
                train_rec_loss += avg_rec_loss.item() / args.gradient_accumulation_steps

                avg_perceptual_loss = accelerator.gather(perceptual_loss.repeat(args.train_batch_size)).mean()
                train_perceptual_loss += avg_perceptual_loss.item() / args.gradient_accumulation_steps

                avg_kl_loss = accelerator.gather(weighted_kl_loss.repeat(args.train_batch_size)).mean()
                train_kl_loss += avg_kl_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    if args.offload_ema:
                        ema_vae.to(device="cuda", non_blocking=True)
                    ema_vae.step(vae.parameters())
                    if args.offload_ema:
                        ema_vae.to(device="cpu", non_blocking=True)
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, 'train_rec_loss': train_rec_loss, 'train_perceptual_loss': train_perceptual_loss, 'train_kl_loss': train_kl_loss}, step=global_step)
                train_loss = 0.0
                train_rec_loss = 0.0
                train_perceptual_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        # accelerator.save_state(save_path)
                        if args.use_ema:
                            logger.info(f"Saved ema model to {save_path}")
                            torch.save(ema_vae.state_dict(), os.path.join(save_path, "vae_ema.pt"))
                        logger.info(f"Saved state to {save_path}")

                if accelerator.is_main_process:
                    if global_step % args.validation_steps == 0:
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_vae.store(vae.parameters())
                            ema_vae.copy_to(vae.parameters())
                        log_validation(
                            vae,
                            eval_video_dataset,
                            eval_image_dataset,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_vae.restore(vae.parameters())

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = unwrap_model(vae)
        if args.use_ema:
            ema_vae.copy_to(vae.parameters())


    accelerator.end_training()


if __name__ == "__main__":
    main()

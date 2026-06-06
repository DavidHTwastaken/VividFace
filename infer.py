import nvdiffrast.torch as dr
ctx = dr.RasterizeCudaContext(device='cuda')

from typing import Any, Callable, Dict, List, Optional, Union
from collections import deque
from datetime import datetime
import sys
import time
import os
import cv2
import random

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import retrieve_latents
from diffusers import DDIMScheduler
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from diffusers import DDIMScheduler, AutoencoderKL
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from moviepy.editor import VideoFileClip, ImageSequenceClip
from decord import VideoReader

from model.unet_2d_condition import UNet2DConditionModel
from model.referencenet import ReferenceAttentionControl
from model.unet_motion_model import UNetMotionModel, MotionAdapter
from face3dvae.models.svd_temporal_decoder import AutoencoderKLTemporalEncoderDecoder
from face_encoder import FaceEmbedder

sys.path.insert(0, './Deep3DFaceRecon')
from options.test_options import TestOptions
from face3dmodel import Face3DModel

def process_file(txt_path):
    bboxes = []
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                print('Get Empty Mask')
                line = '0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0'
            bboxes.append(list(map(int, line.split(',')))[:14])
    return bboxes

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

    return Image.fromarray(face)

def grid_videos(input_dir, output_path, grid_size=(5, 5)):
    rows, cols = grid_size

    sample_video_path = os.path.join(input_dir, "0_0.mp4")
    cap = cv2.VideoCapture(sample_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {sample_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    output_width = cols * width
    output_height = rows * height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    caps = [[cv2.VideoCapture(os.path.join(input_dir, f"{i}_{j}.mp4")) for j in range(cols)] for i in range(rows)]

    while True:
        frames = []
        for i in range(rows):
            row_frames = []
            for j in range(cols):
                ret, frame = caps[i][j].read()
                if not ret:
                    break
                row_frames.append(frame)
            if len(row_frames) != cols:
                break
            frames.append(np.hstack(row_frames))

        if len(frames) != rows:
            break

        grid_frame = np.vstack(frames)
        out.write(grid_frame)

    for i in range(rows):
        for j in range(cols):
            caps[i][j].release()
    out.release()


class StableDiffusionGLIGENInpaintPipeline(StableDiffusionPipeline):

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i], num_frames=8, is_image_batch=False)
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image, num_frames=8, is_image_batch=False), generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
        init_noise=None,
    ):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)

            if image.shape[1] == 4:
                image_latents = image
            else:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            if init_noise is not None:
                print('Using Same Noise for every 8 frame')
                noise = init_noise
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    def prepare_ref_latents(
        self,
        refimage: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Union[int, List[int]],
        do_classifier_free_guidance: bool,
    ) -> torch.Tensor:
        refimage = refimage.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        if isinstance(generator, list):
            ref_image_latents = [
                self.vae.encode(refimage[i : i + 1], num_frames=4, is_image_batch=True).latent_dist.sample(generator=generator[i])
                for i in range(batch_size)
            ]
            ref_image_latents = torch.cat(ref_image_latents, dim=0)
        else:
            ref_image_latents = self.vae.encode(refimage, num_frames=4, is_image_batch=True).latent_dist.sample(generator=generator)
        ref_image_latents = self.vae.config.scaling_factor * ref_image_latents

        # duplicate mask and ref_image_latents for each generation per prompt, using mps friendly method
        if ref_image_latents.shape[0] < batch_size:
            if not batch_size % ref_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {ref_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            ref_image_latents = ref_image_latents.repeat(batch_size // ref_image_latents.shape[0], 1, 1, 1)

        # aligning device to prevent device errors when concating it with the latent model input
        ref_image_latents = ref_image_latents.to(device=device, dtype=dtype)
        return ref_image_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image = None,
        mask = None,
        image_3dmm = None,
        mask_3dmm = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        strength: float = 1.0,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        gligen_embeddings: List[torch.FloatTensor] = None,
        attr_scale: float = 1.0,
        dino_scale: float = 1.0,
        ref_image = None,
        init_noise=None,
        enable_3dmm_cfg=False,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        is_strength_max = False
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds_tuple = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=1,
        )
        prompt_embeds = torch.cat([prompt_embeds_tuple[1] , prompt_embeds_tuple[0]])


        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        init_image = image.to(dtype=torch.float32)
        init_image_3dmm = image_3dmm.to(dtype=torch.float32)

        num_channels_latents = self.vae.config.latent_channels

        ref_image = self.image_processor.preprocess(
            ref_image, height=height, width=width
        )

        ref_image_latents = self.prepare_ref_latents(
            ref_image,
            ref_image.size(0),
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )
        ref_image_latents = torch.cat([ref_image_latents, ref_image_latents], dim=0)

        latents, noise, image_latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=True,
            init_noise=init_noise,
        )

        mask_condition = mask
        masked_image = init_image * (1.0 - mask_condition) + (init_image * mask / mask.sum()).sum(dim=(-2, -1), keepdim=True) * mask_condition

        mask_condition_3dmm = mask_3dmm
        masked_image_3dmm = init_image_3dmm * mask_condition

        mask, masked_image_latents = self.prepare_mask_latents(
            mask_condition,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )
        mask_3dmm, masked_image_latents_3dmm = self.prepare_mask_latents(
            mask_condition_3dmm,
            masked_image_3dmm,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance=False,
        )
        if enable_3dmm_cfg:
            masked_image_latents_3dmm = torch.cat([masked_image_latents_3dmm] * 2)
            mask_3dmm = torch.cat([mask_3dmm, mask_3dmm[:]*0.0])
        else:
            masked_image_latents_3dmm = torch.cat([masked_image_latents_3dmm] * 2)
            mask_3dmm = torch.cat([mask_3dmm] * 2)
        # 5.1 Prepare GLIGEN variables
        repeat_batch = batch_size * num_images_per_prompt
        id_embed = gligen_embeddings["id_embed"]
        attr_embed = gligen_embeddings["attr_embed"]
        dino_embed = gligen_embeddings["dino_embed"]

        if do_classifier_free_guidance:
            id_embed = torch.cat([torch.zeros_like(id_embed), id_embed])
            attr_embed = torch.cat([torch.zeros_like(attr_embed), attr_embed])
            dino_embed = torch.cat([torch.zeros_like(dino_embed), dino_embed])

        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        dtype = prompt_embeds.dtype
        cross_attention_kwargs["gligen"] = {
            "id_embed": id_embed.to(dtype), "attr_embed": attr_embed.to(dtype), "attr_scale": attr_scale, "dino_embed": dino_embed.to(dtype), "dino_scale": dino_scale,
        }
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        reference_scale = 0.1
        reference_control_writer = ReferenceAttentionControl(
            self.refnet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
            reference_scale=reference_scale
        )
        reference_control_reader = ReferenceAttentionControl(
            self.unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
            reference_scale=reference_scale
        )


        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i==0:
                    self.refnet(
                        ref_image_latents,
                        torch.zeros_like(t, device=device),
                        encoder_hidden_states=prompt_embeds[:1].repeat_interleave(ref_image_latents.size(0), dim=0),
                        return_dict=False,
                    )

                    reference_control_reader.update(reference_control_writer)
                latents = mask[:mask.size(0)//2] * latents + (1 - mask[:mask.size(0)//2]) * self.scheduler.add_noise(image_latents, noise, t.repeat(batch_size * num_images_per_prompt))
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents * (1.0 - mask) , mask_3dmm[:,:,0:1]* masked_image_latents_3dmm], dim=1)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input.view(2, 8, -1, 64, 64).permute(0, 2, 1, 3, 4),
                    t,
                    encoder_hidden_states=prompt_embeds.view(2, 8, 77, 768)[:, 1],
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4)
                noise_pred = noise_pred.view(2*8, 4, 64, 64)
                noise_pred = noise_pred.to(latent_model_input.dtype)
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, num_frames=8, is_image_batch=False)[0]
        else:
            image = latents
        has_nsfw_concept = None

        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


def make_image_grid(images: List[Image.Image], rows: int, cols: int, resize: int = None) -> Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid



def run(video_path_list, crop_face_path_list, output=None):
    device = "cuda"
    dtype = torch.float16

    face3d_opt = TestOptions().gather_options()
    face3d_opt.isTrain = False
    face3d_opt.use_opengl = False
    face3d_opt.use_ddp = False
    face3d_opt.bfm_folder = os.path.join(
        os.path.dirname(__file__), './Deep3DFaceRecon/BFM')
    face3d_opt.load_path = os.path.join(os.path.dirname(
        __file__), './Deep3DFaceRecon/BFM/checkpoints/base/epoch_20.pth')
    face3dmodel = Face3DModel(face3d_opt, device='cuda:0')

    task_id = datetime.now().strftime("%Y_%m_%d_%H_%M")
    save_file = f'outputs/{task_id if output is None else output}'
    vae_path = 'weights/face3dvae'
    set_strength = 0.8
    enable_3dmm_cfg = False
    remove_id_tex = True

    unet_path = 'weights/checkpoints'

    pipe = "weights/stable-diffusion-v1-5"

    print(f'Using CKPT : {unet_path}')
    face_embedder = FaceEmbedder(
        arcface_path="weights/IResNet100_WebFace42M.pth",
        dino_model_path="weights/dinov2-base"
    ).to(device)

    face_embedder.eval()
    face_embedder.requires_grad_(False)

    unet = UNetMotionModel.from_pretrained(
        os.path.join(unet_path, 'denoising_unet')).to(dtype)
    refnet: OriginalUNet2DConditionModel = OriginalUNet2DConditionModel.from_pretrained(
        os.path.join(unet_path, 'reference_unet'))
    refnet = refnet.to(dtype)
    refnet = refnet.cuda()
    pipe = StableDiffusionGLIGENInpaintPipeline.from_pretrained(
        pipe,
        unet=unet,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=dtype
    ).to('cuda')
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.vae = AutoencoderKLTemporalEncoderDecoder.from_pretrained(
        vae_path).cuda()
    pipe.vae = pipe.vae.to(dtype)
    pipe.refnet = refnet
    pipe.set_progress_bar_config(ncols=50)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5), std=(0.5)),
    ])

    face_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5), std=(0.5)),
    ])
    attr_transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
    ])

    path_id = unet_path.split('/')[-2]
    this_path_save_path = f'{save_file}_{path_id}'
    os.mkdir(this_path_save_path)
    frames_saved_root_path = f'{this_path_save_path}/frames'
    os.mkdir(frames_saved_root_path)
    video_saved_path = f'{this_path_save_path}/videos'
    os.mkdir(video_saved_path)

    num_frames = 8
    os.environ['new_pred_frames'] = str(num_frames)
    use_init_noise = 'n'
    use_init_noise = use_init_noise == 'y'
    init_noise = randn_tensor((num_frames, 4, 64, 64),
                              generator=None, device=device, dtype=dtype)

    dino_sets = [0.6]
    attr_sets = [0.6]

    for dinoid, set_dino_scale in enumerate(dino_sets):
        for attrid, set_attr_scale in enumerate(attr_sets):
            for video_path, face_path in zip(video_path_list, crop_face_path_list):
                try:
                    short_video_path = os.path.splitext(
                        os.path.basename(video_path))[0]
                    short_face_path = os.path.splitext(
                        os.path.basename(face_path))[0]

                    print(
                        f'Using DINO {set_dino_scale}, ATTR {set_attr_scale}, Video {short_video_path}, Face {short_face_path}')
                    scales_video_saved_path = f'{video_saved_path}/{short_video_path}_{short_face_path}.mp4'
                    frames_path = f'{frames_saved_root_path}/{short_video_path}_{short_face_path}'
                    os.mkdir(frames_path)

                    anno_path = video_path.replace(".mp4", ".txt")
                    anno = np.array(process_file(anno_path))
                    video_reader = VideoReader(video_path)
                    video_length = len(video_reader) - (len(video_reader) % 8)
                    video_length = min(video_length, 80)

                    face_pixel_values = []
                    crop_face = Image.open(face_path)
                    crop_face = crop_face.resize(
                        (224, 224), resample=Image.Resampling.LANCZOS)
                    face_pixel_values.append(torch.stack(
                        [face_transform(crop_face), ], dim=0))
                    face_pixel_values = torch.stack(face_pixel_values, dim=0)
                    face_pixel_values = face_pixel_values.repeat_interleave(
                        num_frames, dim=1)
                    face_pixel_values = face_pixel_values.reshape(
                        -1, *face_pixel_values.shape[-3:])
                    face_pixel_values = face_pixel_values.cuda()
                    prev_frames = deque([], maxlen=4)
                    for _ in range(4):
                        prev_frames.append(
                            Image.new("RGB", (512, 512), (127, 127, 127)))

                    for video_start in range(0, video_length, num_frames):

                        pixel_values = []
                        pixel_values_3dmm = []
                        cond_pixel_values = []
                        attr_pixel_values = []
                        mask_values = []
                        mask_values_3dmm = []
                        try:
                            indices = list(
                                range(video_start, video_start+num_frames))
                            frames = video_reader.get_batch(
                                indices).asnumpy()  # sample_frames+1, h, w, 3
                        except:
                            break

                        pixel_value = []
                        pixel_value_3dmm = []
                        attr_pixel_value = []
                        mask_value = []
                        mask_value_3dmm = []

                        for iidx, (i, frame) in enumerate(zip(indices, frames)):
                            pixel_value.append(transform(frame))
                            s_pixel_value_3dmm, s_mask_3dmm = face3dmodel.process_video_for_training(
                                pixel_value[-1].unsqueeze(0), anno[i][4:14].reshape(1, 5, 2), remove_id_tex=remove_id_tex)
                            pixel_value_3dmm.append(s_pixel_value_3dmm)
                            mask_value_3dmm.append(s_mask_3dmm)

                            # only crop
                            bbox, kps5, pose = anno[i][:4], anno[i][4:14], anno[i][14:17]
                            crop_face = extract_face(frame, bbox)

                            attr_pixel_value.append(attr_transform(crop_face))
                            mask = torch.zeros(
                                (1, 512, 512), dtype=torch.float32)
                            x1, y1, x2, y2 = anno[i][:4]
                            enlarge = 0.05
                            l = max(y2-y1, x2-x1)
                            l = l + int(round(enlarge * l))
                            cx, cy = (x1+x2)//2, (y1+y2) // 2
                            x1 = max(0, cx - l // 2)
                            y1 = max(0, cy - l // 2)
                            x2 = min(512, cx + l // 2)
                            y2 = min(512, cy + l // 2)
                            mask[:, y1:y2, x1:x2] = 1
                            mask_value.append(mask)
                        pixel_values.append(torch.stack(pixel_value, dim=0))
                        pixel_values_3dmm.append(
                            torch.stack(pixel_value_3dmm, dim=0))
                        attr_pixel_values.append(
                            torch.stack(attr_pixel_value, dim=0))
                        mask_values.append(torch.stack(mask_value, dim=0))
                        mask_values_3dmm.append(
                            torch.stack(mask_value_3dmm, dim=0))

                        pixel_values = torch.stack(pixel_values, dim=0)
                        pixel_values_3dmm = torch.stack(
                            pixel_values_3dmm, dim=0)
                        attr_pixel_values = torch.stack(
                            attr_pixel_values, dim=0)
                        mask_values = torch.stack(mask_values, dim=0)
                        mask_values_3dmm = torch.stack(mask_values_3dmm, dim=0)

                        pixel_values = pixel_values.reshape(
                            -1, *pixel_values.shape[-3:])
                        pixel_values_3dmm = pixel_values_3dmm.reshape(
                            -1, *pixel_values.shape[-3:])
                        attr_pixel_values = attr_pixel_values.reshape(
                            -1, *attr_pixel_values.shape[-3:])

                        mask_values = mask_values.reshape(
                            -1, *mask_values.shape[-3:])
                        mask_values_3dmm = mask_values_3dmm.reshape(
                            -1, *mask_values.shape[-3:])

                        attr_pixel_values = attr_pixel_values.cuda()

                        face_embeddings = face_embedder(
                            F.interpolate(face_pixel_values, size=(
                                112, 112), mode="bilinear"),
                            torch.ones(face_pixel_values.size(
                                0), device=face_pixel_values.device),
                            attr_pixel_values,
                            face_pixel_values,
                        )
                        seed = 12345
                        images = pipe(
                            [""] * num_frames,
                            image=pixel_values,
                            mask=mask_values,
                            image_3dmm=pixel_values_3dmm,
                            mask_3dmm=mask_values_3dmm,
                            width=512,
                            height=512,
                            num_inference_steps=40,
                            strength=set_strength,
                            num_images_per_prompt=1,
                            gligen_embeddings=face_embeddings,
                            guidance_scale=2.5,
                            attr_scale=set_attr_scale,
                            dino_scale=set_dino_scale,
                            cross_attention_kwargs={"ip_scale": 0.9},
                            negative_prompt=[""] * num_frames,
                            ref_image=list(prev_frames),
                            init_noise=init_noise if use_init_noise else None,
                            enable_3dmm_cfg=enable_3dmm_cfg,
                        ).images
                        for j, image in enumerate(images):
                            image.save(f'{frames_path}/{video_start+j}.jpg')
                            prev_frames.append(image)

                    frames = []
                    for i in range(video_length):
                        try:
                            frame = cv2.imread(f'{frames_path}/{i}.jpg')
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame)
                        except:
                            break
                    result_clip: VideoFileClip = ImageSequenceClip(
                        frames, fps=25)
                    result_clip.write_videofile(
                        scales_video_saved_path, codec="libx264", audio=False)
                    print('Save in : ', scales_video_saved_path)
                except Exception as e:
                    print(e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Inference runner')
    parser.add_argument('data_root', nargs='?',
                        help='data root directory containing `videos/` and `faces/`')
    parser.add_argument(
        '-s', '--source', help='single source image path (overrides data_root)')
    parser.add_argument(
        '-t', '--target', help='single target video path (overrides data_root)')
    parser.add_argument('--output', default=None,
                        help='output directory to save results')
    args = parser.parse_args()

    # If both source and target are provided, use those as single-item lists
    if args.source and args.target:
        video_path_list = [os.path.join(args.data_root, 'videos', args.target)]
        crop_face_path_list = [os.path.join(
            args.data_root, 'faces', args.source)]
    else:
        # Use all video-face image pairs in the provided data_root directory
        if not args.data_root:
            parser.error(
                'Either provide `data_root` positional or both `--source` and `--target`')
        infer_data_root = args.data_root

        video_dir = os.path.join(infer_data_root, 'videos')
        faces_dir = os.path.join(infer_data_root, 'faces')

        video_path_list = [os.path.join(video_dir, x) for x in os.listdir(
            video_dir) if x.endswith('mp4')]
        crop_face_path_list = [os.path.join(
            faces_dir, x) for x in os.listdir(faces_dir)]
    run(video_path_list, crop_face_path_list, output=args.output)
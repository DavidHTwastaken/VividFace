"""
cf:
    https://github.com/fudan-generative-vision/champ/blob/master/models/mutual_self_attention.py
    https://github.com/magic-research/magic-animate/blob/main/magicanimate/models/mutual_self_attention.py
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import UNet2DConditionModel
from model.attention import BasicTransformerBlock
from diffusers.models.attention import BasicTransformerBlock as raw_BasicTransformerBlock
from diffusers.models.transformers.transformer_temporal import (
    TransformerSpatioTemporalModel,
    TransformerTemporalModel,
)
from diffusers.models.transformers.transformer_temporal import TransformerTemporalModelOutput
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from model.unet_3d_blocks import DownBlockMotion, UpBlockMotion
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2DCrossAttn
from diffusers.utils.import_utils import is_torch_version
from diffusers.models.resnet import ResnetBlock2D


def torch_dfs(model: torch.nn.Module, omitted_modules=None):
    result = [model]
    for child in model.children():
        if omitted_modules is not None:
            if isinstance(child, omitted_modules):
                continue
        result += torch_dfs(child, omitted_modules)
    return result


class ReferenceAttentionControl:
    def __init__(
        self,
        unet,
        mode="write",
        do_classifier_free_guidance=False,
        attention_auto_machine_weight=float("inf"),
        gn_auto_machine_weight=1.0,
        style_fidelity=1.0,
        reference_attn=True,
        reference_adain=False,
        fusion_blocks="midup",
        batch_size=1,
        reference_scale: float = 1.0
    ) -> None:
        # 10. Modify self attention and group norm
        self.unet = unet
        self.mode = mode
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            fusion_blocks,
            batch_size=batch_size,
            reference_scale=reference_scale
        )

    def register_reference_hooks(
        self,
        mode,
        do_classifier_free_guidance,
        attention_auto_machine_weight,
        gn_auto_machine_weight,
        style_fidelity,
        reference_attn,
        reference_adain,
        dtype=torch.float16,
        batch_size=1,
        num_images_per_prompt=1,
        device=torch.device("cuda"),
        fusion_blocks="midup",
        reference_scale: float = 1.0
    ):
        MODE = mode
        do_classifier_free_guidance = do_classifier_free_guidance
        attention_auto_machine_weight = attention_auto_machine_weight
        gn_auto_machine_weight = gn_auto_machine_weight
        style_fidelity = style_fidelity
        reference_attn = reference_attn
        reference_adain = reference_adain
        fusion_blocks = fusion_blocks
        num_images_per_prompt = num_images_per_prompt
        dtype = dtype
        if do_classifier_free_guidance:
            uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt
                    + [0] * batch_size * num_images_per_prompt
                ).to(device).bool()
            )
        else:
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2).to(device).bool()
            )

        # reference_scale = self.reference_scale

        def hacked_transformer_2d_inner_forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Dict[str, torch.Tensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
        ):
            if cross_attention_kwargs is not None:
                if cross_attention_kwargs.get("scale", None) is not None:
                    logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
            if attention_mask is not None and attention_mask.ndim == 2:
                # assume that mask is expressed as:
                #   (1 = keep,      0 = discard)
                # convert mask into a bias that can be added to attention scores:
                #       (keep = +0,     discard = -10000.0)
                attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            # convert encoder_attention_mask to a bias the same way we do for attention_mask
            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

            # 1. Input
            if self.is_input_continuous:
                batch, _, height, width = hidden_states.shape
                residual = hidden_states

                hidden_states = self.norm(hidden_states)
                if not self.use_linear_projection:
                    hidden_states = self.proj_in(hidden_states)
                    inner_dim = hidden_states.shape[1]
                    hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                else:
                    inner_dim = hidden_states.shape[1]
                    hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                    hidden_states = self.proj_in(hidden_states)

            elif self.is_input_vectorized:
                hidden_states = self.latent_image_embedding(hidden_states)
            elif self.is_input_patches:
                height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
                hidden_states = self.pos_embed(hidden_states)

                if self.adaln_single is not None:
                    if self.use_additional_conditions and added_cond_kwargs is None:
                        raise ValueError(
                            "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                        )
                    batch_size = hidden_states.shape[0]
                    timestep, embedded_timestep = self.adaln_single(
                        timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
                    )

            # 2. Blocks
            #if self.caption_projection is not None:
            #    batch_size = hidden_states.shape[0]
            #    encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            #    encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

            for block in self.transformer_blocks:
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        attention_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        timestep,
                        cross_attention_kwargs,
                        class_labels,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states = block(
                        hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        timestep=timestep,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=class_labels,
                    )

            # 3. Output
            if self.is_input_continuous:
                if not self.use_linear_projection:
                    hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                    hidden_states = self.proj_out(hidden_states)
                else:
                    hidden_states = self.proj_out(hidden_states)
                    hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

                output = hidden_states + residual
            elif self.is_input_vectorized:
                hidden_states = self.norm_out(hidden_states)
                logits = self.out(hidden_states)
                # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
                logits = logits.permute(0, 2, 1)

                # log(p(x_0))
                output = F.log_softmax(logits.double(), dim=1).float()

            if self.is_input_patches:
                if self.config.norm_type != "ada_norm_single":
                    conditioning = self.transformer_blocks[0].norm1.emb(
                        timestep, class_labels, hidden_dtype=hidden_states.dtype
                    )
                    shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
                    hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
                    hidden_states = self.proj_out_2(hidden_states)
                elif self.config.norm_type == "ada_norm_single":
                    shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
                    hidden_states = self.norm_out(hidden_states)
                    # Modulation
                    hidden_states = hidden_states * (1 + scale) + shift
                    hidden_states = self.proj_out(hidden_states)
                    hidden_states = hidden_states.squeeze(1)

                # unpatchify
                if self.adaln_single is None:
                    height = width = int(hidden_states.shape[1] ** 0.5)
                hidden_states = hidden_states.reshape(
                    shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
                )
                hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
                output = hidden_states.reshape(
                    shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
                )

            self.bank.append(output.clone())

            if not return_dict:
                return (output,)
            return Transformer2DModelOutput(sample=output)

        def hacked_resnet2d_block_inner_forward(self, input_tensor: torch.FloatTensor, temb: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:

            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if self.time_emb_proj is not None:
                if not self.skip_time_act:
                    temb = self.nonlinearity(temb)
                temb = self.time_emb_proj(temb)[:, :, None, None]

            if self.time_embedding_norm == "default":
                if temb is not None:
                    hidden_states = hidden_states + temb
                hidden_states = self.norm2(hidden_states)
            elif self.time_embedding_norm == "scale_shift":
                if temb is None:
                    raise ValueError(
                        f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                    )
                time_scale, time_shift = torch.chunk(temb, 2, dim=1)
                hidden_states = self.norm2(hidden_states)
                hidden_states = hidden_states * (1 + time_scale) + time_shift
            else:
                hidden_states = self.norm2(hidden_states)

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            self.bank.append(output_tensor)

            return output_tensor

        def hacked_temporal_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.LongTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: torch.LongTensor = None,
            num_frames: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,
        ) -> TransformerTemporalModelOutput:
            # 1. Input
            batch_frames, channel, height, width = hidden_states.shape
            batch_size = batch_frames // num_frames

            residual = hidden_states

            hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, channel, height, width)
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
            if len(self.bank) > 0:
                addition = self.bank[0]
                img_mode = True if (num_frames % 2 != 0) else False
                total_bs = addition.shape[0]
                if img_mode is False: # video:
                    addition = addition[None, :].reshape(batch_size, 4, channel, height, width)
                    addition = addition.permute(0, 2, 1, 3, 4) # B, C, T, H, W
                    modify_hidden_states = torch.cat([addition, hidden_states], dim=2)
                    raw_add_frames = addition.shape[2]
                    raw_batch_frames = batch_frames
                    num_frames = modify_hidden_states.shape[2]
                    batch_frames = num_frames * batch_size
                else: # image
                    addition = addition[0].unsqueeze(0)[None, :].repeat(batch_size, 1, 1, 1, 1)
                    addition = addition.permute(0, 2, 1, 3, 4) # B, C, T, H, W
                    modify_hidden_states = torch.cat([addition, hidden_states], dim=2)
                    raw_add_frames = addition.shape[2]
                    raw_batch_frames = batch_frames
                    num_frames = modify_hidden_states.shape[2]
                    batch_frames = num_frames * batch_size

            else:
                modify_hidden_states = hidden_states
                # print("Do not read bank data!")


            hidden_states = self.norm(modify_hidden_states)
            hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channel)

            hidden_states = self.proj_in(hidden_states)

            # 2. Blocks
            for block in self.transformer_blocks:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

            # 3. Output
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states[None, None, :]
                .reshape(batch_size, height, width, num_frames, channel)
                .permute(0, 3, 4, 1, 2)
                .contiguous()
            )
            if len(self.bank) > 0:
                hidden_states = hidden_states[:, raw_add_frames:, :, :, :]
                hidden_states = hidden_states.reshape(raw_batch_frames, channel, height, width)
            else:
                hidden_states = hidden_states.reshape(batch_frames, channel, height, width)

            output = hidden_states + residual

            if not return_dict:
                return (output,)

            return TransformerTemporalModelOutput(sample=output)

        if self.reference_attn:
            # ======================= FOR MOTION =======================
            if self.mode == 'read':
                motion_modules = [
                    module for module in torch_dfs(self.unet, omitted_modules=(DownBlockMotion, UpBlockMotion))
                                  if isinstance(module, TransformerTemporalModel)
                ]
                for i, module in enumerate(motion_modules):
                    module._original_inner_forward = module.forward
                    if isinstance(module, TransformerTemporalModel):
                        module.forward = hacked_temporal_transformer_inner_forward.__get__(
                            module, TransformerTemporalModel
                        )
                        module.bank = []

            if self.mode == 'write':
                attn_2d_modules = [
                    module for module in torch_dfs(self.unet)
                                  if isinstance(module, Transformer2DModel)
                ]
                for i, module in enumerate(attn_2d_modules):
                    module._original_inner_forward = module.forward
                    if isinstance(module, Transformer2DModel):
                        module.forward = hacked_transformer_2d_inner_forward.__get__(
                            module, Transformer2DModel
                        )
                        module.bank = []

            if self.mode == 'read':
                motion_modules = [
                    module for module in torch_dfs(self.unet.down_blocks[3]) + torch_dfs(self.unet.up_blocks[0])
                                  if isinstance(module, TransformerTemporalModel)
                ]
                for i, module in enumerate(motion_modules):
                    module._original_inner_forward = module.forward
                    if isinstance(module, TransformerTemporalModel):
                        module.forward = hacked_temporal_transformer_inner_forward.__get__(
                            module, TransformerTemporalModel
                        )
                        module.bank = []

            if self.mode == 'write':
                res_2d_modules = [
                    module for module in torch_dfs(self.unet.down_blocks[3]) + torch_dfs(self.unet.up_blocks[0])
                                  if isinstance(module, ResnetBlock2D)
                ]
                for i, module in enumerate(res_2d_modules):
                    module._original_inner_forward = module.forward
                    if isinstance(module, ResnetBlock2D):
                        module.forward = hacked_resnet2d_block_inner_forward.__get__(
                            module, ResnetBlock2D
                        )
                        module.bank = []

    def update(self, writer, dtype=torch.float16):
        # ======================= FOR MOTION =======================
        reader_temporal_attn_modules = [
            module for module in torch_dfs(self.unet, omitted_modules=(DownBlockMotion, UpBlockMotion))
            if isinstance(module, TransformerTemporalModel)
        ]
        writer_temporal_attn_modules = [
            module for module in torch_dfs(writer.unet)
            if isinstance(module, Transformer2DModel)
        ]
        for r, w in zip(reader_temporal_attn_modules, writer_temporal_attn_modules):
            r.bank = [v.clone().to(dtype) for v in w.bank]

        reader_res_motion_modules = [
            module for module in torch_dfs(self.unet.down_blocks[3]) + torch_dfs(self.unet.up_blocks[0])
            if isinstance(module, TransformerTemporalModel)
            ]

        writer_res_motion_modules = [
            module for module in torch_dfs(writer.unet.down_blocks[3]) + torch_dfs(writer.unet.up_blocks[0])
                          if isinstance(module, ResnetBlock2D)
        ]
        for r, w in zip(reader_res_motion_modules, writer_res_motion_modules):
            r.bank = [v.clone().to(dtype) for v in w.bank]

    def set_motion_feature_zero(self, bool_mask_list):
        reader_temporal_attn_modules = [
            module for module in torch_dfs(self.unet, omitted_modules=(DownBlockMotion, UpBlockMotion))
            if isinstance(module, TransformerTemporalModel)
        ]
        for r in reader_temporal_attn_modules:
            for i in range(len(bool_mask_list)):
                if bool_mask_list[i]:
                    bs = r.bank[0].shape[0]
                    #start_index = bs // 5
                    r.bank[0][start_index+i*4:start_index+i*4+4] = r.bank[0][start_index+i*4:start_index+i*4+4] * 0.0

        reader_res_motion_modules = [
            module for module in torch_dfs(self.unet.down_blocks[3]) + torch_dfs(self.unet.up_blocks[0])
            if isinstance(module, TransformerTemporalModel)
            ]
        for r in reader_res_motion_modules:
            for i in range(len(bool_mask_list)):
                if bool_mask_list[i]:
                    bs = r.bank[0].shape[0]
                    #start_index = bs // 5
                    r.bank[0][start_index+i*4:start_index+i*4+4] = r.bank[0][start_index+i*4:start_index+i*4+4] * 0.0

    def clear(self):
        # ======================= FOR MOTION =======================
        if self.mode == 'read':
            motion_modules = [
                module for module in torch_dfs(self.unet, omitted_modules=(DownBlockMotion, UpBlockMotion))
                              if isinstance(module, TransformerTemporalModel)
            ]
            for m in motion_modules:
                m.bank.clear()

        if self.mode == 'write':
            attn_2d_modules = [
                module for module in torch_dfs(self.unet)
                              if isinstance(module, Transformer2DModel)
            ]
            for a2 in attn_2d_modules:
                a2.bank.clear()
        if self.mode == 'read':
            reader_res_motion_modules = [
                module for module in torch_dfs(self.unet.down_blocks[3]) + torch_dfs(self.unet.up_blocks[0])
                if isinstance(module, TransformerTemporalModel)
                ]
            for m in reader_res_motion_modules:
                m.bank.clear()

        if self.mode == 'write':
            writer_res_motion_modules = [
                module for module in torch_dfs(self.unet.down_blocks[3]) + torch_dfs(self.unet.up_blocks[0])
                              if isinstance(module, ResnetBlock2D)
            ]
            for m in writer_res_motion_modules:
                m.bank.clear()



class ReferenceNet(nn.Module):
    '''
    reference_unet: standard unet2d
    denoising_unet: unet2d + temporal
    '''
    def __init__(
        self,
        denoising_unet,
        reference_unet,
        fusion_blocks='midup',
    ):
        super().__init__()
        self.denoising_unet = denoising_unet
        self.referencenet = reference_unet
        self.reference_control_writer = ReferenceAttentionControl(
            reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            fusion_blocks=fusion_blocks,
        )
        self.reference_control_reader = ReferenceAttentionControl(
            denoising_unet,
            do_classifier_free_guidance=False,
            mode="read",
            fusion_blocks=fusion_blocks,
        )

    def forward(
        self,
        noisy_latents: torch.FloatTensor,
        timesteps: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        ref_latents: torch.FloatTensor = None,  # uncond_fwd
        ref_encoder_hidden_states: torch.FloatTensor = None,
        cross_attention_kwargs =None,
        set_reference_feature_zero = None,
        set_motion_feature_zero = None,
        return_dict = True,
    ):

        #TODO encoder_hidden_states_repeated is not correct
        if ref_latents is not None:
            self.referencenet(
                ref_latents,
                torch.zeros((ref_latents.shape[0],), dtype=timesteps.dtype, device=timesteps.device), #TODO
                encoder_hidden_states=ref_encoder_hidden_states.repeat_interleave(8, dim=0),
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        if set_reference_feature_zero:
            self.reference_control_reader.set_reference_feature_zero(set_reference_feature_zero)

        if set_motion_feature_zero:
            self.reference_control_reader.set_motion_feature_zero(set_motion_feature_zero)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=return_dict,
        )

        return model_pred

    def clear(self):
        self.reference_control_reader.clear()
        self.reference_control_writer.clear()



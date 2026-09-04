from ..models.wan_video_dit import WanModel
from ..models.wan_video_pose_encoder import PoseEncoder
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..schedulers.flow_match import FlowMatchScheduler
from ..schedulers.bride_match import BridgeMatchScheduler
from ..pipelines.base import BasePipeline
from ..prompters.wan_prompter import WanPrompter
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from typing import Optional
from functools import partial
from einops import rearrange

from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from ..models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_dit import RMSNorm, SelfAttention, CrossAttentionSrcCam, ViewPackEmbedding
from ..models.wan_video_vae import RMS_norm, CausalConv3d, Upsample


class WanVideoSpaTemPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ["text_encoder", "dit", "vae"]
        self.height_division_factor = 16
        self.width_division_factor = 16

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                # Local patch: Conv3d deliberately NOT wrapped. The DiT's convs
                # are small patch/viewpack embeddings, and the viewpack path
                # introspects their .kernel_size, which the wrapper hides.
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        self.enable_cpu_offload()

    def init_spatem_modules(
        self,
        disable_video_attn: bool = False,
        use_4d_attn: bool = False,
        use_mvs_attn: bool = False,
        use_src_self_attn: bool = False,
        use_src_cross_attn: bool = False,
        freqs_src_shift: int = 121,
        use_viewpack: bool = True,
        viewpack_dropout_prob: float = 0.0,
        use_pose_encoder: bool = True,
        pose_encoder_type: str = "rgb",
        use_cam_encoder: bool = False,
        range_4d_attn: tuple[int, int, int] = (0, None, 2),
        range_mvs_attn: tuple[int, int, int] = (1, None, 2),
        range_src_self_attn: tuple[int, int, int] = (0, None, 2),
        range_src_cross_attn: tuple[int, int, int] = (0, None, 2),
        use_lbm: bool = False,
        fill_wpmask_with_noise: bool = False,
    ):
        device, dtype = self.dit.patch_embedding.weight.device, self.dit.patch_embedding.weight.dtype

        if disable_video_attn:
            # todo: delete self_attn layers from the model
            if use_4d_attn:
                raise ValueError("Cannot use 4D attention when video attention is disabled")
            for block in self.dit.blocks:
                block.disable_video_attn = True

        if use_4d_attn:
            b, e, s = range_4d_attn
            for block in self.dit.blocks[b:e:s]:
                block.use_4d_attn = True

        if use_mvs_attn:
            b, e, s = range_mvs_attn
            for block in self.dit.blocks[b:e:s]:
                block.use_mvs_attn = True

                dim = block.self_attn.q.weight.shape[0]
                block.modulation_mvs = nn.Parameter(block.modulation[:, :3, :].detach().clone())
                block.norm1_mvs = nn.LayerNorm(dim, eps=block.norm1.eps, elementwise_affine=False).to(
                    device=device, dtype=dtype
                )
                block.self_attn_mvs = SelfAttention(dim, block.self_attn.num_heads, block.self_attn.norm_q.eps).to(
                    device=device, dtype=dtype
                )
                block.self_attn_mvs.load_state_dict(block.self_attn.state_dict(), strict=True)

        if not 0.0 <= viewpack_dropout_prob <= 1.0:
            raise ValueError("viewpack_dropout_prob should be between 0 and 1")
        if viewpack_dropout_prob > 0.0 and not use_viewpack:
            raise ValueError("viewpack_dropout_prob requires use_viewpack=True")

        if use_viewpack:
            viewpack_emb = ViewPackEmbedding(
                in_dim=self.dit.patch_embedding.weight.shape[1],
                dim=self.dit.patch_embedding.weight.shape[0],
                patch_size=list(self.dit.patch_embedding.kernel_size),
            )
            viewpack_emb.initialize_from_patch_embedding(self.dit.patch_embedding)
            self.dit.viewpack_embedding = viewpack_emb.to(device=device, dtype=dtype)
        elif use_src_self_attn:
            if use_src_cross_attn:
                raise ValueError("Cannot use both src self-attention and src cross-attention")

            b, e, s = range_src_self_attn
            for block in self.dit.blocks[b:e:s]:
                block.use_src_self_attn = True

                dim = block.self_attn.q.weight.shape[0]
                block.modulation_src = nn.Parameter(block.modulation[:, :3, :].detach().clone())
                block.norm1_src = nn.LayerNorm(dim, eps=block.norm1.eps, elementwise_affine=False).to(
                    device=device, dtype=dtype
                )
                block.self_attn_src = SelfAttention(dim, block.self_attn.num_heads, block.self_attn.norm_q.eps).to(
                    device=device, dtype=dtype
                )
                block.self_attn_src.load_state_dict(block.self_attn.state_dict(), strict=True)
        elif use_src_cross_attn:
            b, e, s = range_src_cross_attn
            for block in self.dit.blocks[b:e:s]:
                block.use_src_cross_attn = True

                dim = block.self_attn.q.weight.shape[0]
                block.modulation_src = nn.Parameter(block.modulation[:, :3, :].detach().clone())
                block.norm1_src = nn.LayerNorm(dim, eps=block.norm1.eps, elementwise_affine=False).to(
                    device=device, dtype=dtype
                )
                block.cross_attn_src = CrossAttentionSrcCam(
                    dim, block.self_attn.num_heads, block.self_attn.norm_q.eps
                ).to(device=device, dtype=dtype)
                block.cross_attn_src.load_state_dict(block.self_attn.state_dict(), strict=True)

        if use_pose_encoder:
            if pose_encoder_type == "rgb":
                in_channels = 3
            elif pose_encoder_type == "rgbd":
                in_channels = 4
            else:
                raise ValueError(f"Invalid pose_encoder_type: {pose_encoder_type}")
            pose_encoder = PoseEncoder(out_dim=self.dit.patch_embedding.out_channels, in_channels=in_channels)
            self.dit.pose_encoder = pose_encoder.to(device=device, dtype=dtype)

        if use_cam_encoder:
            dim = self.dit.blocks[0].self_attn.q.weight.shape[0]
            for block in self.dit.blocks:
                block.use_cam_encoder = True
                block.cam_encoder = nn.Linear(12, dim).to(device=device, dtype=dtype)
                block.projector = nn.Linear(dim, dim).to(device=device, dtype=dtype)
                block.cam_encoder.weight.data.zero_()
                block.cam_encoder.bias.data.zero_()
                block.projector.weight = nn.Parameter(torch.eye(dim, device=device, dtype=dtype))
                block.projector.bias = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))

        if use_lbm:
            # TODO: hard-code for now
            self.scheduler = BridgeMatchScheduler()
            self.dit.fill_wpmask_with_noise = fill_wpmask_with_noise

        self.dit.use_pose_encoder = use_pose_encoder
        self.dit.use_cam_encoder = use_cam_encoder
        self.dit.use_viewpack = use_viewpack
        self.dit.viewpack_dropout_prob = viewpack_dropout_prob
        self.dit.use_src_attn = use_src_self_attn or use_src_cross_attn
        self.dit.use_lbm = use_lbm
        self.dit.freqs_src_shift = freqs_src_shift

    def denoising_model(self):
        return self.dit

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}

    def tensor2video(self, frames):
        frames = rearrange(frames, "c f h w -> f h w c")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def prepare_extra_input(self, latents=None):
        return {}

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(
            input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        return latents

    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames

    def encode_fmask(self, mask, size):
        f_, h_, w_ = size
        g_ = (mask.shape[2] - 1) // (f_ - 1)

        # union of the frames in each latent (the first frame is encoded independently)
        mask = torch.cat([mask[:, :, :1].repeat(1, 1, g_ - 1, 1, 1), mask], dim=2)
        mask = rearrange(mask, "v c (f g) h w -> v c f g h w", f=f_, g=g_)
        mask = mask.max(dim=3).values

        # interpolate along the spatial dimensions
        mask = rearrange(mask, "v c f h w -> (v f) c h w")
        mask = F.interpolate(mask, size=(h_, w_), mode="area")
        mask = rearrange(mask, "(v f) c h w -> v c f h w", f=f_)
        return mask

    def encode_wpmask(self, mask, size):
        f_, h_, w_ = size
        g_ = (mask.shape[2] - 1) // (f_ - 1)

        # intersection of the frames in each latent (the first frame is encoded independently)
        mask = torch.cat([mask[:, :, :1].repeat(1, 1, g_ - 1, 1, 1), mask], dim=2)
        mask = rearrange(mask, "v c (f g) h w -> v c f g h w", f=f_, g=g_)
        mask = mask.min(dim=3).values

        # interpolate along the spatial dimensions
        mask = rearrange(mask, "v c f h w -> (v f) c h w")
        mask = F.interpolate(mask, size=(h_, w_), mode="area")
        mask = rearrange(mask, "(v f) c h w -> v c f h w", f=f_)
        return mask

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        src_videos: torch.Tensor = None,
        skeletons: torch.Tensor = None,
        wpvideos: torch.Tensor = None,
        wpmasks: torch.Tensor = None,
        cam_emb: torch.Tensor = None,
        input_image: Image.Image = None,
        input_video: torch.Tensor = None,
        denoising_strength: float = 1.0,
        seed: int = None,
        rand_device: str = "cpu",
        height: int = 832,
        width: int = 480,
        num_frames: int = None,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        tiled: bool = True,
        tile_size: tuple[int, int] = (52, 30),
        tile_stride: tuple[int, int] = (26, 15),
        tea_cache_l1_thresh: float = None,
        tea_cache_model_id: str = "",
        progress_bar_cmd=partial(tqdm, desc="Denoising"),
        progress_bar_st=None,
        return_tensor=False,
    ):
        assert num_frames is None, "num_frames is not supported for WanVideoSpaTemPipeline"
        assert input_image is None, "input_image is not supported for WanVideoSpaTemPipeline"
        assert input_video is None, "input_video is not supported for WanVideoSpaTemPipeline"
        assert tea_cache_l1_thresh is None, "tea_cache_l1_thresh is not supported for WanVideoSpaTemPipeline"

        # Parameter check
        height, width = self.check_resize_height_width(height, width)

        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Scheduler
        if self.dit.use_lbm:
            # bridge matching scheduler
            self.scheduler.set_timesteps(num_inference_steps)
        else:
            # flow matching scheduler
            self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        src_videos = src_videos.to(dtype=self.torch_dtype, device=self.device)
        if skeletons is not None:
            skeletons = skeletons.to(dtype=self.torch_dtype, device=self.device)
        if wpvideos is not None:
            wpvideos = wpvideos.to(dtype=self.torch_dtype, device=self.device)
        if wpmasks is not None:
            wpmasks = wpmasks.to(dtype=self.torch_dtype, device=self.device)
        if cam_emb is not None:
            cam_emb = cam_emb.to(dtype=self.torch_dtype, device=self.device)

        if skeletons is not None:
            num_cameras = skeletons.shape[0]
            num_frames = skeletons.shape[2]
        elif wpvideos is not None:
            num_cameras = wpvideos.shape[0]
            num_frames = wpvideos.shape[2]
        else:
            raise ValueError("Either skeletons or wpvideos must be provided")

        # Initialize noise
        noise_shape = (
            num_cameras,
            self.vae.model.z_dim,
            (num_frames - 1) // 4 + 1,
            height // self.vae.upsampling_factor,
            width // self.vae.upsampling_factor,
        )
        noise = self.generate_noise(noise_shape, seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)

        if input_video is not None:
            self.load_models_to_device(["vae"])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise

        # Encode source video
        self.load_models_to_device(["vae"])
        src_latents = self.encode_video(src_videos, **tiler_kwargs)
        src_latents = src_latents.to(dtype=self.torch_dtype, device=self.device)
        src_latents_nega = torch.zeros_like(src_latents)

        # Latent bridge matching
        if self.dit.use_lbm:
            if skeletons is not None:
                # skeleton-based: use primary src_latents as bridge source
                lbm_src_latents = src_latents[:1].expand_as(latents)
            elif wpvideos is not None:
                lbm_src_latents = self.encode_video(wpvideos, **tiler_kwargs).to(
                    dtype=self.torch_dtype, device=self.device
                )
                if self.dit.fill_wpmask_with_noise:
                    wpmask_latents = self.encode_wpmask(wpmasks, size=lbm_src_latents.shape[-3:])
                    lbm_src_latents = lbm_src_latents * wpmask_latents + noise * (1 - wpmask_latents)

            latents = lbm_src_latents

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        image_emb = {}

        # Extra input
        extra_input = self.prepare_extra_input(latents)

        # Denoise
        self.load_models_to_device(["dit"])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            timestep = torch.cat([timestep] * num_cameras, dim=0)

            # Inference
            noise_pred_posi = self.denoising_model()(
                x=latents,
                x_src=src_latents,
                timestep=timestep,
                skeletons=skeletons,
                cam_emb=cam_emb,
                **prompt_emb_posi,
                **image_emb,
                **extra_input,
            )
            if cfg_scale != 1.0:
                noise_pred_nega = self.denoising_model()(
                    x=latents,
                    x_src=src_latents_nega,
                    timestep=timestep,
                    skeletons=skeletons,
                    cam_emb=cam_emb,
                    **prompt_emb_nega,
                    **image_emb,
                    **extra_input,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

        # Decode
        self.load_models_to_device(["vae"])
        pred_videos = self.decode_video(latents, **tiler_kwargs)

        if return_tensor:
            return pred_videos

        self.load_models_to_device([])
        pred_video_list = []
        for pred_video in pred_videos:
            pred_video_list.append(self.tensor2video(pred_video))
        return pred_video_list


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None

        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e04, 9.23041404e03, -5.28275948e02, 1.36987616e01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e05, 4.90537029e04, -2.65530556e03, 5.87365115e01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e05, -3.54229917e04, 1.40286849e03, -1.35890334e01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [8.10705460e03, 2.13393892e03, -3.72934672e02, 1.66203073e01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(
                f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids})."
            )
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                (
                    (modulated_inp - self.previous_modulated_input).abs().mean()
                    / self.previous_modulated_input.abs().mean()
                )
                .cpu()
                .item()
            )
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states

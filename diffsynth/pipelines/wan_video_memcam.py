import os
import torch

import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F

from dataset.poses import compute_relative_pose

from .base import BasePipeline
from ..prompters import WanPrompter
from ..schedulers.flow_match import FlowMatchScheduler

from ..models import ModelManager
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear

from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_overlap import calculate_overlap_from_c2w
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample 
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm

from utils.compressor_utils import pad_for_3d_conv, compute_context_rope

TARGET_LENGTH = 20  # anchor 1l + predict 19l
ANCHOR_LENGTH = 1   # anchor 4f 1l or 1f 1l
PREDICT_FRAMES = 76 # 76 frames to predict per section
FRAMES_PER_SECTION = 77 # section 77f 20l

FOV_HALF_H = 45.0    # 水平半视场角（度）增大→更宽松的重叠判定
FOV_HALF_V = 30.0    # 垂直半视场角（度）增大→更宽松的重叠判定
FOV_SAMPLES = 5000   # 采样点数 增大→更准确但更慢
FOV_RADIUS = 50.0    # 采样球体半径


class WanVideoMemCamPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['text_encoder', 'image_encoder', 'dit', 'vae']
        self.height_division_factor = 16
        self.width_division_factor = 16


    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
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
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype, 
                offload_device="cpu",
                onload_dtype=dtype, 
                onload_device=self.device,
                computation_dtype=self.torch_dtype, 
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
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
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype, 
                offload_device="cpu",
                onload_dtype=dtype, 
                onload_device=self.device,
                computation_dtype=self.torch_dtype, 
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype, 
                    offload_device="cpu",
                    onload_dtype=dtype, 
                    onload_device="cpu",
                    computation_dtype=dtype, 
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()
    

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanVideoMemCamPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe
    
    
    def denoising_model(self):
        return self.dit


    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive, device=self.device)
        return {"context": prompt_emb}


    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames


    def forward(
        self,
        context_latents: torch.Tensor,       # (B, C, context_length, H, W)
        target_latents: torch.Tensor,        # (B, C, 20, H, W)
        context_pose: torch.Tensor,          # (B, context_length, 12)
        target_pose: torch.Tensor,           # (B, 20, 12)
        timestep: torch.Tensor,
        context: torch.Tensor,
    ):
        dit = self.dit
        
        # Time embedding
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        
        # Text embedding
        context = dit.text_embedding(context)

        # Context compression
        context_latents_padded = pad_for_3d_conv(context_latents, (1, 4, 4))  # Pad for kernel (1, 4, 4)
        ctx = dit.context_compressor(context_latents_padded)
        f_ctx, h_ctx, w_ctx = ctx.shape[2], ctx.shape[3], ctx.shape[4]
        ctx = rearrange(ctx, 'b c f h w -> b (f h w) c').contiguous()

        # Target patchify
        tgt, (f_tgt, h_tgt, w_tgt) = dit.patchify(target_latents)
        
        # Cat tokens: [context_tokens, target_tokens]
        x = torch.cat([ctx, tgt], dim=1)
        
        # Spatial sizes for cam_emb expansion
        ctx_spatial = h_ctx * w_ctx
        tgt_spatial = h_tgt * w_tgt
        cam_emb = (context_pose, target_pose)
        
        # ========== MemCam memory-context RoPE ==========
        # Target: positions 0-19 (preserve pretrained positions)
        # Context: positions starting from 20 (sequential assignment)
        context_freqs = compute_context_rope(
            dit=dit,
            f_ctx=f_ctx,
            h_tgt=h_tgt, w_tgt=w_tgt,
            h_ctx=h_ctx, w_ctx=w_ctx,
            device=x.device
        )  # (S_ctx, 1, dim) complex
        
        # Target freqs: positions 0 to f_tgt-1 (0-19)
        target_freqs = torch.cat([
            dit.freqs[0][:f_tgt].view(f_tgt, 1, 1, -1).expand(f_tgt, h_tgt, w_tgt, -1),
            dit.freqs[1][:h_tgt].view(1, h_tgt, 1, -1).expand(f_tgt, h_tgt, w_tgt, -1),
            dit.freqs[2][:w_tgt].view(1, 1, w_tgt, -1).expand(f_tgt, h_tgt, w_tgt, -1)
        ], dim=-1).reshape(f_tgt * h_tgt * w_tgt, 1, -1).to(x.device)
        
        # Concatenate: [context_freqs, target_freqs] -> (S_total, 1, dim)
        freqs = torch.cat([context_freqs, target_freqs], dim=0)
        
        # DiT blocks
        for block in dit.blocks:
            x = block(x, context, cam_emb, t_mod, freqs, ctx_spatial=ctx_spatial, tgt_spatial=tgt_spatial)
        
        # Head & unpatchify
        x = dit.head(x, t)
        ctx_tokens = f_ctx * h_ctx * w_ctx
        tgt_tokens = f_tgt * h_tgt * w_tgt
        x_tgt = x[:, ctx_tokens:ctx_tokens + tgt_tokens, :]
        x_tgt = dit.unpatchify(x_tgt, (f_tgt, h_tgt, w_tgt))
        
        return x_tgt


    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        c2ws=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=352,
        width=640,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        progress_bar_cmd=tqdm
    ):
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Encode Prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
        
        # Sections
        total_frames = c2ws.shape[0]
        print(f"Total frames: {total_frames}")
        assert total_frames % 76 == 1
        total_sections = (total_frames - 1) // 76
        print(f"Total sections: {total_sections}")

        # Encode input image
        self.load_models_to_device(['vae'])
        input_image_tensor = self.preprocess_image(input_image).permute(1, 0, 2, 3).unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
        start_latent = self.encode_video(input_image_tensor, **tiler_kwargs)[0]  # (C, 1, H/8, W/8)
        
        # Latent shape
        latent_C = start_latent.shape[0]  # 16
        latent_H = start_latent.shape[2]  # H // 8
        latent_W = start_latent.shape[3]  # W // 8
        
        # ============ 存储结构 ============
        all_section_latents = []  # (0:start_latent, else:1+19latents)
        all_generated_frames = {} # {frame_idx:frame_tensor}
        section_start_frames = [i * (FRAMES_PER_SECTION - 1) for i in range(total_sections)]
        
        # 初始化: section 0 的 anchor 来自输入图片
        all_section_latents.append(start_latent)  # (C, 1, H, W) 作为 section -1 的 "latent"
        all_generated_frames[0] = input_image_tensor.cpu()  # 帧0
        
        # Vanilla Sampling
        for section_idx in range(total_sections):
            print(f"Generating section {section_idx + 1}/{total_sections}")
            section_start_frame = section_start_frames[section_idx]
            
            # ============ 获取anchor latent + 确定帧范围 ============
            if section_idx == 0:
                # Section 0: anchor = start_latent (1f1l, 用户输入图片)
                anchor_latent = all_section_latents[0]  # (C, 1, H, W) - start_latent
                anchor_latent = anchor_latent.to(dtype=self.torch_dtype, device=self.device)

                # anchor 的 pose frame: 当前clip的起始帧
                anchor_pose_frame = section_start_frame  # frame 0 for clip_idx=0
                
                # anchor 覆盖的帧范围 (单帧)
                anchor_frame_range = [section_start_frame]
                
                # predict 帧索引: frames 1, 5, 9, ..., 73
                predict_latent_frames = [section_start_frame + (i * 4 + 1) for i in range(TARGET_LENGTH - ANCHOR_LENGTH)]
                
                # predict 覆盖的帧范围 (frames 1-76)
                predict_frame_range = list(range(section_start_frame + 1, section_start_frame + FRAMES_PER_SECTION))
            else:
                # Section > 0: anchor = 上一个section的最后一个latent (4f1l)
                prev_section_latent = all_section_latents[section_idx]  # (C, 20, H, W)
                anchor_latent = prev_section_latent[:, -1:, :, :]  # (C, 1, H, W) - 4f1l
                anchor_latent = anchor_latent.to(dtype=self.torch_dtype, device=self.device)

                # anchor 的 pose frame: 用前一个clip最后一个latent的pose帧索引
                anchor_pose_frame = section_start_frame - 3  # frame 73
                
                # anchor 覆盖的帧范围 (前一个section的最后4帧: 73-76)
                anchor_frame_range = list(range(section_start_frame - 3, section_start_frame + 1))
                
                # predict 帧索引: frames 77, 81, 85, ..., 149
                predict_latent_frames = [section_start_frame + 1 + (i * 4) for i in range(TARGET_LENGTH - ANCHOR_LENGTH)]
                
                # predict 覆盖的帧范围 (frames 77-152)
                predict_frame_range = list(range(section_start_frame + 1, section_start_frame + FRAMES_PER_SECTION))
            
            # ============ 构建 Context ============
            context_latent_list = []
            context_frame_indices = []
            
            # 要排除的帧: anchor + predict 覆盖的所有帧
            exclude_frames = set(anchor_frame_range) | set(predict_frame_range)
            context_target_frames = [section_start_frame + 1 + i * 1 for i in range(PREDICT_FRAMES)]
            
            if section_idx == 0:
                # Section 0: context全零 (与训练drop_context一致)
                for _ in range(PREDICT_FRAMES):
                    context_latent_list.append(torch.zeros(latent_C, 1, latent_H, latent_W, dtype=anchor_latent.dtype, device=anchor_latent.device))
                    context_frame_indices.append(anchor_pose_frame)
            else:
                # Section > 0: 按 context_target_frames 选择，每个目标帧选1个最佳重叠 context
                candidate_frame_indices = [fidx for fidx in all_generated_frames.keys() if fidx not in exclude_frames]
                
                print(f"  Selecting context frames (1 per target, {PREDICT_FRAMES} targets)...")
                print(f"  Excluding frames: anchor={anchor_frame_range}, predict={predict_frame_range[0]}-{predict_frame_range[-1]}")
                
                self.load_models_to_device(['vae'])
                for frame_idx in context_target_frames:  # 按 frame_interval 选择的目标帧
                    # 计算与该帧重叠度最高的帧
                    target_c2w = c2ws[frame_idx]
                    best_idx = None
                    best_iou = -1
                    for candidate_idx in candidate_frame_indices:
                        candidate_c2w = c2ws[candidate_idx]
                        iou = calculate_overlap_from_c2w(
                            target_c2w, candidate_c2w,
                            fov_half_h=FOV_HALF_H, fov_half_v=FOV_HALF_V,
                            num_samples=FOV_SAMPLES, radius=FOV_RADIUS,
                            return_details=False
                        )
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = candidate_idx
                    
                    # 选择重叠度最高的1帧
                    if best_idx is not None and best_idx in all_generated_frames:
                        chosen_frame = all_generated_frames[best_idx]
                        chosen_frame = chosen_frame.to(dtype=self.torch_dtype, device=self.device)
                        chosen_latent = self.encode_video(chosen_frame, **tiler_kwargs)[0]
                        context_latent_list.append(chosen_latent)
                        context_frame_indices.append(best_idx)
                    else:
                        # 没有有效context，用零填充
                        context_latent_list.append(torch.zeros(latent_C, 1, latent_H, latent_W, dtype=anchor_latent.dtype, device=anchor_latent.device))
                        context_frame_indices.append(anchor_pose_frame)

                print(f"  Selected [{context_frame_indices}] as context frames")
            
            # 拼接context: (C, context_length, H, W)
            context_latent = torch.cat(context_latent_list, dim=1)
            
            # Context pose: 相对于anchor
            context_pose = compute_relative_pose(c2ws, anchor_pose_frame, context_frame_indices)  # (context_length, 12)
            
            # ============ 准备 Target (1个anchor + 19帧噪声 = 20帧) ============
            # 生成19帧噪声
            noise_latents = self.generate_noise((1, latent_C, TARGET_LENGTH - ANCHOR_LENGTH, latent_H, latent_W), seed=seed, device=rand_device, dtype=torch.float32).to(dtype=self.torch_dtype, device=self.device)

            # Target pose: 20帧 (1 anchor + 19predict)
            target_latent_frames = [anchor_pose_frame] + predict_latent_frames
            target_pose = compute_relative_pose(c2ws, anchor_pose_frame, target_latent_frames)  # (20, 12)
            
            # ============ Denoising ============
            self.load_models_to_device(["dit"])
            
            # Input
            context_latent_input = context_latent.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)  # (1, C, context_length, H, W)
            context_pose_input = context_pose.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)  # (1, context_length, 12)
            target_pose_input = target_pose.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)  # (1, 20, 12)
            anchor_latent_batch = anchor_latent.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)  # (1, C, 1, H, W)
                
            for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
                timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
                
                # Target input: 1个anchor(干净) + 19帧噪声 = 20帧
                target_input = torch.cat([anchor_latent_batch, noise_latents], dim=2)  # (1, C, 20, H, W)
                    
                # 前向传播 (context 和 target 分开传入)
                noise_pred_posi = self.forward(
                    context_latents=context_latent_input,       # (1, C, context_length, H, W)
                    target_latents=target_input,                # (1, C, 20, H, W)
                    context_pose=context_pose_input,            # (1, context_length, 12)
                    target_pose=target_pose_input,              # (1, 20, 12)
                    timestep=timestep,
                    context=prompt_emb_posi["context"],
                )
                    
                if cfg_scale != 1.0:
                    noise_pred_nega = self.forward(
                        context_latents=context_latent_input,
                        target_latents=target_input,
                        context_pose=context_pose_input,
                        target_pose=target_pose_input,
                        timestep=timestep,
                        context=prompt_emb_nega["context"],
                    )
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi
                
                # Scheduler
                noise_pred_rest = noise_pred[:, :, ANCHOR_LENGTH:, :, :]  # (1, C, 19, H, W)
                noise_latents = self.scheduler.step(noise_pred_rest, self.scheduler.timesteps[progress_id], noise_latents)

            # ============ 存储当前section的latent ============
            # start (C, 1, H, W) + noise_latents (C, 19, H, W) = 20个latent
            section_start_latent = self.encode_video(all_generated_frames[section_start_frame].to(dtype=self.torch_dtype, device=self.device), **tiler_kwargs)[0]
            section_full_latent = torch.cat([section_start_latent, noise_latents.squeeze(0)], dim=1)  # (C, 20, H, W)
            all_section_latents.append(section_full_latent.cpu())
            
            # ============ 解码当前section并存储帧 ============
            self.load_models_to_device(['vae'])
            section_frames = self.decode_video(section_full_latent.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device), **tiler_kwargs)  # (1, C, 77, H, W)
            
            # 将解码后的帧逐帧存储到all_generated_frames
            section_frames_cpu = section_frames.cpu()  # (1, C, 77, H, W)
            for local_frame_idx in range(section_frames_cpu.shape[2]):
                global_frame_idx = section_start_frame + local_frame_idx
                frame_tensor = section_frames_cpu[:, :, local_frame_idx:local_frame_idx+1, :, :]  # (1, C, 1, H, W)
                all_generated_frames[global_frame_idx] = frame_tensor
            print(f"Section {section_idx} completed.")

        # ============ Decode ============
        print("Decoding all sections...")
        self.load_models_to_device(['vae'])

        all_frames = []
        for section_idx, section_latent in enumerate(all_section_latents[1:]):
            section_latent = section_latent.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            section_frames = self.decode_video(section_latent, **tiler_kwargs).squeeze(0)

            if section_idx == 0:
                all_frames.append(section_frames)
            else:
                all_frames.append(section_frames[:, 1:, :, :])

        all_frames = torch.cat(all_frames, dim=1)
        frames = self.tensor2video(all_frames.cpu())
        self.load_models_to_device([])
        return frames

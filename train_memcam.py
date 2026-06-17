import os
import glob
import json
import torch
import random
import imageio
import argparse
import numpy as np
import torchvision
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange
from torchvision.transforms import v2

import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from diffsynth import ModelManager
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from diffsynth.pipelines.wan_video_memcam import WanVideoMemCamPipeline

from dataset.poses import compute_c2w_matrix, c2w_to_12dim
from utils.compressor_utils import pad_for_3d_conv, compute_context_rope


TARGET_LENGTH = 20  # anchor 1l + predict 19l
ANCHOR_LENGTH = 1   # anchor 4f 1l or 1f 1l
CONTEXT_LENGTH = 76 # 1v1 predict 76f 76l
FRAMES_PER_CLIP = 77 # clip 77f 20l


class LossRecorder:
    def __init__(self, beta: float = 0.98):
        self.beta = beta
        self.value: float | None = None

    def add(self, *, loss: float, **kwargs) -> None:
        if self.value is None:
            self.value = loss
        else:
            self.value = self.beta * self.value + (1 - self.beta) * loss

    @property
    def moving_average(self) -> float:
        return self.value if self.value is not None else 0.0


class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path):
        self.base_path = base_path
        video_dir = os.path.join(base_path, "videos")
        self.video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
        print(f"Found {len(self.video_files)} videos in {video_dir}")

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # 获取视频总帧数 7601
        reader = imageio.get_reader(video_path)
        total_frames = reader.count_frames()
        reader.close()

        # 计算clip数量 100
        num_clips = ((total_frames - FRAMES_PER_CLIP) // (FRAMES_PER_CLIP - 1)) + 1

        # 加载所有clip的caption
        captions = []
        for clip_idx in range(num_clips):
            text_path = os.path.join(self.base_path, "texts", f"{video_name}_clip{clip_idx:04d}.txt")
            if os.path.exists(text_path):
                with open(text_path, 'r', encoding='utf-8') as f:
                    captions.append(f.read().strip())

        return {
            "video_path": video_path,
            "video_name": video_name,
            "total_frames": total_frames,
            "num_clips": num_clips,
            "captions": captions,
        }

    def __len__(self):
        return len(self.video_files)


class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, output_dir, height=352, width=640, tiled=False, tile_size=(34, 34), tile_stride=(18, 16), skip_existing=True):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([text_encoder_path, vae_path])
        self.pipe = WanVideoMemCamPipeline.from_model_manager(model_manager)
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        self.output_dir = output_dir
        self.height = height
        self.width = width
        self.skip_existing = skip_existing

        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image

    def load_clip_frames(self, reader, start_frame, num_frames):
        frames = []
        for frame_id in range(start_frame, start_frame + num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            frame = self.frame_process(frame)
            frames.append(frame)

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        return frames

    def test_step(self, batch, batch_idx):
        video_path = batch["video_path"][0]
        video_name = batch["video_name"][0]
        total_frames = batch["total_frames"].item()
        num_clips = batch["num_clips"].item()
        captions = batch["captions"]  # list of strings

        # 创建输出目录
        video_output_dir = os.path.join(self.output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        # 检查是否已完成
        prompt_path = os.path.join(video_output_dir, "prompt.pt")
        if self.skip_existing and os.path.exists(prompt_path):
            # 验证所有latent文件是否存在
            all_exist = all(
                os.path.exists(os.path.join(video_output_dir, f"latent_clip{i:04d}.pt"))
                for i in range(num_clips)
            )
            if all_exist:
                print(f"Skipping {video_name} (already processed)")
                return

        self.pipe.device = self.device
        print(f"Processing {video_name}: {total_frames} frames, {num_clips} clips")

        # 编码所有clip的prompt
        prompt_embs = []
        for clip_idx in range(num_clips):
            caption = captions[clip_idx][0] if isinstance(captions[clip_idx], list) else captions[clip_idx]
            if caption:
                prompt_emb = self.pipe.encode_prompt(caption)
                prompt_embs.append(prompt_emb)

        # 保存所有prompt embeddings
        torch.save(prompt_embs, prompt_path)
        print(f"  Saved prompt embeddings: {len([p for p in prompt_embs if p is not None])} valid / {num_clips} total")

        # 打开视频reader
        reader = imageio.get_reader(video_path)

        # 滑动窗口编码每个clip的latent
        for clip_idx in range(num_clips):
            latent_path = os.path.join(video_output_dir, f"latent_clip{clip_idx:04d}.pt")

            # 跳过已存在的
            if self.skip_existing and os.path.exists(latent_path):
                continue

            start_frame = clip_idx * (FRAMES_PER_CLIP - 1)

            # 加载77帧
            clip_frames = self.load_clip_frames(reader, start_frame, FRAMES_PER_CLIP)
            clip_frames = clip_frames.unsqueeze(0)  # (1, C, T, H, W)
            clip_frames = clip_frames.to(dtype=self.pipe.torch_dtype, device=self.device)

            # VAE编码 -> (C, 20, H/8, W/8)
            with torch.no_grad():
                latent = self.pipe.encode_video(clip_frames, **self.tiler_kwargs)[0]

            # 保存latent
            torch.save(latent.cpu(), latent_path)

        reader.close()
        print(f"  Done: {video_name}")


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, overlap_labels_path, singles_path, jsons_path, drop_context_prob=0.1, pose_scale=100.0):
        self.base_path = base_path
        self.overlap_labels_path = overlap_labels_path
        self.singles_path = singles_path
        self.jsons_path = jsons_path
        self.drop_context_prob = drop_context_prob
        self.pose_scale = pose_scale

        tensors_dir = os.path.join(base_path, "tensors")

        # 按场景组织数据
        self.scenes = {}
        self.items = []

        for video_name in sorted(os.listdir(tensors_dir)):
            video_tensor_dir = os.path.join(tensors_dir, video_name)
            if not os.path.isdir(video_tensor_dir):
                continue

            # 检查必要文件
            prompt_path = os.path.join(video_tensor_dir, "prompt.pt")
            json_path = os.path.join(jsons_path, f"{video_name}.json")
            singles_dir = os.path.join(singles_path, video_name)
            overlap_dir = os.path.join(overlap_labels_path, video_name)

            if not all(os.path.exists(p) for p in [prompt_path, json_path]):
                print(f"Warning: Missing files for {video_name}, skipping")
                continue

            if not os.path.isdir(singles_dir):
                print(f"Warning: Singles dir not found for {video_name}, skipping")
                continue

            if not os.path.isdir(overlap_dir):
                print(f"Warning: Overlap labels not found for {video_name}, skipping")
                continue

            # 加载prompt
            prompt_embs = torch.load(prompt_path, map_location="cpu", weights_only=False)

            self.scenes[video_name] = {
                "prompt_embs": prompt_embs,
                "json_path": json_path,
                "singles_dir": singles_dir,
                "overlap_dir": overlap_dir,
                "clips": {},
            }

            # 收集有效的clip
            for clip_idx, prompt_emb in enumerate(prompt_embs):
                if prompt_emb is None:
                    continue

                latent_path = os.path.join(video_tensor_dir, f"latent_clip{clip_idx:04d}.pt")
                if not os.path.exists(latent_path):
                    continue

                self.scenes[video_name]["clips"][clip_idx] = latent_path
                self.items.append({
                    "video_name": video_name,
                    "clip_idx": clip_idx,
                })

        print(f"Found {len(self.items)} valid training samples from {len(self.scenes)} scenes")
        assert len(self.items) > 0, "No valid samples!"

        # Cache
        self.latent_cache = {}
        self.c2w_cache = {}

    def _load_clip_latent(self, video_name, clip_idx):
        cache_key = f"{video_name}_clip{clip_idx}"
        if cache_key not in self.latent_cache:
            latent_path = self.scenes[video_name]["clips"][clip_idx]
            self.latent_cache[cache_key] = torch.load(latent_path, map_location="cpu", weights_only=True)
        return self.latent_cache[cache_key]

    def _load_single_latent(self, video_name, frame_idx):
        single_path = os.path.join(self.scenes[video_name]["singles_dir"], f"{frame_idx:04d}.pt")
        return torch.load(single_path, map_location="cpu", weights_only=True)

    def _load_prompt_emb(self, video_name, clip_idx):
        return self.scenes[video_name]["prompt_embs"][clip_idx]

    def _load_overlap_frames(self, video_name, frame_idx):
        overlap_path = os.path.join(self.scenes[video_name]["overlap_dir"], f"{frame_idx}.json")
        with open(overlap_path, 'r') as f:
            data = json.load(f)
        return [int(x) for x in data.get("overlapping_frames", [])]

    def _load_poses(self, video_name):
        if video_name in self.c2w_cache:
            return self.c2w_cache[video_name]
        json_path = self.scenes[video_name]["json_path"]
        with open(json_path, 'r') as f:
            data = json.load(f)
        camera_data = data['CineCameraActor']
        frame_keys = sorted(camera_data.keys(), key=int)
        c2ws = []
        for frame_key in frame_keys:
            frame_data = camera_data[frame_key]
            c2w = compute_c2w_matrix(frame_data, scale=self.pose_scale)
            c2ws.append(c2w)
        c2ws = np.stack(c2ws, axis=0)
        self.c2w_cache[video_name] = c2ws
        return c2ws

    def _compute_relative_pose(self, c2ws, reference_idx, target_idx):
        c2w_ref = c2ws[reference_idx]
        w2c_ref = np.linalg.inv(c2w_ref)
        relative_c2w = w2c_ref @ c2ws[target_idx]
        return c2w_to_12dim(relative_c2w)

    def __getitem__(self, index):
        item = self.items[index]
        video_name = item["video_name"]
        clip_idx = item["clip_idx"]

        # ============ 加载当前Clip的Latent ============
        clip_latent = self._load_clip_latent(video_name, clip_idx)  # (C, 20, H, W)
        C, _, H, W = clip_latent.shape

        # ============ 计算帧索引 ============
        clip_start_frame = clip_idx * (FRAMES_PER_CLIP - 1)  # clip开始帧的绝对索引

        # ============ 确定是否使用前一个clip的anchor ============
        drop_context = random.random() < self.drop_context_prob
        use_prev_anchor = (clip_idx > 0) and (not drop_context)

        # ============ 构建 Target (1个anchor + 19个predict = 20个latent) ============
        if use_prev_anchor:
            # 使用前一个clip的最后一个latent作为anchor (4f1l)
            prev_clip_latent = self._load_clip_latent(video_name, clip_idx - 1)  # (C, 20, H, W)
            anchor_latent = prev_clip_latent[:, -1:, :, :]  # (C, 1, H, W)

            # predict: 当前clip的latent 1-19
            predict_latent = clip_latent[:, 1:, :, :]  # (C, 19, H, W)

            # anchor 的 pose frame: 用前一个clip最后一个latent的pose帧索引
            anchor_pose_frame = clip_start_frame - 3  # frame 73

            # anchor 覆盖的帧范围 (前一个clip的最后4帧: frames 73-76)
            anchor_frame_range = list(range(clip_start_frame - 3, clip_start_frame + 1))

            # predict 的帧索引 (每个latent对应4帧的第一帧): frames 77, 81, ..., 149
            predict_latent_frames = [clip_start_frame + 1 + (i * 4) for i in range(TARGET_LENGTH - ANCHOR_LENGTH)]

            # predict 覆盖的完整帧范围 (frames 77-152)
            predict_frame_range = list(range(clip_start_frame + 1, clip_start_frame + FRAMES_PER_CLIP))
        else:
            # 使用当前clip的第一帧作为anchor (1f1l)
            anchor_latent = clip_latent[:, :1, :, :]  # (C, 1, H, W)

            # predict: 当前clip的latent 1-19
            predict_latent = clip_latent[:, 1:, :, :]  # (C, 19, H, W)

            # anchor 的 pose frame: 当前clip的起始帧
            anchor_pose_frame = clip_start_frame  # frame 0 for clip_idx=0

            # anchor 覆盖的帧范围 (单帧)
            anchor_frame_range = [clip_start_frame]

            # predict 的帧索引: frames 1, 5, 9, ..., 73
            predict_latent_frames = [clip_start_frame + (i * 4 + 1) for i in range(TARGET_LENGTH - ANCHOR_LENGTH)]

            # predict 覆盖的完整帧范围 (frames 1-76)
            predict_frame_range = list(range(clip_start_frame + 1, clip_start_frame + FRAMES_PER_CLIP))

        # 拼接 target_latents: anchor (1) + predict (19) = 20
        target_latents = torch.cat([anchor_latent, predict_latent], dim=1)  # (C, 20, H, W)

        # target 对应的帧索引 (用于 pose 计算)
        target_frame_indices = [anchor_pose_frame] + predict_latent_frames

        # ============ 构建Context (predict覆盖的76帧，每帧选1个context) ============
        context_latents = []
        context_frame_indices = []

        if drop_context:
            # Drop时用0填充
            for _ in range(CONTEXT_LENGTH):
                context_latents.append(torch.zeros(C, 1, H, W, dtype=clip_latent.dtype))
                context_frame_indices.append(anchor_pose_frame)
        else:
            # 要排除的帧: anchor覆盖的帧 + predict覆盖的帧
            exclude_frames = set(anchor_frame_range) | set(predict_frame_range)

            # 按帧维度选择：predict_frame_range 有76帧，每帧选1个context
            for frame_idx in predict_frame_range:
                # 加载该帧预计算的重叠帧列表
                overlap_frames = self._load_overlap_frames(video_name, frame_idx)
                # 排除不可用的帧
                valid_overlaps = [f for f in overlap_frames if f not in exclude_frames]

                if valid_overlaps:
                    # 随机选择1帧作为该帧的context
                    chosen_frame = random.choice(valid_overlaps)
                    chosen_latent = self._load_single_latent(video_name, chosen_frame)
                    context_latents.append(chosen_latent)
                    context_frame_indices.append(chosen_frame)
                else:
                    # 没有有效重叠帧，用零填充
                    context_latents.append(torch.zeros(C, 1, H, W, dtype=clip_latent.dtype))
                    context_frame_indices.append(anchor_pose_frame)

        # 拼接context: (C, 76, H, W)
        context_latents = torch.cat(context_latents, dim=1)

        # ============ 加载Pose数据 ============
        c2ws = self._load_poses(video_name)

        # ============ 计算Pose (相对于anchor帧) ============
        # Target pose: 20帧
        target_poses = []
        for frame_idx in target_frame_indices:
            pose = self._compute_relative_pose(c2ws, anchor_pose_frame, frame_idx)
            target_poses.append(pose)
        target_pose = torch.tensor(target_poses, dtype=torch.float32)  # (20, 12)

        # Context pose: 76帧
        context_poses = []
        for frame_idx in context_frame_indices:
            pose = self._compute_relative_pose(c2ws, anchor_pose_frame, frame_idx)
            context_poses.append(pose)
        context_pose = torch.tensor(context_poses, dtype=torch.float32)  # (76, 12)

        # ============ 加载Prompt ============
        prompt_emb = self._load_prompt_emb(video_name, clip_idx)

        return {
            "target_latents": target_latents,       # (C, 20, H, W)
            "context_latents": context_latents,     # (C, 76, H, W)
            "target_pose": target_pose,             # (20, 12)
            "context_pose": context_pose,           # (76, 12)
            "prompt_emb": prompt_emb
        }

    def __len__(self):
        return len(self.items)


def collate_fn(batch):
    target_latents = torch.stack([item["target_latents"] for item in batch], dim=0)  # (B, C, 20, H, W)
    context_latents = torch.stack([item["context_latents"] for item in batch], dim=0)  # (B, C, 76, H, W)
    target_pose = torch.stack([item["target_pose"] for item in batch], dim=0)  # (B, 20, 12)
    context_pose = torch.stack([item["context_pose"] for item in batch], dim=0)  # (B, 76, 12)
    prompt_contexts = torch.stack([item["prompt_emb"]["context"] for item in batch], dim=0)
    return {
        "target_latents": target_latents,
        "context_latents": context_latents,
        "target_pose": target_pose,
        "context_pose": context_pose,
        "prompt_emb": {"context": prompt_contexts},
    }


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=5e-5,
        resume_ckpt_path=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        gradient_checkpointing_ratio=1.0,
        height=352, width=640,
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])

        self.pipe = WanVideoMemCamPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Camera Encoder
        dim=self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in self.pipe.dit.blocks:
            block.cam_encoder = nn.Linear(12, dim)
            block.projector = nn.Linear(dim, dim)
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))

        # Context Compressor
        self.pipe.dit.context_compressor = nn.Conv3d(16, dim, kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self._initialize_context_compressor_from_patch_embedding(
            self.pipe.dit.context_compressor,
            self.pipe.dit.patch_embedding
        )

        # Resume
        if resume_ckpt_path is not None:
            state_dict = torch.load(resume_ckpt_path, weights_only=False, map_location="cpu")
            self.pipe.dit.load_state_dict(state_dict, strict=True)

        # Param
        self.freeze_parameters()
        keywords = ["cam_encoder", "projector", "context_compressor", "self_attn", "cross_attn", "ffn"]
        for name, module in self.pipe.denoising_model().named_modules():
            if any(keyword in name for keyword in keywords):
                for param in module.parameters():
                    param.requires_grad = True

        # Trainable
        dit_trainable = sum(p.numel() for p in self.pipe.dit.parameters() if p.requires_grad)
        print(f"Total trainable params: {dit_trainable/1e6:.2f}M")

        # Checkpointing
        if use_gradient_checkpointing:
            num_blocks = len(self.pipe.dit.blocks)
            num_ckpt_blocks = int(num_blocks * gradient_checkpointing_ratio)
            if num_ckpt_blocks > 0 and num_ckpt_blocks < num_blocks:
                step = num_blocks / num_ckpt_blocks
                self.ckpt_block_indices = set(int(i * step) for i in range(num_ckpt_blocks))
            elif num_ckpt_blocks >= num_blocks:
                self.ckpt_block_indices = set(range(num_blocks))
            else:
                self.ckpt_block_indices = set()
            print(f"Gradient checkpointing: {len(self.ckpt_block_indices)}/{num_blocks} blocks")

        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.gradient_checkpointing_ratio = gradient_checkpointing_ratio
        self.loss_recorder = LossRecorder()

    def _initialize_context_compressor_from_patch_embedding(self, context_compressor, patch_embedding):
        """
        Copy weights from patch_embedding and scale by compression ratio.
        patch_embedding: kernel (1, 2, 2), context_compressor: kernel (1, 4, 4)
        Compression ratio: (1, 4, 4) / (1, 2, 2) = (1, 2, 2) -> scale by 1*2*2 = 4
        """
        with torch.no_grad():
            weight = patch_embedding.weight.detach().clone()  # (dim, 16, 1, 2, 2)
            bias = patch_embedding.bias.detach().clone()      # (dim,)

            # Expand kernel from (1, 2, 2) to (1, 4, 4) by repeating and scaling
            # Scale factor = 4 (因为空间扩大了4倍: 2*2=4)
            expanded_weight = einops.repeat(
                weight, 'o i t h w -> o i t (h hk) (w wk)', hk=2, wk=2
            ) / 4.0  # 4 = 2 * 2 (spatial expansion)

            context_compressor.weight.copy_(expanded_weight)
            context_compressor.bias.copy_(bias)

    def freeze_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def forward(
        self,
        context_latents: torch.Tensor,       # (B, C, 76, H, W)
        target_latents: torch.Tensor,        # (B, C, 20, H, W)
        context_pose: torch.Tensor,          # (B, 76, 12)
        target_pose: torch.Tensor,           # (B, 20, 12)
        timestep: torch.Tensor,
        context: torch.Tensor,
    ):
        dit = self.pipe.dit

        # Time embedding
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

        # Text embedding
        context = dit.text_embedding(context)

        # Context compression with kernel-aligned padding
        context_latents_padded = pad_for_3d_conv(context_latents, (1, 4, 4))  # Pad for kernel (1, 4, 4)
        ctx = dit.context_compressor(context_latents_padded)  # (B, dim, T_ctx, H/4, W/4)
        f_ctx, h_ctx, w_ctx = ctx.shape[2], ctx.shape[3], ctx.shape[4]
        ctx = rearrange(ctx, 'b c f h w -> b (f h w) c').contiguous()

        # Target patchify
        tgt, (f_tgt, h_tgt, w_tgt) = dit.patchify(target_latents)  # (B, T_tgt*h*w, dim)

        # Cat tokens: [context_tokens, target_tokens]
        x = torch.cat([ctx, tgt], dim=1)

        # Spatial sizes for cam_emb expansion
        ctx_spatial = h_ctx * w_ctx
        tgt_spatial = h_tgt * w_tgt
        cam_emb = (context_pose, target_pose)

        # ========== MemCam memory-context RoPE ==========
        # Target: positions 0-19 (preserve pretrained positions)
        # Context: positions 20-95 (sequential assignment)
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
        ], dim=-1).reshape(f_tgt * h_tgt * w_tgt, 1, -1).to(x.device)  # (S_tgt, 1, dim) complex

        # Concatenate: [context_freqs, target_freqs] -> (S_total, 1, dim)
        freqs = torch.cat([context_freqs, target_freqs], dim=0)

        # DiT blocks
        for block_idx, block in enumerate(dit.blocks):
            if  self.training and self.use_gradient_checkpointing and block_idx in self.ckpt_block_indices:
                def create_custom_forward(module, ctx_sp, tgt_sp):
                    def custom_forward(x, context, cam_emb, t_mod, freqs):
                        return module(x, context, cam_emb, t_mod, freqs, ctx_spatial=ctx_sp, tgt_spatial=tgt_sp)
                    return custom_forward
                if self.use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block, ctx_spatial, tgt_spatial),
                            x, context, cam_emb, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block, ctx_spatial, tgt_spatial),
                        x, context, cam_emb, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, cam_emb, t_mod, freqs, ctx_spatial=ctx_spatial, tgt_spatial=tgt_spatial)

        # Head & unpatchify
        x = dit.head(x, t)

        # 分离context和target tokens
        ctx_tokens = f_ctx * h_ctx * w_ctx
        tgt_tokens = f_tgt * h_tgt * w_tgt
        x_tgt = x[:, ctx_tokens:ctx_tokens + tgt_tokens, :]  # 只取target部分

        # unpatchify target
        x_tgt = dit.unpatchify(x_tgt, (f_tgt, h_tgt, w_tgt))

        return x_tgt

    def training_step(self, batch, batch_idx):
        # Data
        self.pipe.device = self.device
        dtype = self.pipe.torch_dtype
        target_latents = batch["target_latents"].to(dtype=dtype, device=self.device)  # (B, C, 20, H, W)
        context_latents = batch["context_latents"].to(dtype=dtype, device=self.device) # (B, C, 76, H, W)
        target_pose = batch["target_pose"].to(dtype=dtype, device=self.device) # (B, 20, 12)
        context_pose = batch["context_pose"].to(dtype=dtype, device=self.device) # (B, 76, 12)
        prompt_emb = batch["prompt_emb"]["context"].squeeze(1).to(dtype=dtype, device=self.device)

        # Target
        noise = torch.randn_like(target_latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=dtype, device=self.device)
        noisy_target = self.pipe.scheduler.add_noise(target_latents, noise, timestep)
        noisy_target = torch.cat([target_latents[:, :, :ANCHOR_LENGTH, ...], noisy_target[:, :, ANCHOR_LENGTH:, ...]], dim=2)
        training_target = self.pipe.scheduler.training_target(target_latents, noise, timestep)

        # Forward
        noise_pred = self.forward(
            context_latents=context_latents,
            target_latents=noisy_target,
            context_pose=context_pose,
            target_pose=target_pose,
            timestep=timestep,
            context=prompt_emb,
        )

        # Loss
        loss = F.mse_loss(
            noise_pred[:, :, ANCHOR_LENGTH:, ...].float(),
            training_target[:, :, ANCHOR_LENGTH:, ...].float()
        )
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Logging
        self.loss_recorder.add(epoch=self.current_epoch, step=self.global_step, loss=loss.detach().item())
        self.log("train_loss", self.loss_recorder.moving_average, prog_bar=True, on_step=True)
        return loss

    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        ckpt_dir = self.trainer.checkpoint_callback.dirpath if self.trainer.checkpoint_callback else os.path.join(self.trainer.default_root_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.trainer.global_rank == 0:torch.save(self.pipe.denoising_model().state_dict(), os.path.join(ckpt_dir, f"dit_step{self.global_step}.ckpt"))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True, choices=["process", "train"])
    parser.add_argument("--dataset_path", type=str, default="./data")
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--overlap_labels_path", type=str, default="Context-as-Memory-Dataset/overlap_labels")

    parser.add_argument("--text_encoder_path", type=str, default=None)
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--dit_path", type=str, default=None)

    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size_height", type=int, default=34)
    parser.add_argument("--tile_size_width", type=int, default=34)
    parser.add_argument("--tile_stride_height", type=int, default=18)
    parser.add_argument("--tile_stride_width", type=int, default=16)

    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--drop_context_prob", type=float, default=0.1)

    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--save_every_n_steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--training_strategy", type=str, default="deepspeed_stage_1")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true")
    parser.add_argument("--gradient_checkpointing_ratio", type=float, default=1.0)

    parser.add_argument("--use_tensorboard", action="store_true", default=True)
    parser.add_argument("--tensorboard_log_dir", type=str, default=None)
    parser.add_argument("--resume_ckpt_path", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="memcam")

    return parser.parse_args()


def data_process(args):
    print("MemCam: Data Preprocessing Stage...")

    tensors_dir = os.path.join(args.dataset_path, "tensors")
    os.makedirs(tensors_dir, exist_ok=True)

    dataset = TextVideoDataset(args.dataset_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=0
    )

    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        vae_path=args.vae_path,
        output_dir=tensors_dir,
        height=args.height,
        width=args.width,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
        skip_existing=args.skip_existing
    )

    trainer = pl.Trainer(
        accelerator="gpu", devices="auto",
        default_root_dir=args.output_path,
    )

    trainer.test(model, dataloader)
    print("Data preprocessing completed!")


def train(args):
    print("MemCam: Training Stage...")

    dataset = TensorDataset(
        base_path=args.dataset_path,
        overlap_labels_path=args.overlap_labels_path,
        singles_path=os.path.join(args.dataset_path, "singles"),
        jsons_path=os.path.join(args.dataset_path, "jsons"),
        drop_context_prob=args.drop_context_prob,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = LightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        resume_ckpt_path=args.resume_ckpt_path,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        gradient_checkpointing_ratio=args.gradient_checkpointing_ratio,
        height=args.height,
        width=args.width,
    )

    logger = None
    if args.use_tensorboard:
        log_dir = args.tensorboard_log_dir or os.path.join(args.output_path, "tensorboard_logs")
        logger = TensorBoardLogger(save_dir=log_dir, name=args.experiment_name)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(
            save_top_k=-1,
            every_n_train_steps=args.save_every_n_steps
        )],
        logger=logger,
        log_every_n_steps=1,
    )

    ckpt_path = args.resume_from_checkpoint if args.resume_from_checkpoint else None
    if ckpt_path:
        print(f"Resuming from: {ckpt_path}")

    trainer.fit(model, dataloader, ckpt_path=ckpt_path)
    print("Training completed!")


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    if args.task == "process":
        data_process(args)
    elif args.task == "train":
        train(args)

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import torch
import imageio
import argparse
import torchvision
from PIL import Image
import lightning as pl
from torchvision.transforms import v2
from concurrent.futures import ThreadPoolExecutor

from diffsynth import ModelManager
from diffsynth.pipelines.wan_video_memcam import WanVideoMemCamPipeline


class VideosDataset(torch.utils.data.Dataset):
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
        
        return {
            "video_path": video_path,
            "video_name": video_name,
            "total_frames": total_frames,
        }
    
    def __len__(self):
        return len(self.video_files)


class LightningModelForSingleFrame(pl.LightningModule):
    def __init__(self, vae_path, output_dir, height=352, width=640, tiled=False, tile_size=(34, 34), tile_stride=(18, 16), skip_existing=True, batch_size=32):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([vae_path])
        self.pipe = WanVideoMemCamPipeline.from_model_manager(model_manager)
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        self.output_dir = output_dir
        self.height = height
        self.width = width
        self.skip_existing = skip_existing
        self.batch_size = batch_size
        
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
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image
    
    def load_single_frame(self, reader, frame_idx):
        frame = reader.get_data(frame_idx)
        frame = Image.fromarray(frame)
        frame = self.crop_and_resize(frame)
        frame = self.frame_process(frame)  # (C, H, W)
        return frame
    
    def test_step(self, batch, batch_idx):
        video_path = batch["video_path"][0]
        video_name = batch["video_name"][0]
        total_frames = batch["total_frames"].item()
        
        # 创建输出目录
        video_output_dir = os.path.join(self.output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # 收集需要处理的帧
        frames_to_process = []
        if self.skip_existing:
            for i in range(total_frames):
                latent_path = os.path.join(video_output_dir, f"{i:04d}.pt")
                if not os.path.exists(latent_path):
                    frames_to_process.append(i)
            if not frames_to_process:
                print(f"Skipping {video_name} (already processed)")
                return
        else:
            frames_to_process = list(range(total_frames))
        
        self.pipe.device = self.device
        print(f"Processing {video_name}: {len(frames_to_process)}/{total_frames} frames to encode")
        
        # 打开视频reader
        reader = imageio.get_reader(video_path)
        
        # 批量编码 - batch维度，每帧独立编码 (B, C, 1, H, W)
        for batch_start in range(0, len(frames_to_process), self.batch_size):
            batch_indices = frames_to_process[batch_start:batch_start + self.batch_size]
            
            # 批量加载帧
            frames = []
            for frame_idx in batch_indices:
                frame = self.load_single_frame(reader, frame_idx)  # (C, H, W)
                frames.append(frame)
            
            # Stack: (B, C, H, W) -> (B, C, 1, H, W) 每帧独立，T=1
            frames_tensor = torch.stack(frames, dim=0).unsqueeze(2)  # (B, C, 1, H, W)
            frames_tensor = frames_tensor.to(dtype=self.pipe.torch_dtype, device=self.device)
            
            # 批量 VAE 编码 (VAE 内部遍历 batch，每帧独立编码)
            with torch.no_grad():
                latents = self.pipe.encode_video(frames_tensor, **self.tiler_kwargs)  # (B, C, 1, H/8, W/8)
            
            # 异步保存
            latents_cpu = latents.cpu()
            def save_latent(idx, frame_idx, output_dir, latents_ref):
                latent_path = os.path.join(output_dir, f"{frame_idx:04d}.pt")
                torch.save(latents_ref[idx], latent_path)
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                for i, frame_idx in enumerate(batch_indices):
                    executor.submit(save_latent, i, frame_idx, video_output_dir, latents_cpu)
            
            # 进度
            processed = min(batch_start + self.batch_size, len(frames_to_process))
            print(f"  {processed}/{len(frames_to_process)} ({100*processed/len(frames_to_process):.1f}%)")
        
        reader.close()
        print(f"  Done: {video_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Encode video frames to latents individually")
    parser.add_argument("--videos_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="../data/singles")
    parser.add_argument("--vae_path", type=str, default="../models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=128)
    
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size_height", type=int, default=34)
    parser.add_argument("--tile_size_width", type=int, default=34)
    parser.add_argument("--tile_stride_height", type=int, default=18)
    parser.add_argument("--tile_stride_width", type=int, default=16)
    
    parser.add_argument("--skip_existing", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    print("Single Frame Encoding Stage...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    dataset = VideosDataset(args.videos_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=0
    )
    
    model = LightningModelForSingleFrame(
        vae_path=args.vae_path,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
        skip_existing=args.skip_existing,
        batch_size=args.batch_size
    )
    
    trainer = pl.Trainer(
        accelerator="gpu", devices="auto",
        default_root_dir=args.output_dir,
    )
    
    trainer.test(model, dataloader)
    print("Single frame encoding completed!")


if __name__ == "__main__":
    main()
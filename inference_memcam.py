import torch
import torch.nn as nn
from PIL import Image

from dataset.poses import load_c2ws_from_json
from utils.camera_rotation_utils import rotate_c2w_z_swing, rotate_c2w_z_360

from diffsynth import ModelManager, save_video
from diffsynth.pipelines.wan_video_memcam import WanVideoMemCamPipeline


def setup_pipeline(
    dit_path,
    text_encoder_path,
    vae_path,
    dit_ckpt_path,
    device="cuda"
):
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([dit_path, text_encoder_path, vae_path])
    pipe = WanVideoMemCamPipeline.from_model_manager(model_manager, device=device)

    dim=pipe.dit.blocks[0].self_attn.q.weight.shape[0] # 1536
    for block in pipe.dit.blocks:
        block.cam_encoder = nn.Linear(12, dim)
        block.projector = nn.Linear(dim, dim)
        block.cam_encoder.weight.data.zero_()
        block.cam_encoder.bias.data.zero_()
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))
    pipe.dit.context_compressor = nn.Conv3d(16, dim, kernel_size=(1, 4, 4), stride=(1, 4, 4))

    dit_state_dict = torch.load(dit_ckpt_path, map_location="cpu", weights_only=False)
    pipe.dit.load_state_dict(dit_state_dict, strict=True)
    pipe.to(device=device, dtype=torch.bfloat16)
    pipe.device = device; pipe.torch_dtype = torch.bfloat16
    print(f"Loaded DiT checkpoint from {dit_ckpt_path}")
    return pipe


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit_path", type=str, default="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    parser.add_argument("--text_encoder_path", type=str, default="models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--vae_path", type=str, default="models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    parser.add_argument("--dit_ckpt_path", type=str, default="models/MemCam/dit_step20000.ckpt")

    parser.add_argument("--input_image", type=str, default="assets/test.png")
    parser.add_argument("--pose_path", type=str, default="assets/test.json")
    parser.add_argument("--prompt", type=str, default="The video begins with a scene where sunlight filters through an unseen source, creating a hazy atmosphere over a rocky landscape. As the video progresses, the haze gradually clears to reveal more of the environment, including greenery and rocks that suggest a natural setting, possibly near a water body given the presence of reflections on the surface. The light continues to play a significant role in altering the visibility and mood of the scene. As time passes, the clarity improves significantly, allowing for a detailed view of the lush vegetation and various rock formations within what appears to be a serene outdoor area. The camera's subtle movements offer different perspectives of this tranquil setting, emphasizing the textures and colors of the environment under changing lighting conditions. Towards the latter part of the video, the focus shifts slightly to include architectural elements like columns or structures, hinting at human influence or historical significance in the otherwise untouched natural surroundings. This new addition suggests a blend of nature and civilization, enhancing the narrative depth of the location being showcased. Throughout the video, there is no visible movement of objects or characters, indicating a static observation of the environment. The consistent quality of light and the gradual unveiling of details create a sense of progression and discovery, culminating in a richer understanding of the setting without any discernible action or dynamic change occurring.")

    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_frames", type=int, default=8*76+1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    pipe = setup_pipeline(
        dit_path=args.dit_path,
        text_encoder_path=args.text_encoder_path,
        vae_path=args.vae_path,
        dit_ckpt_path=args.dit_ckpt_path,
        device=args.device
    )

    input_image = Image.open(args.input_image).convert("RGB").resize((args.width, args.height), resample=Image.BICUBIC)
    c2ws = load_c2ws_from_json(json_path=str(args.pose_path), num_frames=args.num_frames)

    video = pipe(
        prompt=args.prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，畸形的，静止不动的画面，杂乱的背景",
        input_image=input_image,
        c2ws=rotate_c2w_z_swing(c2ws[0], num_frames=args.num_frames, max_angle_deg=360.0),
        height=args.height,
        width=args.width,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        tiled=False
    )
    save_video(video, "swing.mp4", fps=30, quality=5)

    video = pipe(
        prompt=args.prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，畸形的，静止不动的画面，杂乱的背景",
        input_image=input_image,
        c2ws=rotate_c2w_z_360(c2ws[0], num_frames=args.num_frames),
        height=args.height,
        width=args.width,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        tiled=False
    )
    save_video(video, "360.mp4", fps=30, quality=5)

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import glob
import argparse
import logging
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

FRAMES_PER_CLIP = 77
STEP_SIZE = FRAMES_PER_CLIP - 1  # 1帧重叠

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


"""
输出结构：
  data/
    videos/
      {video_name}.mp4
    texts/
      {video_name}_clip{idx:04d}.txt
"""


def save_frames_as_video(
    frames: np.ndarray,
    output_path: str,
    fps: int = 30,
    codec: str = 'mp4v'
):
    """
    将帧序列保存为视频文件

    Args:
        frames: (T, H, W, C) numpy array
        output_path: 输出视频路径
        fps: 帧率
        codec: 视频编码器
    """
    T, H, W, C = frames.shape

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # 写入帧
    for i in range(T):
        frame = frames[i]  # H, W, C (RGB)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    logger.debug(f"Saved video: {output_path} ({T} frames, {H}x{W})")


def load_frames(
    frames_dir: Path,
    start_frame: int,
    end_frame: int,
    target_height: int,
    target_width: int
) -> np.ndarray:
    """
    加载帧

    Args:
        frames_dir: 帧目录
        start_frame: 起始帧索引
        end_frame: 结束帧索引
        target_height: 目标高度
        target_width: 目标宽度

    Returns:
        frames: (T, H, W, C) numpy array
    """
    frames = []

    for i in range(start_frame, end_frame):
        img_path = frames_dir / f"{i:04d}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Frame not found: {img_path}")

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        frames.append(img)

    return np.stack(frames, axis=0)  # (T, H, W, C)


def load_captions(captions_file: str) -> dict:
    """
    加载 captions.txt 文件

    格式: {video_name}/{start_frame}_{end_frame}.mp4\t{caption}
    例如: AncientTowns_1/0000_0076.mp4\tThe video showcases...

    Returns:
        dict: {"{video_name}/{start:04d}_{end:04d}.mp4": caption}
    """
    captions = {}
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            key, caption = parts
            captions[key] = caption
    logger.info(f"Loaded {len(captions)} captions from {captions_file}")
    return captions


def get_caption_key(video_name: str, start_frame: int, end_frame: int) -> str:
    """
    构造 caption 的查找 key

    Args:
        video_name: 视频名称
        start_frame: 起始帧索引
        end_frame: 结束帧索引 (exclusive, 代码中使用)

    Returns:
        key: "{video_name}/{start:04d}_{end:04d}.mp4"
    """
    # 代码中 end 是 exclusive，captions 中 end 是 inclusive
    caption_end = end_frame - 1
    return f"{video_name}/{start_frame:04d}_{caption_end:04d}.mp4"


def process_single_video(
    video_name: str,
    frames_dir: Path,
    output_videos_dir: Path,
    output_texts_dir: Path,
    target_height: int,
    target_width: int,
    fps: int,
    skip_existing: bool,
    captions: dict
) -> int:
    """
    处理单个视频:
    - 将所有帧保存为一个完整视频
    - 按clip方式保存text

    Returns:
        生成的数量
    """
    # 获取总帧数
    frame_files = sorted(glob.glob(str(frames_dir / "*.png")))
    total_frames = len(frame_files)

    logger.info(f"Processing {video_name}: {total_frames} frames")

    # 完整视频输出路径
    video_output_path = output_videos_dir / f"{video_name}.mp4"

    # 保存完整视频
    if skip_existing and video_output_path.exists():
        logger.info(f"  Skipping existing video: {video_output_path.name}")
    else:
        try:
            # 加载所有帧
            frames = load_frames(
                frames_dir=frames_dir,
                start_frame=0,
                end_frame=total_frames,
                target_height=target_height,
                target_width=target_width
            )

            # 保存完整视频
            save_frames_as_video(
                frames=frames,
                output_path=str(video_output_path),
                fps=fps
            )

            logger.info(f"  ✓ Saved full video: {video_output_path.name} ({total_frames} frames)")

        except Exception as e:
            logger.error(f"  ✗ Failed to save video: {e}")
            return 0

    # 保存text
    clip_index = 0
    start = 0
    texts_generated = 0

    while start < total_frames:
        # 计算结束帧
        end = min(start + FRAMES_PER_CLIP, total_frames)
        frame_count = end - start

        # 跳过帧数不足的clip
        if frame_count < FRAMES_PER_CLIP:
            logger.debug(f"Skipping incomplete clip at start={start}, only {frame_count} frames")
            break

        # text输出路径
        text_output_path = output_texts_dir / f"{video_name}_clip{clip_index:04d}.txt"

        # 获取caption key
        caption_key = get_caption_key(video_name, start, end)
        caption = captions.get(caption_key, "")

        if not caption:
            logger.warning(f"  ! No caption found for {caption_key}, skipping text {clip_index}")
            clip_index += 1
            start += STEP_SIZE  # 有1帧重叠
            continue

        # 检查是否已存在
        if skip_existing and text_output_path.exists():
            logger.debug(f"Skipping existing text: {video_name}_clip{clip_index:04d}.txt")
            clip_index += 1
            start += STEP_SIZE  # 有1帧重叠
            texts_generated += 1
            continue

        try:
            # 保存caption
            with open(text_output_path, 'w', encoding='utf-8') as f:
                f.write(caption)

            logger.debug(f"  ✓ Saved text {clip_index}: frames {start}-{end-1}")
            texts_generated += 1

        except Exception as e:
            logger.error(f"  ✗ Failed to save text {clip_index}: {e}")

        clip_index += 1
        start += STEP_SIZE  # 有1帧重叠

    logger.info(f"  ✓ Generated {texts_generated} text files for {video_name}")
    return texts_generated


def process_video_worker(args_tuple):
    """
    多进程worker函数

    Args:
        args_tuple: (video_name, frames_dir, output_videos_dir, output_texts_dir, target_height, target_width, fps, skip_existing, captions)

    Returns:
        (video_name, texts_generated)
    """
    (video_name, frames_dir, output_videos_dir, output_texts_dir, target_height, target_width, fps, skip_existing, captions) = args_tuple

    try:
        texts_generated = process_single_video(
            video_name=video_name,
            frames_dir=frames_dir,
            output_videos_dir=output_videos_dir,
            output_texts_dir=output_texts_dir,
            target_height=target_height,
            target_width=target_width,
            fps=fps,
            skip_existing=skip_existing,
            captions=captions
        )
        return (video_name, texts_generated)
    except Exception as e:
        logger.error(f"Error processing {video_name}: {e}")
        return (video_name, 0)


def main():
    parser = argparse.ArgumentParser(description="Build videos")
    parser.add_argument(
        "--frames_dir",
        type=str,
        default="Context-as-Memory-Dataset/frames",
        help="Frames directory"
    )
    parser.add_argument(
        "--captions_file",
        type=str,
        default="Context-as-Memory-Dataset/captions.txt",
        help="Captions file path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data",
        help="Output directory for videos, poses, and texts"
    )

    # 视频参数
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--height", type=int, default=352, help="Video height")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")

    # 其他参数
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Skip existing clips")
    parser.add_argument("--video_filter", type=str, default=None, help="Only process videos matching this pattern")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of parallel workers (0 for serial)")

    args = parser.parse_args()

    # 设置路径
    frames_dir = Path(args.frames_dir)
    captions_file = Path(args.captions_file)
    output_dir = Path(args.output_dir)
    output_videos_dir = output_dir / "videos"
    output_texts_dir = output_dir / "texts"

    # 创建输出目录
    output_videos_dir.mkdir(parents=True, exist_ok=True)
    output_texts_dir.mkdir(parents=True, exist_ok=True)

    # 验证输入目录和文件
    if not frames_dir.exists():
        logger.error(f"Frames directory not found: {frames_dir}")
        return
    if not captions_file.exists():
        logger.error(f"Captions file not found: {captions_file}")
        return

    # 加载 captions
    captions = load_captions(str(captions_file))

    logger.info(f"Frames directory: {frames_dir}")
    logger.info(f"Captions file: {captions_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Video resolution: {args.width}x{args.height}")
    logger.info(f"FPS: {args.fps}")
    logger.info(f"Frames per clip: {FRAMES_PER_CLIP}")
    logger.info(f"Step size: {STEP_SIZE} (1 frame overlap)")
    logger.info(f"Skip existing: {args.skip_existing}")
    logger.info(f"Parallel workers: {args.num_workers} (0=serial, max={cpu_count()})")

    # 获取所有视频目录
    video_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])

    # 过滤视频
    if args.video_filter:
        video_dirs = [d for d in video_dirs if args.video_filter in d.name]
        logger.info(f"Filtered to {len(video_dirs)} videos matching '{args.video_filter}'")

    logger.info(f"Found {len(video_dirs)} videos to process")

    # 准备任务列表
    tasks = []
    for video_dir in video_dirs:
        video_name = video_dir.name
        tasks.append((
            video_name,
            video_dir,
            output_videos_dir,
            output_texts_dir,
            args.height,
            args.width,
            args.fps,
            args.skip_existing,
            captions
        ))

    logger.info(f"Prepared {len(tasks)} tasks")

    # 处理视频 - 串行或并行
    total_texts = 0
    successful_videos = 0

    if args.num_workers > 0:
        # 并行处理
        num_workers = min(args.num_workers, cpu_count(), len(tasks))
        logger.info(f"Using {num_workers} parallel workers")

        with Pool(num_workers) as pool:
            results = pool.map(process_video_worker, tasks)

        # 统计结果
        for video_name, texts_generated in results:
            if texts_generated > 0:
                total_texts += texts_generated
                successful_videos += 1
    else:
        # 串行处理
        logger.info("Using serial processing (num_workers=0)")
        for task in tasks:
            video_name, texts_generated = process_video_worker(task)
            if texts_generated > 0:
                total_texts += texts_generated
                successful_videos += 1

    # 统计信息
    logger.info("Summary")
    logger.info(f"Videos processed: {successful_videos} / {len(video_dirs)}")
    logger.info(f"Total text files generated: {total_texts}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"  Videos: {output_videos_dir}")
    logger.info(f"  Texts: {output_texts_dir}")

    # 验证输出
    video_files = list(output_videos_dir.glob("*.mp4"))
    text_files = list(output_texts_dir.glob("*.txt"))
    logger.info(f"Verification:")
    logger.info(f"  Video files: {len(video_files)}")
    logger.info(f"  Text files: {len(text_files)}")


if __name__ == "__main__":
    main()

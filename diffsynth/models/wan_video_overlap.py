"""
FOV Overlap Calculator (适配 UE 坐标系)
======================================

相机视角重叠度计算工具，适配 Z-up 相机坐标系。

使用基于视场(FOV)采样的方法来计算两个相机位姿之间的视角重叠度：
1. 在球体空间内随机采样大量点
2. 检测每个点是否在两个相机的视场(FOV)内
3. 通过共同可见点的比例来估算视角重叠度

UE 坐标系约定 (Z-up):
    - X: Forward (前)
    - Y: Right (右)  
    - Z: Up (上)
    - Yaw: 绕Z轴旋转(左右看)
    - Pitch: 绕Y轴旋转(上下看)
    - Roll: 绕X轴旋转(翻滚)

使用方法:
    # 从JSON数据集计算重叠度
    python wan_video_overlap.py --json path/to/scene.json --frame 0
    
    # 计算两个位姿的重叠度
    python wan_video_overlap.py --pose1 "0,0,0,0,0,0" --pose2 "100,0,0,0,0,30"
"""

import math
import json
import torch
import numpy as np
import argparse
from typing import Tuple, Union, Optional, List, Dict
from pathlib import Path


# ============== UE 坐标系相机位姿转换函数 ==============

def degrees_to_radians(degrees):
    """度数转弧度"""
    return degrees * np.pi / 180


def compute_c2w_matrix(params: dict, scale: float = 100.0) -> np.ndarray:
    """
    从单帧相机参数计算 4x4 c2w 矩阵
    
    Args:
        params: dict包含:
            - 'position': [x, y, z] - 相机位置(UE单位,厘米)
            - 'rotation': [pitch, roll, yaw] (度数,UE导出格式)
        scale: 位置缩放因子(UE使用厘米,除以scale转换)
    
    Returns:
        c2w: 4x4 numpy array, camera-to-world 变换矩阵
    
    坐标系约定 (UE Z-up):
        - X: Forward (前)
        - Y: Right (右)  
        - Z: Up (上)
        - Yaw: 绕Z轴旋转(左右看)
        - Pitch: 绕Y轴旋转(上下看)
        - Roll: 绕X轴旋转(翻滚)
    """
    position = params['position']
    rotation = params['rotation']
    
    # 解析角度(度数)
    # UE导出的rotation格式: [pitch, roll, yaw]
    pitch_deg = rotation[0]  # 俯仰角
    roll_deg = rotation[1]   # 横滚角
    yaw_deg = rotation[2]    # 偏航角
    
    # 转换为弧度
    pitch = degrees_to_radians(pitch_deg)
    roll = degrees_to_radians(roll_deg)
    yaw = degrees_to_radians(yaw_deg)
    
    # 构建各轴旋转矩阵
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    
    # Roll: 绕X轴旋转
    Rx = np.array([
        [1, 0, 0],
        [0, cos_r, -sin_r],
        [0, sin_r, cos_r]
    ], dtype=np.float64)
    
    # Pitch: 绕Y轴旋转
    Ry = np.array([
        [cos_p, 0, sin_p],
        [0, 1, 0],
        [-sin_p, 0, cos_p]
    ], dtype=np.float64)
    
    # Yaw: 绕Z轴旋转
    Rz = np.array([
        [cos_y, -sin_y, 0],
        [sin_y, cos_y, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # 组合旋转: R = Rz @ Ry @ Rx (ZYX旋转顺序,UE默认)
    R = Rz @ Ry @ Rx
    
    # 位置 (缩放处理)
    T = np.array(position, dtype=np.float64) / scale
    
    # 构建 4x4 c2w 矩阵
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R
    c2w[:3, 3] = T
    
    return c2w


def load_poses_from_json(json_path: str, scale: float = 100.0) -> Tuple[np.ndarray, List[dict]]:
    """
    从JSON文件加载所有相机位姿
    
    Args:
        json_path: JSON文件路径
        scale: 位置缩放因子
    
    Returns:
        c2ws: (N, 4, 4) numpy array - 所有帧的c2w矩阵
        raw_params: list of dict - 原始参数 (position, rotation)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    camera_data = data['CineCameraActor']
    frame_keys = sorted(camera_data.keys(), key=int)
    
    c2ws = []
    raw_params = []
    
    for frame_key in frame_keys:
        frame_data = camera_data[frame_key]
        c2w = compute_c2w_matrix(frame_data, scale=scale)
        c2ws.append(c2w)
        raw_params.append({
            'position': frame_data['position'],
            'rotation': frame_data['rotation'],
            'frame_idx': int(frame_key)
        })
    
    return np.stack(c2ws, axis=0), raw_params


# ============== UE 坐标系视场检测函数 ==============

def is_inside_fov_3d_ue(
    points: torch.Tensor,
    center: torch.Tensor,
    center_pitch: Union[torch.Tensor, float],
    center_yaw: Union[torch.Tensor, float],
    fov_half_h: Union[torch.Tensor, float],
    fov_half_v: Union[torch.Tensor, float]
) -> torch.Tensor:
    """
    检测点是否在给定的3D视场(FOV)内。(UE坐标系)
    
    UE 坐标系约定:
        - X: Forward (前向)
        - Y: Right (右)  
        - Z: Up (上)
    
    Args:
        points (torch.Tensor): 采样点坐标，形状 (N, 3)
        center (torch.Tensor): FOV 中心坐标，形状 (3,)
        center_pitch (float or Tensor): 中心视线的俯仰角（度）
        center_yaw (float or Tensor): 中心视线的偏航角（度）
        fov_half_h (float or Tensor): 水平半视场角（度）
        fov_half_v (float or Tensor): 垂直半视场角（度）

    Returns:
        torch.Tensor: 布尔张量，指示每个点是否在 FOV 内
    """
    # 计算相对于中心的向量
    vectors = points - center  # (N, 3)
    x = vectors[..., 0]  # Forward
    y = vectors[..., 1]  # Right
    z = vectors[..., 2]  # Up
    
    # 计算水平角（yaw）：以 X 轴为前向，Y 轴为左右
    # 在UE中，yaw是绕Z轴旋转，正yaw表示向左转
    azimuth = torch.atan2(y, x) * (180 / math.pi)
    
    # 计算垂直角（pitch）：相对于水平面(XY平面)
    # 在UE中，pitch是绕Y轴旋转，正pitch表示向上看
    elevation = torch.atan2(z, torch.sqrt(x**2 + y**2)) * (180 / math.pi)
    
    # 计算与中心视线的角度差（处理循环角度）
    diff_azimuth = (azimuth - center_yaw).abs() % 360
    diff_elevation = (elevation - center_pitch).abs() % 360
    
    # 调整超过180度的角度差
    diff_azimuth = torch.where(diff_azimuth > 180, 360 - diff_azimuth, diff_azimuth)
    diff_elevation = torch.where(diff_elevation > 180, 360 - diff_elevation, diff_elevation)
    
    # 检查是否在水平和垂直 FOV 范围内
    return (diff_azimuth < fov_half_h) & (diff_elevation < fov_half_v)


def generate_points_in_sphere(n_points: int, radius: float) -> torch.Tensor:
    """
    在球体内均匀采样点。

    Args:
        n_points (int): 采样点数量
        radius (float): 球体半径

    Returns:
        torch.Tensor: 采样点坐标，形状 (n_points, 3)
    """
    samples_r = torch.rand(n_points)
    samples_phi = torch.rand(n_points)
    samples_u = torch.rand(n_points)

    # 使用立方根确保体积均匀分布
    r = radius * torch.pow(samples_r, 1/3)
    phi = 2 * math.pi * samples_phi
    theta = torch.acos(1 - 2 * samples_u)

    # 球坐标转直角坐标
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    return torch.stack((x, y, z), dim=1)


# ============== 核心重叠度计算函数 ==============

def calculate_fov_overlap_ue(
    pos1: np.ndarray,
    rot1: List[float],
    pos2: np.ndarray,
    rot2: List[float],
    fov_half_h: float = 45.0,  # 默认90度FOV的一半
    fov_half_v: float = 30.0,  # 垂直FOV
    num_samples: int = 10000,
    radius: float = 50.0,  # 采样球体半径（米为单位，已缩放后的）
    return_details: bool = False
) -> Union[float, Tuple[float, dict]]:
    """
    计算两个相机位姿之间的视角重叠度 (UE坐标系)。

    Args:
        pos1: 第一个相机位置 [x, y, z]（已缩放，米为单位）
        rot1: 第一个相机旋转 [pitch, roll, yaw]（度数）
        pos2: 第二个相机位置 [x, y, z]
        rot2: 第二个相机旋转 [pitch, roll, yaw]（度数）
        fov_half_h (float): 水平半视场角（度）
        fov_half_v (float): 垂直半视场角（度）
        num_samples (int): 采样点数量
        radius (float): 采样球体半径
        return_details (bool): 是否返回详细信息

    Returns:
        float: 视角重叠度 (0.0 - 1.0)
        如果 return_details=True，还返回一个包含详细信息的字典
    """
    device = torch.device('cpu')
    
    # 转换为tensor
    center1 = torch.tensor(pos1, dtype=torch.float32, device=device)
    center2 = torch.tensor(pos2, dtype=torch.float32, device=device)
    
    # UE rotation格式: [pitch, roll, yaw]
    pitch1, roll1, yaw1 = rot1[0], rot1[1], rot1[2]
    pitch2, roll2, yaw2 = rot2[0], rot2[1], rot2[2]
    
    # 以两个相机中点生成采样点
    mid_point = (center1 + center2) / 2
    points = generate_points_in_sphere(num_samples, radius).to(device)
    points = points + mid_point
    
    fov_half_h_t = torch.tensor(fov_half_h, device=device)
    fov_half_v_t = torch.tensor(fov_half_v, device=device)
    
    # 检测点是否在相机1的视场内
    in_fov1 = is_inside_fov_3d_ue(
        points, center1,
        pitch1, yaw1,
        fov_half_h_t, fov_half_v_t
    )
    
    # 检测点是否在相机2的视场内
    in_fov2 = is_inside_fov_3d_ue(
        points, center2,
        pitch2, yaw2,
        fov_half_h_t, fov_half_v_t
    )
    
    # 计算重叠度
    intersection = (in_fov1 & in_fov2).sum().item()
    union = (in_fov1 | in_fov2).sum().item()
    fov1_count = in_fov1.sum().item()
    fov2_count = in_fov2.sum().item()
    
    # 重叠度计算方式
    overlap_ratio_1 = intersection / fov1_count if fov1_count > 0 else 0.0
    overlap_ratio_2 = intersection / fov2_count if fov2_count > 0 else 0.0
    iou = intersection / union if union > 0 else 0.0
    
    if return_details:
        details = {
            "intersection_count": intersection,
            "union_count": union,
            "pose1_visible_count": fov1_count,
            "pose2_visible_count": fov2_count,
            "overlap_ratio_pose1": overlap_ratio_1,
            "overlap_ratio_pose2": overlap_ratio_2,
            "iou": iou,
            "num_samples": num_samples,
            "radius": radius,
            "fov_half_h": fov_half_h,
            "fov_half_v": fov_half_v,
        }
        return iou, details
    
    return iou


def calculate_overlap_from_c2w(
    c2w1: np.ndarray,
    c2w2: np.ndarray,
    fov_half_h: float = 45.0,
    fov_half_v: float = 30.0,
    num_samples: int = 10000,
    radius: float = 50.0,
    return_details: bool = False
) -> Union[float, Tuple[float, dict]]:
    """
    从 c2w 矩阵计算两个相机位姿之间的视角重叠度。

    Args:
        c2w1: 第一个相机的 c2w 矩阵，形状 (4, 4)
        c2w2: 第二个相机的 c2w 矩阵，形状 (4, 4)
        其他参数同 calculate_fov_overlap_ue

    Returns:
        同 calculate_fov_overlap_ue
    """
    # 从 c2w 矩阵提取位置
    pos1 = c2w1[:3, 3]
    pos2 = c2w2[:3, 3]
    
    # 从旋转矩阵提取 pitch, roll, yaw
    def extract_euler_from_rotation_matrix(R):
        """
        从旋转矩阵提取欧拉角 (ZYX顺序, UE格式)
        返回: [pitch, roll, yaw] 度数
        """
        # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        # 提取各角度
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return [np.degrees(pitch), np.degrees(roll), np.degrees(yaw)]
    
    rot1 = extract_euler_from_rotation_matrix(c2w1[:3, :3])
    rot2 = extract_euler_from_rotation_matrix(c2w2[:3, :3])
    
    return calculate_fov_overlap_ue(
        pos1, rot1, pos2, rot2,
        fov_half_h=fov_half_h,
        fov_half_v=fov_half_v,
        num_samples=num_samples,
        radius=radius,
        return_details=return_details
    )


# ============== 数据集重叠度计算函数 ==============

def calculate_overlap_for_frame(
    json_path: str,
    target_frame: int,
    fov_half_h: float = 45.0,
    fov_half_v: float = 30.0,
    num_samples: int = 10000,
    radius: float = 50.0,
    scale: float = 100.0,
    return_sorted: bool = True
) -> Dict:
    """
    计算数据集中指定帧与所有其他帧的重叠度。
    
    Args:
        json_path: JSON文件路径
        target_frame: 目标帧索引
        fov_half_h: 水平半视场角（度）
        fov_half_v: 垂直半视场角（度）
        num_samples: 采样点数量
        radius: 采样球体半径（米为单位）
        scale: 位置缩放因子（UE厘米转米）
        return_sorted: 是否按重叠度排序返回
    
    Returns:
        dict: 包含重叠度结果的字典
            - target_frame: 目标帧索引
            - total_frames: 总帧数
            - overlaps: list of dict, 每帧的重叠度信息
    """
    # 加载所有位姿
    c2ws, raw_params = load_poses_from_json(json_path, scale=scale)
    total_frames = len(c2ws)
    
    if target_frame < 0 or target_frame >= total_frames:
        raise ValueError(f"target_frame {target_frame} 超出范围 [0, {total_frames-1}]")
    
    target_c2w = c2ws[target_frame]
    target_param = raw_params[target_frame]
    
    overlaps = []
    
    for i in range(total_frames):
        if i == target_frame:
            # 自身重叠度为1.0
            overlaps.append({
                'frame_idx': i,
                'iou': 1.0,
                'overlap_ratio_target': 1.0,
                'overlap_ratio_other': 1.0,
                'is_target': True
            })
            continue
        
        other_c2w = c2ws[i]
        iou, details = calculate_overlap_from_c2w(
            target_c2w, other_c2w,
            fov_half_h=fov_half_h,
            fov_half_v=fov_half_v,
            num_samples=num_samples,
            radius=radius,
            return_details=True
        )
        
        overlaps.append({
            'frame_idx': i,
            'iou': iou,
            'overlap_ratio_target': details['overlap_ratio_pose1'],
            'overlap_ratio_other': details['overlap_ratio_pose2'],
            'is_target': False
        })
    
    if return_sorted:
        overlaps = sorted(overlaps, key=lambda x: x['iou'], reverse=True)
    
    return {
        'json_path': json_path,
        'target_frame': target_frame,
        'target_position': target_param['position'],
        'target_rotation': target_param['rotation'],
        'total_frames': total_frames,
        'fov_half_h': fov_half_h,
        'fov_half_v': fov_half_v,
        'overlaps': overlaps
    }


def batch_calculate_overlap(
    json_path: str,
    target_frame: int,
    candidate_frames: List[int],
    fov_half_h: float = 45.0,
    fov_half_v: float = 30.0,
    num_samples: int = 10000,
    radius: float = 50.0,
    scale: float = 100.0
) -> List[Tuple[int, float]]:
    """
    批量计算目标帧与候选帧列表之间的重叠度。
    
    Args:
        json_path: JSON文件路径
        target_frame: 目标帧索引
        candidate_frames: 候选帧索引列表
        其他参数同 calculate_overlap_for_frame
    
    Returns:
        list of (frame_idx, iou): 候选帧索引和对应的IoU
    """
    c2ws, _ = load_poses_from_json(json_path, scale=scale)
    target_c2w = c2ws[target_frame]
    
    results = []
    for frame_idx in candidate_frames:
        if frame_idx == target_frame:
            results.append((frame_idx, 1.0))
            continue
        
        iou = calculate_overlap_from_c2w(
            target_c2w, c2ws[frame_idx],
            fov_half_h=fov_half_h,
            fov_half_v=fov_half_v,
            num_samples=num_samples,
            radius=radius,
            return_details=False
        )
        results.append((frame_idx, iou))
    
    return results


def find_overlapping_frames(
    json_path: str,
    target_frame: int,
    threshold: float = 0.3,
    top_k: Optional[int] = None,
    fov_half_h: float = 45.0,
    fov_half_v: float = 30.0,
    num_samples: int = 10000,
    radius: float = 50.0,
    scale: float = 100.0
) -> List[Tuple[int, float]]:
    """
    找出与目标帧重叠度超过阈值的所有帧。
    
    Args:
        json_path: JSON文件路径
        target_frame: 目标帧索引
        threshold: 重叠度阈值 (0.0 - 1.0)
        top_k: 只返回前k个最高重叠度的帧，None表示返回所有超过阈值的
        其他参数同 calculate_overlap_for_frame
    
    Returns:
        list of (frame_idx, iou): 按重叠度降序排列
    """
    result = calculate_overlap_for_frame(
        json_path, target_frame,
        fov_half_h=fov_half_h,
        fov_half_v=fov_half_v,
        num_samples=num_samples,
        radius=radius,
        scale=scale,
        return_sorted=True
    )
    
    overlapping = [
        (item['frame_idx'], item['iou'])
        for item in result['overlaps']
        if item['iou'] >= threshold and not item['is_target']
    ]
    
    if top_k is not None:
        overlapping = overlapping[:top_k]
    
    return overlapping


# ============== 可视化函数 ==============

def visualize_overlap_distribution(
    json_path: str,
    target_frame: int,
    save_path: Optional[str] = None,
    fov_half_h: float = 45.0,
    fov_half_v: float = 30.0,
    num_samples: int = 10000,
    radius: float = 50.0,
    scale: float = 100.0
):
    """
    可视化目标帧与所有帧的重叠度分布。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要安装 matplotlib: pip install matplotlib")
        return
    
    result = calculate_overlap_for_frame(
        json_path, target_frame,
        fov_half_h=fov_half_h,
        fov_half_v=fov_half_v,
        num_samples=num_samples,
        radius=radius,
        scale=scale,
        return_sorted=False
    )
    
    frame_indices = [item['frame_idx'] for item in result['overlaps']]
    ious = [item['iou'] for item in result['overlaps']]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # 折线图：重叠度随帧变化
    ax1.plot(frame_indices, ious, 'b-', linewidth=0.5, alpha=0.7)
    ax1.scatter(frame_indices, ious, c=ious, cmap='RdYlGn', s=10)
    ax1.axvline(x=target_frame, color='r', linestyle='--', label=f'Target Frame {target_frame}')
    ax1.axhline(y=0.3, color='orange', linestyle=':', alpha=0.7, label='Threshold 0.3')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('IoU')
    ax1.set_title(f'FOV Overlap with Frame {target_frame}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 直方图：重叠度分布
    ax2.hist(ious, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0.3, color='orange', linestyle='--', label='Threshold 0.3')
    ax2.set_xlabel('IoU')
    ax2.set_ylabel('Count')
    ax2.set_title('IoU Distribution')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============== 命令行接口 ==============

def main():
    parser = argparse.ArgumentParser(
        description='FOV 重叠度计算器 (UE坐标系)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从数据集JSON计算指定帧与所有帧的重叠度
  python wan_video_overlap.py --json /path/to/scene.json --frame 0
  
  # 计算并可视化重叠度分布
  python wan_video_overlap.py --json /path/to/scene.json --frame 0 --visualize
  
  # 找出重叠度超过阈值的帧
  python wan_video_overlap.py --json /path/to/scene.json --frame 0 --threshold 0.3
  
  # 输出前10个最高重叠度的帧
  python wan_video_overlap.py --json /path/to/scene.json --frame 0 --top-k 10
  
  # 计算两个位姿的重叠度 (position: x,y,z  rotation: pitch,roll,yaw)
  python wan_video_overlap.py --pos1 "0,0,0" --rot1 "0,0,0" --pos2 "100,0,0" --rot2 "0,0,30"
        """
    )
    
    # 数据集模式参数
    parser.add_argument('--json', type=str, help='JSON数据集文件路径')
    parser.add_argument('--frame', type=int, help='目标帧索引')
    parser.add_argument('--threshold', type=float, default=None,
                        help='重叠度阈值，只输出超过此值的帧')
    parser.add_argument('--top-k', type=int, default=None,
                        help='只输出前k个最高重叠度的帧')
    
    # 直接位姿模式参数
    parser.add_argument('--pos1', type=str, help='第一个相机位置: x,y,z')
    parser.add_argument('--rot1', type=str, help='第一个相机旋转: pitch,roll,yaw')
    parser.add_argument('--pos2', type=str, help='第二个相机位置: x,y,z')
    parser.add_argument('--rot2', type=str, help='第二个相机旋转: pitch,roll,yaw')
    
    # 通用参数
    parser.add_argument('--fov-h', type=float, default=45.0,
                        help='水平半视场角（度），默认 45')
    parser.add_argument('--fov-v', type=float, default=30.0,
                        help='垂直半视场角（度），默认 30')
    parser.add_argument('--samples', type=int, default=10000,
                        help='采样点数量，默认 10000')
    parser.add_argument('--radius', type=float, default=50.0,
                        help='采样球体半径，默认 50.0')
    parser.add_argument('--scale', type=float, default=100.0,
                        help='位置缩放因子（UE厘米转米），默认 100.0')
    parser.add_argument('--visualize', action='store_true',
                        help='可视化重叠度分布')
    parser.add_argument('--save-vis', type=str, default=None,
                        help='保存可视化图像的路径')
    parser.add_argument('--output-json', type=str, default=None,
                        help='输出结果到JSON文件')

    args = parser.parse_args()

    # 模式判断
    if args.json and args.frame is not None:
        # 数据集模式
        print("=" * 70)
        print("FOV 重叠度计算器 (UE坐标系) - 数据集模式")
        print("=" * 70)
        print(f"\n数据集: {args.json}")
        print(f"目标帧: {args.frame}")
        print(f"FOV: 水平 ±{args.fov_h}°, 垂直 ±{args.fov_v}°")
        print(f"采样: {args.samples} 点, 半径 {args.radius}")
        
        if args.threshold is not None or args.top_k is not None:
            # 找出重叠帧
            overlapping = find_overlapping_frames(
                args.json, args.frame,
                threshold=args.threshold or 0.0,
                top_k=args.top_k,
                fov_half_h=args.fov_h,
                fov_half_v=args.fov_v,
                num_samples=args.samples,
                radius=args.radius,
                scale=args.scale
            )
            
            print(f"\n找到 {len(overlapping)} 个重叠帧:")
            print("-" * 40)
            for frame_idx, iou in overlapping:
                print(f"  帧 {frame_idx:5d}: IoU = {iou:.4f} ({iou*100:.2f}%)")
        else:
            # 完整计算
            result = calculate_overlap_for_frame(
                args.json, args.frame,
                fov_half_h=args.fov_h,
                fov_half_v=args.fov_v,
                num_samples=args.samples,
                radius=args.radius,
                scale=args.scale
            )
            
            print(f"\n总帧数: {result['total_frames']}")
            print(f"目标帧位置: {result['target_position']}")
            print(f"目标帧旋转: {result['target_rotation']}")
            
            # 统计
            ious = [item['iou'] for item in result['overlaps'] if not item['is_target']]
            print(f"\n重叠度统计:")
            print(f"  最大: {max(ious):.4f}")
            print(f"  最小: {min(ious):.4f}")
            print(f"  平均: {np.mean(ious):.4f}")
            print(f"  > 0.5 的帧数: {sum(1 for x in ious if x > 0.5)}")
            print(f"  > 0.3 的帧数: {sum(1 for x in ious if x > 0.3)}")
            print(f"  > 0.1 的帧数: {sum(1 for x in ious if x > 0.1)}")
            
            # 显示前10个
            print(f"\n前10个最高重叠度的帧:")
            print("-" * 40)
            for item in result['overlaps'][:10]:
                if item['is_target']:
                    continue
                print(f"  帧 {item['frame_idx']:5d}: IoU = {item['iou']:.4f} ({item['iou']*100:.2f}%)")
            
            # 输出JSON
            if args.output_json:
                with open(args.output_json, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n结果已保存到: {args.output_json}")
        
        # 可视化
        if args.visualize or args.save_vis:
            visualize_overlap_distribution(
                args.json, args.frame,
                save_path=args.save_vis,
                fov_half_h=args.fov_h,
                fov_half_v=args.fov_v,
                num_samples=args.samples,
                radius=args.radius,
                scale=args.scale
            )
    
    elif args.pos1 and args.rot1 and args.pos2 and args.rot2:
        # 直接位姿模式
        def parse_values(s):
            return [float(x.strip()) for x in s.split(',')]
        
        pos1 = parse_values(args.pos1)
        rot1 = parse_values(args.rot1)
        pos2 = parse_values(args.pos2)
        rot2 = parse_values(args.rot2)
        
        # 位置已经是用户指定的，不需要再缩放
        print("=" * 60)
        print("FOV 重叠度计算器 (UE坐标系) - 位姿模式")
        print("=" * 60)
        print(f"\n位姿 1: pos={pos1}, rot=[pitch={rot1[0]}, roll={rot1[1]}, yaw={rot1[2]}]")
        print(f"位姿 2: pos={pos2}, rot=[pitch={rot2[0]}, roll={rot2[1]}, yaw={rot2[2]}]")
        print(f"FOV: 水平 ±{args.fov_h}°, 垂直 ±{args.fov_v}°")
        
        iou, details = calculate_fov_overlap_ue(
            np.array(pos1), rot1,
            np.array(pos2), rot2,
            fov_half_h=args.fov_h,
            fov_half_v=args.fov_v,
            num_samples=args.samples,
            radius=args.radius,
            return_details=True
        )
        
        print("\n" + "-" * 40)
        print("计算结果:")
        print("-" * 40)
        print(f"  IoU (交并比):         {details['iou']:.4f} ({details['iou']*100:.2f}%)")
        print(f"  重叠度 (相对于位姿1): {details['overlap_ratio_pose1']:.4f}")
        print(f"  重叠度 (相对于位姿2): {details['overlap_ratio_pose2']:.4f}")
        print(f"\n详细统计:")
        print(f"  位姿1可见点数: {details['pose1_visible_count']}")
        print(f"  位姿2可见点数: {details['pose2_visible_count']}")
        print(f"  交集点数:      {details['intersection_count']}")
        print(f"  并集点数:      {details['union_count']}")
        print("=" * 60)
    
    else:
        parser.print_help()
        print("\n错误: 请提供 --json 和 --frame 参数，或者 --pos1/--rot1/--pos2/--rot2 参数")


if __name__ == '__main__':
    main()

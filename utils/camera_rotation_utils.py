import math
import numpy as np


def rotate_c2w_z_swing(c2w: np.ndarray, num_frames: int, max_angle_deg: float = 45.0) -> np.ndarray:
    """
    绕 Z 轴摆动旋转（向右再回来）
    
    Args:
        c2w: (4, 4) numpy array - 输入的 c2w 矩阵
        num_frames: 输出帧数
        max_angle_deg: 最大摆动角度 默认45度
    
    Returns:
        c2ws: (num_frames, 4, 4) numpy array - 旋转后的 c2w 矩阵序列
        
    轨迹: 0° -> +max_angle° -> 0°
    """
    c2w = np.asarray(c2w, dtype=np.float64)
    
    # 将角度转换为弧度
    max_angle_rad = max_angle_deg * math.pi / 180.0
    
    # 生成摆动的角度序列: 0 -> +45° -> 0
    half = num_frames // 2
    n1 = half  # 0 -> +max
    n2 = num_frames - half  # +max -> 0
    
    angles_1 = np.linspace(0, max_angle_rad, n1)
    angles_2 = np.linspace(max_angle_rad, 0, n2)
    
    # 合并角度（去除重复的端点）
    thetas = np.concatenate([angles_1[:-1], angles_2])
    
    # 确保长度正确
    if len(thetas) < num_frames:
        thetas = np.concatenate([thetas, np.zeros(num_frames - len(thetas))])
    elif len(thetas) > num_frames:
        thetas = thetas[:num_frames]
    
    # 生成旋转后的 c2w 序列
    c2ws = []
    for theta in thetas:
        # Z-up 坐标系下绕 Z 轴旋转矩阵
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R_z = np.array([
            [cos_t, -sin_t, 0, 0],
            [sin_t, cos_t, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        # 右乘旋转矩阵：相机原地转动，位置不变
        c2w_new = c2w @ R_z
        c2ws.append(c2w_new)
    
    return np.stack(c2ws, axis=0)  # (num_frames, 4, 4)


def rotate_c2w_z_360(c2w: np.ndarray, num_frames: int) -> np.ndarray:
    """
    绕 Z 轴 360 度旋转（相机原地转一圈）
    
    Args:
        c2w: (4, 4) numpy array - 输入的 c2w 矩阵
        num_frames: 输出帧数
    
    Returns:
        c2ws: (num_frames, 4, 4) numpy array - 旋转后的 c2w 矩阵序列
    """
    c2w = np.asarray(c2w, dtype=np.float64)
    
    # 生成绕 Z 轴的旋转角度序列
    thetas = np.linspace(0, 2 * math.pi, num_frames)
    
    # 生成旋转后的 c2w 序列
    c2ws = []
    for theta in thetas:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R_z = np.array([
            [cos_t, -sin_t, 0, 0],
            [sin_t, cos_t, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        # 右乘旋转矩阵：相机原地转动，位置不变
        c2w_new = c2w @ R_z
        c2ws.append(c2w_new)
    
    return np.stack(c2ws, axis=0)  # (num_frames, 4, 4)
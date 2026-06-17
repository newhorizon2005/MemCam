import json
import torch
import numpy as np


def degrees_to_radians(degrees):
    return degrees * np.pi / 180


def compute_c2w_matrix(params, scale=100.0):
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


def get_relative_pose(c2ws):
    """
    计算相对于第一帧的位姿

    Args:
        c2ws: list of 4x4 c2w matrices

    Returns:
        relative_c2ws: list of 4x4 relative c2w matrices
                       第一帧为单位矩阵
    """
    # 计算第一帧的 w2c (world-to-camera)
    c2w_0 = c2ws[0]
    w2c_0 = np.linalg.inv(c2w_0)

    relative_c2ws = []
    for c2w in c2ws:
        # 相对位姿 = w2c_0 @ c2w_i
        # 这使得第一帧变成单位矩阵
        relative_c2w = w2c_0 @ c2w
        relative_c2ws.append(relative_c2w)

    return relative_c2ws


def compute_relative_pose(c2ws, reference_idx, target_indices):
    """
    计算相对于reference帧的相对位姿 (与训练完全一致)

    Args:
        c2ws: (N, 4, 4) numpy array - 所有帧的c2w矩阵
        reference_idx: int - 参考帧索引
        target_indices: list[int] - 目标帧索引列表

    Returns:
        poses: (len(target_indices), 12) torch tensor - 相对位姿
    """
    c2w_ref = c2ws[reference_idx]  # (4, 4)
    w2c_ref = np.linalg.inv(c2w_ref)  # (4, 4)

    poses = []
    for idx in target_indices:
        relative_c2w = w2c_ref @ c2ws[idx]
        pose_12 = c2w_to_12dim(relative_c2w)
        poses.append(pose_12)

    return torch.tensor(np.stack(poses), dtype=torch.float32)  # (N, 12)


def c2w_to_12dim(c2w):
    """
    将 4x4 c2w 矩阵转换为 12 维向量

    格式: 3x4 矩阵按行展平
    [R_00, R_01, R_02, T_x, R_10, R_11, R_12, T_y, R_20, R_21, R_22, T_z]

    MemCam uses this row-major 3x4 camera pose format.
    """
    # 取 3x4 部分并按行展平
    return c2w[:3, :].flatten().tolist()


def load_camera_poses_from_json(json_path, start_frame=0, num_frames=None, scale=100.0):
    """
    从JSON加载相机位姿,返回相对位姿的 12 维列表

    This helper returns MemCam training camera pose vectors.

    Args:
        json_path: JSON文件路径
        start_frame: 起始帧索引
        num_frames: 帧数
        scale: 位置缩放因子(UE使用厘米,默认除以100转换为米)

    Returns:
        cam_params: list of 12-dim vectors
                    格式: [R_00, R_01, R_02, T_x, R_10, R_11, R_12, T_y, R_20, R_21, R_22, T_z]
                    相对于第一帧（第一帧为单位矩阵）
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    camera_data = data['CineCameraActor']
    frame_keys = sorted(camera_data.keys(), key=int)

    if num_frames is None:
        num_frames = len(frame_keys) - start_frame

    frame_keys = frame_keys[start_frame:start_frame + num_frames]

    # 计算所有帧的绝对 c2w
    c2ws = []
    for frame_key in frame_keys:
        frame_data = camera_data[frame_key]
        c2w = compute_c2w_matrix(frame_data, scale=scale)
        c2ws.append(c2w)

    # 计算相对位姿 (相对于第一帧)
    relative_c2ws = get_relative_pose(c2ws)

    # 转换为 12 维列表
    cam_params = []
    for c2w in relative_c2ws:
        rt = c2w_to_12dim(c2w)
        cam_params.append(rt)

    return cam_params


def load_c2ws_from_json(json_path, start_frame=0, num_frames=None, scale=100.0):
    """
    从JSON加载相机位姿,返回绝对的 4x4 c2w 矩阵

    这是供推理使用的函数 返回原始c2w矩阵
    让pipeline内部像训练一样实时计算相对位姿

    Args:
        json_path: JSON文件路径
        start_frame: 起始帧索引
        num_frames: 帧数
        scale: 位置缩放因子(UE使用厘米,默认除以100转换为米)

    Returns:
        c2ws: (N, 4, 4) numpy array - 所有帧的绝对c2w矩阵
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    camera_data = data['CineCameraActor']
    frame_keys = sorted(camera_data.keys(), key=int)

    if num_frames is None:
        num_frames = len(frame_keys) - start_frame

    frame_keys = frame_keys[start_frame:start_frame + num_frames]

    # 计算所有帧的绝对 c2w
    c2ws = []
    for frame_key in frame_keys:
        frame_data = camera_data[frame_key]
        c2w = compute_c2w_matrix(frame_data, scale=scale)
        c2ws.append(c2w)

    return np.stack(c2ws, axis=0)  # (N, 4, 4)

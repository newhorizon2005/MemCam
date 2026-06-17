import torch
import torch.nn.functional as F
TARGET_LENGTH = 76


def pad_for_3d_conv(x, kernel_size):
    if x.dim() == 4:  # (C, T, H, W)
        c, t, h, w = x.shape
        pt, ph, pw = kernel_size
    else:  # (B, C, T, H, W)
        b, c, t, h, w = x.shape
        pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")


def center_down_sample_3d(x, kernel_size):
    return F.avg_pool3d(x, kernel_size, stride=kernel_size)


def compute_context_rope(dit, f_ctx, h_tgt, w_tgt, h_ctx, w_ctx, device):
    """
    RoPE computation for context (batch_size=1 only):
    1. Compute full RoPE at (T_ctx, H_tgt, W_tgt) resolution
    2. Downsample spatially using avg_pool to (T_ctx, H_ctx, W_ctx)

    MemCam memory-context RoPE layout:
    - Target uses positions 0-19 (preserve pretrained positions)
    - Context uses positions 20-95 (sequential assignment, 76 frames)

    Args:
        dit: The DiT model with precomputed freqs (complex tensors)
        f_ctx: Number of context frames (e.g., 76)
        h_tgt, w_tgt: Target spatial dimensions (before compression)
        h_ctx, w_ctx: Context spatial dimensions (after compression)
        device: Target device

    Returns:
        context_rope: (S_ctx, 1, dim) complex - RoPE for context tokens
    """
    # Context tokens use positions starting from TARGET_LENGTH.
    # So context positions are: 20, 21, 22, ..., 20+f_ctx-1
    context_start_pos = TARGET_LENGTH  # 20

    # Get time frequencies for context positions (20, 21, ..., 95)
    time_freqs = dit.freqs[0].to(device)  # (max_frames, dim_t//2) complex
    selected_time_freqs = time_freqs[context_start_pos:context_start_pos + f_ctx]  # (T_ctx, dim_t//2) complex

    # Get spatial frequencies (complex)
    h_freqs = dit.freqs[1][:h_tgt].to(device)  # (H_tgt, dim_h) complex
    w_freqs = dit.freqs[2][:w_tgt].to(device)  # (W_tgt, dim_w) complex

    # Build full RoPE grid at (T_ctx, H_tgt, W_tgt) - complex
    full_rope = torch.cat([
        selected_time_freqs.view(f_ctx, 1, 1, -1).expand(f_ctx, h_tgt, w_tgt, -1),
        h_freqs.view(1, h_tgt, 1, -1).expand(f_ctx, h_tgt, w_tgt, -1),
        w_freqs.view(1, 1, w_tgt, -1).expand(f_ctx, h_tgt, w_tgt, -1)
    ], dim=-1)  # (T_ctx, H_tgt, W_tgt, dim) complex

    # Downsample using avg_pool after kernel-aligned padding.
    full_rope = full_rope.permute(3, 0, 1, 2).unsqueeze(0)  # (1, dim, T_ctx, H_tgt, W_tgt) complex
    spatial_kernel = (1, h_tgt // h_ctx, w_tgt // w_ctx)  # (1, 2, 2)

    # Separate real and imaginary, pad, pool, then combine
    rope_real = pad_for_3d_conv(full_rope.real.float(), spatial_kernel)
    rope_real = center_down_sample_3d(rope_real, spatial_kernel)

    rope_imag = pad_for_3d_conv(full_rope.imag.float(), spatial_kernel)
    rope_imag = center_down_sample_3d(rope_imag, spatial_kernel)

    # Reconstruct complex tensor and reshape: (S_ctx, 1, dim)
    context_rope = torch.complex(rope_real, rope_imag)
    context_rope = context_rope.squeeze(0).permute(1, 2, 3, 0)  # (T_ctx, H_ctx, W_ctx, dim)
    context_rope = context_rope.reshape(f_ctx * h_ctx * w_ctx, 1, -1)  # (S_ctx, 1, dim)

    return context_rope

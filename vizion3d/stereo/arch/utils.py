"""
Image manipulation, positional encoding, and numeric utilities for S2M2.

This module contains pure tensor operations shared across the model architecture:
padding/cropping helpers, sinc-based positional encoding, cost-volume sampling,
and numerically stable log-sum-exp.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def image_pad(img: Tensor, factor: int = 32) -> Tensor:
    """Pad spatial dims to the next multiple of *factor* with content-aware fill.

    The border pixels are filled by downsampling then upsampling the padded region
    so edge artefacts do not bleed into the network.  The original image content is
    then written back into the centre of the padded result.

    Args:
        img: Float tensor of shape ``(B, C, H, W)``.
        factor: Divisibility factor (default 32).

    Returns:
        Padded tensor of shape ``(B, C, H', W')`` where ``H'`` and ``W'`` are the
        nearest multiples of *factor* ≥ ``H`` and ``W``.
    """
    H, W = img.shape[-2:]
    H_new = math.ceil(H / factor) * factor
    W_new = math.ceil(W / factor) * factor
    pad_h = H_new - H
    pad_w = W_new - W
    img_pad = F.pad(img, (pad_w // 2, pad_w - pad_w // 2, 0, 0), "constant", 0)
    img_pad = F.pad(img_pad, (0, 0, pad_h // 2, pad_h - pad_h // 2), "constant", 0)
    img_pad_down = F.adaptive_avg_pool2d(img_pad.float(), output_size=[H // factor, W // factor])
    img_pad = F.interpolate(img_pad_down, size=[H_new, W_new], mode="bilinear")
    h_s, h_e = pad_h // 2, pad_h - pad_h // 2
    w_s, w_e = pad_w // 2, pad_w - pad_w // 2
    if h_e == 0 and w_e == 0:
        img_pad[:, :, h_s:, w_s:] = img
    elif h_e == 0:
        img_pad[:, :, h_s:, w_s:-w_e] = img
    elif w_e == 0:
        img_pad[:, :, h_s:-h_e, w_s:] = img
    else:
        img_pad[:, :, h_s:-h_e, w_s:-w_e] = img
    return img_pad


def image_crop(img: Tensor, img_shape: tuple) -> Tensor:
    """Remove symmetric padding added by :func:`image_pad`.

    Args:
        img: Padded tensor of shape ``(B, C, H_pad, W_pad)``.
        img_shape: ``(H_orig, W_orig)`` — the target crop size.

    Returns:
        Cropped tensor of shape ``(B, C, H_orig, W_orig)``.
    """
    H, W = img.shape[-2:]
    H_new, W_new = img_shape
    crop_h = H - H_new
    if crop_h > 0:
        s, e = crop_h // 2, crop_h - crop_h // 2
        img = img[:, :, s:-e]
    crop_w = W - W_new
    if crop_w > 0:
        s, e = crop_w // 2, crop_w - crop_w // 2
        img = img[:, :, :, s:-e]
    return img


def custom_sinc(x: Tensor) -> Tensor:
    """Numerically stable sinc: ``sin(π·x)/(π·x)`` with the correct limit 1 at x=0.

    Args:
        x: Input tensor (any shape, any dtype).

    Returns:
        Sinc values, same shape and dtype as *x*.
    """
    return torch.where(
        torch.abs(x) < 1e-6,
        torch.ones_like(x),
        (torch.sin(3.1415 * x) / (3.1415 * x)).to(x.dtype),
    )


def custom_unfold(x: Tensor, kernel_size: int = 3, padding: int = 1) -> Tensor:
    """Expand a feature map into per-pixel 3×3 neighbourhood patches.

    The output channels contain the ``kernel_size²`` shifted copies of the input,
    concatenated along the channel dimension — equivalent to ``nn.Unfold`` but
    implemented as a stack of slices to avoid large intermediate tensors.

    Args:
        x: Input tensor ``(B, C, H, W)``.
        kernel_size: Neighbourhood size (default 3).
        padding: Replicate padding applied before extraction (default 1).

    Returns:
        Tensor ``(B, C·kernel_size², H, W)``.
    """
    B, C, H, W = x.shape
    x_pad = F.pad(x, (padding, padding, padding, padding), "replicate")
    parts = []
    for i in range(kernel_size):
        for j in range(kernel_size):
            parts.append(x_pad[:, :, i : i + H, j : j + W])
    return torch.cat(parts, dim=1)


def get_pe(h: int, w: int, pe_dim: int, dtype, device) -> Tensor:
    """Compute 2D relative positional encoding via sinc interpolation.

    Produces a ``(h·w, h·w, pe_dim)`` tensor whose ``[i, j, :]`` entry encodes
    the relative 2D offset between spatial positions *i* and *j*.

    Args:
        h: Feature-map height (at the scale where PE is applied).
        w: Feature-map width.
        pe_dim: Encoding dimensionality (split equally between x and y).
        dtype: Torch dtype for the output.
        device: Torch device for the output.

    Returns:
        Positional encoding tensor ``(h·w, h·w, pe_dim)``.
    """
    with torch.no_grad():
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, h - 1, h, device=device, dtype=dtype),
            torch.linspace(0, w - 1, w, device=device, dtype=dtype),
            indexing="ij",
        )
        rel_x = (grid_x.reshape(-1, 1) - grid_x.reshape(1, -1)).long()
        rel_y = (grid_y.reshape(-1, 1) - grid_y.reshape(1, -1)).long()

        sig = 5 / pe_dim
        x_pos = torch.linspace(-3, 3, 2 * w + 1, device=device, dtype=dtype).tanh()
        dim_t = torch.linspace(-1, 1, pe_dim // 2, device=device, dtype=dtype)
        pe_x = F.normalize(custom_sinc((dim_t[None] - x_pos[:, None]) / sig), p=2, dim=-1)
        rel_pe_x = pe_x[rel_x + w - 1].reshape(h * w, h * w, pe_dim // 2)

        y_pos = torch.linspace(-3, 3, 2 * h + 1, device=device, dtype=dtype).tanh()
        pe_y = F.normalize(custom_sinc((dim_t[None] - y_pos[:, None]) / sig), p=2, dim=-1)
        rel_pe_y = pe_y[rel_y + h - 1].reshape(h * w, h * w, pe_dim // 2)

        pe = 0.5 * torch.cat([rel_pe_x, rel_pe_y], dim=2)
    return pe.clone()


def bilinear_sampler(img: Tensor, coords: Tensor, mode: str = "bilinear") -> Tensor:
    """Sample *img* at fractional pixel coordinates using bilinear interpolation.

    Args:
        img: Feature map ``(B·H, 1, H_feat, W_feat)`` (squeezed batch format used
             by :class:`CostVolume`).
        coords: Sampling coordinates ``(B·H, W, K, 2)`` where the last dim is
                ``(x, y)`` in pixel units (not normalised).
        mode: Interpolation mode forwarded to ``F.grid_sample``.

    Returns:
        Sampled values, same shape prefix as *coords* with the channel squeezed.
    """
    W = torch.tensor(img.shape[-1], dtype=img.dtype, device=img.device)
    H = torch.tensor(img.shape[-2], dtype=img.dtype, device=img.device)
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    grid = torch.cat([2 * xgrid / (W - 1) - 1, 2 * ygrid / (H - 1) - 1], dim=-1)
    return F.grid_sample(img, grid, mode=mode, align_corners=True)


def logsumexp_stable(x: Tensor, dim: int, keepdim: bool = False, eps: float = 1e-30) -> Tensor:
    """Numerically stable log-sum-exp along *dim* using the max-subtraction trick.

    Args:
        x: Input tensor.
        dim: Reduction dimension.
        keepdim: Whether to keep the reduced dimension.
        eps: Floor clamped onto the inner sum to avoid ``log(0)``.

    Returns:
        Log-sum-exp values, with *dim* reduced (or kept as size 1 if *keepdim*).
    """
    m, _ = x.max(dim=dim, keepdim=True)
    y = (x - m).exp().sum(dim=dim, keepdim=True)
    y = m + torch.log(torch.clamp(y, min=eps))
    return y if keepdim else y.squeeze(dim)

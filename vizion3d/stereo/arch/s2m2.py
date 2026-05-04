"""
S2M2 stereo matching transformer — model definition and checkpoint helpers.

Model architecture constants are hardcoded here and must not be exposed as user
configuration.  The only external knob is the checkpoint variant (S / M / L / XL),
which is detected automatically from the checkpoint filename.

Architecture constants (fixed for all inference):
    _DIM_EXPANSION   = 1     — Q/K/V width multiplier (not a tunable hyperparameter).
    _OT_ITER         = 3     — Sinkhorn iterations for disparity initialisation.
    _USE_POSITIVITY  = True  — clamp disparity ≥ 0 (correct for rectified pairs).
    _OUTPUT_UPSAMPLE = False — training-only 2× flag; never needed at inference.
    _REFINE_ITER     = 3     — local GRU refinement steps per inference call.

Variant configs (feature_channels, num_transformer):
    S  → (128, 1)
    M  → (192, 2)
    L  → (256, 3)   ← default / recommended
    XL → (384, 3)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import BasicAttnBlock, GlobalAttnBlock
from .components import CNNEncoder, CostVolume, DispInit, FeatureFusion, Unet
from .refiners import GlobalRefiner, LocalRefiner, UpsampleMask1x, UpsampleMask4x
from .utils import custom_unfold

# ── Model-level constants (must not be user-configurable) ─────────────────────

_DIM_EXPANSION: int = 1
_OT_ITER: int = 3
_USE_POSITIVITY: bool = True
_OUTPUT_UPSAMPLE: bool = False
_REFINE_ITER: int = 3

# Per-variant channel / transformer counts — detected from checkpoint filename
_VARIANT_CONFIGS: dict[str, dict] = {
    "S": {"feature_channels": 128, "num_transformer": 1},
    "M": {"feature_channels": 192, "num_transformer": 2},
    "L": {"feature_channels": 256, "num_transformer": 3},
    "XL": {"feature_channels": 384, "num_transformer": 3},
}
_DEFAULT_VARIANT: str = "L"


def s2m2_config_from_checkpoint(model_path: str) -> dict:
    """Return the architecture config for the S2M2 variant inferred from *model_path*.

    Detection checks the filename (case-insensitive) for ``-S``, ``-M``, ``-L``,
    or ``-XL`` suffixes (with or without the ``.pth``/``.pt`` extension).
    Falls back to the L variant when no match is found.

    Args:
        model_path: Local file path to the checkpoint.

    Returns:
        Dict with ``"feature_channels"`` and ``"num_transformer"`` keys.
    """
    stem = Path(model_path).stem.upper()
    for variant in ("XL", "L", "M", "S"):  # XL before L to avoid partial match
        if stem.endswith(f"-{variant}") or stem.endswith(f"_{variant}"):
            return _VARIANT_CONFIGS[variant]
    return _VARIANT_CONFIGS[_DEFAULT_VARIANT]


# ── MRT transformer block ─────────────────────────────────────────────────────


class MRT(nn.Module):
    """Multi-scale recurrent transformer block — one stage of the StackedMRT.

    Implements a U-Net-shaped encoder-decoder path where each level applies
    one :class:`BasicAttnBlock` (cross + self attention) and the bottleneck
    applies two :class:`GlobalAttnBlock` steps with cross-attention.

    Args:
        dims: Channel counts at three resolution levels ``[d0, d1, d2]``.
        num_heads: Base head count (multiplied per level: 1×, 2×, 4×, 8×).
        dim_expansion: Attention expansion factor.
        use_gate_fusion: Whether to use gated skip connections.
    """

    def __init__(self, dims: list, num_heads: int, dim_expansion: int, use_gate_fusion: bool):
        super().__init__()
        self.down_conv0 = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(dims[0], dims[1], 1))
        self.down_conv1 = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(dims[1], dims[2], 1))
        self.down_conv2 = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(dims[2], dims[2], 1))
        self.up_conv0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dims[1], dims[0], 1),
        )
        self.up_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dims[2], dims[1], 1),
        )
        self.up_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dims[2], dims[2], 1),
        )
        self.down_concat1 = FeatureFusion(dims[1], 1, use_gate_fusion)
        self.down_concat2 = FeatureFusion(dims[2], 1, use_gate_fusion)
        self.down_concat3 = FeatureFusion(dims[2], 1, use_gate_fusion)
        self.up_concat0 = FeatureFusion(dims[0], 1, use_gate_fusion)
        self.up_concat1 = FeatureFusion(dims[1], 1, use_gate_fusion)
        self.up_concat2 = FeatureFusion(dims[2], 1, use_gate_fusion)
        self.enc_attn0 = BasicAttnBlock(dims[0], 1 * num_heads, dim_expansion)
        self.enc_attn1 = BasicAttnBlock(dims[1], 2 * num_heads, dim_expansion)
        self.enc_attn2 = BasicAttnBlock(dims[2], 4 * num_heads, dim_expansion)
        self.enc_attn3s = nn.ModuleList(
            [GlobalAttnBlock(dims[2], 8 * num_heads, dim_expansion, True) for _ in range(2)]
        )
        self.dec_attn0 = BasicAttnBlock(dims[0], 1 * num_heads, dim_expansion)
        self.dec_attn1 = BasicAttnBlock(dims[1], 2 * num_heads, dim_expansion)
        self.dec_attn2 = BasicAttnBlock(dims[2], 4 * num_heads, dim_expansion)
        self.dec_attn3s = nn.ModuleList(
            [GlobalAttnBlock(dims[2], 8 * num_heads, dim_expansion, True) for _ in range(2)]
        )

    def forward(self, z0, z1, z2, z3):
        """Apply one MRT encoder-decoder pass to four multi-scale feature maps.

        Args:
            z0, z1, z2, z3: Feature maps at strides 1×, 2×, 4×, 8× (all ``(2B, C, H, W)``).

        Returns:
            Updated ``(z0, z1, z2, z3)`` at the same shapes.
        """
        z0 = self.enc_attn0(z0)
        z1 = self.enc_attn1(self.down_concat1(z1, self.down_conv0(z0)))
        z2 = self.enc_attn2(self.down_concat2(z2, self.down_conv1(z1)))
        z3 = self.down_concat3(z3, self.down_conv2(z2))
        for blk in self.enc_attn3s:
            z3 = blk(z3)
        for blk in self.dec_attn3s:
            z3 = blk(z3)
        z2 = self.dec_attn2(self.up_concat2(z2, self.up_conv2(z3)))
        z1 = self.dec_attn1(self.up_concat1(z1, self.up_conv1(z2)))
        z0 = self.dec_attn0(self.up_concat0(z0, self.up_conv0(z1)))
        return z0, z1, z2, z3


class StackedMRT(nn.Module):
    """Stack of *num_transformer* MRT blocks that refine multi-scale features iteratively.

    Args:
        num_transformer: Number of MRT stages to stack.
        dims: Channel counts per pyramid level ``[d0, d1, d2]``.
        num_heads: Base attention head count.
        dim_expansion: Attention expansion factor.
        use_gate_fusion: Whether to use gated skip connections.
    """

    def __init__(
        self,
        num_transformer: int,
        dims: list,
        num_heads: int,
        dim_expansion: int,
        use_gate_fusion: bool,
    ):
        super().__init__()
        self.uformer_list = nn.ModuleList(
            [MRT(dims, num_heads, dim_expansion, use_gate_fusion) for _ in range(num_transformer)]
        )

    def forward(self, z0, z1, z2, z3) -> Tensor:
        """Run all MRT stages in sequence and return the finest-resolution output.

        Returns:
            ``z0`` after the final MRT stage, shape ``(2B, dims[0], H/4, W/4)``.
        """
        for blk in self.uformer_list:
            z0, z1, z2, z3 = blk(z0, z1, z2, z3)
        return z0.contiguous()


# ── S2M2 top-level model ──────────────────────────────────────────────────────


class S2M2(nn.Module):
    """Stereo Matching Model with Multi-scale transformer (S2M2).

    Full forward pass: CNN backbone → feature pyramid → stacked MRT transformer
    → OT disparity init → global refinement → iterative local GRU refinement
    → learned 4× and 1× upsampling.

    Instantiate via :func:`build_s2m2` rather than calling this constructor
    directly, so architecture constants are applied consistently.

    Args:
        feature_channels: Per-level channel count (variant-dependent).
        dim_expansion: Q/K/V expansion factor (hardcoded to ``_DIM_EXPANSION``).
        num_transformer: Number of StackedMRT stages (variant-dependent).
        use_positivity: Clamp disparity ≥ 0 (hardcoded to ``_USE_POSITIVITY``).
        output_upsample: Enable 2× output head (hardcoded to ``_OUTPUT_UPSAMPLE``).
        refine_iter: Local GRU refinement steps (hardcoded to ``_REFINE_ITER``).
    """

    def __init__(
        self,
        feature_channels: int,
        dim_expansion: int,
        num_transformer: int,
        use_positivity: bool = _USE_POSITIVITY,
        output_upsample: bool = _OUTPUT_UPSAMPLE,
        refine_iter: int = _REFINE_ITER,
    ):
        super().__init__()
        ch = feature_channels
        self.use_positivity = use_positivity
        self.refine_iter = refine_iter
        self.output_upsample = output_upsample
        self.cnn_backbone = CNNEncoder(ch)
        self.feat_pyramid = Unet(
            [ch, ch, 2 * ch],
            dim_expansion,
            True,
            n_attn=num_transformer * 2,
            use_gate_fusion=True,
        )
        self.transformer = StackedMRT(num_transformer, [ch, ch, 2 * ch], 1, dim_expansion, True)
        self.disp_init = DispInit(ch, _OT_ITER, use_positivity)
        self.upsample_mask_1x = UpsampleMask1x(ch)
        self.upsample_mask_4x = UpsampleMask4x(ch)
        self.global_refiner = GlobalRefiner(ch)
        self.feat_fusion_layer = FeatureFusion(ch, 3, True)
        self.refiner = LocalRefiner(ch, dim_expansion, 4, True)
        self.ctx_feat = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.GELU(), nn.Conv2d(ch, ch, 1))

    def my_load_state_dict(self, state_dict: dict):
        """Load *state_dict* with shape-mismatch tolerance.

        Silently skips keys whose tensor shapes differ from the current model
        (e.g. when fine-tuning a pretrained model on a different image size).

        Args:
            state_dict: Dict of parameter tensors, typically from
                        ``torch.load(...).get("state_dict", ckpt)``.
        """
        own = self.state_dict()
        for k in state_dict:
            if k in own and state_dict[k].shape != own[k].shape:
                state_dict[k] = own[k]
        self.load_state_dict(state_dict, strict=False)

    @staticmethod
    def _normalize(img0: Tensor, img1: Tensor):
        """Normalise uint8 pixel values to ``[-1, 1]``."""
        return (img0 / 255.0 - 0.5) * 2, (img1 / 255.0 - 0.5) * 2

    def _upsample4x(self, x: Tensor, up_weights: Tensor) -> Tensor:
        """Apply the learned 4× convex-combination upsampler."""
        b, c, h, w = x.shape
        x_unfold = custom_unfold(x.reshape(b, c, h, w), 3, 1)
        x_unfold = F.interpolate(x_unfold, (h * 4, w * 4), mode="nearest").reshape(
            b, 9, h * 4, w * 4
        )
        return (x_unfold * up_weights.softmax(dim=1)).sum(1, keepdim=True)

    def _upsample1x(self, disp: Tensor, filter_weights: Tensor) -> Tensor:
        """Apply the guided 1× (or 2× when ``output_upsample``) disparity upsampler."""
        disp_unfold = custom_unfold(disp, 3, 1)
        if self.output_upsample:
            disp_unfold = F.interpolate(disp_unfold, scale_factor=2, mode="nearest")
            filter_weights = F.interpolate(
                filter_weights, scale_factor=2, mode="bilinear", align_corners=False
            )
        return (disp_unfold * filter_weights.softmax(dim=1).to(disp.dtype)).sum(1, keepdim=True)

    def forward(self, img0: Tensor, img1: Tensor):
        """Run full stereo forward pass.

        Args:
            img0: Left image ``(B, 3, H, W)`` with pixel values in ``[0, 255]``.
            img1: Right image ``(B, 3, H, W)`` with pixel values in ``[0, 255]``.

        Returns:
            ``(disp_up, occ_up, conf_up)`` — disparity (in pixels at input resolution),
            occlusion, and confidence maps, all ``(B, 1, H, W)``.
        """
        img0_nor, img1_nor = self._normalize(img0, img1)
        feature_4x, feature_2x = self.cnn_backbone(torch.cat([img0_nor, img1_nor], dim=0))
        feature0_2x, _ = feature_2x.chunk(2, dim=0)
        feature_py_4x, feature_py_8x, feature_py_16x, feature_py_32x = self.feat_pyramid(feature_4x)
        feature_tr_4x = self.transformer(
            feature_py_4x, feature_py_8x, feature_py_16x, feature_py_32x
        )
        disp, conf, occ, cv = self.disp_init(feature_tr_4x)
        feature0_tr_4x, _ = feature_tr_4x.chunk(2, dim=0)
        feature0_py_4x, _ = feature_py_4x.chunk(2, dim=0)
        disp = self.global_refiner(feature0_tr_4x.contiguous(), disp.detach(), conf.detach())
        if self.use_positivity:
            disp = disp.clamp(min=0)
        feature0_fusion_4x = self.feat_fusion_layer(feature0_tr_4x, feature0_py_4x)
        ctx0 = self.ctx_feat(feature0_fusion_4x)
        hidden = torch.tanh(ctx0)
        b, c, h, w = feature0_fusion_4x.shape
        coords_4x = torch.arange(
            w, device=feature0_fusion_4x.device, dtype=feature0_fusion_4x.dtype
        )
        cv_fn = CostVolume(cv, coords_4x.reshape(1, 1, w, 1).repeat(b, h, 1, 1), radius=4)
        for _ in range(self.refine_iter):
            hidden, disp, conf, occ = self.refiner(hidden, ctx0, disp, conf, occ, cv_fn)
            if self.use_positivity:
                disp = disp.clamp(min=0)
            occ = occ * torch.ge(coords_4x.reshape(1, 1, 1, -1) - disp, 0)
        upsample_mask = self.upsample_mask_4x(hidden, feature0_2x)
        disp_up = self._upsample4x(disp * 4, upsample_mask)
        occ_up = self._upsample4x(occ, upsample_mask)
        conf_up = self._upsample4x(conf, upsample_mask)
        filter_weights = self.upsample_mask_1x(disp_up, img0_nor, feature0_2x)
        disp_up = self._upsample1x(disp_up, filter_weights)
        occ_up = self._upsample1x(occ_up, filter_weights)
        conf_up = self._upsample1x(conf_up, filter_weights)
        if self.output_upsample:
            disp_up = 2 * disp_up
        return disp_up, occ_up, conf_up


def build_s2m2(model_path: str) -> S2M2:
    """Construct an :class:`S2M2` instance with the correct architecture for *model_path*.

    Architecture constants (_DIM_EXPANSION, _USE_POSITIVITY, _REFINE_ITER,
    _OUTPUT_UPSAMPLE) are applied automatically.  The variant (S/M/L/XL) is
    detected from the checkpoint filename.

    Args:
        model_path: Path to the ``.pth`` checkpoint.

    Returns:
        Un-loaded :class:`S2M2` in eval-ready configuration (call
        ``my_load_state_dict`` + ``.eval()`` to complete initialisation).
    """
    cfg = s2m2_config_from_checkpoint(model_path)
    return S2M2(
        feature_channels=cfg["feature_channels"],
        dim_expansion=_DIM_EXPANSION,
        num_transformer=cfg["num_transformer"],
        use_positivity=_USE_POSITIVITY,
        output_upsample=_OUTPUT_UPSAMPLE,
        refine_iter=_REFINE_ITER,
    )

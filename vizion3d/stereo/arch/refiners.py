"""
Disparity refinement modules for S2M2.

Contains:
- :class:`ConvGRU`          — separable convolutional GRU for iterative updates.
- :class:`UpsampleMask4x`   — learned 4× disparity upsampler.
- :class:`UpsampleMask1x`   — learned 1× (or 2×) guided upsampler.
- :class:`GlobalRefiner`    — U-Net-based global gap-filling pass.
- :class:`LocalRefiner`     — cost-volume-guided local GRU refinement.
"""

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from .components import Unet


class ConvGRU(nn.Module):
    """Separable convolutional GRU: horizontal then vertical 1D convolutions.

    Runs two sequential GRU steps — one with a vertical kernel then one with a
    horizontal kernel — to propagate context efficiently across 2D feature maps.

    Args:
        hidden_dim: Recurrent state channel count.
        input_dim: Input feature channel count.
        kernel_size: Kernel size along the non-unit spatial dimension.
    """

    def __init__(self, hidden_dim: int = 128, input_dim: int = 128, kernel_size: int = 3):
        super().__init__()
        self.convz1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, [kernel_size, 1], padding=[kernel_size // 2, 0]
        )
        self.convr1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, [kernel_size, 1], padding=[kernel_size // 2, 0]
        )
        self.convq1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, [kernel_size, 1], padding=[kernel_size // 2, 0]
        )
        self.convz2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, [1, kernel_size], padding=[0, kernel_size // 2]
        )
        self.convr2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, [1, kernel_size], padding=[0, kernel_size // 2]
        )
        self.convq2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, [1, kernel_size], padding=[0, kernel_size // 2]
        )

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        """
        Args:
            h: Current hidden state ``(B, hidden_dim, H, W)``.
            x: Input features ``(B, input_dim, H, W)``.

        Returns:
            Updated hidden state ``(B, hidden_dim, H, W)``, dtype matches *x*.
        """
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        h = (1 - z) * h + z * torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        h = (1 - z) * h + z * torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        return h.to(x.dtype)


class UpsampleMask4x(nn.Module):
    """Learned convex-combination 4× upsampler using multi-scale feature context.

    Produces a ``(B, 9, H·4, W·4)`` weight map (via a transposed-conv step) that is
    used to combine 3×3 neighbourhood patches of the 4×-downsampled disparity map.

    Args:
        dim: Context feature channel count.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv_x = nn.ConvTranspose2d(dim, 64, 2, stride=2)
        self.conv_y = nn.Conv2d(dim, 64, 3, padding=1)
        self.conv_concat = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(128, 9, 2, stride=2),
        )

    def forward(self, feat_x: Tensor, feat_y: Tensor) -> Tensor:
        """
        Args:
            feat_x: GRU hidden state ``(B, dim, H, W)``.
            feat_y: 2× feature map ``(B, dim, H·2, W·2)``.

        Returns:
            9-channel upsample weight map ``(B, 9, H·4, W·4)``.
        """
        return self.conv_concat(torch.cat([self.conv_x(feat_x), self.conv_y(feat_y)], dim=1))


class UpsampleMask1x(nn.Module):
    """Guided 1× (or 2×) disparity upsampler using disparity, RGB, and context.

    Produces a ``(B, 9, H, W)`` (or ``(B, 9, H·2, W·2)`` when ``output_upsample``
    is enabled in S2M2) weight map for subpixel-accurate disparity refinement.

    Args:
        dim: Context feature channel count.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv_disp = nn.Sequential(
            nn.ConvTranspose2d(1, 16, 3, padding=1), nn.ReLU(inplace=False)
        )
        self.conv_rgb = nn.Sequential(
            nn.ConvTranspose2d(3, 16, 3, padding=1), nn.ReLU(inplace=False)
        )
        self.conv_ctx = nn.ConvTranspose2d(dim, 16, 2, stride=2)
        self.conv_concat = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(48, 9, 1),
        )

    def forward(self, disp: Tensor, rgb: Tensor, ctx: Tensor) -> Tensor:
        """
        Args:
            disp: 4×-upsampled disparity ``(B, 1, H, W)``.
            rgb: Normalised left image ``(B, 3, H, W)``.
            ctx: 2× feature map ``(B, dim, H/2, W/2)``.

        Returns:
            9-channel upsample filter weights ``(B, 9, H, W)``.
        """
        return self.conv_concat(
            torch.cat([self.conv_disp(disp), self.conv_rgb(rgb), self.conv_ctx(ctx)], dim=1)
        )


class GlobalRefiner(nn.Module):
    """U-Net-based global gap-filling pass that corrects low-confidence regions.

    Uses the confidence mask to suppress reliable disparities and predict
    corrections only where the OT initialisation was uncertain.

    Args:
        feature_channels: Context feature channel count.
    """

    def __init__(self, feature_channels: int):
        super().__init__()
        ch = feature_channels
        self.init_feat = nn.Sequential(
            nn.Conv2d(2 + ch, ch, 3, padding=1), nn.GELU(), nn.Conv2d(ch, ch, 1)
        )
        self.refine_unet = Unet([ch, ch, ch], 1, False, n_attn=1, use_gate_fusion=True)
        self.out_feat = nn.Sequential(nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, ctx: Tensor, disp: Tensor, conf: Tensor) -> Tensor:
        """
        Args:
            ctx: Left context features ``(B, C, H, W)``.
            disp: Initial disparity estimate ``(B, 1, H, W)``.
            conf: Per-pixel confidence ``(B, 1, H, W)`` in ``[0, 1]``.

        Returns:
            Refined disparity ``(B, 1, H, W)`` — confident regions are kept,
            uncertain regions are replaced by the U-Net prediction.
        """
        mask = 1.0 * (conf > 0.2)
        conf_logit = (mask * conf).logit(eps=1e-1)
        feat = self.init_feat(torch.cat([disp / 1e2 * mask, conf_logit, ctx], dim=1).to(disp.dtype))
        disp_update = self.out_feat(self.refine_unet(feat)[0]) * 1e2
        return (mask * disp + (1 - mask) * disp_update).to(disp.dtype)


class LocalRefiner(nn.Module):
    """Cost-volume-guided iterative local refinement using a ConvGRU.

    Each call to :meth:`forward` performs one GRU update step, reading cost-volume
    correlations around the current disparity estimate and jointly updating the
    disparity, confidence, and occlusion maps.

    Args:
        feature_channels: Context feature channel count.
        dim_expansion: U-Net expansion factor.
        radius: Cost-volume lookup radius.
        use_gate_fusion: Whether to use gated U-Net skip connections.
    """

    def __init__(
        self,
        feature_channels: int,
        dim_expansion: int,
        radius: int,
        use_gate_fusion: bool,
    ):
        super().__init__()
        ch = feature_channels
        r = radius
        self.disp_feat = nn.Sequential(
            nn.Conv2d(1, 96, 3, padding=1), nn.GELU(), nn.Conv2d(96, 96, 3, padding=1)
        )
        self.corr_feat1 = nn.Sequential(
            nn.Conv2d((2 * r + 1), 96, 1), nn.GELU(), nn.Conv2d(96, 64, 1)
        )
        self.corr_feat2 = nn.Sequential(
            nn.Conv2d((2 * r + 1), 96, 1), nn.GELU(), nn.Conv2d(96, 64, 1)
        )
        self.conf_occ_feat = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 32, 1)
        )
        self.disp_corr_ctx_cat = nn.Sequential(
            nn.Conv2d(256 + ch, 2 * ch, 1), nn.GELU(), nn.Conv2d(2 * ch, ch, 3, padding=1)
        )
        self.refine_unet = Unet(
            [ch, ch, 2 * ch], dim_expansion, False, n_attn=1, use_gate_fusion=use_gate_fusion
        )
        self.disp_update = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, 1, 3, padding=1, bias=False),
        )
        self.conf_occ_update = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, 2, 3, padding=1, bias=False),
        )
        self.gru = ConvGRU(ch, ch, 3)

    def forward(
        self,
        hidden: Tensor,
        ctx: Tensor,
        disp: Tensor,
        conf: Tensor,
        occ: Tensor,
        cv_fn: Callable,
    ):
        """Perform one refinement step.

        Args:
            hidden: GRU hidden state ``(B, C, H, W)``.
            ctx: Left context features ``(B, C, H, W)``.
            disp: Current disparity ``(B, 1, H, W)``.
            conf: Current confidence ``(B, 1, H, W)``.
            occ: Current occlusion ``(B, 1, H, W)``.
            cv_fn: Callable that accepts *disp* and returns ``(corrs, corrs_2x)``.

        Returns:
            ``(hidden_new, disp_new, conf_new, occ_new)`` — all updated tensors.
        """
        conf_logit = conf.logit(eps=1e-2)
        occ_logit = occ.logit(eps=1e-2)
        corr1, corr2 = cv_fn(disp)
        cat_in = torch.cat(
            [
                self.disp_feat(disp / 1e2),
                self.corr_feat1(corr1 / 16),
                self.corr_feat2(corr2 / 16),
                ctx,
                self.conf_occ_feat(torch.cat([conf_logit, occ_logit], dim=1).to(disp.dtype)),
            ],
            dim=1,
        ).to(disp.dtype)
        refine_feat = self.refine_unet(self.disp_corr_ctx_cat(cat_in))[0]
        hidden_new = self.gru(hidden, refine_feat)
        disp_update = self.disp_update(hidden_new)
        conf_update, occ_update = self.conf_occ_update(hidden_new).chunk(2, dim=1)
        return (
            hidden_new.to(disp.dtype),
            (disp + disp_update).to(disp.dtype),
            torch.sigmoid(conf_update + conf_logit).to(disp.dtype),
            torch.sigmoid(occ_update + occ_logit).to(disp.dtype),
        )

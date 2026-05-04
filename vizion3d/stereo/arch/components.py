"""
Feature extraction, U-Net pyramid, cost volume, and disparity initialisation for S2M2.

Contains:
- :class:`FeatureFusion` — gated blending of two same-dimension feature maps.
- :class:`ConvBlock2D`   — residual conv block used throughout the U-Net.
- :class:`CNNEncoder`    — 4× downsampling CNN backbone.
- :class:`Unet`          — multi-scale feature pyramid with bottleneck attention.
- :class:`CostVolume`    — efficient stereo cost-volume sampler.
- :class:`DispInit`      — optimal-transport disparity initialisation from correlations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import GlobalAttnBlock
from .utils import bilinear_sampler, get_pe, logsumexp_stable


class FeatureFusion(nn.Module):
    """Gated fusion of two same-resolution feature maps.

    Learns a soft gate *w ∈ (0, 1)* per spatial position so the output is
    ``fusion(z0, z1) + w·z0 + (1−w)·z1`` when ``use_gate=True``, or just
    ``fusion(z0, z1)`` otherwise.

    Args:
        dim: Input and output channel count for both *z0* and *z1*.
        kernel_size: Convolution kernel size for the gate and fusion networks.
        use_gate: Whether to include the learnable gate (default ``True``).
    """

    def __init__(self, dim: int, kernel_size: int, use_gate: bool = True):
        super().__init__()
        pad = kernel_size // 2
        self.use_gate = use_gate
        if use_gate:
            self.feature_gate = nn.Sequential(
                nn.Conv2d(2 * dim, dim, kernel_size=kernel_size, padding=pad),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.Sigmoid(),
            )
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1),
        )

    def forward(self, z0: Tensor, z1: Tensor) -> Tensor:
        z = torch.cat([z0, z1], dim=1)
        if self.use_gate:
            eps = 0.01
            w = self.feature_gate(z).clamp(min=eps, max=1 - eps)
            return self.feature_fusion(z) + w * z0 + (1 - w) * z1
        return self.feature_fusion(z)


class ConvBlock2D(nn.Module):
    """Residual 2D conv block: kxk conv path fused with a parallel 1×1 skip path.

    Args:
        dim: Channel count (input = output).
        kernel_size: Kernel size for the main conv path.
        dim_expansion: Hidden-width multiplier for both paths.
    """

    def __init__(self, dim: int, kernel_size: int, dim_expansion: int):
        super().__init__()
        p = kernel_size // 2
        self.convs = nn.Sequential(
            nn.Conv2d(dim, dim_expansion * dim, kernel_size, padding=p),
            nn.GELU(),
            nn.Conv2d(dim_expansion * dim, dim, kernel_size, padding=p),
        )
        self.convs_1x = nn.Sequential(
            nn.Conv2d(dim, dim_expansion * dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim_expansion * dim, dim, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.convs(z) + self.convs_1x(z)


class CNNEncoder(nn.Module):
    """4× downsampling CNN backbone producing features at 2× and 4× strides.

    Processes concatenated left+right images (along the batch dim) so both
    images share weights.  The two outputs are split externally.

    Args:
        output_dim: Output channel count at both stride levels.
    """

    def __init__(self, output_dim: int):
        super().__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(3, 16, 1), nn.GELU(), nn.Conv2d(16, 16, 1))
        self.conv1_down = nn.Sequential(
            nn.Conv2d(16, 64, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(64, output_dim, 3, padding=1),
        )
        self.norm1 = nn.GroupNorm(8, output_dim)
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(output_dim, output_dim, 3, padding=1),
        )
        self.conv2_down = nn.Sequential(nn.Conv2d(output_dim, output_dim, 3, stride=2, padding=1))

    def forward(self, x: Tensor):
        """
        Args:
            x: ``(2B, 3, H, W)`` — left and right images stacked along batch.

        Returns:
            ``(feature_4x, feature_2x)`` — downsampled 4× and 2× feature maps,
            both ``(2B, output_dim, H/4, W/4)`` and ``(2B, output_dim, H/2, W/2)``.
        """
        x = self.conv0(x)
        x_2x = self.norm1(self.conv1_down(x))
        x_2x = self.conv2(x_2x) + x_2x
        return self.conv2_down(x_2x), x_2x


class Unet(nn.Module):
    """Multi-scale U-Net feature pyramid with global attention at the bottleneck.

    Encoder downsamples three times via average pooling; decoder upsamples back
    with gated skip connections.  Global attention blocks sit at the lowest
    resolution for long-range context.

    Args:
        dims: Channel counts at the three pyramid levels ``[d0, d1, d2]``.
        dim_expansion: Conv/attention expansion factor.
        use_pe: Whether to use relative PE in bottleneck attention.
        n_attn: Number of global-attention blocks at the bottleneck.
        use_gate_fusion: Whether to use gated skip connections.
    """

    def __init__(
        self,
        dims: list,
        dim_expansion: int,
        use_pe: bool,
        n_attn: int = 1,
        use_gate_fusion: bool = True,
    ):
        super().__init__()
        self.use_pe = use_pe
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
        self.concat_conv0 = FeatureFusion(dims[0], 1, use_gate_fusion)
        self.concat_conv1 = FeatureFusion(dims[1], 1, use_gate_fusion)
        self.concat_conv2 = FeatureFusion(dims[2], 1, use_gate_fusion)
        self.enc0 = ConvBlock2D(dims[0], 3, dim_expansion)
        self.enc1 = ConvBlock2D(dims[1], 3, dim_expansion)
        self.enc2 = ConvBlock2D(dims[2], 3, dim_expansion)
        self.enc3s = nn.ModuleList(
            [GlobalAttnBlock(dims[2], 8, dim_expansion, False, use_pe) for _ in range(n_attn)]
        )
        self.dec0 = ConvBlock2D(dims[0], 3, dim_expansion)
        self.dec1 = ConvBlock2D(dims[1], 3, dim_expansion)
        self.dec2 = ConvBlock2D(dims[2], 3, dim_expansion)
        self.dec3s = nn.ModuleList(
            [GlobalAttnBlock(dims[2], 8, dim_expansion, False, False) for _ in range(n_attn)]
        )

    def forward(self, z: Tensor):
        """
        Args:
            z: Input feature map ``(B, dims[0], H, W)``.

        Returns:
            ``(z0, z1, z2, z3)`` — multi-scale outputs at strides 1×, 2×, 4×, 8×
            relative to the input spatial size.
        """
        pe = None
        if self.use_pe:
            H, W = z.shape[-2:]
            pe = get_pe(H // 8, W // 8, 32, z.dtype, z.device)
        z0 = self.enc0(z)
        z1 = self.enc1(self.down_conv0(z0))
        z2 = self.enc2(self.down_conv1(z1))
        z3 = self.down_conv2(z2)
        for blk in self.enc3s:
            z3 = blk(z3, pe)
        for blk in self.dec3s:
            z3 = blk(z3, pe)
        z2_new = self.dec2(self.concat_conv2(z2, self.up_conv2(z3)))
        z1_new = self.dec1(self.concat_conv1(z1, self.up_conv1(z2_new)))
        z0_new = self.dec0(self.concat_conv0(z0, self.up_conv0(z1_new)))
        return z0_new, z1_new, z2_new, z3


class CostVolume:
    """Efficient stereo cost-volume sampler for local refinement.

    Pre-computes the dense correlation matrix and supports fast local lookups
    around a running disparity estimate during GRU-based refinement.

    Args:
        cv: Dense correlation volume ``(B, H, W, W)`` — left × right feature dot-products.
        coords: Left-image x-coordinates ``(B, H, W, 1)`` used to anchor lookups.
        radius: Disparity search radius; lookups cover ``[disp−radius, disp+radius]``.
    """

    def __init__(self, cv: Tensor, coords: Tensor, radius: int):
        self.radius = radius
        dx = torch.linspace(-radius, radius, 2 * radius + 1, device=cv.device, dtype=cv.dtype)
        self.dx = dx.reshape(1, 1, 2 * radius + 1, 1)
        b, h, w, w2 = cv.shape
        self.cv = cv.reshape(b * h * w, 1, 1, w2)
        self.cv_2x = F.avg_pool2d(self.cv, kernel_size=[1, 2])
        self.cv = self.cv.reshape(b * h, 1, w, w2)
        self.cv_2x = self.cv_2x.reshape(b * h, 1, w, w2 // 2)
        self.coords = coords.reshape(b * h * w, 1, 1, 1)

    def __call__(self, disp: Tensor):
        """Sample full-res and half-res cost values around the current disparity.

        Args:
            disp: Current disparity estimate ``(B, 1, H, W)``.

        Returns:
            ``(corrs, corrs_2x)`` — sampled cost volumes at full and half resolution,
            each ``(B, 2·radius+1, H, W)``.
        """
        b, _, h, w = disp.shape
        dx = self.dx
        x0 = (self.coords - disp.reshape(b * h * w, 1, 1, 1) + dx).reshape(b * h, w, -1, 1)
        y0 = (self.coords + 0 * dx).reshape(b * h, w, -1, 1)
        corrs = bilinear_sampler(self.cv, torch.cat([x0, y0], dim=-1))
        corrs = corrs.reshape(b, h, w, 2 * self.radius + 1).permute(0, 3, 1, 2)
        x0_2 = (self.coords / 2 - disp.reshape(b * h * w, 1, 1, 1) / 2 + dx).reshape(
            b * h, w, -1, 1
        )
        corrs_2x = bilinear_sampler(self.cv_2x, torch.cat([x0_2, y0], dim=-1))
        corrs_2x = corrs_2x.reshape(b, h, w, 2 * self.radius + 1).permute(0, 3, 1, 2)
        return corrs, corrs_2x


class DispInit(nn.Module):
    """Optimal-transport disparity initialisation from dense feature correlations.

    Computes the dense left-right correlation volume, applies a Sinkhorn OT solver
    to produce a soft correspondence distribution, and extracts the expected disparity
    via a soft-argmax over a local window around the peak.

    Args:
        dim: Feature channel count.
        ot_iter: Number of Sinkhorn iterations.
        use_positivity: Mask the upper triangle so only positive (left→right) disparities
                        are considered — appropriate for well-rectified stereo pairs.
    """

    def __init__(self, dim: int, ot_iter: int, use_positivity: bool):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=True)
        self.ot_iter = ot_iter
        self.use_positivity = use_positivity

    def _sinkhorn(self, attn: Tensor, log_mu: Tensor, log_nu: Tensor) -> Tensor:
        """Run *ot_iter* steps of the Sinkhorn algorithm in log-space."""
        v = log_nu - logsumexp_stable(attn, dim=2)
        u = log_mu - logsumexp_stable(attn + v.unsqueeze(2), dim=3)
        for _ in range(self.ot_iter - 1):
            v = log_nu - logsumexp_stable(attn + u.unsqueeze(3), dim=2)
            u = log_mu - logsumexp_stable(attn + v.unsqueeze(2), dim=3)
        return attn + u.unsqueeze(3) + v.unsqueeze(2)

    def _optimal_transport(self, attn: Tensor) -> Tensor:
        """Convert a raw attention matrix to a normalised transport plan."""
        bs, h, w, _ = attn.shape
        dtype = attn.dtype
        marginal = torch.cat(
            [torch.ones([w], device=attn.device), torch.tensor([w], device=attn.device)]
        ) / (2 * w)
        log_mu = marginal.log().reshape(1, 1, w + 1)
        log_nu = marginal.log().reshape(1, 1, w + 1)
        attn = F.pad(attn, (0, 1, 0, 1), "constant", 0)
        attn = self._sinkhorn(attn, log_mu, log_nu)
        log_const = torch.log(torch.tensor(w, dtype=dtype, device=attn.device) * 2)
        return (attn[:, :, :-1, :-1] + log_const).exp().to(dtype)

    def forward(self, feature: Tensor):
        """
        Args:
            feature: Joint left+right feature tensor ``(2B, C, H, W)``.

        Returns:
            ``(disparity, confidence, occlusion, cost_volume)`` — all at the same
            spatial resolution as *feature*.
        """
        dtype = feature.dtype
        device = feature.device
        w = feature.shape[-1]
        x_grid = torch.linspace(0, w - 1, w, device=device, dtype=dtype)
        mask = (
            torch.triu(torch.ones((w, w), dtype=torch.bool, device=device), diagonal=1)
            if self.use_positivity
            else torch.zeros((w, w), dtype=torch.bool, device=device)
        )

        feature0, feature1 = self.layer_norm(feature.permute(0, 2, 3, 1)).chunk(2, dim=0)
        cv = torch.einsum("...hic,...hjc->...hij", feature0, feature1)
        cv_mask = cv.masked_fill(mask, -1e4)
        prob = self._optimal_transport(cv_mask)
        masked_prob = prob.masked_fill(mask, 0)

        prob_max_ind = masked_prob.argmax(dim=3).unsqueeze(3)
        prob_l = 2
        masked_prob_pad = F.pad(masked_prob, (prob_l, prob_l), "constant", 0)
        conf = 0
        correspondence_left = 0
        for idx in range(2 * prob_l + 1):
            weight = torch.gather(masked_prob_pad, index=prob_max_ind + idx, dim=3)
            conf += weight
            correspondence_left += weight * (prob_max_ind + idx - prob_l)
        eps = 1e-4
        correspondence_left = (correspondence_left + eps) / (conf + eps)
        disparity = (x_grid.reshape(1, 1, w) - correspondence_left.squeeze(3)).unsqueeze(1)
        conf = conf.unsqueeze(1).squeeze(-1)
        occ = masked_prob.sum(dim=3).unsqueeze(1)
        return disparity, conf, occ, cv

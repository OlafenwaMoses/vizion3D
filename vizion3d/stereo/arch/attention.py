"""
Multi-head attention modules for the S2M2 stereo matching transformer.

Provides self-attention and bidirectional cross-attention in both 1D (row-wise)
and 2D (global/flattened) variants, plus feed-forward network layers and the
combined ``GlobalAttnBlock`` / ``BasicAttnBlock`` building blocks used by the
feature pyramid and transformer stages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SelfAttn(nn.Module):
    """Multi-head self-attention with optional sinc-based relative positional encoding.

    Args:
        dim: Input and output channel count.
        num_heads: Number of attention heads.
        dim_expansion: Channel expansion factor inside Q/K/V projections.
        use_pe: If ``True``, adds a learned projection of the 2D relative PE to
                each head's output.
    """

    def __init__(self, dim: int, num_heads: int, dim_expansion: int, use_pe: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_expansion * dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_pe = use_pe
        self.q = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.k = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.v = nn.Linear(dim, dim_expansion * dim, bias=True)
        self.proj = nn.Linear(dim_expansion * dim, dim, bias=False)
        if use_pe:
            self.pe_proj = nn.Linear(32, self.head_dim)

    def forward(self, x: Tensor, pe: Tensor = None) -> Tensor:
        """
        Args:
            x: Token sequence ``(B, N, C)``.
            pe: Relative positional encoding ``(N, N, 32)`` (only used when
                ``use_pe=True``).

        Returns:
            Attended output ``(B, N, C)``.
        """
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if self.use_pe:
            score = torch.einsum("...ic,...jc->...ij", self.scale * q, k)
            attn = score.softmax(dim=-1)
            out = torch.einsum("...ij,...jc->...ic", attn, v)
            pe_sum = torch.einsum("...nij,ijc->...nic", attn, pe)
            out = out + self.pe_proj(pe_sum)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim))


class CrossAttn(nn.Module):
    """Bidirectional cross-attention that updates both left and right feature sequences.

    Each direction attends the other's keys/values, allowing both images to
    exchange information in a single block.

    Args:
        dim: Input and output channel count.
        num_heads: Number of attention heads.
        dim_expansion: Channel expansion factor inside Q/K/V projections.
    """

    def __init__(self, dim: int, num_heads: int, dim_expansion: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_expansion * dim // num_heads
        self.q = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.k = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.v = nn.Linear(dim, dim_expansion * dim, bias=True)
        self.proj = nn.Linear(dim_expansion * dim, dim, bias=False)

    def forward(self, x: Tensor, y: Tensor):
        """
        Args:
            x: Left feature sequence ``(B, N, C)``.
            y: Right feature sequence ``(B, N, C)``.

        Returns:
            ``(x_out, y_out)`` — updated left and right sequences ``(B, N, C)``.
        """
        B, N, _ = x.shape
        qx = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        ky = self.k(y).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        vy = self.v(y).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        x_out = F.scaled_dot_product_attention(qx, ky, vy)
        kx = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        qy = self.q(y).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        vx = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        y_out = F.scaled_dot_product_attention(qy, kx, vx)
        x_out = self.proj(x_out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim))
        y_out = self.proj(y_out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim))
        return x_out, y_out


class FFN(nn.Module):
    """Position-wise two-layer feed-forward network with pre-LayerNorm and residual.

    Args:
        dim: Input and output channel count.
        dim_expansion: Hidden layer size multiplier.
    """

    def __init__(self, dim: int, dim_expansion: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_expansion * dim),
            nn.GELU(),
            nn.Linear(dim_expansion * dim, dim),
        )
        self.norm_pre = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, z: Tensor) -> Tensor:
        return self.ffn(self.norm_pre(z)) + z


class SelfAttnBlock1D(nn.Module):
    """Row-wise self-attention: treats each image row as an independent sequence.

    Reshapes ``(B, H, W, C)`` → ``(B·H, W, C)`` so attention operates along W.

    Args:
        dim: Channel count.
        num_heads: Attention heads.
        dim_expansion: Q/K/V expansion factor.
        use_pe: Whether to apply relative PE.
    """

    def __init__(self, dim: int, num_heads: int, dim_expansion: int, use_pe: bool):
        super().__init__()
        self.attn = SelfAttn(dim, num_heads, dim_expansion, use_pe)
        self.norm_pre = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, z: Tensor, pe: Tensor = None) -> Tensor:
        B, H, W, C = z.shape
        z = z.reshape(B * H, W, C)
        z = self.attn(self.norm_pre(z), pe) + z
        return z.reshape(B, H, W, C)


class CrossAttnBlock1D(nn.Module):
    """Row-wise bidirectional cross-attention for paired left/right feature maps.

    Expects *z* to hold ``[left, right]`` concatenated along the batch dimension,
    i.e. ``z.shape[0]`` is ``2·B``.

    Args:
        dim: Channel count.
        num_heads: Attention heads.
        dim_expansion: Q/K/V expansion factor.
    """

    def __init__(self, dim: int, num_heads: int, dim_expansion: int):
        super().__init__()
        self.attn = CrossAttn(dim, num_heads, dim_expansion)
        self.norm_pre = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, z: Tensor) -> Tensor:
        x, y = self.norm_pre(z).chunk(2, dim=0)
        B, H, W, C = x.shape
        x, y = x.reshape(B * H, W, C), y.reshape(B * H, W, C)
        x, y = self.attn(x, y)
        x, y = x.reshape(B, H, W, C), y.reshape(B, H, W, C)
        return torch.cat([x, y], dim=0) + z


class SelfAttnBlock2D(nn.Module):
    """Global self-attention: flattens H×W into one long sequence per image.

    Args:
        dim: Channel count.
        num_heads: Attention heads.
        dim_expansion: Q/K/V expansion factor.
        use_pe: Whether to apply relative PE.
    """

    def __init__(self, dim: int, num_heads: int, dim_expansion: int, use_pe: bool):
        super().__init__()
        self.attn = SelfAttn(dim, num_heads, dim_expansion, use_pe)
        self.norm_pre = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, z: Tensor, pe: Tensor = None) -> Tensor:
        B, H, W, C = z.shape
        z = z.reshape(B, H * W, C)
        z = self.attn(self.norm_pre(z), pe) + z
        return z.reshape(B, H, W, C).contiguous()


class CrossAttnBlock2D(nn.Module):
    """Global bidirectional cross-attention for paired left/right feature maps.

    Expects *z* to hold ``[left, right]`` concatenated along batch.

    Args:
        dim: Channel count.
        num_heads: Attention heads.
        dim_expansion: Q/K/V expansion factor.
    """

    def __init__(self, dim: int, num_heads: int, dim_expansion: int):
        super().__init__()
        self.attn = CrossAttn(dim, num_heads, dim_expansion)
        self.norm_pre = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, z: Tensor) -> Tensor:
        x, y = self.norm_pre(z).chunk(2, dim=0)
        B, H, W, C = x.shape
        x, y = x.reshape(B, H * W, C), y.reshape(B, H * W, C)
        x, y = self.attn(x, y)
        x, y = x.reshape(B, H, W, C), y.reshape(B, H, W, C)
        return torch.cat([x, y], dim=0) + z


class GlobalAttnBlock(nn.Module):
    """Global self-attention (+ optional cross-attention) block used in U-Net stages.

    Operates on ``(B, C, H, W)`` conv-format tensors by permuting to
    ``(B, H, W, C)`` internally.

    Args:
        dim: Channel count.
        num_heads: Attention heads.
        dim_expansion: Expansion factor.
        use_cross_attn: If ``True``, applies 2D cross-attention before self-attention.
        use_pe: Whether to apply relative PE in self-attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_expansion: int,
        use_cross_attn: bool = False,
        use_pe: bool = False,
    ):
        super().__init__()
        self.self_attn = SelfAttnBlock2D(dim, num_heads, dim_expansion, use_pe)
        if use_cross_attn:
            self.cross_attn = CrossAttnBlock2D(dim, num_heads, dim_expansion)
            self.ffn_c = FFN(dim, dim_expansion)
        else:
            self.cross_attn = None
        self.ffn = FFN(dim, dim_expansion)

    def forward(self, z: Tensor, pe: Tensor = None) -> Tensor:
        z = z.permute(0, 2, 3, 1)
        if self.cross_attn is not None:
            z = self.ffn_c(self.cross_attn(z))
        z = self.ffn(self.self_attn(z, pe))
        return z.permute(0, 3, 1, 2).contiguous()


class BasicAttnBlock(nn.Module):
    """Row-wise cross-attention then row-wise self-attention, with FFN after each.

    The standard building block for the MRT transformer stages.

    Args:
        dim: Channel count.
        num_heads: Attention heads.
        dim_expansion: Expansion factor.
        use_pe: Whether to apply relative PE in self-attention.
    """

    def __init__(self, dim: int, num_heads: int, dim_expansion: int, use_pe: bool = False):
        super().__init__()
        self.cross_attn = CrossAttnBlock1D(dim, num_heads, dim_expansion)
        self.self_attn = SelfAttnBlock1D(dim, num_heads, dim_expansion, use_pe)
        self.ffn_c = FFN(dim, dim_expansion)
        self.ffn = FFN(dim, dim_expansion)

    def forward(self, z: Tensor, pe: Tensor = None) -> Tensor:
        z = z.permute(0, 2, 3, 1)
        z = self.ffn_c(self.cross_attn(z))
        z = self.ffn(self.self_attn(z, pe))
        return z.permute(0, 3, 1, 2)

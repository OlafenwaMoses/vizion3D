"""
Vendored SegFormer-B4 (MiT-B4) semantic segmentation backbone for the
SceneMaskAnnotation3D task.

The module names mirror the HuggingFace ``SegformerForSemanticSegmentation``
checkpoint layout, so the released ``segformer_b4_ade20k.bin`` state_dict loads
directly with no key remapping and no ``transformers`` dependency at runtime.

Public API
----------
- ``ADE20K_CLASSES``      — 150 class names (COCO-aligned where a 1:1 synonym
                            exists: sofa→couch, table→dining table, etc.).
- ``ADE20K_PALETTE``      — deterministic fixed RGB colour per class id.
- ``load_segformer``      — build the model and load weights onto a device.
- ``infer_semantic``      — run inference on a PIL image, returning an HxW int
                            class-id map at the *original* image resolution.
- ``colorize``            — map a class-id map to an RGB colour mask.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ── Architecture configuration (SegFormer-B4 / MiT-B4) ──────────────────────────
DEPTHS = [3, 8, 27, 3]
HIDDEN_SIZES = [64, 128, 320, 512]
NUM_HEADS = [1, 2, 5, 8]
SR_RATIOS = [8, 4, 2, 1]
PATCH_SIZES = [7, 3, 3, 3]
STRIDES = [4, 2, 2, 2]
MLP_RATIOS = [4, 4, 4, 4]
DECODER_HIDDEN = 768
NUM_CLASSES = 150
LN_EPS = 1e-6

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# ADE20K class names, renamed to the matching COCO name where a single synonym
# exists so masks flow straight into COCO-keyed pipelines (e.g. ScaleObservation
# size priors).  Ambiguous many-to-one cases (animal, pot) keep the ADE name.
ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
    "dining table", "mountain", "potted plant", "curtain", "chair", "car",
    "water", "painting", "couch", "shelf", "house", "sea", "mirror", "rug",
    "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
    "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen island", "laptop",
    "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel",
    "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth",
    "tv", "airplane", "dirt track", "apparel", "pole", "land", "bannister",
    "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van",
    "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything",
    "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag",
    "motorcycle", "cradle", "oven", "ball", "food", "step", "tank",
    "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light",
    "tray", "ashcan", "fan", "pier", "crt screen", "plate", "monitor",
    "bulletin board", "shower", "radiator", "wine glass", "clock", "flag",
]


def _hsv_to_rgb(h: float, s: float, v: float):
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


def _build_palette(num_colors: int) -> np.ndarray:
    palette = np.zeros((num_colors, 3), dtype=np.uint8)
    for i in range(num_colors):
        hue = (i * 0.6180339887498949) % 1.0
        sat = 0.55 + 0.35 * ((i // 6) % 2)
        val = 0.65 + 0.30 * ((i // 3) % 2)
        r, g, b = _hsv_to_rgb(hue, sat, val)
        palette[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return palette


ADE20K_PALETTE = _build_palette(NUM_CLASSES)


# ── Architecture (module names mirror the HF checkpoint) ─────────────────────────
class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, patch_size, stride, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.layer_norm = nn.LayerNorm(out_ch, eps=LN_EPS)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.layer_norm(x), h, w


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = dim // num_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.layer_norm = nn.LayerNorm(dim, eps=LN_EPS)

    def _shape(self, x):
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_size).permute(0, 2, 1, 3)

    def forward(self, x, h, w):
        b, n, c = x.shape
        q = self._shape(self.query(x))
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.layer_norm(x_)
        else:
            x_ = x
        k = self._shape(self.key(x_))
        v = self._shape(self.value(x_))
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_size)
        probs = F.softmax(scores, dim=-1)
        ctx = (probs @ v).permute(0, 2, 1, 3).reshape(b, n, c)
        return ctx


class SelfOutput(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)

    def forward(self, x):
        return self.dense(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.self = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.output = SelfOutput(dim)

    def forward(self, x, h, w):
        return self.output(self.self(x, h, w))


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.dense1 = nn.Linear(dim, hidden)
        self.dwconv = DWConv(hidden)
        self.dense2 = nn.Linear(hidden, dim)

    def forward(self, x, h, w):
        x = self.dense1(x)
        x = self.dwconv(x, h, w)
        x = F.gelu(x)
        return self.dense2(x)


class SegformerLayer(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, mlp_ratio):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(dim, eps=LN_EPS)
        self.attention = Attention(dim, num_heads, sr_ratio)
        self.layer_norm_2 = nn.LayerNorm(dim, eps=LN_EPS)
        self.mlp = MixFFN(dim, dim * mlp_ratio)

    def forward(self, x, h, w):
        x = x + self.attention(self.layer_norm_1(x), h, w)
        x = x + self.mlp(self.layer_norm_2(x), h, w)
        return x


class SegformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embeddings = nn.ModuleList()
        self.block = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        in_ch = 3
        for i in range(4):
            self.patch_embeddings.append(
                OverlapPatchEmbeddings(PATCH_SIZES[i], STRIDES[i], in_ch, HIDDEN_SIZES[i]))
            self.block.append(nn.ModuleList([
                SegformerLayer(HIDDEN_SIZES[i], NUM_HEADS[i], SR_RATIOS[i], MLP_RATIOS[i])
                for _ in range(DEPTHS[i])
            ]))
            self.layer_norm.append(nn.LayerNorm(HIDDEN_SIZES[i], eps=LN_EPS))
            in_ch = HIDDEN_SIZES[i]

    def forward(self, x):
        b = x.shape[0]
        features = []
        for i in range(4):
            x, h, w = self.patch_embeddings[i](x)
            for blk in self.block[i]:
                x = blk(x, h, w)
            x = self.layer_norm[i](x)
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            features.append(x)
        return features


class SegformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SegformerEncoder()

    def forward(self, x):
        return self.encoder(x)


class SegformerMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, DECODER_HIDDEN)

    def forward(self, x):
        return self.proj(x.flatten(2).transpose(1, 2))


class SegformerDecodeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_c = nn.ModuleList([SegformerMLP(c) for c in HIDDEN_SIZES])
        self.linear_fuse = nn.Conv2d(DECODER_HIDDEN * 4, DECODER_HIDDEN, 1, bias=False)
        self.batch_norm = nn.BatchNorm2d(DECODER_HIDDEN)
        self.classifier = nn.Conv2d(DECODER_HIDDEN, NUM_CLASSES, 1)

    def forward(self, features):
        b = features[0].shape[0]
        target_size = features[0].shape[2:]
        outs = []
        for feat, mlp in zip(features, self.linear_c):
            h, w = feat.shape[2], feat.shape[3]
            x = mlp(feat).permute(0, 2, 1).reshape(b, -1, h, w)
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            outs.append(x)
        x = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        x = F.relu(self.batch_norm(x))
        return self.classifier(x)


class SegformerForSemanticSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.segformer = SegformerModel()
        self.decode_head = SegformerDecodeHead()

    def forward(self, x):
        return self.decode_head(self.segformer(x))


# ── Loading & inference ─────────────────────────────────────────────────────────
def load_segformer(weights_path: str, device) -> SegformerForSemanticSegmentation:
    """Build SegFormer-B4 and load weights onto *device* (always float32)."""
    model = SegformerForSemanticSegmentation()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    missing, _ = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing:
        raise RuntimeError(f"Missing SegFormer weights for: {missing[:8]} ...")
    return model.to(device=device, dtype=torch.float32).eval()


def _preprocess(image: Image.Image, inference_size: int):
    orig_w, orig_h = image.size
    if inference_size and inference_size > 0:
        scale = inference_size / min(orig_w, orig_h)
        resized = image.resize(
            (max(1, round(orig_w * scale)), max(1, round(orig_h * scale))), Image.BILINEAR)
    else:
        resized = image
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGE_MEAN, dtype=np.float32)) / np.array(IMAGE_STD, dtype=np.float32)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, (orig_w, orig_h)


@torch.no_grad()
def infer_semantic(model, image: Image.Image, device, inference_size: int = 512) -> np.ndarray:
    """Return an HxW int32 class-id map at the *original* image resolution.

    SegFormer accepts arbitrary input sizes; logits are bilinearly upsampled
    back to the source resolution before argmax, so the result always matches
    the input image dimensions.
    """
    tensor, (orig_w, orig_h) = _preprocess(image, inference_size)
    tensor = tensor.to(device=device, dtype=torch.float32)
    logits = model(tensor)
    logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    return logits.argmax(dim=1).squeeze(0).to("cpu").numpy().astype(np.int32)


def colorize(seg: np.ndarray) -> np.ndarray:
    """Map an HxW class-id array to an HxWx3 uint8 RGB colour mask."""
    return ADE20K_PALETTE[seg]

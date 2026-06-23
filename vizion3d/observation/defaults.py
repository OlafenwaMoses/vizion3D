"""Defaults and semantic priors for ScaleObservation.

How these tables were generated
-------------------------------
ScaleObservation uses two distinct kinds of per-class knowledge, derived two
different ways:

1. **Size priors** (``SCALE_SIZE_PRIORS_M`` = COCO + YOLOE tables below). These
   are *hand-authored* broad physical priors, NOT fitted from any dataset. Each
   entry is ``{dimension: (mean_m, sigma_m), ..., "r": reliability}`` where
   ``mean_m``/``sigma_m`` are a real-world size estimate and its spread in
   metres, and ``r`` is a coarse per-class trust weight. Means/sigmas are taken
   from public reference catalogues (anthropometric stature tables, furniture
   and appliance size guides, common product specifications); the per-class
   ``# Source:`` comments record where each came from. Sigmas are intentionally
   wide -- these are size *priors*, not exact dimensions.

2. **Calibration corrections** (``CALIBRATED_SCALE_CORRECTION_BY_LABEL_DIM``).
   These ARE learned from data -- see the comment on that table for the method
   and the reproduction script.

The per-dimension reliability weights (``DIMENSION_RELIABILITY_BY_LABEL``) are
also hand-tuned: they encode which axes of a class are stable enough to drive
scene scale (e.g. a person's height is trustworthy, their depth is not).
"""

from __future__ import annotations

# COCO-aligned size priors, in metres. Intentionally broad physical priors, not
# exact object dimensions. Each value is ``(mean_m, sigma_m)``; ``r`` is the
# per-class reliability weight. Sources are cited per class below.
COCO_SIZE_PRIORS_M: dict[str, dict[str, object]] = {
    # Source: CDC/NCHS adult stature references for height; width/depth are broad
    # body-envelope priors, not a fixed anthropometric standard.
    "person": {"height": (1.70, 0.15), "width": (0.45, 0.10), "depth": (0.30, 0.10), "r": 0.80},
    # Source: BIFMA/ergonomic chair ranges and common dining/office chair product dimensions.
    "chair": {"height": (0.85, 0.18), "width": (0.50, 0.12), "depth": (0.55, 0.14), "r": 0.70},
    # Source: Dimensions.com sofa/couch references and common 2-3 seat sofa product ranges.
    "couch": {"height": (0.85, 0.18), "width": (2.00, 0.45), "depth": (0.90, 0.20), "r": 0.65},
    # Source: Dimensions.com queen bed / Sleep Foundation queen mattress dimensions;
    # height includes a broad mattress/frame allowance.
    "bed": {"height": (0.60, 0.18), "width": (1.60, 0.35), "depth": (2.05, 0.30), "r": 0.75},
    # Source: Dimensions.com dining table collection and common 4-person table dimensions.
    "dining table": {
        "height": (0.75, 0.08),
        "width": (1.40, 0.40),
        "depth": (0.90, 0.25),
        "r": 0.65,
    },
    # Source: Rempros/Angi toilet dimension guides; height/depth cover tank and bowl envelope.
    "toilet": {"height": (0.75, 0.10), "width": (0.38, 0.08), "depth": (0.70, 0.12), "r": 0.80},
    # Source: Dimensions.com TV display references and common 43-55 inch TV sizes.
    "tv": {"height": (0.65, 0.25), "width": (1.10, 0.40), "depth": (0.08, 0.05), "r": 0.60},
    # Source: common 13-15 inch laptop product specifications; very weak height prior.
    "laptop": {"height": (0.02, 0.01), "width": (0.34, 0.06), "depth": (0.24, 0.04), "r": 0.45},
    # Source: common full-size keyboard specifications, approximately 17 x 5.5 inches.
    "keyboard": {"height": (0.03, 0.01), "width": (0.43, 0.08), "depth": (0.14, 0.04), "r": 0.45},
    # Source: common desktop mouse product specifications.
    "mouse": {"height": (0.04, 0.02), "width": (0.065, 0.02), "depth": (0.11, 0.03), "r": 0.35},
    # Source: ISO/US common book trim sizes; intentionally broad due to high variation.
    "book": {"height": (0.03, 0.02), "width": (0.18, 0.08), "depth": (0.25, 0.08), "r": 0.35},
    # Source: RTINGS refrigerator size guide and common full-size fridge product ranges.
    "refrigerator": {
        "height": (1.70, 0.25),
        "width": (0.75, 0.15),
        "depth": (0.75, 0.15),
        "r": 0.80,
    },
    # Source: KitchenAid/Wayfair microwave size guides; typical countertop/OTR envelope.
    "microwave": {"height": (0.30, 0.08), "width": (0.50, 0.10), "depth": (0.40, 0.08), "r": 0.65},
    # Source: common 24 inch built-in/range oven product dimensions.
    "oven": {"height": (0.75, 0.15), "width": (0.60, 0.10), "depth": (0.60, 0.10), "r": 0.70},
    # Source: Dimensions.com kitchen sink collection and common 22 x 30 inch sink guides.
    "sink": {"height": (0.20, 0.10), "width": (0.55, 0.18), "depth": (0.45, 0.15), "r": 0.45},
    # Source: common decorative vase product dimensions; high variance, low reliability.
    "vase": {"height": (0.30, 0.18), "width": (0.16, 0.10), "depth": (0.16, 0.10), "r": 0.30},
    # Source: common beverage bottle dimensions; high category variance.
    "bottle": {"height": (0.25, 0.12), "width": (0.08, 0.04), "depth": (0.08, 0.04), "r": 0.35},
    # Source: Dimensions.com coffee mug/cup references and common mug product dimensions.
    "cup": {"height": (0.10, 0.04), "width": (0.08, 0.03), "depth": (0.08, 0.03), "r": 0.30},
    # Source: common cereal/soup bowl product dimensions; high category variance.
    "bowl": {"height": (0.08, 0.04), "width": (0.18, 0.08), "depth": (0.18, 0.08), "r": 0.30},
    # Source: common indoor potted plant product ranges; intentionally weak prior.
    "potted plant": {
        "height": (0.70, 0.45),
        "width": (0.45, 0.30),
        "depth": (0.45, 0.30),
        "r": 0.25,
    },
}

DEFAULT_DIMENSION_RELIABILITY = {"height": 0.75, "width": 0.65, "depth": 0.35}

# Hand-tuned per-class/per-dimension trust weights in [0, 1]. The prior means
# above stay as broad size references; these weights control whether a given
# axis is stable enough to drive scene scale (e.g. person height is reliable,
# person depth is not; a tv's thin depth is near-useless).
DIMENSION_RELIABILITY_BY_LABEL: dict[str, dict[str, float]] = {
    "person": {"height": 0.90, "width": 0.25, "depth": 0.10},
    "chair": {"height": 0.65, "width": 0.55, "depth": 0.45},
    "couch": {"height": 0.35, "width": 0.85, "depth": 0.65},
    "bed": {"height": 0.30, "width": 0.80, "depth": 0.90},
    "dining table": {"height": 0.85, "width": 0.55, "depth": 0.45},
    "toilet": {"height": 0.75, "width": 0.55, "depth": 0.65},
    "tv": {"height": 0.75, "width": 0.85, "depth": 0.05},
    "laptop": {"height": 0.05, "width": 0.35, "depth": 0.30},
    "keyboard": {"height": 0.05, "width": 0.35, "depth": 0.20},
    "mouse": {"height": 0.05, "width": 0.15, "depth": 0.10},
    "book": {"height": 0.05, "width": 0.20, "depth": 0.20},
    "refrigerator": {"height": 0.90, "width": 0.75, "depth": 0.70},
    "microwave": {"height": 0.55, "width": 0.65, "depth": 0.55},
    "oven": {"height": 0.65, "width": 0.65, "depth": 0.55},
    "sink": {"height": 0.20, "width": 0.45, "depth": 0.40},
    "vase": {"height": 0.20, "width": 0.15, "depth": 0.15},
    "bottle": {"height": 0.20, "width": 0.10, "depth": 0.10},
    "cup": {"height": 0.15, "width": 0.10, "depth": 0.10},
    "bowl": {"height": 0.10, "width": 0.15, "depth": 0.15},
    "potted plant": {"height": 0.15, "width": 0.10, "depth": 0.10},
}

COCO_PRIOR_LABELS = frozenset(COCO_SIZE_PRIORS_M)


# Expanded prompt-free YOLOE size priors, in metres. Same authoring method and
# format as COCO_SIZE_PRIORS_M above: hand-set ``(mean_m, sigma_m)`` from common
# furniture/fixture/appliance/electronics product-dimension references, not
# fitted from data. These are consumed only when ObjectMaskAnnotation3D is run
# with the YOLOE prompt-free checkpoint, which emits this wider label set.
# NOTE: these classes have no entry in CALIBRATED_SCALE_CORRECTION_BY_LABEL_DIM
# yet, so their calibration factor defaults to 1.0 (uncalibrated).
YOLOE_SIZE_PRIORS_M: dict[str, dict[str, object]] = {
    "armchair": {"height": (0.90, 0.18), "width": (0.78, 0.22), "depth": (0.82, 0.20), "r": 0.62},
    "office chair": {
        "height": (0.95, 0.18),
        "width": (0.62, 0.16),
        "depth": (0.62, 0.16),
        "r": 0.68,
    },
    "stool": {"height": (0.55, 0.18), "width": (0.38, 0.12), "depth": (0.38, 0.12), "r": 0.52},
    "bench": {"height": (0.48, 0.12), "width": (1.30, 0.45), "depth": (0.45, 0.16), "r": 0.58},
    "desk": {"height": (0.75, 0.08), "width": (1.20, 0.35), "depth": (0.65, 0.18), "r": 0.70},
    "office desk": {
        "height": (0.75, 0.08),
        "width": (1.40, 0.35),
        "depth": (0.70, 0.18),
        "r": 0.70,
    },
    "computer desk": {
        "height": (0.75, 0.08),
        "width": (1.20, 0.35),
        "depth": (0.65, 0.18),
        "r": 0.68,
    },
    "side table": {"height": (0.55, 0.12), "width": (0.55, 0.18), "depth": (0.45, 0.16), "r": 0.48},
    "coffee table": {
        "height": (0.42, 0.10),
        "width": (1.05, 0.35),
        "depth": (0.55, 0.20),
        "r": 0.52,
    },
    "cabinet": {"height": (1.20, 0.45), "width": (0.85, 0.35), "depth": (0.45, 0.18), "r": 0.60},
    "file cabinet": {
        "height": (1.05, 0.35),
        "width": (0.45, 0.12),
        "depth": (0.60, 0.12),
        "r": 0.62,
    },
    "kitchen cabinet": {
        "height": (0.85, 0.20),
        "width": (0.75, 0.35),
        "depth": (0.55, 0.12),
        "r": 0.56,
    },
    "bookshelf": {"height": (1.60, 0.45), "width": (0.85, 0.35), "depth": (0.32, 0.10), "r": 0.62},
    "bookcase": {"height": (1.60, 0.45), "width": (0.85, 0.35), "depth": (0.32, 0.10), "r": 0.62},
    "shelf": {"height": (1.20, 0.55), "width": (0.85, 0.40), "depth": (0.30, 0.12), "r": 0.45},
    "dresser": {"height": (0.95, 0.25), "width": (1.20, 0.35), "depth": (0.50, 0.12), "r": 0.58},
    "nightstand": {"height": (0.60, 0.15), "width": (0.50, 0.14), "depth": (0.42, 0.12), "r": 0.52},
    "door": {"height": (2.03, 0.08), "width": (0.82, 0.12), "depth": (0.04, 0.03), "r": 0.72},
    "window": {"height": (1.20, 0.45), "width": (1.00, 0.45), "depth": (0.05, 0.04), "r": 0.42},
    "mirror": {"height": (0.90, 0.40), "width": (0.60, 0.25), "depth": (0.04, 0.03), "r": 0.42},
    "lamp": {"height": (0.55, 0.25), "width": (0.25, 0.12), "depth": (0.25, 0.12), "r": 0.30},
    "table lamp": {"height": (0.55, 0.18), "width": (0.28, 0.10), "depth": (0.28, 0.10), "r": 0.38},
    "monitor": {"height": (0.36, 0.10), "width": (0.58, 0.16), "depth": (0.06, 0.04), "r": 0.62},
    "computer monitor": {
        "height": (0.36, 0.10),
        "width": (0.58, 0.16),
        "depth": (0.06, 0.04),
        "r": 0.62,
    },
    "computer": {"height": (0.40, 0.18), "width": (0.18, 0.08), "depth": (0.42, 0.14), "r": 0.42},
    "desktop computer": {
        "height": (0.42, 0.18),
        "width": (0.20, 0.08),
        "depth": (0.45, 0.14),
        "r": 0.45,
    },
    "printer": {"height": (0.22, 0.10), "width": (0.45, 0.14), "depth": (0.36, 0.12), "r": 0.45},
    "phone": {"height": (0.025, 0.02), "width": (0.08, 0.03), "depth": (0.16, 0.08), "r": 0.18},
    "smartphone": {
        "height": (0.01, 0.005),
        "width": (0.075, 0.015),
        "depth": (0.15, 0.03),
        "r": 0.18,
    },
    "tablet": {"height": (0.01, 0.005), "width": (0.17, 0.04), "depth": (0.24, 0.05), "r": 0.20},
    "remote": {"height": (0.02, 0.01), "width": (0.05, 0.02), "depth": (0.18, 0.06), "r": 0.18},
    "toaster": {"height": (0.20, 0.06), "width": (0.30, 0.08), "depth": (0.20, 0.06), "r": 0.42},
    "blender": {"height": (0.40, 0.10), "width": (0.18, 0.06), "depth": (0.18, 0.06), "r": 0.32},
    "coffee machine": {
        "height": (0.35, 0.12),
        "width": (0.25, 0.10),
        "depth": (0.35, 0.12),
        "r": 0.36,
    },
    "dish washer": {
        "height": (0.86, 0.05),
        "width": (0.60, 0.05),
        "depth": (0.60, 0.05),
        "r": 0.65,
    },
    "washing machine": {
        "height": (0.85, 0.08),
        "width": (0.60, 0.06),
        "depth": (0.60, 0.08),
        "r": 0.65,
    },
    "trash bin": {"height": (0.65, 0.25), "width": (0.35, 0.15), "depth": (0.35, 0.15), "r": 0.38},
    "waste container": {
        "height": (0.65, 0.25),
        "width": (0.35, 0.15),
        "depth": (0.35, 0.15),
        "r": 0.38,
    },
    "backpack": {"height": (0.45, 0.12), "width": (0.32, 0.10), "depth": (0.18, 0.08), "r": 0.35},
    "suitcase": {"height": (0.65, 0.18), "width": (0.42, 0.12), "depth": (0.25, 0.10), "r": 0.42},
    "luggage": {"height": (0.65, 0.20), "width": (0.42, 0.14), "depth": (0.25, 0.10), "r": 0.38},
    "pillow": {"height": (0.12, 0.05), "width": (0.65, 0.18), "depth": (0.45, 0.14), "r": 0.28},
    "mattress": {"height": (0.28, 0.08), "width": (1.50, 0.38), "depth": (2.00, 0.35), "r": 0.55},
    "basket": {"height": (0.28, 0.12), "width": (0.38, 0.16), "depth": (0.32, 0.14), "r": 0.28},
    "bucket": {"height": (0.30, 0.10), "width": (0.28, 0.08), "depth": (0.28, 0.08), "r": 0.28},
    "box": {"height": (0.35, 0.20), "width": (0.45, 0.25), "depth": (0.35, 0.20), "r": 0.22},
    "plant": {"height": (0.70, 0.45), "width": (0.45, 0.30), "depth": (0.45, 0.30), "r": 0.25},
    "houseplant": {"height": (0.65, 0.38), "width": (0.40, 0.24), "depth": (0.40, 0.24), "r": 0.28},
    "faucet": {"height": (0.20, 0.08), "width": (0.16, 0.08), "depth": (0.18, 0.08), "r": 0.20},
    "shower": {"height": (2.00, 0.25), "width": (0.85, 0.25), "depth": (0.85, 0.25), "r": 0.35},
    "bathtub": {"height": (0.55, 0.12), "width": (0.76, 0.10), "depth": (1.55, 0.20), "r": 0.48},
}


SIZE_PRIOR_ALIASES: dict[str, str] = {
    "fridge": "refrigerator",
    "television": "tv",
    "screen": "monitor",
    "display": "monitor",
    "computer screen": "monitor",
    "dinning table": "dining table",
    "kitchen table": "dining table",
    "round table": "dining table",
    "cocktail table": "coffee table",
    "writing desk": "desk",
    "table": "dining table",
    "sofa": "couch",
    "loveseat": "couch",
    "bean bag chair": "armchair",
    "folding chair": "chair",
    "swivel chair": "office chair",
    "bar stool": "stool",
    "step stool": "stool",
    "bathroom sink": "sink",
    "kitchen sink": "sink",
    "stove": "oven",
    "gas stove": "oven",
    "dishwasher": "dish washer",
    "washer": "washing machine",
    "laundry basket": "basket",
    "garbage": "trash bin",
    "bin": "trash bin",
    "garbage bin": "trash bin",
    "tv cabinet": "cabinet",
    "side cabinet": "cabinet",
    "bathroom cabinet": "cabinet",
    "bucket cabinet": "cabinet",
    "book shelf": "bookshelf",
    "supermarket shelf": "shelf",
    "table lamp": "table lamp",
    "cell phone": "smartphone",
    "mobile phone": "smartphone",
    "iphone": "smartphone",
    "tablet computer": "tablet",
    "paper cup": "cup",
    "coffee cup": "cup",
    "mug": "cup",
    "glass bottle": "bottle",
    "wine bottle": "bottle",
    "mixing bowl": "bowl",
    "glass bowl": "bowl",
    "salad bowl": "bowl",
    "soup bowl": "bowl",
    "toilet seat": "toilet",
    "toilet bowl": "toilet",
    "potted plant": "potted plant",
    "flower pot": "potted plant",
}


SCALE_SIZE_PRIORS_M: dict[str, dict[str, object]] = {
    **COCO_SIZE_PRIORS_M,
    **YOLOE_SIZE_PRIORS_M,
}

PRIOR_SOURCE_BY_LABEL: dict[str, str] = {
    **{label: "coco" for label in COCO_SIZE_PRIORS_M},
    **{label: "yoloe_pf" for label in YOLOE_SIZE_PRIORS_M},
}


YOLOE_DIMENSION_RELIABILITY_BY_LABEL: dict[str, dict[str, float]] = {
    "armchair": {"height": 0.55, "width": 0.70, "depth": 0.65},
    "office chair": {"height": 0.70, "width": 0.55, "depth": 0.45},
    "stool": {"height": 0.75, "width": 0.35, "depth": 0.30},
    "bench": {"height": 0.55, "width": 0.75, "depth": 0.35},
    "desk": {"height": 0.90, "width": 0.50, "depth": 0.45},
    "office desk": {"height": 0.90, "width": 0.50, "depth": 0.45},
    "computer desk": {"height": 0.90, "width": 0.48, "depth": 0.42},
    "side table": {"height": 0.70, "width": 0.40, "depth": 0.35},
    "coffee table": {"height": 0.80, "width": 0.50, "depth": 0.45},
    "cabinet": {"height": 0.65, "width": 0.55, "depth": 0.45},
    "file cabinet": {"height": 0.75, "width": 0.55, "depth": 0.55},
    "kitchen cabinet": {"height": 0.65, "width": 0.45, "depth": 0.55},
    "bookshelf": {"height": 0.80, "width": 0.55, "depth": 0.35},
    "bookcase": {"height": 0.80, "width": 0.55, "depth": 0.35},
    "shelf": {"height": 0.35, "width": 0.35, "depth": 0.25},
    "dresser": {"height": 0.60, "width": 0.65, "depth": 0.50},
    "nightstand": {"height": 0.70, "width": 0.45, "depth": 0.40},
    "door": {"height": 0.95, "width": 0.80, "depth": 0.05},
    "window": {"height": 0.35, "width": 0.35, "depth": 0.05},
    "mirror": {"height": 0.35, "width": 0.35, "depth": 0.05},
    "lamp": {"height": 0.35, "width": 0.20, "depth": 0.20},
    "table lamp": {"height": 0.45, "width": 0.25, "depth": 0.25},
    "monitor": {"height": 0.80, "width": 0.85, "depth": 0.05},
    "computer monitor": {"height": 0.80, "width": 0.85, "depth": 0.05},
    "computer": {"height": 0.35, "width": 0.25, "depth": 0.35},
    "desktop computer": {"height": 0.40, "width": 0.25, "depth": 0.40},
    "printer": {"height": 0.45, "width": 0.55, "depth": 0.45},
    "phone": {"height": 0.03, "width": 0.10, "depth": 0.12},
    "smartphone": {"height": 0.02, "width": 0.10, "depth": 0.12},
    "tablet": {"height": 0.02, "width": 0.15, "depth": 0.18},
    "remote": {"height": 0.05, "width": 0.10, "depth": 0.20},
    "toaster": {"height": 0.45, "width": 0.45, "depth": 0.35},
    "blender": {"height": 0.55, "width": 0.25, "depth": 0.25},
    "coffee machine": {"height": 0.55, "width": 0.35, "depth": 0.45},
    "dish washer": {"height": 0.90, "width": 0.80, "depth": 0.80},
    "washing machine": {"height": 0.90, "width": 0.80, "depth": 0.80},
    "trash bin": {"height": 0.45, "width": 0.30, "depth": 0.30},
    "waste container": {"height": 0.45, "width": 0.30, "depth": 0.30},
    "backpack": {"height": 0.45, "width": 0.35, "depth": 0.25},
    "suitcase": {"height": 0.55, "width": 0.45, "depth": 0.30},
    "luggage": {"height": 0.50, "width": 0.40, "depth": 0.30},
    "pillow": {"height": 0.10, "width": 0.30, "depth": 0.25},
    "mattress": {"height": 0.30, "width": 0.75, "depth": 0.85},
    "basket": {"height": 0.25, "width": 0.25, "depth": 0.25},
    "bucket": {"height": 0.35, "width": 0.25, "depth": 0.25},
    "box": {"height": 0.20, "width": 0.20, "depth": 0.20},
    "plant": {"height": 0.15, "width": 0.10, "depth": 0.10},
    "houseplant": {"height": 0.18, "width": 0.12, "depth": 0.12},
    "faucet": {"height": 0.20, "width": 0.15, "depth": 0.15},
    "shower": {"height": 0.45, "width": 0.25, "depth": 0.25},
    "bathtub": {"height": 0.55, "width": 0.55, "depth": 0.75},
}

DIMENSION_RELIABILITY_BY_LABEL = {
    **DIMENSION_RELIABILITY_BY_LABEL,
    **YOLOE_DIMENSION_RELIABILITY_BY_LABEL,
}


# Learned (not hand-authored) per-class/per-dimension scale corrections, applied
# as multipliers on each candidate's proposed scale at
# scale.py:calibration_factor(). Derivation: from a full SUN RGB-D pipeline run
# (originally the first v2_scene_object_consensus pass), each accepted
# object/dimension candidate's *uncalibrated* scale was compared to the
# ground-truth dimension-specific scene scale (gt_bounds[dim] / generated_bounds[dim]);
# the per-(label, dimension) correction is the robust median of those ratios,
# shrunk toward 1.0 when class support is low. Values < 1.0 dominate because the
# monocular-depth backend systematically over-sizes objects. Classes/dimensions
# absent here default to 1.0.
#
# Calibration values are stored here so runtime scale estimation does not depend
# on an external experiment directory.
CALIBRATED_SCALE_CORRECTION_BY_LABEL_DIM: dict[str, dict[str, float]] = {
    "bed": {"depth": 0.6757},
    "book": {"height": 1.0, "width": 0.6788, "depth": 0.4039},
    "bottle": {"height": 0.6005, "width": 0.6926, "depth": 0.5758},
    "bowl": {"height": 0.9005, "width": 0.8674, "depth": 0.8759},
    "chair": {"height": 0.5693, "width": 0.6804, "depth": 0.5350},
    "couch": {"height": 0.8647, "width": 0.6511, "depth": 0.6479},
    "cup": {"height": 0.9180, "width": 0.8010, "depth": 0.8179},
    "dining table": {"height": 0.6521, "width": 0.5367, "depth": 0.7159},
    "keyboard": {"width": 0.6101, "depth": 0.8980},
    "laptop": {"width": 0.8395},
    "microwave": {"height": 0.8466, "width": 0.7965, "depth": 0.6279},
    "oven": {"width": 0.9099},
    "person": {"height": 0.7114, "width": 0.7749},
    "potted plant": {"height": 0.6073, "width": 0.5768, "depth": 0.6538},
    "refrigerator": {"height": 0.7629, "width": 0.8662, "depth": 0.7342},
    "sink": {"height": 0.7884, "width": 0.7406, "depth": 0.5623},
    "toilet": {"height": 0.7675},
    "tv": {"height": 0.4586, "width": 0.2846},
    "vase": {"height": 0.8664, "width": 0.8110, "depth": 0.9268},
}

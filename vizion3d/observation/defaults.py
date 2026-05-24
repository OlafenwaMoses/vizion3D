"""Defaults and semantic priors for ScaleObservation."""

from __future__ import annotations

COCO_SIZE_PRIORS_M: dict[str, dict[str, object]] = {
    "person": {"height": (1.70, 0.15), "width": (0.45, 0.10), "depth": (0.30, 0.10), "r": 0.80},
    "chair": {"height": (0.85, 0.18), "width": (0.50, 0.12), "depth": (0.55, 0.14), "r": 0.70},
    "couch": {"height": (0.85, 0.18), "width": (2.00, 0.45), "depth": (0.90, 0.20), "r": 0.65},
    "bed": {"height": (0.60, 0.18), "width": (1.60, 0.35), "depth": (2.05, 0.30), "r": 0.75},
    "dining table": {
        "height": (0.75, 0.08),
        "width": (1.40, 0.40),
        "depth": (0.90, 0.25),
        "r": 0.65,
    },
    "toilet": {"height": (0.75, 0.10), "width": (0.38, 0.08), "depth": (0.70, 0.12), "r": 0.80},
    "tv": {"height": (0.65, 0.25), "width": (1.10, 0.40), "depth": (0.08, 0.05), "r": 0.60},
    "laptop": {"height": (0.02, 0.01), "width": (0.34, 0.06), "depth": (0.24, 0.04), "r": 0.45},
    "keyboard": {"height": (0.03, 0.01), "width": (0.43, 0.08), "depth": (0.14, 0.04), "r": 0.45},
    "mouse": {"height": (0.04, 0.02), "width": (0.065, 0.02), "depth": (0.11, 0.03), "r": 0.35},
    "book": {"height": (0.03, 0.02), "width": (0.18, 0.08), "depth": (0.25, 0.08), "r": 0.35},
    "refrigerator": {
        "height": (1.70, 0.25),
        "width": (0.75, 0.15),
        "depth": (0.75, 0.15),
        "r": 0.80,
    },
    "microwave": {"height": (0.30, 0.08), "width": (0.50, 0.10), "depth": (0.40, 0.08), "r": 0.65},
    "oven": {"height": (0.75, 0.15), "width": (0.60, 0.10), "depth": (0.60, 0.10), "r": 0.70},
    "sink": {"height": (0.20, 0.10), "width": (0.55, 0.18), "depth": (0.45, 0.15), "r": 0.45},
    "vase": {"height": (0.30, 0.18), "width": (0.16, 0.10), "depth": (0.16, 0.10), "r": 0.30},
    "bottle": {"height": (0.25, 0.12), "width": (0.08, 0.04), "depth": (0.08, 0.04), "r": 0.35},
    "cup": {"height": (0.10, 0.04), "width": (0.08, 0.03), "depth": (0.08, 0.03), "r": 0.30},
    "bowl": {"height": (0.08, 0.04), "width": (0.18, 0.08), "depth": (0.18, 0.08), "r": 0.30},
    "potted plant": {
        "height": (0.70, 0.45),
        "width": (0.45, 0.30),
        "depth": (0.45, 0.30),
        "r": 0.25,
    },
}

DEFAULT_DIMENSION_RELIABILITY = {"height": 0.75, "width": 0.65, "depth": 0.35}

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

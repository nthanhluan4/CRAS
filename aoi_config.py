"""
aoi_config.py
Persistent AOI station configuration (survives app restarts, shared by all browser sessions).
One JSON file per inspection station; values are edited from Admin > AOI parameters.
"""

import json
import os
from typing import Any, Dict

CONFIG_PATH = os.path.join("results", "config", "aoi_config.json")

DEFAULTS: Dict[str, Any] = {
    # Optical calibration (measure with a checkerboard / ruler lying on the fabric plane)
    "px_per_inch": 100.0,
    # Blob filtering / classification
    "min_area": 150,
    "aspect_ratio": 2.8,
    # Inspection ROI across the fabric width, in original-image pixel columns (roll runs vertically).
    # roi_right_px = 0 means "up to the right image border".
    "roi_left_px": 0,
    "roi_right_px": 0,
    # Defects closer than this to the ROI side edges (selvedge) are ignored, as is usual in ASTM D5430 practice
    "selvedge_margin_in": 1.0,
    # Roll geometry
    "frame_overlap_in": 0.0,      # overlap between consecutive frames along the roll
    "usable_width_in": 0.0,       # cuttable fabric width for points/100yd2; 0 = derive from ROI
    # Buyer acceptance (points per 100 square yards, whole roll)
    "grade_a_limit": 20.0,
    "acceptance_limit": 40.0,
    # Hardware
    "backlight_available": False,  # when False, suspected holes are routed to manual review
}


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                stored = json.load(f)
            cfg.update({k: stored[k] for k in DEFAULTS if k in stored})
        except (OSError, ValueError):
            pass
    return cfg


def save_config(cfg: Dict[str, Any], path: str = CONFIG_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean = {k: cfg.get(k, v) for k, v in DEFAULTS.items()}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

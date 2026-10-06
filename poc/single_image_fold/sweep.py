"""
sweep.py  (POC) - threshold sweep: how well can a detector separate folds from fabric texture at ALL?

For each detector score map and each threshold we measure, over all labelled images:
  recall      : required folds found (>= 50 % of length within TOL px of detected pixels)
  precision   : share of detected fabric pixels that lie within TOL px of a labelled fold (required or optional)
  false area  : detected fabric pixels farther than TOL from any labelled fold, in % of the fabric area
This replaces "counting separate false components", which hides false detections that touch a real fold.

    python poc/single_image_fold/sweep.py
"""

import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import baseline_current  # noqa: E402
import detector_mf  # noqa: E402
from evaluate import TOL, label_mask, polyline_points  # noqa: E402


def score_maps(path):
    pil = Image.open(path).convert("RGB")
    gray = np.array(pil.convert("L"))
    H, W = gray.shape
    cur = baseline_current.detect_crease_defect(pil)[3]
    res = detector_mf.detect(gray)
    mf = cv2.resize(res.zmax, (W, H), interpolation=cv2.INTER_LINEAR)
    fabric = cv2.resize(res.valid.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return {"current": cur, "mf_z": mf}, fabric, res.threshold


def main():
    with open(os.path.join(HERE, "labels.json"), encoding="utf-8") as f:
        labels = json.load(f)
    data = []
    for fname, ann in labels["images"].items():
        maps, fabric, thr = score_maps(os.path.join(labels["image_dir"], fname))
        H, W = fabric.shape
        for x0, y0, x1, y1 in ann.get("ignore", []):
            fabric[y0:y1, x0:x1] = False
        band = label_mask((H, W), ann["required"] + ann.get("optional", []), 2 * TOL + 1)
        req = []
        for fo in ann["required"]:
            p = polyline_points(fo["polyline"])
            req.append((np.clip(p[:, 0].astype(int), 0, W - 1), np.clip(p[:, 1].astype(int), 0, H - 1)))
        data.append((maps, fabric, band, req))
        print(f"loaded {fname} (NFA threshold for mf_z would be {thr:.2f})")

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOL + 1, 2 * TOL + 1))
    grids = {"current": [0.10, 0.18, 0.25, 0.40, 0.60, 0.80, 0.95],
             "mf_z": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]}
    for det, thresholds in grids.items():
        print(f"\n--- {det}")
        print(f"{'threshold':>10} {'recall':>8} {'precision':>10} {'false area %':>13}")
        for t in thresholds:
            found = total = 0
            det_px = in_band = false_px = fab_px = 0
            for maps, fabric, band, req in data:
                m = (maps[det] >= t) & fabric
                near = cv2.dilate(m.astype(np.uint8), kern) > 0
                for xi, yi in req:
                    total += 1
                    found += near[yi, xi].mean() >= 0.5
                det_px += int(m.sum())
                in_band += int((m & band).sum())
                false_px += int((m & ~band).sum())
                fab_px += int(fabric.sum())
            prec = in_band / det_px if det_px else float("nan")
            print(f"{t:>10.2f} {found:>4}/{total:<3} {prec:>10.0%} {100.0 * false_px / fab_px:>12.2f}%")


if __name__ == "__main__":
    main()

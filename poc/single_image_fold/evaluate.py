"""
evaluate.py  (POC) - objective comparison of fold detectors on labelled single flat images.

    python poc/single_image_fold/evaluate.py [--labels labels.json] [--out out_eval]

Metrics (original-image pixels):
  * a REQUIRED fold is "found" when >= 50 % of its length lies within TOL px of detected pixels
    (coverage is reported too);
  * a FALSE ALARM is a detected component (area >= MIN_AREA) lying entirely farther than TOL from every
    labelled fold (required or optional) and outside the ignore boxes.
Detectors:
  * current   : clone of the production crease map (baseline_current.py) at the crease-mode (0.18)
                and ensemble (0.40) thresholds;
  * mf_nfa    : whitened matched filter with a-contrario threshold (detector_mf.py), epsilon = 1.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import baseline_current  # noqa: E402
import detector_mf  # noqa: E402

TOL = 30          # px: how far a detection may be from the labelled fold line
MIN_AREA = 150    # px: same as the production profiler's min_area


def polyline_points(poly, step=5.0):
    pts = []
    for (x0, y0), (x1, y1) in zip(poly[:-1], poly[1:]):
        n = max(int(np.hypot(x1 - x0, y1 - y0) / step), 1)
        for t in np.linspace(0, 1, n, endpoint=False):
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    pts.append(tuple(poly[-1]))
    return np.array(pts, np.float32)


def label_mask(shape, folds, thickness):
    m = np.zeros(shape, np.uint8)
    for f in folds:
        cv2.polylines(m, [np.round(np.array(f["polyline"])).astype(np.int32)], False, 1, thickness)
    return m > 0


def evaluate_mask(mask, ann, shape):
    h, w = shape
    ignore = np.zeros(shape, bool)
    for x0, y0, x1, y1 in ann.get("ignore", []):
        ignore[y0:y1, x0:x1] = True
    mask = mask & ~ignore
    near = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOL + 1, 2 * TOL + 1))) > 0

    folds = []
    for f in ann["required"]:
        pts = polyline_points(f["polyline"])
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        cov = float(near[yi, xi].mean())
        folds.append({"name": f["name"], "coverage": cov, "found": cov >= 0.5})

    all_labels = label_mask(shape, ann["required"] + ann.get("optional", []), 2 * TOL + 1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    false_alarms = 0
    fa_area = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        comp = lab == i
        if not (comp & all_labels).any():
            false_alarms += 1
            fa_area += int(stats[i, cv2.CC_STAT_AREA])
    return folds, false_alarms, fa_area


def clean_small(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_AREA
    return keep[lab]


def run_detectors(path):
    pil = Image.open(path).convert("RGB")
    gray = np.array(pil.convert("L"))
    out = {}
    t = time.time()
    cmap = baseline_current.detect_crease_defect(pil)[3]
    t_cur = time.time() - t
    out["current@0.18"] = (clean_small(cmap >= 0.18), t_cur, None)
    out["current@0.40"] = (clean_small(cmap >= 0.40), t_cur, None)
    t = time.time()
    res = detector_mf.detect(gray)
    out["mf_nfa"] = (clean_small(res.mask), time.time() - t, res)
    return pil, out


def overlay(pil, mask, ann, title):
    img = np.array(pil).copy()
    img = np.clip(img.astype(np.float32) * 2.2, 0, 255).astype(np.uint8)   # brighten dark fabric for display
    img[mask] = (0.45 * img[mask] + 0.55 * np.array([255, 60, 60])).astype(np.uint8)
    for f in ann["required"]:
        cv2.polylines(img, [np.array(f["polyline"], np.int32)], False, (60, 255, 60), 6)
    for f in ann.get("optional", []):
        cv2.polylines(img, [np.array(f["polyline"], np.int32)], False, (255, 220, 0), 4)
    for x0, y0, x1, y1 in ann.get("ignore", []):
        cv2.rectangle(img, (x0, y0), (x1, y1), (120, 120, 120), 6)
    img = cv2.resize(img, (512, 612))
    cv2.putText(img, title, (6, 22), 0, 0.6, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join(HERE, "labels.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "out_eval"))
    args = ap.parse_args()
    with open(args.labels, encoding="utf-8") as f:
        labels = json.load(f)
    os.makedirs(args.out, exist_ok=True)

    summary = {}
    rows_img = []
    for fname, ann in labels["images"].items():
        path = os.path.join(labels["image_dir"], fname)
        pil, dets = run_detectors(path)
        shape = (pil.size[1], pil.size[0])
        tiles = []
        print(f"\n=== {fname}")
        for det_name, (mask, secs, extra) in dets.items():
            folds, fa, fa_area = evaluate_mask(mask, ann, shape)
            s = summary.setdefault(det_name, {"found": 0, "total": 0, "fa": 0, "secs": []})
            s["found"] += sum(f["found"] for f in folds)
            s["total"] += len(folds)
            s["fa"] += fa
            s["secs"].append(secs)
            cov = "  ".join(f"{f['name'].split(' ')[0]}={f['coverage']:.0%}" for f in folds)
            print(f"  {det_name:13} found {sum(f['found'] for f in folds)}/{len(folds)} | false alarms {fa} | coverage: {cov}")
            if extra is not None:
                print(f"  {'':13} {extra.notes[0]}")
                # margin: median max-z along each required fold vs threshold
                zfull = cv2.resize(extra.zmax, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
                zs = []
                for f in ann["required"]:
                    pts = polyline_points(f["polyline"])
                    band = cv2.dilate(zfull, cv2.getStructuringElement(cv2.MORPH_RECT, (TOL, TOL)))
                    xi = np.clip(pts[:, 0].astype(int), 0, shape[1] - 1)
                    yi = np.clip(pts[:, 1].astype(int), 0, shape[0] - 1)
                    zs.append(f"{f['name'].split(' ')[0]}={np.median(band[yi, xi]):.1f}")
                print(f"  {'':13} median z along folds (threshold {extra.threshold:.1f}): " + "  ".join(zs))
            tiles.append(overlay(pil, mask, ann, f"{fname[-14:-8]} {det_name}"))
        cv2.imwrite(os.path.join(args.out, f"{fname[:-4]}_eval.jpg"), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))
        rows_img.append(np.hstack(tiles))

    print("\n=== SUMMARY (required folds found / false alarms over all images)")
    for det_name, s in summary.items():
        print(f"  {det_name:13} recall {s['found']}/{s['total']} ({s['found'] / max(s['total'], 1):.0%}) | "
              f"false alarms {s['fa']} | {np.mean(s['secs']) * 1000:.0f} ms/frame")
    cv2.imwrite(os.path.join(args.out, "all_eval.jpg"), cv2.cvtColor(np.vstack(rows_img), cv2.COLOR_RGB2BGR))
    print(f"\nOverlays: {os.path.abspath(args.out)}  (green = required label, yellow = optional, red = detection)")


if __name__ == "__main__":
    main()

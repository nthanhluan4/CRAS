"""
run_benchmark.py - regression benchmark of the production AOI pipeline (app.py + defect_profiler.py).

Run it BEFORE and AFTER any algorithm change, then compare:
    python benchmark/run_benchmark.py --tag before
    ... change code ...
    python benchmark/run_benchmark.py --tag after
    python benchmark/compare.py benchmark/results/<before>.json benchmark/results/<after>.json

For every labelled defect it reports WHERE the pipeline fails, stage by stage:
    S1 map        : does any detector map reach the mode threshold on the defect?      (detector problem)
    S2 extraction : does an extracted instance cover the defect?                       (linking / merging problem)
    S3 class      : is the instance classified into an accepted category?              (classification problem)
and, per image, false alarms: instances that touch no labelled defect (outside ignore zones), plus the
share of detected pixels lying outside labelled defects (catches detections that sprawl from a real defect).
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import logging  # noqa: E402
logging.disable(logging.WARNING)
import app  # noqa: E402
import aoi_config  # noqa: E402
import defect_profiler as dp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 30                                   # px tolerance around a labelled fold line / box
MODES = {                                  # mode -> default UI threshold
    "ensemble": 0.40,
    "crease": 0.18,
    "stain": 0.45,
}


# ----------------------------------------------------------------------------------------------- geometry
def gt_region(shape, d, tol=TOL):
    m = np.zeros(shape, np.uint8)
    if "polyline" in d:
        cv2.polylines(m, [np.round(np.array(d["polyline"])).astype(np.int32)], False, 1, 2 * tol + 1)
    else:
        x, y, w, h = d["bbox"]
        t = tol // 2
        m[max(y - t, 0):y + h + t, max(x - t, 0):x + w + t] = 1
    return m > 0


def polyline_points(poly, step=5.0):
    pts = []
    for (x0, y0), (x1, y1) in zip(poly[:-1], poly[1:]):
        n = max(int(np.hypot(x1 - x0, y1 - y0) / step), 1)
        pts += [(x0 + t * (x1 - x0), y0 + t * (y1 - y0)) for t in np.linspace(0, 1, n, endpoint=False)]
    pts.append(tuple(poly[-1]))
    return np.array(pts)


def ignore_mask(shape, boxes):
    m = np.zeros(shape, bool)
    for x0, y0, x1, y1 in boxes:
        m[y0:y1, x0:x1] = True
    return m


# ----------------------------------------------------------------------------------------------- pipeline
def run_pipeline(pil, mode, thr, cfg):
    """Replicates the app's inspection path for one mode. Returns (defects, instance label image, maps)."""
    W, H = pil.size
    ppi = float(cfg["px_per_inch"])
    x0, x1 = dp.inspection_columns(W, cfg["roi_left_px"], cfg["roi_right_px"], int(round(cfg["selvedge_margin_in"] * ppi)))
    kw = dict(min_area=int(cfg["min_area"]), aspect_ratio_crease_thresh=float(cfg["aspect_ratio"]),
              pixels_per_inch=ppi, hole_needs_review=not cfg["backlight_available"])
    if mode == "ensemble":
        _, _, _, details, _ = app.run_hybrid_ensemble(pil)
        maps = {k: dp.apply_inspection_roi(v, x0, x1) for k, v in details["component_maps"].items()}
        defects, mask = dp.extract_from_component_maps(pil, maps, threshold=thr, **kw)
    else:
        m = app.detect_crease_defect(pil)[3] if mode == "crease" else app.run_stain_enhancement(pil)[3]
        maps = {mode: dp.apply_inspection_roi(m, x0, x1)}
        defects, mask = dp.extract_defect_instances(pil, maps[mode], threshold=thr, **kw)
    # instance label image: pixel -> index of the defect that owns it
    lab = np.full((H, W), -1, np.int32)
    for i, d in enumerate(defects):
        x, y, w, h = d["bbox"]
        sub = lab[y:y + h, x:x + w]
        own = (mask[y:y + h, x:x + w] > 0) & (sub < 0)
        sub[own] = i
    return defects, lab, maps


# ----------------------------------------------------------------------------------------------- scoring
def score_image(ann, defects, lab, maps, thr, shape):
    H, W = shape
    ign = ignore_mask(shape, ann.get("ignore", []))
    near = cv2.dilate((lab >= 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOL + 1, 2 * TOL + 1))) > 0
    all_gt = np.zeros(shape, bool)
    for d in ann["defects"]:
        all_gt |= gt_region(shape, d)
    fabric = ~ign

    # background level of every map (outside labelled defects) -> how much margin a defect has
    bg = {k: float(np.percentile(v[fabric & ~all_gt], 99.9)) for k, v in maps.items()}

    gts = []
    for d in ann["defects"]:
        reg = gt_region(shape, d)
        peak = {k: float(np.percentile(v[reg], 95)) for k, v in maps.items()}
        map_ok = any(p >= thr for p in peak.values())
        if "polyline" in d:
            pts = polyline_points(d["polyline"])
            xi = np.clip(pts[:, 0].astype(int), 0, W - 1)
            yi = np.clip(pts[:, 1].astype(int), 0, H - 1)
            coverage = float(near[yi, xi].mean())
            found = coverage >= 0.5
        else:
            x, y, w, h = d["bbox"]
            box = np.zeros(shape, bool)
            box[y:y + h, x:x + w] = True
            hit = (lab >= 0) & box
            coverage = float(hit.sum() / max(box.sum(), 1))
            found = hit.sum() >= min(0.1 * box.sum(), 150)
        owners = lab[reg & (lab >= 0)]
        category = None
        if owners.size:
            idx = int(np.bincount(owners).argmax())
            category = defects[idx]["category"]
        correct = bool(found and category in d["accept"])
        stage = "ok" if correct else ("S1 map" if not map_ok else ("S2 extraction" if not found else "S3 class"))
        gts.append({"id": d["id"], "type": d["type"], "required": d["required"], "found": bool(found),
                    "coverage": round(coverage, 3), "category": category, "correct": correct,
                    "fail_stage": stage, "map_peak": {k: round(v, 3) for k, v in peak.items()}})

    false_alarms = []
    fa_px = det_px = 0
    for i, d in enumerate(defects):
        own = lab == i
        n = int(own.sum())
        if n == 0 or (own & ign).sum() > 0.5 * n:
            continue
        det_px += n
        outside = own & ~all_gt & ~ign
        fa_px += int(outside.sum())
        if not (own & all_gt).any():
            false_alarms.append({"category": d["category"], "bbox": d["bbox"], "length_in": d["length_in"],
                                 "points": d["astm_points"], "source": d.get("source", "")})
    counted = [d for d in defects if not d.get("needs_review")]
    return {
        "gt": gts,
        "false_alarms": false_alarms,
        "sprawl_share": round(fa_px / det_px, 3) if det_px else 0.0,
        "n_defects": len(defects),
        "astm_points_raw": int(sum(d["astm_points"] for d in counted)),
        "map_background_p999": {k: round(v, 3) for k, v in bg.items()},
    }


def summarize(results, scope="roll"):
    out = {}
    for mode in MODES:
        req = [g for img in results.values() if img["scope"] == scope for g in img["modes"][mode]["gt"] if g["required"]]
        fa = sum(len(img["modes"][mode]["false_alarms"]) for img in results.values() if img["scope"] == scope)
        stages = {}
        for g in req:
            stages[g["fail_stage"]] = stages.get(g["fail_stage"], 0) + 1
        by_type = {}
        for g in req:
            t = by_type.setdefault(g["type"], [0, 0, 0])
            t[0] += g["found"]
            t[1] += g["correct"]
            t[2] += 1
        out[mode] = {"required": len(req), "found": sum(g["found"] for g in req),
                     "correct": sum(g["correct"] for g in req), "false_alarms": fa,
                     "fail_stages": stages, "by_type": by_type}
    return out


def print_report(results, summary):
    print("\n" + "=" * 100)
    for fname, img in results.items():
        print(f"\n{fname}  [{img['scope']}]")
        for mode, r in img["modes"].items():
            miss = [f"{g['id'].split('-')[1]}:{g['fail_stage']}({g['category']})" for g in r["gt"] if g["required"] and not g["correct"]]
            print(f"  {mode:9} defects={r['n_defects']:2d} FA={len(r['false_alarms'])} sprawl={r['sprawl_share']:.0%} "
                  f"pts={r['astm_points_raw']:3d} | not-ok: {', '.join(miss) or '-'}")
    print("\n" + "=" * 100 + "\nSUMMARY (scope = roll; required defects)")
    for mode, s in summary.items():
        types = "  ".join(f"{t}: found {v[0]}/{v[2]}, correct {v[1]}/{v[2]}" for t, v in s["by_type"].items())
        print(f"  {mode:9} found {s['found']}/{s['required']} | correct class {s['correct']}/{s['required']} | "
              f"false alarms {s['false_alarms']} | fail stages {s['fail_stages']}\n            {types}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--gt", default=os.path.join(HERE, "ground_truth.json"))
    args = ap.parse_args()
    with open(args.gt, encoding="utf-8") as f:
        G = json.load(f)
    cfg = dict(aoi_config.DEFAULTS)   # fixed reference settings: independent of the station's saved config
    results = {}
    t0 = time.time()
    for fname, ann in G["images"].items():
        pil = Image.open(os.path.join(G["image_dir"], fname)).convert("RGB")
        shape = (pil.size[1], pil.size[0])
        img = {"scope": ann["scope"], "modes": {}}
        for mode, thr in MODES.items():
            defects, lab, maps = run_pipeline(pil, mode, thr, cfg)
            img["modes"][mode] = score_image(ann, defects, lab, maps, thr, shape)
        results[fname] = img
    summary = summarize(results)
    print_report(results, summary)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"{datetime.now():%Y%m%d_%H%M%S}_{args.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"tag": args.tag, "created": datetime.now().isoformat(timespec="seconds"),
                   "config": cfg, "modes": MODES, "summary": summary, "results": results}, f, ensure_ascii=False, indent=1)
    print(f"\nSaved {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

"""
run_ps.py  (POC - standalone CLI)

Synthetic self-check:
    python run_ps.py --synthetic --out out_synth

Real 4-light capture (+ optional current front-lit image for comparison, + fold segments to measure):
    python run_ps.py --left L.png --right R.png --top T.png --bottom B.png --front F.png \
                     --lines lines.json --elevation 18 --out out_real

lines.json: [{"name": "1 horizontal fold", "seg": [x0, y0, x1, y1]}, ...]   (original-image pixels)

Outputs in --out: albedo.png, shape.png (folds), fold_strength.png, normals.png, p.png, q.png,
front.png / rake_left.png (if given), snr.csv and a printed comparison table.
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ps_core  # noqa: E402
import snr  # noqa: E402


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:  # 12/16-bit cameras: keep linearity, rescale to 0..255 float
        img = img.astype(np.float32) * (255.0 / float(np.iinfo(img.dtype).max))
    return img


def compare(images, segments, cfg, band=40):
    """SNR of every fold in: current front light, one raking light alone, and the PS shape map."""
    res = ps_core.compute({k: images[k] for k in ps_core.DIRECTIONS}, cfg)
    maps = {}
    if "front" in images:
        maps["front light (hien tai)"] = (ps_core.to_gray_float(images["front"]), None)
    maps["1 raking light (left)"] = (ps_core.to_gray_float(images["left"]), None)
    maps["1 raking light (top)"] = (ps_core.to_gray_float(images["top"]), None)
    maps["PS shape (4 lights)"] = (res.shape, res.valid)
    rows = []
    for map_name, (m, valid) in maps.items():
        for r in snr.measure_map(m, segments, valid=valid, band=band):
            rows.append({"map": map_name, "fold": r.name, "median_snr": round(r.median_snr, 2),
                         "share_snr_ge_2": round(r.share_snr_ge_2, 2), "profile_snr": round(r.profile_snr, 1)})
    return res, rows


def save_outputs(res, images, out_dir, rows):
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "albedo.png"), ps_core.to_u8(res.albedo))
    cv2.imwrite(os.path.join(out_dir, "shape.png"), ps_core.to_u8(res.shape, symmetric=True))
    cv2.imwrite(os.path.join(out_dir, "fold_strength.png"),
                cv2.applyColorMap(np.clip(ps_core.fold_strength(res.shape, res.valid) / 12 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET))
    cv2.imwrite(os.path.join(out_dir, "normals.png"), cv2.cvtColor(ps_core.normals_to_rgb(res.normals), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "p.png"), ps_core.to_u8(res.p, symmetric=True))
    cv2.imwrite(os.path.join(out_dir, "q.png"), ps_core.to_u8(res.q, symmetric=True))
    if "front" in images:
        cv2.imwrite(os.path.join(out_dir, "front.png"), ps_core.to_u8(ps_core.to_gray_float(images["front"])))
    cv2.imwrite(os.path.join(out_dir, "rake_left.png"), ps_core.to_u8(ps_core.to_gray_float(images["left"])))
    if rows:
        with open(os.path.join(out_dir, "snr.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def print_table(rows):
    if not rows:
        print("(no fold segments given -> no SNR table)")
        return
    folds = list(dict.fromkeys(r["fold"] for r in rows))
    maps = list(dict.fromkeys(r["map"] for r in rows))
    print(f"\n{'fold':24}" + "".join(f"{m[:24]:>26}" for m in maps))
    print(f"{'':24}" + "".join(f"{'SNR/px  >=2   avg':>26}" for _ in maps))
    for fold in folds:
        line = f"{fold[:24]:24}"
        for m in maps:
            r = next(x for x in rows if x["fold"] == fold and x["map"] == m)
            line += f"{r['median_snr']:>12.2f}{r['share_snr_ge_2'] * 100:>6.0f}%{r['profile_snr']:>7.1f}"
        print(line)
    print("\nSNR/px: typical visibility per position (fold vs pixel noise); >=2: share of the fold clearly visible;"
          " avg: visibility after averaging along the whole fold.")


def main():
    ap = argparse.ArgumentParser(description="Photometric stereo POC for fabric folds")
    ap.add_argument("--synthetic", action="store_true")
    for d in ps_core.DIRECTIONS + ("front",):
        ap.add_argument(f"--{d}")
    ap.add_argument("--lines", help="JSON list of {name, seg:[x0,y0,x1,y1]}")
    ap.add_argument("--elevation", type=float, default=18.0)
    ap.add_argument("--band", type=int, default=40, help="half-width (px) of the profile across a fold")
    ap.add_argument("--out", default="out_ps")
    args = ap.parse_args()

    cfg = ps_core.PSConfig(elevation_deg=args.elevation)
    if args.synthetic:
        import synthetic
        images, segments = synthetic.make_capture_set()
        band = 40
    else:
        missing = [d for d in ps_core.DIRECTIONS if not getattr(args, d)]
        if missing:
            ap.error(f"missing images for: {missing} (or use --synthetic)")
        images = {d: load_gray(getattr(args, d)) for d in ps_core.DIRECTIONS}
        if args.front:
            images["front"] = load_gray(args.front)
        segments = []
        if args.lines:
            with open(args.lines, encoding="utf-8") as f:
                segments = [(x["name"], x["seg"]) for x in json.load(f)]
        band = args.band

    if segments:
        res, rows = compare(images, segments, cfg, band=band)
    else:
        res, rows = ps_core.compute({k: images[k] for k in ps_core.DIRECTIONS}, cfg), []
    for n in res.notes:
        print("NOTE:", n)
    save_outputs(res, images, args.out, rows)
    print_table(rows)
    print(f"\nOutputs written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

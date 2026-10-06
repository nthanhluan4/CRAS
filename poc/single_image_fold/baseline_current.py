"""
baseline_current.py  (POC - CLONE, not imported from production)

Verbatim copy of the crease/fold detector currently in app.py (cloned 2026-09-25), so the POC can
measure it against labelled folds without touching production code. Do not edit to "fix" it here:
it is the reference being evaluated.
"""

from typing import Tuple

import cv2
import numpy as np
from PIL import Image


PROC_MAX_SIDE = 1024
BORDER_PAD = 8


def _prepare(pil_image: Image.Image):
    """Downscale to <=1024px and build the shared local background / texture-noise estimate."""
    np_img = np.array(pil_image)
    if np_img.ndim == 2:
        np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
    orig_h, orig_w = np_img.shape[:2]

    scale = min(float(PROC_MAX_SIDE) / max(orig_h, orig_w), 1.0)
    proc_w, proc_h = int(orig_w * scale), int(orig_h * scale)
    scaled = cv2.resize(np_img, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(scaled, cv2.COLOR_RGB2GRAY).astype(np.float32)

    ksize = int(round(proc_w * 0.045)) | 1
    sigma = ksize / 3.0
    bg_gray = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    local_sq = cv2.GaussianBlur(gray**2, (ksize, ksize), sigma)
    texture_noise = float(np.median(np.sqrt(np.maximum(local_sq - bg_gray**2, 1.0))))

    return {
        "np_img": np_img,
        "orig_size": (orig_w, orig_h),
        "scaled": scaled,
        "gray": gray,
        "bg_gray": bg_gray,
        "texture_noise": texture_noise,
    }


def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((length, length), np.uint8)
    c = (length - 1) / 2.0
    dx = np.cos(np.deg2rad(angle_deg)) * c
    dy = np.sin(np.deg2rad(angle_deg)) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, 1)
    return k


CREASE_ANGLES = tuple(range(0, 180, 15))


def _crease_anomaly_map(ctx) -> np.ndarray:
    """
    Hessian ridge/valley strength followed by directional path opening along 12 orientations,
    so only long straight-ish structures survive (creases, folds) at ANY angle,
    while short isotropic fabric grain is removed.
    """
    gray = ctx["gray"]
    proc_w = gray.shape[1]

    smooth = cv2.GaussianBlur(gray, (9, 9), 2.0)
    d2x = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=5)
    d2y = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=5)
    dxy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=5)

    trace = d2x + d2y
    det = d2x * d2y - dxy**2
    disc = np.sqrt(np.maximum(trace**2 / 4.0 - det, 0.0))
    eig1 = trace / 2.0 + disc
    eig2 = trace / 2.0 - disc
    ridge = np.maximum(-eig2, eig1)

    k_len = max(int(proc_w * 0.035), 21) | 1
    line_struct = np.zeros_like(ridge)
    for ang in CREASE_ANGLES:
        opened = cv2.morphologyEx(ridge, cv2.MORPH_OPEN, _line_kernel(k_len, ang))
        np.maximum(line_struct, opened, out=line_struct)

    crease_floor = max(ctx["texture_noise"] * 4.5, 18.0)
    sharp_map = np.clip((line_struct - crease_floor) / 45.0, 0.0, 1.0)

    # Soft folds (broad, low-contrast valleys/ridges) are invisible to the fine Hessian above
    fold_map = cv2.resize(_fold_anomaly_map(gray), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.maximum(sharp_map, fold_map)


# Soft-fold detector, working on a thumbnail with a fixed long side so parameters are in "thumbnail px".
FOLD_LONG_SIDE = 612
FOLD_S_ACROSS = 4.0      # fold half-width (~40 px wide folds on a 2448 px frame)
FOLD_S_ALONG = 22.0
FOLD_ZCLIP = 12.0
FOLD_BLOB_ZMIN = 3.0
# Line-averaging length -> score at which the fold reaches the ensemble decision level (0.40).
# Set just above the texture background measured on clean areas of the dark-fabric test frames;
# re-tune with labelled production frames.
FOLD_LEVELS = {81: 7.8, 161: 6.0, 241: 4.9}


def _fold_kernel(angle_deg: float, s_across: float, s_along: float) -> np.ndarray:
    """Elongated second-derivative-of-Gaussian: long averaging along the fold, curvature across it."""
    half = int(np.ceil(3 * max(s_across, s_along)))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    t = np.deg2rad(angle_deg)
    u = xx * np.cos(t) + yy * np.sin(t)
    v = -xx * np.sin(t) + yy * np.cos(t)
    g = np.exp(-u**2 / (2 * s_along**2) - v**2 / (2 * s_across**2))
    k = (v**2 / s_across**4 - 1 / s_across**2) * g
    k -= k.mean()
    return (k / np.abs(k).sum()).astype(np.float32)


def _line_mean_kernel(length: int, angle_deg: float) -> np.ndarray:
    k = _line_kernel(length, angle_deg).astype(np.float32)
    return k / k.sum()


def _fold_anomaly_map(gray: np.ndarray) -> np.ndarray:
    """
    Soft crease / fold detector for low-contrast folds (a few grey levels deep, tens of px wide) that sit
    below the fabric texture noise pixel-wise:
    1. elongated curvature filter at 12 orientations, turned into a robust z-score (median / MAD) per image;
    2. blob-like responses (all orientations high: stains, prints, holes) are removed;
    3. the z-score is averaged along long straight segments (81/161/241 px) at the same orientation, so
       short texture streaks fade while folds that stay straight over inches keep their score.
    Returns a 0..1 map on the thumbnail grid; 0.40 = decision level of the ensemble.
    """
    h, w = gray.shape
    sc = FOLD_LONG_SIDE / float(max(h, w))
    g = cv2.resize(gray, (max(int(w * sc), 16), max(int(h * sc), 16)), interpolation=cv2.INTER_AREA).astype(np.float32)

    zs = []
    for ang in CREASE_ANGLES:
        r = np.abs(cv2.filter2D(g, cv2.CV_32F, _fold_kernel(ang, FOLD_S_ACROSS, FOLD_S_ALONG), borderType=cv2.BORDER_REFLECT))
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med))) * 1.4826 + 1e-6
        zs.append(np.clip((r - med) / mad, 0.0, FOLD_ZCLIP))
    zs = np.stack(zs)
    blob = cv2.dilate((zs.min(axis=0) > FOLD_BLOB_ZMIN).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    zs[:, blob] = 0.0

    fold = np.zeros_like(g)
    for length, level in FOLD_LEVELS.items():
        for i, ang in enumerate(CREASE_ANGLES):
            s = cv2.filter2D(zs[i], cv2.CV_32F, _line_mean_kernel(length, ang), borderType=cv2.BORDER_REFLECT)
            # 0.40 at `level`, 0.18 (crease-mode default) at 0.9 * level, saturates at ~1.27 * level
            np.maximum(fold, np.clip(0.40 + 2.2 * (s - level) / level, 0.0, 1.0), out=fold)
    return fold


def _finalize_map(ctx, anomaly_proc: np.ndarray, pad_border: bool = True):
    """Zero the border, upscale to original size, and render score / heatmap / overlay."""
    anomaly_proc = anomaly_proc.copy()
    if pad_border:
        anomaly_proc[:BORDER_PAD, :] = 0.0
        anomaly_proc[-BORDER_PAD:, :] = 0.0
        anomaly_proc[:, :BORDER_PAD] = 0.0
        anomaly_proc[:, -BORDER_PAD:] = 0.0

    orig_w, orig_h = ctx["orig_size"]
    anomaly = cv2.resize(anomaly_proc, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    anomaly = np.clip(anomaly, 0.0, 1.0)
    score = float(np.percentile(anomaly, 99.8)) if anomaly.max() > 0 else 0.0

    heatmap_rgb = cv2.cvtColor(cv2.applyColorMap((anomaly * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(ctx["np_img"], 0.60, heatmap_rgb, 0.40, 0)
    return score, heatmap_rgb, overlay, anomaly


def detect_crease_defect(pil_image: Image.Image) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Principled Hessian Curvature Tensor with multi-orientation Directional Path Opening.
    Detects physical linear creases and folds at any angle while rejecting natural fabric grain.
    """
    ctx = _prepare(pil_image)
    return _finalize_map(ctx, _crease_anomaly_map(ctx))

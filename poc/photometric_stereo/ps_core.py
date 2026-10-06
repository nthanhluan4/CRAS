"""
ps_core.py  (POC - standalone, does not import anything from the production app)

Photometric stereo for fabric fold / crease visualisation.

Input : 4 images of the SAME fabric area, fixed camera, one raking light at a time, coming from the
        LEFT, RIGHT, TOP and BOTTOM side of the image (light elevation ~15-20 deg above the fabric).
Output: - gradient maps p = dz/dx, q = dz/dy  (albedo-free, from opposing-light ratios)
        - shape map  = Laplacian of the height (div of the gradient): folds / creases / ridges
        - albedo map (Woodham least squares): colour / print / stains, without shading
        - normal map

Why it works (Lambertian model, light l, albedo rho, surface normal n):
    I = rho * (n . l)
    Opposing lights R / L at elevation e:  (I_R - I_L) / (I_R + I_L) = -p * cot(e)
    -> rho (the fabric's mottled texture) cancels out exactly, only the surface slope remains.
    Folds are geometry, texture mottling is albedo, so the shape map keeps the former and drops the latter.

Conventions: image x to the right, y downward. "left" image = light placed on the left side of the image.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

DIRECTIONS = ("left", "right", "top", "bottom")


@dataclass
class PSConfig:
    elevation_deg: float = 18.0      # light elevation above the fabric plane (only scales p/q)
    smooth_sigma: float = 1.5        # px, denoise the gradient maps (fabric grain)
    flat_sigma_frac: float = 0.12    # high-pass: removes light fall-off of a near light (fraction of min side)
    dark_level: float = 3.0          # grey level: below -> shadow, unreliable ratio
    saturation_level: float = 250.0  # grey level: above -> clipped highlight, unreliable
    align: bool = True               # sub-pixel translation alignment of the 4 images to the first one


@dataclass
class PSResult:
    p: np.ndarray
    q: np.ndarray
    shape: np.ndarray                # Laplacian of height (valley > 0, ridge < 0)
    albedo: np.ndarray
    normals: np.ndarray              # HxWx3, unit
    valid: np.ndarray                # bool mask: pixels with usable intensities in all 4 images
    shifts: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    notes: list = field(default_factory=list)


def to_gray_float(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.astype(np.float32)


def light_vectors(elevation_deg: float) -> Dict[str, np.ndarray]:
    e = np.deg2rad(elevation_deg)
    c, s = np.cos(e), np.sin(e)
    return {
        "left": np.array([-c, 0.0, s]),
        "right": np.array([c, 0.0, s]),
        "top": np.array([0.0, -c, s]),
        "bottom": np.array([0.0, c, s]),
    }


def align_to_reference(images: Dict[str, np.ndarray], ref_key: str = "left") -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]]:
    """Translation-only alignment with phase correlation on high-passed images (lighting-insensitive enough)."""
    def hp(a):
        return a - cv2.GaussianBlur(a, (0, 0), 8)

    ref = hp(images[ref_key])
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    out, shifts = {}, {}
    for k, im in images.items():
        if k == ref_key:
            out[k], shifts[k] = im, (0.0, 0.0)
            continue
        (dx, dy), _ = cv2.phaseCorrelate(ref, hp(im), win)
        shifts[k] = (float(dx), float(dy))
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        out[k] = cv2.warpAffine(im, m, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out, shifts


def _highpass(a: np.ndarray, sigma: float) -> np.ndarray:
    return a - cv2.GaussianBlur(a, (0, 0), sigma)


def compute(images: Dict[str, np.ndarray], cfg: Optional[PSConfig] = None) -> PSResult:
    cfg = cfg or PSConfig()
    missing = [d for d in DIRECTIONS if d not in images]
    if missing:
        raise ValueError(f"Missing light directions: {missing}")
    imgs = {k: to_gray_float(images[k]) for k in DIRECTIONS}
    shapes = {v.shape for v in imgs.values()}
    if len(shapes) != 1:
        raise ValueError(f"All 4 images must have the same size, got {shapes}")

    notes = []
    shifts = {}
    if cfg.align:
        imgs, shifts = align_to_reference(imgs)
        worst = max(np.hypot(*s) for s in shifts.values())
        if worst > 1.0:
            notes.append(f"Camera/fabric moved between shots (max shift {worst:.1f} px) - aligned by translation only.")

    I = {k: imgs[k] for k in DIRECTIONS}
    stack = np.stack([I[k] for k in DIRECTIONS])
    valid = np.all(stack > cfg.dark_level, axis=0) & np.all(stack < cfg.saturation_level, axis=0)
    invalid_pct = 100.0 * (1.0 - valid.mean())
    if invalid_pct > 2.0:
        notes.append(f"{invalid_pct:.1f}% pixels are shadowed or saturated in at least one image (adjust exposure).")

    # Albedo-free gradients from opposing-light ratios
    e = np.deg2rad(cfg.elevation_deg)
    eps = 1e-3
    rx = (I["right"] - I["left"]) / (I["right"] + I["left"] + eps)
    ry = (I["bottom"] - I["top"]) / (I["bottom"] + I["top"] + eps)
    p = (-np.tan(e) * rx).astype(np.float32)
    q = (-np.tan(e) * ry).astype(np.float32)

    # Near-light fall-off makes the two lights of a pair unequal across the frame -> low-frequency bias.
    # Folds are local, so a high-pass removes the bias without touching them.
    flat_sigma = max(cfg.flat_sigma_frac * min(p.shape), 5.0)
    p = _highpass(p, flat_sigma)
    q = _highpass(q, flat_sigma)
    p[~valid] = 0.0
    q[~valid] = 0.0
    if cfg.smooth_sigma > 0:
        p = cv2.GaussianBlur(p, (0, 0), cfg.smooth_sigma)
        q = cv2.GaussianBlur(q, (0, 0), cfg.smooth_sigma)

    # Shape map: Laplacian of height = dp/dx + dq/dy (valley/fold bottom > 0, ridge < 0)
    shape = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3) / 8.0 + cv2.Sobel(q, cv2.CV_32F, 0, 1, ksize=3) / 8.0

    # Woodham least squares: G = rho * n = pinv(L) I
    Lv = light_vectors(cfg.elevation_deg)
    Lmat = np.stack([Lv[k] for k in DIRECTIONS])            # 4x3
    G = np.tensordot(np.linalg.pinv(Lmat), stack, axes=1)    # 3xHxW
    rho = np.linalg.norm(G, axis=0) + 1e-6
    normals = np.moveaxis(G / rho, 0, -1)
    albedo = rho.astype(np.float32)

    return PSResult(p=p.astype(np.float32), q=q.astype(np.float32), shape=shape.astype(np.float32),
                    albedo=albedo, normals=normals.astype(np.float32), valid=valid, shifts=shifts, notes=notes)


def robust_z(a: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    vals = a[mask] if mask is not None else a.ravel()
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1e-9
    return (a - med) / mad


def fold_strength(shape: np.ndarray, valid: Optional[np.ndarray] = None, scales=(1.5, 3.0, 6.0)) -> np.ndarray:
    """
    Simple, un-tuned fold map for the POC: |shape| smoothed at a few scales, as a robust z-score
    (max over scales). Folds are expected to stand out clearly; no orientation or length tricks.
    """
    best = None
    for s in scales:
        z = np.abs(robust_z(cv2.GaussianBlur(shape, (0, 0), s), valid))
        best = z if best is None else np.maximum(best, z)
    if valid is not None:
        best = np.where(valid, best, 0.0)
    return best.astype(np.float32)


def to_u8(a: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0, symmetric: bool = False) -> np.ndarray:
    """Contrast-stretch any map to uint8 for display."""
    if symmetric:
        m = float(np.percentile(np.abs(a), hi_pct)) + 1e-9
        return np.clip((a / m) * 127.5 + 127.5, 0, 255).astype(np.uint8)
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    return np.clip((a - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)


def normals_to_rgb(normals: np.ndarray) -> np.ndarray:
    return np.clip((normals * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)

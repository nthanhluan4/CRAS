"""
synthetic.py  (POC - standalone)

Generates a physically-motivated test set to validate the photometric-stereo code BEFORE real captures:
dark fabric with strong albedo mottling (like the real black-fabric frames) + folds of known position and
depth, rendered with a Lambertian model under
  - "front"   : the current set-up (steep light, ~75 deg elevation)
  - "left/right/top/bottom": raking lights at 18 deg, with near-light fall-off across the frame
plus sensor noise and 8-bit quantisation.

It is a sanity check of the code and of the principle, NOT a proof for real fabric (real fabric is not
perfectly Lambertian, has self-shadowing and fibre sheen). Real 4-light captures are the actual test.
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np


def _smooth_noise(rng, shape, sigma):
    n = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), sigma)
    return n / (n.std() + 1e-9)


def _fold_segment(h, w, x0, y0, x1, y1, depth, width):
    """Gaussian-profile valley (depth > 0) or ridge (depth < 0) along a segment, height in px units."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / L2, 0, 1)
    d2 = (xx - (x0 + t * dx)) ** 2 + (yy - (y0 + t * dy)) ** 2
    return -depth * np.exp(-d2 / (2 * width * width))


def make_scene(h: int = 1024, w: int = 860, seed: int = 0):
    """Height map (px), albedo map and ground-truth fold segments."""
    rng = np.random.default_rng(seed)
    folds = [
        # name, segment, depth(px), width sigma(px)
        ("1 horizontal fold", (60, 240, 600, 240), 1.2, 9.0),
        ("2 vertical fold", (600, 240, 600, 980), 1.2, 9.0),
        ("3 faint vertical", (360, 260, 360, 560), 0.6, 10.0),
        ("4 broad diagonal", (120, 620, 260, 1000), 1.6, 22.0),
    ]
    z = np.zeros((h, w), np.float32)
    for _, (x0, y0, x1, y1), depth, width in folds:
        z += _fold_segment(h, w, x0, y0, x1, y1, depth, width)
    # a curved ridge (not in the SNR table, only for visual check)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xx - 820) ** 2 + (yy - 60) ** 2)
    z += 0.8 * np.exp(-((r - 170) ** 2) / (2 * 8.0 ** 2)) * (xx < 820)

    # Albedo: dark fabric with large mottling (the "noise" that hides folds in front light) + fine grain
    albedo = 0.20 * (1.0 + 0.15 * _smooth_noise(rng, (h, w), 25) + 0.10 * _smooth_noise(rng, (h, w), 6)
                     + 0.06 * _smooth_noise(rng, (h, w), 1.0))
    albedo = np.clip(albedo, 0.02, 1.0).astype(np.float32)
    segments = [(name, seg) for name, seg, _, _ in folds]
    return z, albedo, segments


def _normals(z):
    p = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    q = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    n = np.stack([-p, -q, np.ones_like(p)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def render(z, albedo, light, gain, falloff=None, ambient=0.02, noise_std=1.5, seed=1):
    rng = np.random.default_rng(seed)
    n = _normals(z)
    l = np.asarray(light, np.float32)
    l = l / np.linalg.norm(l)
    shading = np.clip(n @ l, 0, None)
    img = 255.0 * gain * albedo * (shading + ambient)
    if falloff is not None:
        img *= falloff
    img += rng.normal(0, noise_std, img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def make_capture_set(seed: int = 0) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, tuple]]]:
    z, albedo, segments = make_scene(seed=seed)
    h, w = z.shape
    e_rake = np.deg2rad(18.0)
    c, s = np.cos(e_rake), np.sin(e_rake)
    ramp_x = np.linspace(1.25, 0.75, w, dtype=np.float32)[None, :].repeat(h, 0)   # near light on the left
    ramp_y = np.linspace(1.25, 0.75, h, dtype=np.float32)[:, None].repeat(w, 1)   # near light at the top
    lights = {
        "left": ((-c, 0, s), ramp_x),
        "right": ((c, 0, s), ramp_x[:, ::-1]),
        "top": ((0, -c, s), ramp_y),
        "bottom": ((0, c, s), ramp_y[::-1, :]),
    }
    out = {}
    for i, (k, (l, fall)) in enumerate(lights.items()):
        out[k] = render(z, albedo, l, gain=1.2 / s * 0.9, falloff=fall, seed=10 + i)
    e_front = np.deg2rad(75.0)
    out["front"] = render(z, albedo, (0, -np.cos(e_front), np.sin(e_front)), gain=0.95, seed=99)
    return out, segments

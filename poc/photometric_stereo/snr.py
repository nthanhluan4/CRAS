"""
snr.py  (POC - standalone)

Objective, identical measurement of how visible a fold is in ANY map (front-lit grey image, PS shape map,
p/q gradients...). Used to decide the lighting set-up with numbers instead of eyeballing.

For a fold marked as a straight segment (x0, y0) -> (x1, y1):
  1. the map is detrended (minus a wide Gaussian) and lightly smoothed - same settings for every map;
  2. pixel noise sigma = robust std (1.4826 * MAD) of the whole detrended map;
  3. perpendicular profiles are sampled along the segment; one common sub-pixel offset of the fold
     centre (within +-band/3) is chosen on the AVERAGED profile (no per-sample max -> no noise bias);
  4. per sample: contrast = |centre mean - side mean|; SNR = contrast / sigma.
Reported: median per-sample SNR, share of samples with SNR >= 2, and the SNR of the averaged profile.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class LineSNR:
    name: str
    median_snr: float        # typical per-position visibility of the fold
    share_snr_ge_2: float    # fraction of positions where the fold is clearly above noise
    profile_snr: float       # visibility after averaging along the whole segment
    contrast: float          # median contrast in map units
    noise: float             # pixel noise (map units)
    offset_px: float         # chosen centre offset from the marked segment


def prepare_map(a: np.ndarray, detrend_sigma: float = 50.0, smooth_sigma: float = 3.0,
                valid: Optional[np.ndarray] = None) -> np.ndarray:
    a = a.astype(np.float32)
    out = a - cv2.GaussianBlur(a, (0, 0), detrend_sigma)
    if smooth_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), smooth_sigma)
    if valid is not None:
        out = np.where(valid, out, 0.0).astype(np.float32)
    return out


def robust_sigma(a: np.ndarray, valid: Optional[np.ndarray] = None, margin: int = 20) -> float:
    m = np.zeros_like(a, bool)
    m[margin:-margin, margin:-margin] = True
    if valid is not None:
        m &= valid
    v = a[m]
    return float(np.median(np.abs(v - np.median(v)))) * 1.4826 + 1e-9


def _sample(a: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return cv2.remap(a, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def measure_line(prepared: np.ndarray, sigma: float, seg: Sequence[float], band: int = 40,
                 n_samples: int = 80, name: str = "") -> LineSNR:
    x0, y0, x1, y1 = [float(v) for v in seg]
    length = np.hypot(x1 - x0, y1 - y0)
    if length < 5:
        raise ValueError(f"Segment {name} too short")
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    nx, ny = -uy, ux                                            # unit normal
    t = np.linspace(0.05, 0.95, n_samples)
    cx, cy = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    offs = np.arange(-band, band + 1, dtype=np.float32)
    xs = cx[:, None] + offs[None, :] * nx
    ys = cy[:, None] + offs[None, :] * ny
    prof = _sample(prepared, xs, ys)                            # n_samples x (2*band+1)

    c_half = max(2, band // 6)
    side_in = int(band * 0.6)

    def contrast_at(profiles, shift):
        o = offs - shift
        centre = profiles[:, np.abs(o) <= c_half].mean(axis=1)
        sides = profiles[:, (np.abs(o) > side_in) & (np.abs(o) <= band)].mean(axis=1)
        return centre - sides

    mean_prof = prof.mean(axis=0, keepdims=True)
    shifts = np.arange(-band // 3, band // 3 + 1)
    best_shift = max(shifts, key=lambda s: abs(float(contrast_at(mean_prof, s)[0])))
    signed = contrast_at(prof, best_shift)
    sign = np.sign(np.median(signed)) or 1.0
    per = sign * signed / sigma
    avg_contrast = abs(float(contrast_at(mean_prof, best_shift)[0]))
    return LineSNR(
        name=name,
        median_snr=float(np.median(per)),
        share_snr_ge_2=float(np.mean(per >= 2.0)),
        profile_snr=avg_contrast / (sigma / np.sqrt(n_samples)),
        contrast=float(np.median(np.abs(signed))),
        noise=sigma,
        offset_px=float(best_shift),
    )


def measure_map(a: np.ndarray, segments: List[Tuple[str, Sequence[float]]], valid: Optional[np.ndarray] = None,
                detrend_sigma: float = 50.0, smooth_sigma: float = 3.0, band: int = 40) -> List[LineSNR]:
    prepared = prepare_map(a, detrend_sigma, smooth_sigma, valid)
    sigma = robust_sigma(prepared, valid)
    return [measure_line(prepared, sigma, seg, band=band, name=name) for name, seg in segments]

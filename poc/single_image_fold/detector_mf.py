"""
detector_mf.py  (POC - single flat image fold detector, no hand-tuned thresholds)

Principle: a soft fold is a long, roughly straight, narrow valley/ridge whose per-pixel contrast is BELOW
the fabric texture. It can only be detected by accumulating evidence along its length, and the decision
threshold must come from statistics, not from tuning on a few images.

1. Backdrop / saturated pixels are neutralised; illumination is detrended.
2. Isotropic pre-whitening: the fabric mottling is spatially correlated ("coloured") noise. Dividing the
   spectrum by the radially-averaged texture power spectrum turns it into ~white noise, which is the
   condition under which a matched filter is optimal. Radial (isotropic) averaging keeps the oriented
   energy of a fold.
3. Cross-profile response: steerable 2nd derivative of Gaussian at a few fold widths (exact at any angle).
4. Along-fold accumulation: line integral over segments of a few lengths, angular step adapted to the
   length (long segments need finer angles).
5. Each response map is converted to a z-score (robust: median / MAD, background-dominated).
6. a-contrario threshold (Desolneux-Moisan-Morel, as in the LSD line detector): with N_tests tests,
   accept z > t where N_tests * P(Z > t) = epsilon, i.e. on average at most `epsilon` false detections
   per image under the texture-only hypothesis. No per-image or per-fabric tuning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.stats import norm


@dataclass
class MFConfig:
    downsample: int = 4                  # work at 1/4 resolution (1 px = 4 original px)
    widths: Tuple[float, ...] = (2.0, 3.5, 6.0)   # Gaussian sigma of the fold cross-section (downsampled px)
    lengths: Tuple[int, ...] = (40, 80, 160)      # segment lengths (downsampled px) = 160 / 320 / 640 original px
    epsilon: float = 1.0                 # expected number of false detections per image (NFA)
    two_sided: bool = True               # folds can be dark valleys or bright ridges
    backdrop_level: float = 200.0        # grey level above which a pixel is backdrop / saturated
    detrend_sigma: float = 40.0          # downsampled px
    border: int = 6                      # downsampled px ignored at the border


@dataclass
class MFResult:
    mask: np.ndarray                     # bool, original resolution: pixels covered by detected segments
    zmax: np.ndarray                     # float, downsampled: max z over all tests (for display / margins)
    threshold: float
    n_tests: float
    valid: np.ndarray                    # bool, downsampled: pixels considered fabric
    notes: List[str] = field(default_factory=list)


def _prepare(gray_full: np.ndarray, cfg: MFConfig):
    g = gray_full.astype(np.float32)
    backdrop = g >= cfg.backdrop_level
    fabric_med = float(np.median(g[~backdrop])) if (~backdrop).any() else float(np.median(g))
    g = np.where(backdrop, fabric_med, g)
    h, w = g.shape
    ds = cv2.resize(g, (w // cfg.downsample, h // cfg.downsample), interpolation=cv2.INTER_AREA)
    bd = cv2.resize(backdrop.astype(np.uint8), (ds.shape[1], ds.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    # grow the backdrop mask so the fabric/backdrop edge itself is not tested
    bd = cv2.dilate(bd.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))) > 0
    ds = ds - cv2.GaussianBlur(ds, (0, 0), cfg.detrend_sigma)
    ds[bd] = 0.0
    valid = ~bd
    b = cfg.border
    valid[:b] = valid[-b:] = False
    valid[:, :b] = valid[:, -b:] = False
    return ds, valid


def whiten(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Isotropic pre-whitening with the radially averaged power spectrum of the image itself."""
    h, w = img.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = np.fft.fft2(img * win)
    P = np.abs(F) ** 2
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    nb = 128
    bins = np.minimum((r / r.max() * nb).astype(int), nb - 1)
    radial = np.bincount(bins.ravel(), weights=P.ravel(), minlength=nb) / np.maximum(np.bincount(bins.ravel(), minlength=nb), 1)
    radial = np.exp(cv2.GaussianBlur(np.log(radial + 1e-12).astype(np.float32).reshape(1, -1), (0, 0), 2.0).ravel())
    Wf = 1.0 / np.sqrt(radial[bins] + 1e-12)
    Wf[0, 0] = 0.0                                  # drop DC
    out = np.real(np.fft.ifft2(np.fft.fft2(img) * Wf)).astype(np.float32)
    v = out[valid]
    out = (out - np.median(v)) / (1.4826 * np.median(np.abs(v - np.median(v))) + 1e-9)
    out[~valid] = 0.0
    return out


def _hessian(img: np.ndarray, sigma: float):
    s = cv2.GaussianBlur(img, (0, 0), sigma)
    ixx = cv2.Sobel(s, cv2.CV_32F, 2, 0, ksize=3) / 4.0
    iyy = cv2.Sobel(s, cv2.CV_32F, 0, 2, ksize=3) / 4.0
    ixy = cv2.Sobel(s, cv2.CV_32F, 1, 1, ksize=3) / 4.0
    return ixx * sigma ** 2, iyy * sigma ** 2, ixy * sigma ** 2     # scale-normalised


def _angles_for(length: int, width: float) -> np.ndarray:
    step = np.degrees(np.arctan2(2.0 * width, length))            # a segment tolerates ~width/length rad
    step = float(np.clip(step, 1.5, 15.0))
    n = int(np.ceil(180.0 / step))
    return np.linspace(0.0, 180.0, n, endpoint=False)


def detect(gray_full: np.ndarray, cfg: Optional[MFConfig] = None) -> MFResult:
    cfg = cfg or MFConfig()
    ds, valid = _prepare(gray_full, cfg)
    wimg = whiten(ds, valid)
    h, w = wimg.shape
    centre = (w / 2.0, h / 2.0)

    # count tests first (for the a-contrario threshold)
    n_px = float(valid.sum())
    combos = [(sw, L, _angles_for(L, sw)) for sw in cfg.widths for L in cfg.lengths]
    n_tests = sum(n_px * len(angs) for _, _, angs in combos) * (2.0 if cfg.two_sided else 1.0)
    thr = float(norm.isf(cfg.epsilon / n_tests))

    zmax = np.zeros((h, w), np.float32)
    mask_ds = np.zeros((h, w), np.uint8)
    for sw, L, angs in combos:
        ixx, iyy, ixy = _hessian(wimg, sw)
        for a in angs:
            t = np.deg2rad(a)
            # fold runs along direction a -> cross-profile curvature is along the normal (a + 90 deg)
            nx, ny = -np.sin(t), np.cos(t)
            cross = nx * nx * ixx + 2 * nx * ny * ixy + ny * ny * iyy
            cross = np.where(valid, cross, 0.0).astype(np.float32)
            # rotate so the fold direction is horizontal, integrate along it, rotate back
            M = cv2.getRotationMatrix2D(centre, a, 1.0)
            Minv = cv2.getRotationMatrix2D(centre, -a, 1.0)
            rot = cv2.warpAffine(cross, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
            acc = cv2.blur(rot, (L, 1), borderType=cv2.BORDER_CONSTANT)
            back = cv2.warpAffine(acc, Minv, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
            vv = back[valid]
            med = float(np.median(vv))
            z = (back - med) / (1.4826 * float(np.median(np.abs(vv - med))) + 1e-9)
            if cfg.two_sided:
                z = np.abs(z)
            z = np.where(valid, z, 0.0).astype(np.float32)
            np.maximum(zmax, z, out=zmax)
            hit = (z > thr).astype(np.uint8)
            if hit.any():
                # paint the whole supporting segment of each detected centre
                hit_rot = cv2.warpAffine(hit, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
                seg = cv2.dilate(hit_rot, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
                mask_ds |= cv2.warpAffine(seg, Minv, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    mask_ds &= valid.astype(np.uint8)
    H, W = gray_full.shape
    mask = cv2.resize(mask_ds, (W, H), interpolation=cv2.INTER_NEAREST) > 0
    notes = [f"a-contrario threshold z > {thr:.2f} for {n_tests:.3g} tests (epsilon = {cfg.epsilon})"]
    return MFResult(mask=mask, zmax=zmax, threshold=thr, n_tests=n_tests, valid=valid, notes=notes)

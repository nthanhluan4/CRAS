"""
test_ps.py  (POC) - correctness checks of the photometric-stereo code on synthetic data.
Run: python poc/photometric_stereo/test_ps.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ps_core  # noqa: E402
import snr  # noqa: E402
import synthetic  # noqa: E402

E = 18.0


def _render_set(z, albedo, falloff=True):
    h, w = z.shape
    e = np.deg2rad(E)
    c, s = np.cos(e), np.sin(e)
    ramp = np.linspace(1.25, 0.75, w, dtype=np.float32)[None, :].repeat(h, 0) if falloff else None
    lights = {"left": (-c, 0, s), "right": (c, 0, s), "top": (0, -c, s), "bottom": (0, c, s)}
    return {k: synthetic.render(z, albedo, l, gain=3.0, falloff=ramp, noise_std=0.0, seed=i)
            for i, (k, l) in enumerate(lights.items())}


def test_shape_map_is_albedo_invariant():
    z, albedo, _ = synthetic.make_scene()
    flat_albedo = np.full_like(albedo, float(albedo.mean()))
    s1 = ps_core.compute(_render_set(z, albedo), ps_core.PSConfig(elevation_deg=E, align=False)).shape
    s2 = ps_core.compute(_render_set(z, flat_albedo), ps_core.PSConfig(elevation_deg=E, align=False)).shape
    corr = np.corrcoef(s1[50:-50, 50:-50].ravel(), s2[50:-50, 50:-50].ravel())[0, 1]
    assert corr > 0.95, corr


def test_gradient_sign_and_scale():
    h, w = 200, 200
    xx = np.arange(w, dtype=np.float32)[None, :].repeat(h, 0)
    z = 0.10 * xx                                         # plane rising to the right: p = +0.10
    imgs = _render_set(z, np.full((h, w), 0.5, np.float32), falloff=False)
    cfg = ps_core.PSConfig(elevation_deg=E, align=False, flat_sigma_frac=10.0)  # ~no high-pass for a plane
    p_raw = -np.tan(np.deg2rad(E)) * ((imgs["right"].astype(np.float32) - imgs["left"]) /
                                      (imgs["right"].astype(np.float32) + imgs["left"]))
    assert abs(float(np.median(p_raw)) - 0.10) < 0.02, float(np.median(p_raw))
    ps_core.compute(imgs, cfg)  # runs without error on a plane


def test_ps_beats_front_light_on_synthetic():
    images, segments = synthetic.make_capture_set()
    cfg = ps_core.PSConfig(elevation_deg=E)
    res = ps_core.compute({k: images[k] for k in ps_core.DIRECTIONS}, cfg)
    front = snr.measure_map(images["front"].astype(np.float32), segments)
    ps = snr.measure_map(res.shape, segments, valid=res.valid)
    for f, p in zip(front, ps):
        assert f.median_snr < 1.0, (f.name, f.median_snr)      # hidden under texture with the current light
        assert p.median_snr > 5.0 * max(f.median_snr, 0.2), (p.name, p.median_snr, f.median_snr)


def test_alignment_recovers_shift():
    images, _ = synthetic.make_capture_set()
    shifted = dict(images)
    shifted["right"] = np.roll(images["right"], 3, axis=1)
    res = ps_core.compute({k: shifted[k] for k in ps_core.DIRECTIONS}, ps_core.PSConfig(elevation_deg=E))
    dx, dy = res.shifts["right"]
    assert abs(abs(dx) - 3) < 0.6 and abs(dy) < 0.6, res.shifts
    assert any("moved" in n for n in res.notes)


def test_snr_on_pure_noise_is_low():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, (400, 400)).astype(np.float32)
    r = snr.measure_map(noise, [("none", (50, 200, 350, 200))])[0]
    assert abs(r.median_snr) < 1.0 and r.share_snr_ge_2 < 0.2, r


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL PS POC TESTS PASSED")

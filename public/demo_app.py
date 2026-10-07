"""
public/demo_app.py - CRAS algorithm demo (single page, read left to right).

Run from the repo root:
    streamlit run public/demo_app.py --server.port 12000

Reuses the detectors in app.py (no duplicated algorithm code); this file only adds the demo UI:
input -> algorithm + live threshold -> 4 views, then step-by-step / algorithm comparison / defect list.
"""

import os
import sys
import glob
import time
import hashlib
from io import BytesIO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # app.py / aoi_config use paths relative to the repo root

import numpy as np
import cv2
import pandas as pd
import streamlit as st
from PIL import Image

import app as core  # noqa: E402  (sets page config + shared CSS on import)
import aoi_config  # noqa: E402
import defect_profiler  # noqa: E402
import golden_learner  # noqa: E402

SAMPLE_DIRS = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples"),
               os.path.join(ROOT, "test_samples")]
CRAS_MODEL_DIR = "results/models/backbone_0/itdd_cotton_fabric"

MODES = {
    "ensemble": "⚡ Ensemble (Crease + Stain + Contrast)",
    "crease": "📏 Nếp gấp (Hessian)",
    "stain": "🟡 Vết ố (CIE Lab)",
    "golden": "🎓 Golden Memory",
    "cras": "🤖 Học sâu CRAS",
}
DEFAULT_THRESHOLD = {"ensemble": 0.40, "crease": 0.18, "stain": 0.45, "golden": 0.50, "cras": 0.50}
THRESHOLD_RANGE = {
    "ensemble": (0.10, 1.00, 0.05), "crease": (0.05, 0.60, 0.02), "stain": (0.10, 1.50, 0.05),
    "golden": (0.20, 1.00, 0.05), "cras": (0.30, 0.99, 0.01),
}


# ---------------------------------------------------------------- inputs
def synthetic_fabric(seed: int = 7) -> Image.Image:
    """Built-in sample so the demo runs on any machine: woven texture + a diagonal fold + a yellow stain."""
    rng = np.random.default_rng(seed)
    h, w = 768, 1024
    yy, xx = np.mgrid[0:h, 0:w]
    weave = 0.5 + 0.05 * np.sin(xx * 1.1) * np.sin(yy * 1.1)
    img = np.stack([weave * 150, weave * 160, weave * 175], axis=-1) + rng.normal(0, 2, (h, w, 3))
    # fold: dark ridge along a diagonal line
    dist = np.abs((xx - 180) * np.sin(np.radians(70)) - (yy - 120) * np.cos(np.radians(70)))
    on_line = (xx > 180) & (xx < 880)
    img -= (np.exp(-(dist / 4.0) ** 2) * on_line)[..., None] * 55
    # stain: yellowish blotch
    stain = np.exp(-(((xx - 720) / 70.0) ** 2 + ((yy - 540) / 48.0) ** 2) * 1.2)
    img[..., 0] += stain * 25
    img[..., 1] += stain * 14
    img[..., 2] -= stain * 70
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def list_samples() -> dict:
    out = {"🧪 Mẫu tổng hợp (có sẵn)": None}
    for d in SAMPLE_DIRS:
        for p in sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg"))):
            out[f"📸 {os.path.basename(p)}"] = p
    return out


def to_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def from_bytes(b: bytes) -> Image.Image:
    return Image.open(BytesIO(b)).convert("RGB")


# ---------------------------------------------------------------- algorithms (cached per image+mode)
@st.cache_resource(show_spinner=False)
def _learner():
    return core.get_golden_learner()


@st.cache_data(show_spinner=False, max_entries=24)
def run_algorithm(img_bytes: bytes, mode: str, profile_path: str, threshold_for_ensemble: float):
    """Heavy step. Returns the anomaly map(s); the threshold is applied afterwards, so dragging the
    slider never re-runs the model (except Ensemble+Golden, whose golden map is rescaled by the threshold)."""
    pil = from_bytes(img_bytes)
    t0 = time.time()
    details, note = {}, ""
    if mode == "crease":
        score, hm, ov, anomaly = core.detect_crease_defect(pil)
    elif mode == "stain":
        score, hm, ov, anomaly = core.run_stain_enhancement(pil)
    elif mode == "golden":
        learner = _learner()
        learner.load_profile(profile_path)
        score, hm, ov, anomaly = learner.predict(pil)
    elif mode == "cras":
        model, device = core.load_cras_model(CRAS_MODEL_DIR)
        score, hm, ov, anomaly, ood = core.run_cras_inference(model, device, pil)
        if ood > 0.30:
            note = f"⚠️ CRAS ngoài miền huấn luyện ({ood * 100:.0f}% vùng bị chấm > 0.5) - kết quả không đáng tin."
    else:
        golden = None
        if profile_path:
            golden = _learner()
            golden.load_profile(profile_path)
        score, hm, ov, details, anomaly = core.run_hybrid_ensemble(pil, golden=golden, decision_level=threshold_for_ensemble)
    return {
        "score": score, "heatmap": hm, "overlay": ov, "anomaly": anomaly,
        "component_maps": details.get("component_maps"), "details": details,
        "ms": (time.time() - t0) * 1000.0, "note": note,
    }


def extract(pil: Image.Image, res: dict, threshold: float, cfg: dict):
    """Cheap step: threshold the stored map -> instances, mask, annotated image."""
    kwargs = dict(
        min_area=int(cfg["min_area"]), aspect_ratio_crease_thresh=float(cfg["aspect_ratio"]),
        pixels_per_inch=float(cfg["px_per_inch"]), hole_needs_review=not cfg["backlight_available"],
    )
    if res["component_maps"]:
        defects, mask = defect_profiler.extract_from_component_maps(pil, res["component_maps"], threshold=threshold, **kwargs)
    else:
        defects, mask = defect_profiler.extract_defect_instances(pil, res["anomaly"], threshold=threshold, **kwargs)
    return defects, mask, defect_profiler.draw_defect_bounding_boxes(pil, defects)


def available_modes(have_profile: bool) -> list:
    keys = ["ensemble", "crease", "stain"]
    if have_profile:
        keys.append("golden")
    if os.path.isdir(CRAS_MODEL_DIR):
        keys.append("cras")
    return keys


# ---------------------------------------------------------------- UI helpers
def card(title: str, tag: str = ""):
    st.html(f'<div class="st-view-card"><div class="st-view-header"><span>{title}</span>'
            f'<span class="st-view-tag">{tag}</span></div></div>')


def show(img, width_px: int = 640):
    arr = np.array(img) if isinstance(img, Image.Image) else img
    h, w = arr.shape[:2]
    s = min(width_px / w, 1.0)
    st.image(cv2.resize(arr, (int(w * s), int(h * s))), use_container_width=True)


THEME_CSS = """
<style>
    :root { --primary: oklch(48.2% .211 264.05); }
    /* Nền trắng + gradient dọc từ primary (7%) mờ dần xuống trong suốt, như notes.hots.one */
    [data-testid="stAppViewContainer"], .stApp {
        background: linear-gradient(to bottom in oklab, color-mix(in oklab, var(--primary) 7%, transparent), transparent 520px), #ffffff !important;
        background-repeat: no-repeat !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, color-mix(in oklab, var(--primary) 16%, white), color-mix(in oklab, var(--primary) 8%, white)) !important;
        border-right: 1px solid color-mix(in oklab, var(--primary) 18%, white) !important;
    }
    /* Full width */
    .block-container, [data-testid="stMainBlockContainer"] {
        max-width: 100% !important; width: 100% !important;
        padding-left: 2rem !important; padding-right: 2rem !important;
    }
    .st-page-title {
        font-size: 1.9rem; font-weight: 700; letter-spacing: -0.01em;
        color: transparent; background-clip: text; -webkit-background-clip: text;
        background-image: linear-gradient(100deg, oklch(55% .22 264), oklch(60% .23 300), oklch(70% .15 220), oklch(55% .22 264));
        background-size: 220% 100%; animation: cras-pan 8s ease-in-out infinite;
    }
    @keyframes cras-pan { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
</style>
"""


def main():
    cfg = aoi_config.load_config()
    st.html(THEME_CSS)

    # ── sidebar: ① input ────────────────────────────────────────────
    st.sidebar.markdown("**① ẢNH VÀO**")
    samples = list_samples()
    src = st.sidebar.radio("Nguồn ảnh", ["Mẫu có sẵn", "Tải ảnh lên"], horizontal=True, label_visibility="collapsed")
    pil, name = None, ""
    if src == "Mẫu có sẵn":
        name = st.sidebar.selectbox("Mẫu", list(samples.keys()))
        pil = synthetic_fabric() if samples[name] is None else Image.open(samples[name]).convert("RGB")
    else:
        up = st.sidebar.file_uploader("Ảnh vải", type=["png", "jpg", "jpeg", "bmp"])
        if up is not None:
            pil, name = Image.open(up).convert("RGB"), up.name
    if pil is None:
        st.info("👈 Chọn mẫu hoặc tải ảnh vải ở thanh bên trái.")
        return
    if st.sidebar.checkbox("Tự động cắt mép vải", value=False):
        pil, _ = core.auto_crop_fabric(pil)
    st.sidebar.image(pil, use_container_width=True, caption=f"{name} · {pil.size[0]}×{pil.size[1]}")

    profiles = {p["display_name"]: p["path"] for p in golden_learner.GoldenMemoryLearner.list_available_profiles()}
    prof_label = st.sidebar.selectbox("Hồ sơ Golden (tuỳ chọn)", ["(không nạp)"] + list(profiles.keys()),
                                      help="Chỉ chọn hồ sơ của ĐÚNG mã vải; bật chế độ Golden và gộp vào Ensemble.")
    profile_path = profiles.get(prof_label, "")

    img_bytes = to_bytes(pil)

    # ── main: ② algorithm ───────────────────────────────────────────
    st.html('<div class="st-page-title">Demo nhận diện khuyết điểm vải</div>')
    keys = available_modes(bool(profile_path))
    c_mode, c_thr = st.columns([3, 2])
    mode = c_mode.radio("② THUẬT TOÁN", keys, format_func=MODES.get, horizontal=False)
    lo, hi, step = THRESHOLD_RANGE[mode]
    threshold = c_thr.slider("Ngưỡng (kéo để thấy mask/overlay đổi ngay)", lo, hi, DEFAULT_THRESHOLD[mode], step, format="%.2f")

    with st.spinner("Đang phân tích..."):
        res = run_algorithm(img_bytes, mode, profile_path if mode in ("golden", "ensemble") else "",
                            round(threshold, 2) if (mode == "ensemble" and profile_path) else 0.0)
    if res["note"]:
        if mode == "cras":
            st.error(
                f"**🚫 Ảnh này nằm NGOÀI miền huấn luyện của CRAS - kết quả bên dưới KHÔNG đáng tin.**\n\n"
                f"{res['note']}\n\n"
                "CRAS chỉ được huấn luyện trên vải cotton của bộ ITDD (ảnh 512×512), nên với loại vải/camera khác "
                "nó coi gần như toàn bộ ảnh là bất thường (heatmap đỏ, mask phủ kín, nhiều lỗi giả). "
                "👉 Với ảnh camera nhà máy hãy chọn **Ensemble**, **Nếp gấp** hoặc **Vết ố**; "
                "muốn dùng CRAS cần huấn luyện lại trên ảnh vải tốt của chính camera đó.")
        else:
            st.warning(res["note"])
    defects, mask, annotated = extract(pil, res, threshold, cfg)
    stats = defect_profiler.compute_defect_statistics(
        defects, (pil.size[1], pil.size[0]), pixels_per_inch=float(cfg["px_per_inch"]),
        inspected_width_px=pil.size[0])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số khuyết điểm", len(defects))
    m2.metric("Diện tích lỗi", f"{stats.get('defect_area_pct', 0.0):.2f}%")
    m3.metric("Điểm bất thường max", f"{res['score']:.2f}")
    m4.metric("Thời gian chạy model", f"{int(res['ms'])} ms", help="Chỉ tính lần chạy model; kéo ngưỡng không chạy lại.")

    # ── ③ result: 4 views ───────────────────────────────────────────
    st.markdown("**③ KẾT QUẢ**")
    v1, v2, v3, v4 = st.columns(4)
    with v1:
        card("📸 Gốc", f"{pil.size[0]}×{pil.size[1]}"); show(pil)
    with v2:
        card("🔥 Heatmap", "JET"); show(res["heatmap"])
    with v3:
        card("🎯 Mask", f"ngưỡng {threshold:.2f}"); show(mask)
    with v4:
        card("📐 Định vị", f"{len(defects)} lỗi"); show(annotated)

    # ── ④ step by step ──────────────────────────────────────────────
    st.divider()
    st.markdown("**④ TỪNG BƯỚC XỬ LÝ**")
    if True:
        ctx = core._prepare(pil)
        diff = np.abs(ctx["gray"] - ctx["bg_gray"])
        diff_u8 = np.clip(diff / max(float(np.percentile(diff, 99.5)), 1.0) * 255, 0, 255).astype(np.uint8)
        steps = [
            ("1 · Gốc", ctx["np_img"], f"Thu nhận; thu nhỏ về ≤{core.PROC_MAX_SIDE}px để xử lý."),
            ("2 · Nền cục bộ", np.clip(ctx["bg_gray"], 0, 255).astype(np.uint8), "Làm mờ Gauss: ước lượng nền vải."),
            ("3 · Lệch khỏi nền", diff_u8, f"|ảnh − nền|; nhiễu vân dệt ≈ {ctx['texture_noise']:.1f}."),
            ("4 · Heatmap", res["heatmap"], f"Điểm bất thường theo chế độ «{MODES[mode]}»."),
            ("5 · Ngưỡng", mask, f"Cắt tại {threshold:.2f} + lọc diện tích tối thiểu."),
            ("6 · Khoanh vùng", np.array(annotated), "Thành phần liên thông → hộp + loại lỗi."),
        ]
        for row in (steps[:3], steps[3:]):
            for col, (title, im, cap) in zip(st.columns(3), row):
                with col:
                    card(title, ""); show(im, 480)
                    st.caption(cap)

    # ── ⑤ algorithm comparison ──────────────────────────────────────
    st.divider()
    st.markdown("**⑤ SO SÁNH THUẬT TOÁN**")
    if True:
        st.caption("Chạy cùng một ảnh qua từng thuật toán, mỗi thuật toán dùng ngưỡng mặc định của nó.")
        if st.button("▶ Chạy so sánh tất cả", key="cmp_btn") or st.session_state.get("cmp_done") == hashlib.md5(img_bytes).hexdigest():
            st.session_state["cmp_done"] = hashlib.md5(img_bytes).hexdigest()
            cols = st.columns(len(keys))
            for col, k in zip(cols, keys):
                with col:
                    try:
                        thr = DEFAULT_THRESHOLD[k]
                        r = run_algorithm(img_bytes, k, profile_path if k in ("golden", "ensemble") else "",
                                          thr if (k == "ensemble" and profile_path) else 0.0)
                        d, _, ann = extract(pil, r, thr, cfg)
                        card(MODES[k], ""); show(ann, 400)
                        st.caption(f"{len(d)} lỗi · {int(r['ms'])} ms · ngưỡng {thr:.2f}")
                    except Exception as e:  # one failing detector must not hide the others
                        card(MODES[k], "")
                        st.error(f"Lỗi: {e}")

    # ── ⑥ defect list ───────────────────────────────────────────────
    st.divider()
    st.markdown("**⑥ DANH SÁCH LỖI**")
    if True:
        if not defects:
            st.success("Không phát hiện khuyết điểm ở ngưỡng này. Thử hạ ngưỡng.")
        else:
            df = pd.DataFrame(defects)
            df = df[[c for c in df.columns if df[c].map(lambda v: isinstance(v, (int, float, str, bool))).all()]]
            st.dataframe(df, use_container_width=True)


main()

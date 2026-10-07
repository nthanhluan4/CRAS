"""
app.py - CRAS Fabric Defect Inspection & Cognitive Vision-LLM Platform
Design System:
- Soft pastel background aura (Sky -> Lavender -> Peach -> White)
- Pure white clean sidebar with orange brand badge
- 5 signature pastel StatCards (Blue, Lavender, Purple, Mint, Warm Peach) with arrows
- 4-View Workstation framed with 16px rounded white cards and clean borders
- Security Gateway for Admin with matching pastel aesthetics and password Admin@2026!
"""

import os
import glob
import json
import time
import secrets
import hashlib
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
import sys
import pandas as pd
import streamlit as st

# Auto-redirect to streamlit run if user executes 'python app.py' directly
if __name__ == "__main__" and not st.runtime.exists():
    import subprocess
    streamlit_exe = os.path.join(os.path.dirname(sys.executable), "streamlit.exe")
    if not os.path.exists(streamlit_exe):
        streamlit_exe = "streamlit"
    print("=================================================================")
    print("  [CRAS] Web Dashboard can chay bang Streamlit de hien thi giao dien.")
    print("  Dang tu dong khoi chay: streamlit run app.py --server.port 8501")
    print("=================================================================")
    cmd = [streamlit_exe, "run", os.path.abspath(__file__), "--server.port", "8501"]
    subprocess.run(cmd)
    sys.exit(0)

import backbones
import cras
import utils
import defect_profiler
import vlm_inspector
import golden_learner
import aoi_config
import roll_session

# ==========================================
# 0. PAGE CONFIG & 100% MATCHED DESIGN SYSTEM
# ==========================================
st.set_page_config(
    page_title="CRAS - Hệ Thống Giám Định Khuyết Tật Vải",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Admin security hash (Password: Admin@2026!)
ADMIN_PASS_HASH = hashlib.sha256("Admin@2026!".encode("utf-8")).hexdigest()

# 100% EXACT CSS FROM USER'S SCREENSHOT
st.html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global Background: Soft Pastel Aura (Sky -> Lavender -> Peach -> White) */
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: #0f172a;
    }

    [data-testid="stAppViewContainer"] {
        background: radial-gradient(at 0% 0%, #edf6fd 0px, transparent 40%),
                    radial-gradient(at 35% 0%, #f6f1fe 0px, transparent 35%),
                    radial-gradient(at 75% 0%, #fff7eb 0px, transparent 35%),
                    #ffffff !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }

    /* Clean Pure White Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #edf2f7 !important;
        box-shadow: 1px 0 3px rgba(0, 0, 0, 0.01) !important;
    }

    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 98% !important;
    }

    /* ── Clean Modern Sidebar Brand Header ── */
    .st-sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        padding: 0.2rem 0.2rem 0.9rem 0.2rem;
        margin-bottom: 0.6rem;
    }
    .st-brand-icon {
        width: 1.85rem;
        height: 1.85rem;
        background: #f95721;
        border-radius: 7px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        font-size: 0.95rem;
        box-shadow: 0 1px 3px rgba(249, 87, 33, 0.3);
    }
    .st-brand-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.01em;
    }

    /* ── Sidebar Navigation List ── */
    .st-nav-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.45rem 0.6rem;
        border-radius: 8px;
        color: #475569;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 0.2rem;
        cursor: pointer;
        transition: background 0.15s ease;
    }
    .st-nav-item:hover {
        background: #f8fafc;
        color: #0f172a;
    }
    .st-nav-left {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }

    /* ── Sidebar Footer (User info & switch) ── */
    .st-sidebar-footer {
        border-top: 1px solid #f1f5f9;
        padding-top: 0.85rem;
        margin-top: 1.2rem;
    }
    .st-user-pill {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.4rem 0.65rem;
        font-size: 0.78rem;
        color: #1e293b;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.65rem;
        background: #ffffff;
    }
    .st-profile-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.35rem 0.2rem;
    }
    .st-avatar-badge {
        width: 1.85rem;
        height: 1.85rem;
        border-radius: 50%;
        background: #3b82f6;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 700;
    }

    /* ── Main Page Header ── */
    .st-page-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }

    /* ── The 5 Exact Signature Pastel StatCards ── */
    .st-stat-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.85rem;
        margin-bottom: 1.25rem;
    }
    .st-stat-card {
        border-radius: 16px;
        padding: 1rem 1.15rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .st-stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }
    .st-stat-left {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }
    .st-stat-icon-box {
        width: 2.6rem;
        height: 2.6rem;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        flex-shrink: 0;
    }
    .st-stat-title {
        font-size: 0.73rem;
        color: #64748b;
        line-height: 1.25;
        margin-bottom: 0.2rem;
    }
    .st-stat-num {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .st-stat-arrow {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 300;
    }

    /* Card 1: Pastel Blue */
    .st-card-blue {
        background: #f0f8ff;
        border: 1px solid #dbeafe;
    }
    .st-icon-blue {
        background: #dbeafe;
        color: #0284c7;
    }

    /* Card 2: Pastel Lavender */
    .st-card-lavender {
        background: #f5f3ff;
        border: 1px solid #ede9fe;
    }
    .st-icon-lavender {
        background: #ede9fe;
        color: #7c3aed;
    }

    /* Card 3: Pastel Purple */
    .st-card-purple {
        background: #faf5ff;
        border: 1px solid #f3e8ff;
    }
    .st-icon-purple {
        background: #f3e8ff;
        color: #9333ea;
    }

    /* Card 4: Pastel Mint */
    .st-card-mint {
        background: #f0fdf4;
        border: 1px solid #dcfce7;
    }
    .st-icon-mint {
        background: #dcfce7;
        color: #16a34a;
    }

    /* Card 5: Pastel Warm Peach */
    .st-card-peach {
        background: #fffbeb;
        border: 1px solid #fef3c7;
    }
    .st-icon-peach {
        background: #fef3c7;
        color: #d97706;
    }

    /* ── The Attention Card ('Cần chú ý') ── */
    .st-attention-card {
        background: #ffffff;
        border: 1px solid #edf2f7;
        border-radius: 16px;
        padding: 1.15rem 1.4rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
        margin-bottom: 1.25rem;
    }
    .st-attention-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.88rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.75rem;
    }
    .st-attention-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.65rem 0;
        border-bottom: 1px solid #f8fafc;
        font-size: 0.84rem;
        color: #334155;
    }
    .st-attention-row:last-child {
        border-bottom: none;
        padding-bottom: 0.1rem;
    }
    .st-action-btn {
        border: 1px solid #e2e8f0;
        background: #ffffff;
        color: #334155;
        font-size: 0.76rem;
        font-weight: 500;
        padding: 0.3rem 0.75rem;
        border-radius: 7px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        cursor: pointer;
        transition: background 0.15s ease;
    }
    .st-action-btn:hover {
        background: #f8fafc;
        border-color: #cbd5e1;
    }

    /* ── 4-View Workstation Cards ── */
    .st-view-card {
        background: #ffffff;
        border: 1px solid #edf2f7;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        margin-bottom: 0.75rem;
    }
    .st-view-header {
        padding: 0.55rem 0.85rem;
        background: #ffffff;
        border-bottom: 1px solid #f1f5f9;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.78rem;
        font-weight: 600;
        color: #1e293b;
    }
    .st-view-tag {
        font-size: 0.7rem;
        padding: 0.12rem 0.45rem;
        border-radius: 9999px;
        background: #f1f5f9;
        color: #64748b;
        font-weight: 600;
    }

    /* ── Admin Login Card ── */
    .st-login-card {
        max-width: 420px;
        margin: 3.5rem auto;
        background: #ffffff;
        border: 1px solid #edf2f7;
        border-radius: 18px;
        padding: 2.2rem 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .st-login-icon {
        width: 3.2rem;
        height: 3.2rem;
        border-radius: 12px;
        background: #fff7ed;
        color: #ea580c;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        margin: 0 auto 1.1rem auto;
        border: 1px solid #ffedd5;
    }

    /* ── Streamlit Tabs Modern Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        background-color: #f1f5f9;
        padding: 0.25rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        padding: 0.4rem 0.85rem;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    button[kind="primary"] {
        background-color: #f95721 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        border: none !important;
        box-shadow: 0 1px 3px rgba(249, 87, 33, 0.3) !important;
    }

    @media print {
        [data-testid="stSidebar"], [data-testid="stSidebarCollapseButton"], header[data-testid="stHeader"] {
            display: none !important;
        }
        .main, [data-testid="stMain"], .stApp {
            overflow: visible !important;
            height: auto !important;
        }
    }
</style>
""")


# ==========================================
# 1. CORE ALGORITHMS & LOGIC
# ==========================================
@st.cache_resource(show_spinner="Đang nạp mô hình CRAS...")
def load_cras_model(model_dir: str, backbone_name: str = "wideresnet50", img_size: int = 288):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone = backbones.load(backbone_name)
    backbone.name = backbone_name
    backbone.seed = None

    cras_model = cras.CRAS(device)
    cras_model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, img_size, img_size),
        pretrain_embed_dimension=1536,
        target_embed_dimension=1536,
        patchsize=3,
        meta_epochs=1,
        eval_epochs=1,
        dsc_layers=3,
        train_backbone=False,
        pre_proj=1,
        noise=0.015,
        k=0.3,
        lr=0.0001,
        limit=-1,
    )

    ckpt_files = sorted(glob.glob(os.path.join(model_dir, "ckpt_best*.pth")))
    if not ckpt_files:
        ckpt_files = glob.glob(os.path.join(model_dir, "*.pth"))
        ckpt_files = [f for f in ckpt_files if "center" not in os.path.basename(f)]

    if not ckpt_files:
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại {model_dir}")

    ckpt_path = ckpt_files[-1]
    state_dict = torch.load(ckpt_path, map_location=device)

    if "discriminator" in state_dict:
        cras_model.discriminator.load_state_dict(state_dict["discriminator"])
        if "pre_projection" in state_dict and cras_model.pre_proj > 0:
            cras_model.pre_projection.load_state_dict(state_dict["pre_projection"])
    else:
        cras_model.load_state_dict(state_dict, strict=False)

    center_path = os.path.join(model_dir, "center.pth")
    if not os.path.exists(center_path):
        raise FileNotFoundError(f"Center file not found at: {center_path}")
    cras_model.c2 = torch.load(center_path, map_location=device)

    cras_model.to(device)
    cras_model.eval()
    return cras_model, device


@st.cache_resource(show_spinner="Đang khởi tạo Golden Memory Learner...")
def get_golden_learner() -> golden_learner.GoldenMemoryLearner:
    return golden_learner.GoldenMemoryLearner()


def auto_crop_fabric(pil_image: Image.Image) -> Tuple[Image.Image, bool]:
    np_img = np.array(pil_image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        total_area = np_img.shape[0] * np_img.shape[1]
        if 0.15 * total_area < area < 0.95 * total_area:
            x, y, w, h = cv2.boundingRect(largest)
            pad = 8
            x0 = max(0, x + pad)
            y0 = max(0, y + pad)
            x1 = min(np_img.shape[1], x + w - pad)
            y1 = min(np_img.shape[0], y + h - pad)
            return Image.fromarray(np_img[y0:y1, x0:x1]), True
    return pil_image, False


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


def _median_background(channel_u8: np.ndarray, frac: float, out_size: Tuple[int, int]) -> np.ndarray:
    """
    Robust large-scale background via median filter on a 256px thumbnail.
    A median ignores any blob covering < 50% of its window, so stains up to ~frac/1.25 of the
    short image side are fully recovered (a Gaussian only leaves their rim).
    """
    h, w = channel_u8.shape[:2]
    thumb_scale = 256.0 / max(h, w)
    tw, th = max(int(w * thumb_scale), 8), max(int(h * thumb_scale), 8)
    thumb = cv2.resize(channel_u8, (tw, th), interpolation=cv2.INTER_AREA)
    k = max(int(round(min(tw, th) * frac)) | 1, 3)
    bg = cv2.medianBlur(thumb, k)
    return cv2.resize(bg, out_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _drop_border_blobs(anomaly: np.ndarray, level: float = 0.2, margin: int = BORDER_PAD + 2) -> np.ndarray:
    """Zero every connected blob that touches the frame border (table strips, fabric selvedge, backdrop)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((anomaly > level).astype(np.uint8), connectivity=8)
    h, w = anomaly.shape
    x0, y0 = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    x1, y1 = x0 + stats[:, cv2.CC_STAT_WIDTH], y0 + stats[:, cv2.CC_STAT_HEIGHT]
    touches = (x0 <= margin) | (y0 <= margin) | (x1 >= w - margin) | (y1 >= h - margin)
    touches[0] = True  # background label
    keep = ~touches[labels]
    # Dilate the kept support a little so the soft rim of each blob survives
    keep = cv2.dilate(keep.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    return np.where(keep, anomaly, 0.0).astype(np.float32)


def _chromatic_anomaly_map(ctx) -> np.ndarray:
    """
    CIE L*a*b* Delta-E with a texture-adaptive JND floor, at two scales:
    - local: Delta-E against a local Gaussian background (small stains / spots, tracks shading)
    - large: chroma shift against a wide median background, so the interior of a large stain is
      filled in instead of only its rim. Blobs touching the frame border are dropped from this term
      (table / backdrop strips are not stains, and a median cannot model thin border bands).
    """
    scaled = ctx["scaled"]
    proc_h, proc_w = scaled.shape[:2]
    lab_u8 = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    lab = lab_u8.astype(np.float32)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    stain_thresh = max(ctx["texture_noise"] * 1.8, 6.0)

    ksize = int(round(proc_w * 0.045)) | 1
    sigma = ksize / 3.0
    bg_L = cv2.GaussianBlur(L, (ksize, ksize), sigma)
    bg_A = cv2.GaussianBlur(A, (ksize, ksize), sigma)
    bg_B = cv2.GaussianBlur(B, (ksize, ksize), sigma)
    delta_E = np.sqrt((L - bg_L)**2 + (A - bg_A)**2 + (B - bg_B)**2)
    local_map = np.maximum(
        np.clip((delta_E - stain_thresh) / 25.0, 0.0, 1.0),
        np.clip((B - bg_B - 6.0) / 20.0, 0.0, 1.0),
    )

    # Chroma is illumination-stable, so a very wide window is safe; lightness is left to the local term.
    big_A = _median_background(lab_u8[:, :, 1], 0.60, (proc_w, proc_h))
    big_B = _median_background(lab_u8[:, :, 2], 0.60, (proc_w, proc_h))
    delta_ab = np.sqrt((A - big_A)**2 + (B - big_B)**2)
    large_map = np.maximum(
        np.clip((delta_ab - stain_thresh) / 25.0, 0.0, 1.0),
        np.clip((B - big_B - 6.0) / 20.0, 0.0, 1.0),
    )
    large_map = _drop_border_blobs(large_map)

    return np.maximum(local_map, large_map)


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


def _contrast_anomaly_map(ctx) -> np.ndarray:
    """Photometric outliers: chalk lines, foreign yarn, bleached spots."""
    contrast_diff = np.abs(ctx["gray"] - ctx["bg_gray"])
    contrast_floor = max(ctx["texture_noise"] * 5.0, 28.0)
    contrast_anomaly = np.clip((contrast_diff - contrast_floor) / 50.0, 0.0, 1.0)
    k_dot = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(contrast_anomaly, cv2.MORPH_OPEN, k_dot)


def _upscale_map(ctx, anomaly_proc: np.ndarray, pad_border: bool = True) -> np.ndarray:
    """Zero the border and upscale a processing-resolution map to the original image size (0..1)."""
    anomaly_proc = anomaly_proc.copy()
    if pad_border:
        anomaly_proc[:BORDER_PAD, :] = 0.0
        anomaly_proc[-BORDER_PAD:, :] = 0.0
        anomaly_proc[:, :BORDER_PAD] = 0.0
        anomaly_proc[:, -BORDER_PAD:] = 0.0
    orig_w, orig_h = ctx["orig_size"]
    return np.clip(cv2.resize(anomaly_proc, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)


def _finalize_map(ctx, anomaly_proc: np.ndarray, pad_border: bool = True):
    """Zero the border, upscale to original size, and render score / heatmap / overlay."""
    anomaly = _upscale_map(ctx, anomaly_proc, pad_border)
    score = float(np.percentile(anomaly, 99.8)) if anomaly.max() > 0 else 0.0

    heatmap_rgb = cv2.cvtColor(cv2.applyColorMap((anomaly * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(ctx["np_img"], 0.60, heatmap_rgb, 0.40, 0)
    return score, heatmap_rgb, overlay, anomaly


def run_stain_enhancement(pil_image: Image.Image) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Principled CIE L*a*b* Perceptual Chromatic Anomaly Detection.
    Evaluates Delta-E against a robust background with industrial JND noise floor.
    Prevents false amplification of sensor noise on dark/monochrome fabrics.
    """
    ctx = _prepare(pil_image)
    return _finalize_map(ctx, _chromatic_anomaly_map(ctx))


def detect_crease_defect(pil_image: Image.Image) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Principled Hessian Curvature Tensor with multi-orientation Directional Path Opening.
    Detects physical linear creases and folds at any angle while rejecting natural fabric grain.
    """
    ctx = _prepare(pil_image)
    return _finalize_map(ctx, _crease_anomaly_map(ctx))


def cras_crop_box(orig_w: int, orig_h: int, resize: int = 329, imagesize: int = 288) -> Tuple[int, int, int, int]:
    """Region of the original image seen by Resize(resize) + CenterCrop(imagesize), as (x0, y0, x1, y1)."""
    scale = resize / float(min(orig_w, orig_h))
    crop = imagesize / scale
    x0 = int(round((orig_w - crop) / 2.0))
    y0 = int(round((orig_h - crop) / 2.0))
    return max(x0, 0), max(y0, 0), min(int(round(x0 + crop)), orig_w), min(int(round(y0 + crop)), orig_h)


def run_cras_inference(model, device, pil_image: Image.Image, imagesize: int = 288, resize: int = 329):
    """
    Deep CRAS inference with the SAME preprocessing as training (Resize 329 + CenterCrop 288).
    The discriminator ends in a Sigmoid, so the mask is already an absolute defect probability:
    it is NOT min-max stretched (that would invent defects on good fabric).
    Returns the probability map in original-image coordinates (0 outside the analysed center crop)
    plus `ood_ratio`: fraction of the crop scored >0.5. A high value means the fabric is outside the
    model's training domain and the result must not be trusted.
    """
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(imagesize),
        transforms.ToTensor(),
        transforms.Normalize(mean=utils.IMAGENET_MEAN, std=utils.IMAGENET_STD),
    ])
    orig_w, orig_h = pil_image.size
    img_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        scores, masks = model._predict(img_tensor)

    score = float(scores[0])
    mask = np.clip(np.asarray(masks[0], dtype=np.float32), 0.0, 1.0)
    ood_ratio = float((mask > 0.5).mean())

    x0, y0, x1, y1 = cras_crop_box(orig_w, orig_h, resize, imagesize)
    anomaly = np.zeros((orig_h, orig_w), np.float32)
    anomaly[y0:y1, x0:x1] = cv2.resize(mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)

    heatmap_rgb = cv2.cvtColor(cv2.applyColorMap((anomaly * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    raw_rgb = np.array(pil_image)
    overlay = cv2.addWeighted(raw_rgb, 0.65, heatmap_rgb, 0.35, 0)
    cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 3)
    return score, heatmap_rgb, overlay, anomaly, ood_ratio


def run_hybrid_ensemble(pil_image: Image.Image, golden: Optional[golden_learner.GoldenMemoryLearner] = None,
                        decision_level: float = 0.40) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
    """
    Principled Orthogonal Multi-Modal Fabric Anomaly Fusion (max-fusion):
    - Chromatic Anomaly (CIE Delta-E with JND noise floor)
    - Structural Linear Crease Anomaly (Hessian eigenvalues, 12-orientation path opening)
    - Photometric Outliers (Chalk lines, Foreign yarn, Bleached spots)
    - Golden Memory Bank texture anomaly (weave faults the rules cannot see), only when a profile of
      THIS fabric is loaded. Its map is calibrated to 0.5 at the decision level, so it is rescaled to put
      that level on `decision_level` (the ensemble threshold) before max-fusion.
    The deep CRAS map is intentionally NOT fused: its checkpoint must first be trained on the
    production fabric (see run_cras_inference `ood_ratio`), otherwise it floods the fusion.
    """
    ctx = _prepare(pil_image)
    chromatic_map = _chromatic_anomaly_map(ctx)
    crease_map = _crease_anomaly_map(ctx)
    contrast_map = _contrast_anomaly_map(ctx)

    fused = np.maximum(np.maximum(chromatic_map, crease_map), contrast_map)
    golden_score = None
    component_proc = {"chromatic": chromatic_map, "crease": crease_map, "contrast": contrast_map}
    if golden is not None and golden.memory_bank is not None:
        _, _, _, golden_map = golden.predict(pil_image)
        proc_h, proc_w = fused.shape
        golden_map = cv2.resize(golden_map, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        golden_map = np.clip(golden_map * (decision_level / golden_learner.DEFAULT_DECISION_LEVEL), 0.0, 1.0)
        golden_score = float(golden_map.max())
        fused = np.maximum(fused, golden_map)
        component_proc["golden"] = golden_map
    score, heatmap_rgb, overlay, master_anomaly = _finalize_map(ctx, fused)
    # Per-detector maps at full resolution: defects must be extracted per map, otherwise a fold / fabric-edge
    # response bridged to a stain swallows it into one mixed blob (and its colour shift is averaged away).
    component_maps = {k: _upscale_map(ctx, v) for k, v in component_proc.items()}

    details = {
        "crease_score": float(crease_map.max()),
        "stain_score": float(chromatic_map.max()),
        "contrast_score": float(contrast_map.max()),
        "texture_noise": ctx["texture_noise"],
        "golden_score": golden_score,
        "component_maps": component_maps,
    }
    return score, heatmap_rgb, overlay, details, master_anomaly


# ==========================================
# 2. APPLICATION STATE & SECURITY
# ==========================================
def is_admin_authenticated() -> bool:
    auth_state = st.session_state.get("admin_authenticated", False)
    token = st.session_state.get("admin_session_token", "")
    return auth_state and bool(token) and len(token) >= 32


def grade_short_label(grade_text: str) -> str:
    if "HẠNG A" in grade_text:
        return "Hạng A"
    if "HẠNG B" in grade_text:
        return "Hạng B"
    if "HẠNG C" in grade_text:
        return "Reject"
    return "Chưa xếp hạng"

if "gemini_api_key_store" not in st.session_state:
    st.session_state["gemini_api_key_store"] = os.environ.get("GEMINI_API_KEY", "")
if "gemini_model_store" not in st.session_state:
    st.session_state["gemini_model_store"] = "gemini-2.0-flash-lite"
if "mes_endpoint_store" not in st.session_state:
    st.session_state["mes_endpoint_store"] = "http://mes.factory.internal/api/v1/quality/fabric-inspection"


# ==========================================
# 3. MAIN CONTROLLER & ROUTING
# ==========================================
def main():
    # ── Sidebar Brand (Modern orange badge & clean typography) ──
    st.sidebar.html("""
    <div class="st-sidebar-brand">
        <div class="st-brand-icon">🧵</div>
        <div class="st-brand-title">Hệ Thống Giám Định Vải AOI</div>
    </div>
    """)

    app_section = st.sidebar.radio(
        "Chọn phân hệ làm việc:",
        [
            "🔍 Giám Định & Nhận Diện Khuyết Tật (Demo)",
            "⚙️ Cấu Hình Nền Tảng & Huấn Luyện [🔒]"
        ],
        index=0
    )
    st.sidebar.markdown("---")

    learner = get_golden_learner()

    # Route
    if "Cấu Hình" in app_section or "Quản Trị" in app_section:
        render_admin_section(learner)
    else:
        render_demo_inspection_section(learner)

    # ── Sidebar Footer (Clean system status) ──
    st.sidebar.html("""
    <div class="st-sidebar-footer">
        <div class="st-user-pill">
            <span style="font-weight: 500;">Hệ thống: Sẵn sàng</span>
            <span style="color: #16a34a; font-size: 0.85rem;">●</span>
        </div>
        <div style="font-size: 0.72rem; color: #94a3b8; text-align: center; padding-top: 0.2rem;">
            CRAS Vision AI • v2.5 Enterprise
        </div>
    </div>
    """)


# =========================================================================
# 4. PHẦN 1: TRUNG TÂM KIỂM ĐỊNH & DEMO (EXACT PASTEL COLORWAY)
# =========================================================================
def render_demo_inspection_section(learner: golden_learner.GoldenMemoryLearner):
    active_profile_str = learner.active_profile_name if learner.memory_bank is not None else "Khuôn mẫu vải chuẩn"

    # Main Page Title
    st.html('<div class="st-page-title">Hệ Thống Giám Định & Nhận Diện Khuyết Tật Vải (AOI)</div>')

    # Sidebar parameters
    st.sidebar.markdown("<p style='font-size: 0.8rem; font-weight: 700; color: #64748b;'>THIẾT LẬP THUẬT TOÁN</p>", unsafe_allow_html=True)
    mode_options = [
        "⚡ Hỗn Hợp Toàn Diện (Ensemble: Crease + Stain + Contrast)",
        "🎓 Khuôn Mẫu Vàng (Golden Memory Bank Few-Shot)",
        "📏 Bắt Nếp Gấp / Vết Nhăn (Hessian 2nd Deriv)",
        "🟡 Bắt Vết Ố Vàng / Dầu Mỡ (CIE Lab Shift)",
        "🤖 AI Sâu CRAS (Deep Residual Synthesis)"
    ]
    selected_mode = st.sidebar.selectbox("Chế độ:", mode_options, index=0)

    if "AI Sâu" in selected_mode:
        threshold = st.sidebar.slider("Ngưỡng xác suất lỗi CRAS (Threshold)", 0.30, 0.99, 0.50, 0.01, format="%.2f")
    elif "Khuôn Mẫu Vàng" in selected_mode:
        threshold = st.sidebar.slider("Ngưỡng Mẫu Vàng (Threshold)", 0.20, 1.00, 0.50, 0.05, format="%.2f")
    elif "Nếp Gấp" in selected_mode:
        threshold = st.sidebar.slider("Ngưỡng nếp gấp (Threshold)", 0.05, 0.60, 0.18, 0.02, format="%.2f")
    elif "Vết Ố" in selected_mode:
        threshold = st.sidebar.slider("Ngưỡng vết ố màu (Threshold)", 0.1, 1.5, 0.45, 0.05, format="%.2f")
    else:
        threshold = st.sidebar.slider("Ngưỡng tổng hợp (Threshold)", 0.1, 1.0, 0.40, 0.05, format="%.2f")

    # Sample images library
    sample_images = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_test_folder = os.path.join(base_dir, "test_samples")
    fallback_test_folder = r"d:\Luan.Nguyen\Tools\fabric_defect\test"
    test_folder = local_test_folder if os.path.isdir(local_test_folder) else fallback_test_folder

    if os.path.isdir(test_folder):
        cam_tests = {
            "export_20260923_154200_raw.png": "📸 Camera 1: Vải đen - Nếp gấp chữ T dọc & ngang",
            "export_20260923_154231_raw.png": "📸 Camera 2: Vải đen - Nếp gấp gợn sóng ngang",
            "export_20260923_154250_raw.png": "📸 Camera 3: Vải đen - Nếp gấp sắc nét rõ",
            "export_20260923_154325_raw.png": "📸 Camera 4: Vải đen - Bề mặt ít nếp gấp",
            "export_20260923_154829_raw.png": "📸 Camera 5: Vải trắng - Có vết ố vàng / dầu mỡ",
            "export_20260923_155041_raw.png": "📸 Camera 6: Vải đen - Nếp gấp đậm"
        }
        for f, label in cam_tests.items():
            full_p = os.path.join(test_folder, f)
            if os.path.exists(full_p):
                sample_images[label] = full_p

    for p in [
        os.path.join(test_folder, "user_fabric_sample.jpg"),
        os.path.join(base_dir, "results", "inference", "user_fabric_sample.jpg"),
    ]:
        if os.path.exists(p):
            sample_images["Mẫu vải trắng có vết ố (Ảnh ban đầu)"] = p
            break

    st.sidebar.markdown("<p style='font-size: 0.8rem; font-weight: 700; color: #64748b;'>NGUỒN HÌNH ẢNH</p>", unsafe_allow_html=True)
    upload_choice = st.sidebar.selectbox(
        "Nguồn ảnh:",
        ["Ảnh chụp từ Camera Nhà Máy"] + ["Tải ảnh riêng của bạn (Upload)"],
        index=0
    )

    pil_img = None
    sample_name = "Fabric Inspection Sample"

    if "Tải ảnh" in upload_choice:
        uploaded_file = st.sidebar.file_uploader("Tải lên ảnh vải", type=["png", "jpg", "jpeg", "bmp"])
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
            sample_name = uploaded_file.name
    else:
        sample_choice = st.sidebar.selectbox("Mẫu ảnh camera:", list(sample_images.keys()), index=0)
        if sample_choice in sample_images:
            img_path = sample_images[sample_choice]
            pil_img = Image.open(img_path).convert("RGB")
            sample_name = sample_choice

    auto_crop = st.sidebar.checkbox("Tự động cắt mép vải (Auto-Crop)", value=False)
    if pil_img is not None and auto_crop:
        pil_img, _ = auto_crop_fabric(pil_img)

    # Active profile selection
    avail_profiles = golden_learner.GoldenMemoryLearner.list_available_profiles()
    # Nothing is loaded by default: a Golden profile of another fabric would flood the ensemble with false alarms
    profile_options = {"(Không nạp - Dùng mặc định)": None}
    profile_options.update({p["display_name"]: p["path"] for p in avail_profiles})

    selected_p_label = st.sidebar.selectbox(
        "Khuôn mẫu đang nạp:",
        list(profile_options.keys()),
        index=0,
        help="Chọn Mẫu Vàng của ĐÚNG mã vải đang chạy. Khi đã chọn, chế độ Hỗn Hợp sẽ gộp thêm bản đồ Mẫu Vàng.",
    )
    selected_p_path = profile_options[selected_p_label]
    if selected_p_path and os.path.exists(selected_p_path):
        profile_file_name = os.path.basename(selected_p_path).replace(".bank", "")
        if learner.memory_bank is None or learner.active_profile_name != profile_file_name:
            try:
                learner.load_profile(selected_p_path)
            except Exception as e:
                st.sidebar.error(f"Lỗi: {e}")
    golden_active = selected_p_path is not None and learner.memory_bank is not None

    # Roll session (ASTM D5430 is graded on the whole roll, never on one frame)
    cfg = aoi_config.load_config()
    st.sidebar.markdown("<p style='font-size: 0.8rem; font-weight: 700; color: #64748b;'>PHIÊN KIỂM CUỘN (ASTM D5430)</p>", unsafe_allow_html=True)
    roll = st.session_state.get("roll_session")
    roll_info = st.sidebar.empty()
    start_roll_clicked = add_frame_clicked = finish_roll_clicked = False
    if roll is None:
        with st.sidebar.form("roll_start_form"):
            new_roll_id = st.text_input("Mã cuộn:", value=f"ROLL-{datetime.now():%Y%m%d-%H%M}")
            new_sku = st.text_input("Mã vải (SKU):", value=learner.active_profile_name if golden_active else "")
            new_width_in = st.number_input(
                "Khổ vải hữu dụng (inch):", min_value=0.0, max_value=200.0,
                value=float(cfg["usable_width_in"]), step=1.0,
                help="0 = lấy theo bề rộng vùng kiểm tra (ROI trừ biên vải) của camera.",
            )
            start_roll_clicked = st.form_submit_button("▶ Bắt đầu cuộn mới", use_container_width=True)
    else:
        add_frame_clicked = st.sidebar.button("➕ Ghi khung hiện tại vào cuộn", use_container_width=True)
        finish_roll_clicked = st.sidebar.button("🏁 Kết thúc cuộn & lưu báo cáo", use_container_width=True)

    if pil_img is None:
        st.info("👈 Vui lòng chọn hoặc tải ảnh bề mặt vải ở thanh bên trái để bắt đầu kiểm tra.")
        return

    cras_model, device = None, None
    if "AI Sâu" in selected_mode:
        model_dir = "results/models/backbone_0/itdd_cotton_fabric"
        if os.path.exists(model_dir):
            try:
                cras_model, device = load_cras_model(model_dir)
            except Exception:
                pass

    # Inference Execution
    t_start = time.time()
    details = {}
    master_anomaly = None

    if "Khuôn Mẫu Vàng" in selected_mode:
        if golden_active:
            score, heatmap_rgb, overlay, master_anomaly = learner.predict(pil_img)
        else:
            st.warning("Chưa chọn Mẫu Vàng cho mã vải này — đang dùng chế độ Hỗn Hợp thay thế.")
            score, heatmap_rgb, overlay, details, master_anomaly = run_hybrid_ensemble(pil_img)
    elif "Nếp Gấp" in selected_mode:
        score, heatmap_rgb, overlay, master_anomaly = detect_crease_defect(pil_img)
    elif "Vết Ố" in selected_mode:
        score, heatmap_rgb, overlay, master_anomaly = run_stain_enhancement(pil_img)
    elif "AI Sâu" in selected_mode and cras_model is not None:
        score, heatmap_rgb, overlay, master_anomaly, ood_ratio = run_cras_inference(cras_model, device, pil_img)
        if ood_ratio > 0.30:
            st.warning(
                f"⚠️ Mô hình CRAS đang ngoài miền dữ liệu huấn luyện: {ood_ratio * 100:.0f}% vùng phân tích bị chấm "
                "xác suất lỗi > 0.5. Kết quả KHÔNG đáng tin — cần huấn luyện lại CRAS trên chính loại vải này."
            )
        st.caption("Vùng phân tích CRAS: khung trắng ở giữa ảnh (Resize 329 + CenterCrop 288, giống lúc huấn luyện).")
    else:
        if "AI Sâu" in selected_mode:
            st.warning("Không nạp được mô hình CRAS — đang dùng chế độ Hỗn Hợp thay thế.")
        score, heatmap_rgb, overlay, details, master_anomaly = run_hybrid_ensemble(
            pil_img, golden=learner if golden_active else None, decision_level=threshold)
        if golden_active:
            st.caption(f"Hỗn Hợp đang gộp thêm Mẫu Vàng: **{learner.active_profile_name}**")

    t_infer_ms = (time.time() - t_start) * 1000.0
    px_per_inch = float(cfg["px_per_inch"])

    # Inspection ROI: graded columns = configured ROI minus the selvedge margin on both sides
    img_w, img_h = pil_img.size
    roi_x0, roi_x1 = defect_profiler.inspection_columns(
        img_w, cfg["roi_left_px"], cfg["roi_right_px"], int(round(cfg["selvedge_margin_in"] * px_per_inch)))
    master_anomaly = defect_profiler.apply_inspection_roi(master_anomaly, roi_x0, roi_x1)

    # Defect Profiler Execution (Calibrated Absolute Thresholding, no percentile forcing)
    profiler_kwargs = dict(
        min_area=int(cfg["min_area"]),
        aspect_ratio_crease_thresh=float(cfg["aspect_ratio"]),
        pixels_per_inch=px_per_inch,
        hole_needs_review=not cfg["backlight_available"],
    )
    component_maps = details.get("component_maps")
    if component_maps:
        # Ensemble: one instance extraction per detector map, then duplicate merge (no cross-detector blobs)
        component_maps = {k: defect_profiler.apply_inspection_roi(v, roi_x0, roi_x1) for k, v in component_maps.items()}
        defects, defect_mask = defect_profiler.extract_from_component_maps(
            pil_img, component_maps, threshold=threshold, **profiler_kwargs)
    else:
        defects, defect_mask = defect_profiler.extract_defect_instances(
            pil_img, master_anomaly, threshold=threshold, **profiler_kwargs)
    annotated_img = defect_profiler.draw_defect_bounding_boxes(pil_img, defects)
    if roi_x0 > 0 or roi_x1 < img_w:
        ann = np.array(annotated_img)
        for xl in (roi_x0, roi_x1 - 1):
            cv2.line(ann, (xl, 0), (xl, img_h - 1), (148, 163, 184), 4)
        annotated_img = Image.fromarray(ann)
    stats = defect_profiler.compute_defect_statistics(
        defects, (img_h, img_w), pixels_per_inch=px_per_inch, inspected_width_px=roi_x1 - roi_x0)

    # Roll session bookkeeping
    if start_roll_clicked:
        width_in = new_width_in if new_width_in > 0 else (roi_x1 - roi_x0) / px_per_inch
        roll = roll_session.RollSession(
            roll_id=new_roll_id.strip() or "ROLL", sku=new_sku.strip(), usable_width_in=width_in,
            pixels_per_inch=px_per_inch, frame_length_px=img_h,
            frame_overlap_in=float(cfg["frame_overlap_in"]),
            grade_a_limit=float(cfg["grade_a_limit"]), acceptance_limit=float(cfg["acceptance_limit"]),
        )
        st.session_state["roll_session"] = roll
        st.rerun()
    if roll is not None and add_frame_clicked:
        if abs(img_h / px_per_inch - roll.frame_length_in) > 0.05:
            st.warning("Kích thước khung khác với khung đầu cuộn — vị trí trên cuộn có thể bị lệch.")
        res = roll.add_frame(defects, frame_name=sample_name)
        st.toast(f"Đã ghi khung #{len(roll.frames)}: {res['added']} lỗi mới, {res['merged']} lỗi trùng đã gộp")
    if roll is not None and finish_roll_clicked:
        report_path = roll.finish()
        st.session_state["last_roll_summary"] = roll.summary()
        st.session_state["roll_session"] = None
        st.success(f"✅ Đã kết thúc cuộn **{roll.roll_id}** — báo cáo: `{report_path}`")
        roll = None

    roll_summary = roll.summary() if roll is not None else None
    if roll_summary is not None:
        roll_info.markdown(
            f"**Cuộn:** `{roll_summary['roll_id']}` · {roll_summary['frames']} khung · "
            f"{roll_summary['inspected_length_m']} m\n\n"
            f"**{roll_summary['total_astm_points']} điểm** · {roll_summary['points_per_100yd2']}/100yd² · "
            f"{grade_short_label(roll_summary['roll_grade'])} (tạm tính)"
        )
        grade_short = grade_short_label(roll_summary["roll_grade"]) + "*"
    else:
        roll_info.caption("Chưa có cuộn đang kiểm. Hạng A/B/C chỉ có khi kiểm theo cuộn.")
        grade_short = "Chưa xếp hạng"

    # ── THE 5 EXACT SIGNATURE PASTEL STATCARDS (100% MATCHED DESIGN SYSTEM) ──
    st.html(f"""
    <div class="st-stat-grid">
        <div class="st-stat-card st-card-blue">
            <div class="st-stat-left">
                <div class="st-stat-icon-box st-icon-blue">🧵</div>
                <div>
                    <div class="st-stat-title">Điểm Phạt ASTM D5430</div>
                    <div class="st-stat-num">{stats.get('total_astm_points', 0)} <span style="font-size: 0.82rem; font-weight: 500; color: #64748b;">điểm · {stats.get('points_per_100yd2', 0.0):.0f}/100yd²</span></div>
                </div>
            </div>
            <div class="st-stat-arrow">→</div>
        </div>

        <div class="st-stat-card st-card-lavender">
            <div class="st-stat-left">
                <div class="st-stat-icon-box st-icon-lavender">🔍</div>
                <div>
                    <div class="st-stat-title">Số Lượng Khuyết Tật</div>
                    <div class="st-stat-num">{len(defects)} <span style="font-size: 0.82rem; font-weight: 500; color: #64748b;">vết lỗi</span></div>
                </div>
            </div>
            <div class="st-stat-arrow">→</div>
        </div>

        <div class="st-stat-card st-card-purple">
            <div class="st-stat-left">
                <div class="st-stat-icon-box st-icon-purple">📐</div>
                <div>
                    <div class="st-stat-title">Tỷ Lệ Diện Tích Lỗi</div>
                    <div class="st-stat-num">{stats.get('defect_area_pct', 0.0):.2f}%</div>
                </div>
            </div>
            <div class="st-stat-arrow">→</div>
        </div>

        <div class="st-stat-card st-card-mint">
            <div class="st-stat-left">
                <div class="st-stat-icon-box st-icon-mint">🏷️</div>
                <div>
                    <div class="st-stat-title">Phân Hạng Cuộn Vải</div>
                    <div class="st-stat-num">{grade_short}</div>
                </div>
            </div>
            <div class="st-stat-arrow">→</div>
        </div>

        <div class="st-stat-card st-card-peach">
            <div class="st-stat-left">
                <div class="st-stat-icon-box st-icon-peach">⚡</div>
                <div>
                    <div class="st-stat-title">Thời Gian Phân Tích</div>
                    <div class="st-stat-num">{int(t_infer_ms)} <span style="font-size: 0.82rem; font-weight: 500; color: #64748b;">ms</span></div>
                </div>
            </div>
            <div class="st-stat-arrow">→</div>
        </div>
    </div>
    """)

    # ── 4-VIEW WORKSTATION CARDS (Pastel Clean) ──
    col1, col2, col3, col4 = st.columns(4)
    w_px, h_px = pil_img.size

    # Responsive display thumbnails for instantaneous browser loading
    disp_w = 768
    disp_scale = min(disp_w / w_px, 1.0)
    disp_size = (int(w_px * disp_scale), int(h_px * disp_scale))

    disp_raw = pil_img.resize(disp_size, Image.Resampling.BILINEAR)
    disp_hm = cv2.resize(heatmap_rgb, disp_size)
    disp_mask = cv2.resize(defect_mask, disp_size)
    disp_ann = annotated_img.resize(disp_size, Image.Resampling.BILINEAR)

    with col1:
        st.html(f"""
        <div class="st-view-card">
            <div class="st-view-header">
                <span>📸 1. Ảnh Gốc Thu Nhận</span>
                <span class="st-view-tag">{w_px}×{h_px}</span>
            </div>
        </div>
        """)
        st.image(disp_raw, use_container_width=True)

    with col2:
        st.html("""
        <div class="st-view-card">
            <div class="st-view-header">
                <span>🔥 2. Bản Đồ Nhiệt Anomaly</span>
                <span class="st-view-tag">JET MAP</span>
            </div>
        </div>
        """)
        st.image(disp_hm, use_container_width=True)

    with col3:
        st.html("""
        <div class="st-view-card">
            <div class="st-view-header">
                <span>🎯 3. Mặt Nạ Phân Vùng Lỗi</span>
                <span class="st-view-tag">BINARY MASK</span>
            </div>
        </div>
        """)
        st.image(disp_mask, use_container_width=True)

    with col4:
        st.html(f"""
        <div class="st-view-card">
            <div class="st-view-header">
                <span>📐 4. Định Vị Khuyết Tật ASTM</span>
                <span class="st-view-tag" style="background: #e0f2fe; color: #0284c7;">{grade_short}</span>
            </div>
        </div>
        """)
        st.image(disp_ann, use_container_width=True)

    st.html("<div style='height: 0.8rem;'></div>")

    if stats.get("review_count"):
        st.warning(
            f"🔍 {stats['review_count']} vùng nghi lỗ thủng cần người kiểm xác nhận (chưa có đèn nền) — "
            "chưa tính vào điểm phạt."
        )

    if roll_summary is not None:
        with st.expander(f"📜 Phiên kiểm cuộn {roll_summary['roll_id']} — {roll_summary['roll_grade']} (tạm tính)", expanded=False):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Chiều dài đã kiểm", f"{roll_summary['inspected_length_m']} m")
            m2.metric("Điểm phạt (đã giới hạn 4đ/yd)", roll_summary["total_astm_points"])
            m3.metric("Điểm / 100 yd²", roll_summary["points_per_100yd2"])
            m4.metric("Chờ xác nhận", roll_summary["review_count"])
            st.caption(
                f"Khổ hữu dụng {roll_summary['usable_width_in']:.1f} in · Ngưỡng buyer: Hạng A ≤ {roll_summary['grade_a_limit']:.0f}, "
                f"chấp nhận ≤ {roll_summary['acceptance_limit']:.0f} điểm/100yd² · Vị trí trên cuộn = số khung × chiều dài khung (chưa có encoder)."
            )
            if roll.defects:
                st.dataframe(pd.DataFrame(roll.defects), use_container_width=True, height=200)
            for t in [t for t in roll.defects if t["needs_review"]]:
                c_r1, c_r2, c_r3 = st.columns([3, 1, 1])
                c_r1.markdown(f"`{t['uid']}` {t['category']} tại {t['along'][0] / 36:.2f} yd, {t['length_in']:.1f} in")
                if c_r2.button("✔ Lỗi thật", key=f"confirm_{t['uid']}"):
                    roll.resolve_review(t["uid"], True)
                    st.rerun()
                if c_r3.button("✖ Không phải lỗi", key=f"reject_{t['uid']}"):
                    roll.resolve_review(t["uid"], False)
                    st.rerun()

    # ── DEEP ANALYSIS TABS ──
    tab_list, tab_pareto, tab_vlm, tab_pipe, tab_export = st.tabs([
        "📋 Danh Sách Khuyết Tật",
        "📊 Phân Tích Pareto",
        "🧠 Báo Cáo Kỹ Sư VLM",
        "🔄 Luồng Xử Lý 6 Bước",
        "💾 Xuất Dữ Liệu QC"
    ])

    df_defects = pd.DataFrame(defects)

    with tab_list:
        st.markdown(f"#### 📋 Chi tiết các khuyết tật định vị ({len(defects)} vết lỗi)")
        if not df_defects.empty:
            st.dataframe(df_defects, use_container_width=True, height=260)
        else:
            st.success("✅ Mẫu vải hoàn toàn sạch, không có khuyết tật nào vượt qua ngưỡng quy định.")

    with tab_pareto:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("#### 📊 Phân bố theo chủng loại khuyết tật:")
            if stats.get("type_counts"):
                type_df = pd.DataFrame(list(stats["type_counts"].items()), columns=["Chủng Loại Khuyết Tật", "Số Lượng"]).set_index("Chủng Loại Khuyết Tật")
                st.bar_chart(type_df, color="#f95721")
            else:
                st.info("Mẫu vải đạt chuẩn, không có lỗi để thống kê.")
        with col_p2:
            st.markdown("#### 🎯 Cơ cấu điểm phạt ASTM D5430:")
            p1 = sum(1 for d in defects if d.get("astm_points") == 1)
            p2 = sum(1 for d in defects if d.get("astm_points") == 2)
            p3 = sum(1 for d in defects if d.get("astm_points") == 3)
            p4 = sum(1 for d in defects if d.get("astm_points") == 4)
            points_dist = pd.DataFrame({
                "Mức điểm phạt": ["1 Điểm (≤3\")", "2 Điểm (3-6\")", "3 Điểm (6-9\")", "4 Điểm (>9\"/Thủng)"],
                "Số lượng": [p1, p2, p3, p4]
            }).set_index("Mức điểm phạt")
            st.bar_chart(points_dist, color="#7c3aed")

    with tab_vlm:
        st.markdown("#### 🧠 Báo Cáo Chẩn Đoán Căn Nguyên Cơ Khí & Tuân Thủ Tiêu Chuẩn")
        api_key_to_use = st.session_state.get("gemini_api_key_store", "")
        model_to_use = st.session_state.get("gemini_model_store", "gemini-2.0-flash-lite")

        report = vlm_inspector.generate_vlm_inspection_report(
            defects=defects,
            stats=stats,
            pil_image=pil_img,
            sample_name=sample_name,
            api_key=api_key_to_use,
            gemini_model=model_to_use
        )

        diag_list = report.get("diagnostics", [])
        diag_paragraphs = [f"• **{d.get('category', 'Lỗi')}**: {d.get('diagnosis', '')}" for d in diag_list]
        root_cause_display = "\n\n".join(diag_paragraphs) if diag_paragraphs else report.get("executive_summary", "Không ghi nhận khuyết tật bất thường.")

        action_list = report.get("corrective_actions", [])
        actions_display = "\n".join([f"{i+1}. {act}" for i, act in enumerate(action_list)]) if action_list else "Duy trì quy trình kiểm soát chất lượng cuộn vải hiện tại."

        v_c1, v_c2 = st.columns([1, 1])
        with v_c1:
            st.markdown(f"**Chuyên Gia:** `{report.get('ai_engine', 'Hệ Thống Phân Tích')}`")
            st.markdown(f"**Mã Báo Cáo:** `{report.get('report_id', 'QC-REPORT')}`")
            st.markdown(f"**Thời Gian:** `{report.get('timestamp', '')}`")
            st.info(f"**Căn Nguyên Kỹ Thuật (Root-Cause):**\n\n{root_cause_display}")

        with v_c2:
            st.markdown(f"**Đánh Giá Xếp Hạng:** **{report.get('roll_grade', stats.get('roll_grade', 'HẠNG A'))}**")
            st.markdown(f"**Trạng Thái Nghiệm Thu:** **{report.get('acceptance_status', stats.get('acceptance_status', 'ĐẠT CHUẨN'))}**")
            st.warning(f"**Khuyến Nghị Kỹ Sư Cơ Khí / Vận Hành:**\n\n{actions_display}")

    with tab_pipe:
        st.markdown("#### 🔄 Sơ Đồ Kiến Trúc Luồng Xử Lý 6 Bước Thời Gian Thực")
        flow_path = "docs/pipeline_flow.html"
        if os.path.exists(flow_path):
            with open(flow_path, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=420, scrolling=True)

    with tab_export:
        st.markdown("#### 💾 Xuất Dữ Liệu Báo Cáo Kiểm Định Chất Lượng (ERP / MES Integration)")
        c_exp1, c_exp2 = st.columns(2)
        rep_id = report.get('report_id', f"QC-{int(time.time())}")
        with c_exp1:
            json_str = json.dumps(report, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Tải Báo Cáo QC Toàn Diện (JSON)",
                data=json_str,
                file_name=f"fabric_qc_report_{rep_id}.json",
                mime="application/json",
                use_container_width=True
            )
        with c_exp2:
            csv_bytes = df_defects.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Tải Nhật Ký Khuyết Tật (CSV)",
                data=csv_bytes,
                file_name=f"fabric_defects_log_{rep_id}.csv",
                mime="text/csv",
                use_container_width=True
            )


# =========================================================================
# 5. PHẦN 2: QUẢN TRỊ NỀN TẢNG & HUẤN LUYỆN DỮ LIỆU
# =========================================================================
def render_admin_section(learner: golden_learner.GoldenMemoryLearner):
    if not is_admin_authenticated():
        st.html("""
        <div class="st-login-card">
            <div class="st-login-icon">🔑</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 0.35rem;">Quản Trị Hệ Thống AOI - Xác Thực</div>
            <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 1.4rem;">
                Khu vực quản trị cấu hình hệ thống & huấn luyện Mẫu Vàng.<br>
                Yêu cầu mật khẩu của Quản Trị Viên để tiếp tục.
            </div>
        </div>
        """)

        col_l1, col_l2, col_l3 = st.columns([1, 1.2, 1])
        with col_l2:
            with st.form("admin_login_form"):
                admin_pass_input = st.text_input(
                    "Mật khẩu quản trị viên:",
                    type="password",
                    placeholder="Nhập mật khẩu quản trị..."
                )
                submit_login = st.form_submit_button("Xác Thực & Mở Khóa", use_container_width=True)

                if submit_login:
                    input_hash = hashlib.sha256(admin_pass_input.encode("utf-8")).hexdigest()
                    if input_hash == ADMIN_PASS_HASH:
                        session_token = secrets.token_hex(32)
                        st.session_state["admin_authenticated"] = True
                        st.session_state["admin_session_token"] = session_token
                        st.session_state["admin_login_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.success("✅ Xác thực thành công! Đang chuyển hướng...")
                        time.sleep(0.3)
                        st.rerun()
                    else:
                        st.error("❌ Mật khẩu không chính xác. Yêu cầu truy cập bị từ chối.")
                        st.stop()
        return

    # Authenticated Admin View
    token_preview = st.session_state.get("admin_session_token", "")[:12] + "..."
    login_time = st.session_state.get("admin_login_time", "Vừa xong")

    st.html('<div class="st-page-title">⚙️ Quản Trị Nền Tảng & Huấn Luyện Mẫu Chuẩn</div>')

    profiles_list = golden_learner.GoldenMemoryLearner.list_available_profiles()
    coreset_patches = learner.memory_bank.shape[0] if learner.memory_bank is not None else 0

    # 5 Exact Pastel Cards for Admin Platform
    st.html(f"""
    <div class="st-stat-grid">
        <div class="st-stat-card st-card-blue">
            <div class="st-stat-left">
                <div class="st-stat-icon-box st-icon-blue">📦</div>
                <div>
                    <div class="st-stat-title">Hồ Sơ Mẫu Vàng (.bank)</div>
                    <div class="st-stat-num">{len(profiles_list)} <span style="font-size: 0.82rem; font-weight: 500; color: #64748b;">hồ sơ</span></div>
                </div>
            </div>
            <div class="st-stat-arrow">→</div>
        </div>

        <div class="st-stat-card st-card-lavender">
            <div class="st-stat-left">
                <div class="st-stat-icon-box st-icon-lavender">🧠</div>
                <div>
                    <div class="st-stat-title">Bộ Nhớ Coreset</div>
                    <div class="st-stat-num">{coreset_patches} <span style="font-size: 0.82rem; font-weight: 500; color: #64748b;">patches</span></div>
                </div>
            </div>
            <div class="st-stat-arrow">→</div>
        </div>

        <div class="st-stat-card st-card-purple">
            <div class="st-stat-left">
                <div class="st-stat-icon-box st-icon-purple">⚙️</div>
                <div>
                    <div class="st-stat-title">Mô Hình Cốt Lõi</div>
                    <div class="st-stat-num" style="font-size: 1.05rem;">PatchCore/CRAS</div>
                </div>
            </div>
            <div class="st-stat-arrow">→</div>
        </div>

        <div class="st-stat-card st-card-mint">
            <div class="st-stat-left">
                <div class="st-stat-icon-box st-icon-mint">🛡️</div>
                <div>
                    <div class="st-stat-title">Lọc Nhiễm Độc</div>
                    <div class="st-stat-num" style="font-size: 1.05rem;">Anti-Poison (p98)</div>
                </div>
            </div>
            <div class="st-stat-arrow">→</div>
        </div>

        <div class="st-stat-card st-card-peach">
            <div class="st-stat-left">
                <div class="st-stat-icon-box st-icon-peach">🔒</div>
                <div>
                    <div class="st-stat-title">Trạng Thái Đóng Băng</div>
                    <div class="st-stat-num" style="font-size: 1.05rem;">Production Lock</div>
                </div>
            </div>
            <div class="st-stat-arrow">→</div>
        </div>
    </div>
    """)

    col_hdr1, col_hdr2 = st.columns([4.2, 1])
    with col_hdr2:
        if st.button("🚪 Đăng Xuất (Khóa Trang)", use_container_width=True):
            st.session_state["admin_authenticated"] = False
            st.session_state["admin_session_token"] = None
            st.success("Đã khóa phân hệ quản trị.")
            st.rerun()

    admin_tab1, admin_tab2, admin_tab3, admin_tab4 = st.tabs([
        "🎓 Huấn Luyện Mẫu Chuẩn",
        "📦 Kho Hồ Sơ Khuôn Mẫu (.bank)",
        "🛠️ Tham Số Quang Học & Thuật Toán AOI",
        "🤖 Cognitive VLM & Tích Hợp MES/PLC"
    ])

    with admin_tab1:
        st.markdown("#### 🎓 Huấn Luyện Vân Dệt Mới Bằng PatchCore Coreset (< 5 Giây)")
        st.info("💡 Không cần huấn luyện lặp (0 epochs). Chỉ cần tải lên 3–10 tấm ảnh vải mẫu sạch đạt chuẩn, hệ thống tự động ghi nhớ phân phối chuẩn và tự động kích hoạt màng lọc chống nhiễm độc dữ liệu.")

        col_t1, col_t2 = st.columns([1.5, 1])
        with col_t1:
            sku_name = st.text_input("Tên Mã Hàng / SKU:", value=f"Vai_Det_SKU_{int(time.time()) % 1000}")
            golden_files = st.file_uploader(
                "Tải lên các ảnh vải chuẩn sạch (Good Samples):",
                type=["png", "jpg", "jpeg", "bmp"],
                accept_multiple_files=True,
                key="admin_golden_upload"
            )

        with col_t2:
            st.markdown("**🛡️ Cơ Chế Bảo Vệ Chống Nhiễm Độc (Anti-Poisoning):**")
            st.caption("• Tự động tính ma trận phân vị khoảng cách chéo $p_{98}$ giữa các patch.")
            st.caption("• Nếu ảnh mẫu bị dính nếp gấp hoặc đốm ố ($p_{98} > 0.58$), hệ thống sẽ cảnh báo và từ chối nạp.")
            st.caption("• Tự động đóng băng sản xuất (**Production Freeze**) và xuất file `.bank` chống sửa đổi.")

        if st.button("⚡ Bắt Đầu Huấn Luyện & Tạo Hồ Sơ Chuẩn", use_container_width=True):
            if not golden_files:
                st.warning("Vui lòng tải lên ít nhất 1 ảnh vải chuẩn sạch.")
            else:
                with st.spinner("Đang trích xuất đặc trưng đa tầng và chọn tập Coreset..."):
                    pil_samples = [Image.open(f).convert("RGB") for f in golden_files]
                    cal_meta = learner.calibrate(pil_samples, profile_name=sku_name)

                    st.success(f"🎉 Huấn luyện thành công Mã Hàng **{sku_name}** trong **{cal_meta['learning_time_sec']} giây**!")

                    res_c1, res_c2, res_c3 = st.columns(3)
                    with res_c1:
                        st.metric("Kích thước Coreset Memory", f"{cal_meta['bank_vectors']} patches")
                    with res_c2:
                        st.metric("Ngưỡng Chuẩn Hóa", f"{cal_meta['calibrated_threshold']:.4f}")
                    with res_c3:
                        st.metric("Trạng Thái Khóa", "🔒 ĐÃ ĐÓNG BĂNG")

                    if cal_meta.get("outlier_warnings"):
                        for w in cal_meta["outlier_warnings"]:
                            st.error(w)

    with admin_tab2:
        st.markdown("#### 📦 Kho Lưu Trữ Hồ Sơ Mẫu Vàng Đã Khóa Sản Xuất")
        st.caption("Các hồ sơ `.bank` lưu trữ vector chuẩn hóa cùng mã băm xác thực toàn vẹn SHA-256.")

        profiles_list = golden_learner.GoldenMemoryLearner.list_available_profiles()
        if profiles_list:
            df_profiles = pd.DataFrame(profiles_list)
            st.dataframe(df_profiles[["display_name", "size_kb", "modified_time", "path"]], use_container_width=True)

            sel_prof_name = st.selectbox("Chọn hồ sơ để thao tác:", [p["display_name"] for p in profiles_list])
            sel_prof_obj = next(p for p in profiles_list if p["display_name"] == sel_prof_name)

            col_p_act1, col_p_act2, col_p_act3 = st.columns(3)
            with col_p_act1:
                if st.button("⚡ Kích Hoạt Hồ Sơ Này (Active Profile)", use_container_width=True):
                    learner.load_profile(sel_prof_obj["path"])
                    st.success(f"Đã kích hoạt hồ sơ: **{sel_prof_name}** cho ca sản xuất.")
            with col_p_act2:
                with open(sel_prof_obj["path"], "rb") as f_bank:
                    st.download_button(
                        label="📥 Tải File Hồ Sơ (.bank)",
                        data=f_bank.read(),
                        file_name=os.path.basename(sel_prof_obj["path"]),
                        mime="application/octet-stream",
                        use_container_width=True
                    )
            with col_p_act3:
                if st.button("🗑️ Xóa Hồ Sơ Này", use_container_width=True):
                    try:
                        os.remove(sel_prof_obj["path"])
                        st.warning(f"Đã xóa file: {sel_prof_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Lỗi: {e}")
        else:
            st.info("Chưa có hồ sơ Mẫu Vàng nào được tạo trong `results/profiles/`.")

    with admin_tab3:
        st.markdown("#### 🛠️ Hiệu Chuẩn Tham Số Quang Học & Thuật Toán Tầng 1 (AOI)")
        st.caption("Các tham số này điều khiển ngưỡng lọc nhiễu sợi dệt và độ nhạy nhận diện nếp gấp / vết ố.")

        cfg = aoi_config.load_config()
        with st.form("aoi_params_form"):
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                new_min_area = st.number_input(
                    "Diện tích lỗi tối thiểu (Min Area px) - Lọc nhiễu sợi vải:",
                    min_value=20, max_value=2000,
                    value=int(cfg["min_area"]),
                    step=10,
                    help="Các vùng khuyết tật có diện tích pixel nhỏ hơn giá trị này sẽ bị loại bỏ để tránh báo động giả do xơ vải bình thường."
                )
                new_aspect_ratio = st.number_input(
                    "Tỷ lệ dạng nếp gấp linear (Aspect Ratio Threshold):",
                    min_value=1.5, max_value=10.0,
                    value=float(cfg["aspect_ratio"]),
                    step=0.2,
                    help="Tỷ lệ Chiều Dài / Chiều Rộng lớn hơn ngưỡng này sẽ được phân loại vào nhóm Nếp Gấp / Vết Nhăn Dài."
                )
                st.markdown("**Vùng kiểm tra theo bề ngang khổ vải (ROI, pixel ảnh gốc):**")
                new_roi_left = st.number_input("Cột trái (px):", min_value=0, max_value=20000, value=int(cfg["roi_left_px"]), step=10)
                new_roi_right = st.number_input("Cột phải (px, 0 = hết ảnh):", min_value=0, max_value=20000, value=int(cfg["roi_right_px"]), step=10)
                new_selvedge = st.number_input(
                    "Biên vải bỏ qua mỗi bên (inch):", min_value=0.0, max_value=6.0,
                    value=float(cfg["selvedge_margin_in"]), step=0.25,
                    help="Lỗi nằm sát mép vải trong khoảng này không bị tính điểm (thông lệ 1 inch).",
                )

            with col_a2:
                st.markdown("**Hiệu Chuẩn Kích Thước Quang Học Thực Địa (FOV Calibration):**")
                px_per_inch = st.number_input(
                    "Số điểm ảnh trên mỗi Inch (Pixels per Inch):",
                    min_value=5.0, max_value=2000.0,
                    value=float(cfg["px_per_inch"]),
                    step=1.0,
                    help="Đo bằng bảng bàn cờ / thước chuẩn đặt trên mặt vải: số pixel ảnh gốc ứng với 1 inch (25.4 mm). Quyết định trực tiếp điểm phạt ASTM.",
                )
                st.caption(f"Tương đương: 1 pixel ≈ **{(25.4 / px_per_inch):.3f} mm**")
                st.markdown("**Cuộn vải & tiêu chí buyer (điểm / 100 yd², cả cuộn):**")
                new_overlap = st.number_input("Độ chồng giữa 2 khung liên tiếp (inch):", min_value=0.0, max_value=24.0,
                                              value=float(cfg["frame_overlap_in"]), step=0.5)
                new_usable_width = st.number_input("Khổ vải hữu dụng mặc định (inch, 0 = theo ROI):", min_value=0.0,
                                                   max_value=200.0, value=float(cfg["usable_width_in"]), step=1.0)
                new_grade_a = st.number_input("Hạng A khi ≤ (điểm/100yd²):", min_value=0.0, max_value=200.0,
                                              value=float(cfg["grade_a_limit"]), step=1.0)
                new_accept = st.number_input("Chấp nhận khi ≤ (điểm/100yd²):", min_value=0.0, max_value=200.0,
                                             value=float(cfg["acceptance_limit"]), step=1.0)
                new_backlight = st.checkbox(
                    "Trạm đã có đèn nền (backlight)", value=bool(cfg["backlight_available"]),
                    help="Khi chưa có, vùng nghi lỗ thủng được chuyển cho người kiểm xác nhận thay vì tự chấm 4 điểm.",
                )

            save_params_btn = st.form_submit_button("💾 Lưu Thiết Lập Tham Số", use_container_width=True)
            if save_params_btn:
                if new_accept < new_grade_a:
                    st.error("Ngưỡng chấp nhận phải ≥ ngưỡng Hạng A.")
                else:
                    cfg.update({
                        "min_area": int(new_min_area),
                        "aspect_ratio": float(new_aspect_ratio),
                        "px_per_inch": float(px_per_inch),
                        "roi_left_px": int(new_roi_left),
                        "roi_right_px": int(new_roi_right),
                        "selvedge_margin_in": float(new_selvedge),
                        "frame_overlap_in": float(new_overlap),
                        "usable_width_in": float(new_usable_width),
                        "grade_a_limit": float(new_grade_a),
                        "acceptance_limit": float(new_accept),
                        "backlight_available": bool(new_backlight),
                    })
                    aoi_config.save_config(cfg)
                    st.success(f"✅ Đã lưu cấu hình vào `{aoi_config.CONFIG_PATH}` (giữ nguyên sau khi khởi động lại).")

    with admin_tab4:
        st.markdown("#### 🤖 Cấu Hình Cognitive VLM & Kết Nối Công Nghiệp MES / PLC")

        with st.form("cloud_mes_form"):
            st.markdown("**1. Cấu hình Trí Tuệ Nhân Tạo Google Gemini Vision:**")
            new_api_key = st.text_input(
                "Google Gemini API Key:",
                value=st.session_state.get("gemini_api_key_store", ""),
                type="password",
                placeholder="Nhập API Key Gemini..."
            )
            model_options = [
                "gemini-2.0-flash-lite",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-1.5-flash-8b",
                "gemini-2.5-flash"
            ]
            cur_model = st.session_state.get("gemini_model_store", "gemini-2.0-flash-lite")
            new_model = st.selectbox(
                "Mô hình VLM triển khai:",
                model_options,
                index=model_options.index(cur_model) if cur_model in model_options else 0
            )

            st.markdown("---")
            st.markdown("**2. Cấu hình Kết Nối Máy Chủ Quản Lý Sản Xuất (MES Endpoint):**")
            new_mes_endpoint = st.text_input(
                "MES REST API URL Endpoint:",
                value=st.session_state.get("mes_endpoint_store", "http://mes.factory.internal/api/v1/quality/fabric-inspection")
            )

            st.markdown("---")
            st.markdown("**3. Bản Đồ Kênh Digital Output Điều Khiển PLC (24V DC):**")
            plc_c1, plc_c2 = st.columns(2)
            with plc_c1:
                st.checkbox("DO_01: Van khí nén phun mực đánh dấu mép vải", value=True)
                st.checkbox("DO_02: Còi hú & Đèn tháp cảnh báo Grade B", value=True)
            with plc_c2:
                st.checkbox("DO_03: Rơ-le dừng khẩn cấp (E-Stop) khi Reject cuộn", value=True)
                st.checkbox("DI_01: Đồng bộ xung Rotary Encoder đo mét vải", value=True)

            save_cloud_btn = st.form_submit_button("💾 Lưu Toàn Bộ Thiết Lập Kết Nối", use_container_width=True)
            if save_cloud_btn:
                st.session_state["gemini_api_key_store"] = new_api_key.strip()
                st.session_state["gemini_model_store"] = new_model
                st.session_state["mes_endpoint_store"] = new_mes_endpoint.strip()
                st.success("✅ Đã cập nhật cấu hình Cognitive VLM & Tích hợp MES/PLC.")


if __name__ == "__main__":
    main()

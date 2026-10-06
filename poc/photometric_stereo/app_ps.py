"""
app_ps.py  (POC - standalone Streamlit app, separate from the production app.py)

    streamlit run poc/photometric_stereo/app_ps.py --server.port 8503

Upload the 4 raking-light captures (+ optionally the current front-lit capture), mark the folds you can
see, and compare their SNR: current light vs one raking light vs photometric stereo (4 lights).
"""

import io
import os
import sys
import zipfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ps_core  # noqa: E402
import run_ps  # noqa: E402
import synthetic  # noqa: E402

st.set_page_config(page_title="POC Photometric Stereo - Nếp gấp vải", layout="wide")
st.title("POC · Photometric Stereo cho nếp gấp vải")
st.caption(
    "Thử nghiệm độc lập với hệ thống chính. Mục tiêu: đo bằng số liệu xem chiếu sáng 4 hướng xiên có làm nếp gấp "
    "nổi rõ hơn đèn hiện tại không, trước khi quyết định phần cứng."
)

LABELS = {"left": "Đèn từ TRÁI", "right": "Đèn từ PHẢI", "top": "Đèn từ TRÊN", "bottom": "Đèn từ DƯỚI",
          "front": "Đèn hiện tại (để so sánh, tuỳ chọn)"}


def decode(uploaded):
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Không đọc được ảnh {uploaded.name}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = img.astype(np.float32) * (255.0 / float(np.iinfo(img.dtype).max))
    return img


def with_grid(u8, step=200):
    vis = cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB) if u8.ndim == 2 else u8.copy()
    h, w = vis.shape[:2]
    for x in range(0, w, step):
        cv2.line(vis, (x, 0), (x, h - 1), (0, 200, 255), 1)
        cv2.putText(vis, str(x), (x + 3, 18), 0, 0.5, (0, 200, 255), 1)
    for y in range(0, h, step):
        cv2.line(vis, (0, y), (w - 1, y), (0, 200, 255), 1)
        cv2.putText(vis, str(y), (3, y - 4), 0, 0.5, (0, 200, 255), 1)
    return vis


def draw_segments(u8, segments):
    vis = cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB) if u8.ndim == 2 else u8.copy()
    for name, (x0, y0, x1, y1) in segments:
        cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), (255, 60, 60), 2)
        cv2.putText(vis, name.split(" ")[0], (int(x0) + 5, int(y0) + 18), 0, 0.7, (255, 60, 60), 2)
    return vis


# ------------------------------------------------------------------ inputs
with st.sidebar:
    source = st.radio("Nguồn dữ liệu", ["Dữ liệu tổng hợp (kiểm tra code)", "Ảnh chụp thật (4 hướng đèn)"])
    elevation = st.slider("Góc nâng của đèn so với mặt vải (độ)", 5, 45, 18,
                          help="Chỉ ảnh hưởng tỉ lệ độ dốc, không ảnh hưởng vị trí nếp.")
    band = st.slider("Nửa bề rộng vùng đo quanh nếp (px)", 10, 120, 40)
    align = st.checkbox("Tự căn chỉnh lệch vị trí giữa các ảnh", value=True)

images, default_segments = {}, []
if source.startswith("Dữ liệu tổng hợp"):
    images, default_segments = synthetic.make_capture_set()
    st.info("Dữ liệu tổng hợp: vải đen có vân loang + 4 nếp đã biết vị trí. Chỉ để kiểm tra code và nguyên lý; "
            "vải thật không lý tưởng như mô hình này, nên con số thật sẽ thấp hơn.")
else:
    st.markdown("**Quy trình chụp:** camera và vải cố định; tắt đèn trần; mỗi ảnh chỉ bật 1 đèn đặt thấp ~15-20° "
                "so với mặt vải; khóa phơi sáng/gain thủ công, giống nhau cho cả 4 ảnh.")
    cols = st.columns(5)
    for col, key in zip(cols, ["left", "right", "top", "bottom", "front"]):
        with col:
            up = st.file_uploader(LABELS[key], type=["png", "bmp", "tif", "tiff", "jpg", "jpeg"], key=f"up_{key}")
            if up is not None:
                images[key] = decode(up)
                st.image(ps_core.to_u8(images[key]), use_container_width=True)
    missing = [LABELS[k] for k in ps_core.DIRECTIONS if k not in images]
    if missing:
        st.warning("Còn thiếu: " + ", ".join(missing))
        st.stop()

# ------------------------------------------------------------------ fold segments to measure
st.subheader("1. Đánh dấu các nếp cần đo")
st.caption("Nhập toạ độ đầu-cuối mỗi nếp (pixel ảnh gốc). Dùng lưới toạ độ trên ảnh 'shape' bên dưới để đọc vị trí.")
seg_df = pd.DataFrame(
    [{"name": n, "x0": s[0], "y0": s[1], "x1": s[2], "y1": s[3]} for n, s in default_segments]
    or [{"name": "1 nep", "x0": 0, "y0": 0, "x1": 0, "y1": 0}]
)
seg_df = st.data_editor(seg_df, num_rows="dynamic", use_container_width=True, key="segments")
segments = [(str(r["name"]), (float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])))
            for _, r in seg_df.iterrows()
            if np.hypot(float(r["x1"]) - float(r["x0"]), float(r["y1"]) - float(r["y0"])) >= 10]

# ------------------------------------------------------------------ compute
cfg = ps_core.PSConfig(elevation_deg=float(elevation), align=align)
with st.spinner("Đang tính photometric stereo..."):
    if segments:
        res, rows = run_ps.compare(images, segments, cfg, band=band)
    else:
        res, rows = ps_core.compute({k: images[k] for k in ps_core.DIRECTIONS}, cfg), []
for n in res.notes:
    st.warning(n)

# ------------------------------------------------------------------ SNR table + verdict
st.subheader("2. So sánh độ rõ của nếp (SNR)")
if rows:
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="fold", columns="map", values="median_snr")
    share = df.pivot(index="fold", columns="map", values="share_snr_ge_2")
    st.markdown("**SNR mỗi vị trí** (tín hiệu nếp / nhiễu pixel). < 1: chìm trong nhiễu · 2-3: nhìn thấy · > 5: rõ, thuật toán đơn giản là đủ.")
    st.dataframe(pivot.style.format("{:.2f}").background_gradient(cmap="RdYlGn", vmin=0, vmax=5), use_container_width=True)
    st.markdown("**Tỷ lệ chiều dài nếp có SNR ≥ 2**")
    st.dataframe(share.style.format("{:.0%}").background_gradient(cmap="RdYlGn", vmin=0, vmax=1), use_container_width=True)
    ps_col = "PS shape (4 lights)"
    base_col = "front light (hien tai)" if "front light (hien tai)" in pivot.columns else None
    ps_med = float(pivot[ps_col].median())
    if base_col:
        gain = ps_med / max(float(pivot[base_col].median()), 1e-6)
        st.metric("Cải thiện SNR trung vị so với đèn hiện tại", f"x{gain:.1f}")
    if ps_med >= 5:
        st.success("Nếp gấp nổi rõ trên ảnh hình dạng → hướng chiếu sáng 4 hướng khả thi; bộ dò nếp có thể rất đơn giản.")
    elif ps_med >= 2:
        st.warning("Cải thiện nhưng chưa dư dả → thử góc đèn thấp hơn, đèn mạnh hơn, hoặc vải căng phẳng hơn.")
    else:
        st.error("Chưa đủ rõ → kiểm tra lại quy trình chụp (lệch vị trí, bão hòa, đèn quá cao) trước khi kết luận.")
else:
    st.info("Chưa có nếp nào được đánh dấu → chỉ hiển thị ảnh kết quả.")

# ------------------------------------------------------------------ images
st.subheader("3. Ảnh kết quả")
shape_u8 = ps_core.to_u8(res.shape, symmetric=True)
fold_u8 = cv2.cvtColor(cv2.applyColorMap(np.clip(ps_core.fold_strength(res.shape, res.valid) / 12 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
c1, c2, c3 = st.columns(3)
with c1:
    if "front" in images:
        st.markdown("**Đèn hiện tại**")
        st.image(draw_segments(ps_core.to_u8(ps_core.to_gray_float(images["front"])), segments), use_container_width=True)
    st.markdown("**1 đèn xiên (trái)**")
    st.image(ps_core.to_u8(ps_core.to_gray_float(images["left"])), use_container_width=True)
with c2:
    st.markdown("**Ảnh HÌNH DẠNG (shape) - nếp gấp, đã loại vân màu** · có lưới toạ độ")
    st.image(with_grid(draw_segments(shape_u8, segments)), use_container_width=True)
    st.markdown("**Độ mạnh nếp (fold strength, z-score)**")
    st.image(fold_u8, use_container_width=True)
with c3:
    st.markdown("**Ảnh ALBEDO - màu/vân, đã loại bóng nếp** (dùng cho vết ố)")
    st.image(ps_core.to_u8(res.albedo), use_container_width=True)
    st.markdown("**Bản đồ pháp tuyến (normal)**")
    st.image(ps_core.normals_to_rgb(res.normals), use_container_width=True)

# ------------------------------------------------------------------ download
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for name, arr in {"shape.png": shape_u8, "albedo.png": ps_core.to_u8(res.albedo),
                      "fold_strength.png": cv2.cvtColor(fold_u8, cv2.COLOR_RGB2BGR),
                      "normals.png": cv2.cvtColor(ps_core.normals_to_rgb(res.normals), cv2.COLOR_RGB2BGR)}.items():
        z.writestr(name, cv2.imencode(".png", arr)[1].tobytes())
    if rows:
        z.writestr("snr.csv", pd.DataFrame(rows).to_csv(index=False))
st.download_button("📥 Tải kết quả (ZIP)", buf.getvalue(), file_name="ps_poc_results.zip", mime="application/zip")

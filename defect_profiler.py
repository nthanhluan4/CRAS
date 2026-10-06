"""
defect_profiler.py
Industrial Fabric Defect Profiling & ASTM D5430 4-Point System Calculator
Stage 1: Deterministic Machine Vision / AOI Engine
"""

import math
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Default optical calibration. Must be re-measured per camera / lens / working distance
# (Admin > FOV Calibration) because every ASTM length and every size-based rule below depends on it.
DEFAULT_PIXELS_PER_INCH = 100.0

# Geometric rules expressed in inches (identical to the former pixel constants at 100 px/inch)
LINK_GAP_IN = 0.31           # morphological bridging of fragmented creases / chalk strokes
LINEAR_MAX_THICKNESS_IN = 0.25
LINEAR_MIN_LENGTH_IN = 0.50
HOLE_MIN_AREA_SQIN = 0.12
HOLE_MIN_CONTRAST = 60.0     # grey levels vs. fabric median (hole shows the backing/backlight)
HOLE_MIN_FILL = 0.45         # share of the blob that really shows the backing: ~0.5-1 for a hole (blurred rim included), ~0.3 for a print / logo


def inspection_columns(image_width: int, roi_left_px: int = 0, roi_right_px: int = 0,
                       selvedge_margin_px: int = 0) -> Tuple[int, int]:
    """Pixel columns [x0, x1) that are graded: the configured ROI minus the selvedge margin on both sides."""
    left = max(int(roi_left_px), 0)
    right = int(roi_right_px) if roi_right_px and roi_right_px > left else image_width
    right = min(right, image_width)
    x0 = min(left + max(int(selvedge_margin_px), 0), image_width)
    x1 = max(right - max(int(selvedge_margin_px), 0), x0)
    return x0, x1


def apply_inspection_roi(anomaly_map: np.ndarray, x0: int, x1: int) -> np.ndarray:
    """Zero the anomaly map outside the graded columns (backdrop, selvedge, machine frame)."""
    out = anomaly_map.copy()
    out[:, :x0] = 0
    out[:, x1:] = 0
    return out


def grade_from_density(points_per_100yd2: float, total_points: int,
                       grade_a_limit: float = 20.0, acceptance_limit: float = 40.0) -> Dict[str, str]:
    """Buyer grade from ASTM D5430 point density (points per 100 square yards)."""
    if total_points == 0:
        return {"roll_grade": "HẠNG A (XUẤT KHẨU - PREMIUM)", "grade_color": "#10B981",
                "acceptance_status": "ĐẠT CHUẨN (PASSED)"}
    if points_per_100yd2 <= grade_a_limit:
        return {"roll_grade": "HẠNG A (ĐẠT CHUẨN THƯƠNG MẠI)", "grade_color": "#3B82F6",
                "acceptance_status": "ĐẠT CHUẨN (PASSED)"}
    if points_per_100yd2 <= acceptance_limit:
        return {"roll_grade": "HẠNG B (CHẤP NHẬN CÓ ĐIỀU KIỆN)", "grade_color": "#F59E0B",
                "acceptance_status": "CẦN XỬ LÝ NHIỆT / LÀ ÉP"}
    return {"roll_grade": "HẠNG C / REJECT (PHẾ PHẨM)", "grade_color": "#EF4444",
            "acceptance_status": "KHÔNG ĐẠT (REJECTED)"}


FRAME_ONLY_GRADE = {
    "roll_grade": "CHƯA XẾP HẠNG (cần dữ liệu cả cuộn)",
    "grade_color": "#64748B",
    "acceptance_status": "CHỜ KẾT QUẢ CUỘN",
}


def astm_d5430_points(length_inches: float, is_hole: bool = False) -> int:
    """
    ASTM D5430 4-point system.
      Length of defect:  <=3 in -> 1 | 3-6 in -> 2 | 6-9 in -> 3 | >9 in -> 4
      Holes / openings:  <=1 in -> 2 | >1 in -> 4
    """
    if is_hole:
        return 2 if length_inches <= 1.0 else 4
    if length_inches <= 3.0:
        return 1
    if length_inches <= 6.0:
        return 2
    if length_inches <= 9.0:
        return 3
    return 4


def extract_defect_instances(
    pil_image: Image.Image,
    heatmap_or_mask: np.ndarray,
    threshold: float = 0.35,
    min_area: int = 150,
    aspect_ratio_crease_thresh: float = 2.8,
    pixels_per_inch: float = DEFAULT_PIXELS_PER_INCH,
    hole_needs_review: bool = True,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Extract individual defect instances (blobs) using Connected Component Analysis (CCA)
    and compute geometric invariants + ASTM D5430 penalty points.

    `heatmap_or_mask` must be an anomaly MAP (0..1 float or 0..255 uint8), not a colour-mapped image.
    Uses absolute calibrated thresholding (NO percentile forcing) so that defect-free
    fabrics produce zero false positive detections.
    `hole_needs_review`: without a backlight a hole cannot be told apart from a bright print, so
    suspected holes are flagged `needs_review` and kept out of the ASTM total until an operator confirms.
    """
    np_img = np.array(pil_image)
    orig_h, orig_w = np_img.shape[:2]
    ppi = max(float(pixels_per_inch), 1e-3)

    if heatmap_or_mask.ndim == 3:
        raise ValueError("extract_defect_instances expects a single-channel anomaly map, not an RGB heatmap.")

    # Convert heatmap to float 0.0 .. 1.0
    if heatmap_or_mask.dtype == np.uint8:
        hm_float = heatmap_or_mask.astype(np.float32) / 255.0
    else:
        hm_float = np.clip(heatmap_or_mask.astype(np.float32), 0.0, 1.0)

    if hm_float.shape[:2] != (orig_h, orig_w):
        hm_float = cv2.resize(hm_float, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # 1. Absolute calibrated thresholding
    binary = (hm_float >= threshold).astype(np.uint8) * 255

    # 2. Multi-directional morphological line linking (connects fragmented creases and chalk strokes)
    link = max(int(round(LINK_GAP_IN * ppi)) | 1, 3)
    k_link_h = cv2.getStructuringElement(cv2.MORPH_RECT, (link, 3))
    k_link_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, link))
    binary_linked = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_link_h)
    binary_linked = cv2.morphologyEx(binary_linked, cv2.MORPH_CLOSE, k_link_v)

    # 3. Connected Components with Stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_linked, connectivity=8)

    # Photometric analysis (median = robust to the defects themselves)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    bg_level = float(np.median(gray))

    # Color space for stain/discoloration check
    lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
    b_chan = lab[:, :, 2]
    b_median = float(np.median(b_chan))

    defects = []
    defect_id = 1
    clean_mask = np.zeros_like(binary)

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])

        # Work inside the bounding box only (full-image `labels == i` is O(components x pixels))
        roi = (slice(y, y + h), slice(x, x + w))
        component_mask = (labels[roi] == i)
        # Photometry on truly anomalous pixels only, not on the gaps bridged by the linking step
        measure_mask = component_mask & (binary[roi] > 0)
        if not measure_mask.any():
            measure_mask = component_mask

        # Get contour for oriented bounding box
        contours, _ = cv2.findContours(component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)

        # Min area rotated rectangle
        rect = cv2.minAreaRect(c)
        (rcx, rcy), (rw, rh), angle = rect
        dim_length = max(rw, rh)
        dim_width = max(min(rw, rh), 1.0)
        aspect_ratio = float(dim_length / dim_width)

        # Average line thickness = Area / Length
        mean_thickness = float(area / max(dim_length, 1.0))

        # Photometric fill: share of the (filled) blob whose grey level departs strongly from the fabric.
        # The anomaly map of a large opening only lights up its rim, so it cannot be used for this.
        filled = np.zeros((h, w), np.uint8)
        cv2.drawContours(filled, [c], -1, 1, thickness=cv2.FILLED)
        filled_px = gray[roi][filled > 0].astype(np.float32)
        photometric_fill = float(np.mean(np.abs(filled_px - bg_level) > HOLE_MIN_CONTRAST / 2.0)) if filled_px.size else 0.0
        hole_contrast = float(abs(np.median(filled_px) - bg_level)) if filled_px.size else 0.0

        # Color shift analysis in ROI
        delta_b = float(np.mean(b_chan[roi][measure_mask]) - b_median)

        # Mean intensity & contrast against substrate
        mean_intensity = float(np.mean(gray[roi][measure_mask]))
        contrast_diff = abs(mean_intensity - bg_level)

        # Physical defect classification
        is_linear = (aspect_ratio >= aspect_ratio_crease_thresh) or (
            mean_thickness < LINEAR_MAX_THICKNESS_IN * ppi and dim_length >= LINEAR_MIN_LENGTH_IN * ppi
        )

        if delta_b > 12.0 or (bg_level > 120 and delta_b > 6.0):
            defect_type = "Vết Ố / Dầu Mỡ (Stain / Contamination)"
            sub_type = "Ố vàng / Dầu bôi trơn"
            category = "Stain"
        elif is_linear:
            if contrast_diff > 45.0:
                defect_type = "Vạch Đánh Dấu / Sợi Lạ (Marking / Foreign Yarn)"
                sub_type = "Vết phấn / Sợi tương phản cao"
                category = "ChalkMark"
            else:
                defect_type = "Nếp Gấp / Vệt Gập (Crease / Fold)"
                sub_type = "Nếp gấp dọc" if h > w else "Nếp gấp ngang / xiên"
                category = "Crease"
        elif (area > HOLE_MIN_AREA_SQIN * ppi * ppi and aspect_ratio < 2.0
              and hole_contrast > HOLE_MIN_CONTRAST and photometric_fill >= HOLE_MIN_FILL):
            # Relative contrast, not absolute intensity: on black fabric every region is < 25 grey levels.
            # Without backlight a bright print/logo can still look like this -> confirm holes with backlight.
            defect_type = "Lỗ Thủng / Rách Vải (Hole / Tear)"
            sub_type = "Nghi lỗ thủng - cần xác nhận (chưa có đèn nền)" if hole_needs_review else "Rách cấu trúc dệt"
            category = "Hole"
        else:
            defect_type = "Dị Tật Dệt / Đốm Lạ (Weave / Surface Defect)"
            sub_type = "Đốm dị biệt cục bộ"
            category = "Weave"

        # ASTM D5430 (4-Point Fabric Inspection Standard)
        length_inches = dim_length / ppi
        astm_points = astm_d5430_points(length_inches, is_hole=(category == "Hole"))

        # Severity level
        if astm_points >= 4:
            severity = "Nghiêm Trọng (Major)"
        elif astm_points >= 2:
            severity = "Trung Bình (Moderate)"
        else:
            severity = "Nhẹ (Minor)"

        clean_mask[roi][component_mask] = 255
        defects.append({
            "id": f"DEF-{defect_id:02d}",
            "type": defect_type,
            "sub_type": sub_type,
            "category": category,
            "bbox": [x, y, w, h],
            "center": [round(cx, 1), round(cy, 1)],
            "length_px": round(dim_length, 1),
            "length_in": round(length_inches, 2),
            "width_px": round(dim_width, 1),
            "area_px": area,
            "aspect_ratio": round(aspect_ratio, 2),
            "photometric_fill": round(photometric_fill, 2),
            "astm_points": astm_points,
            "needs_review": bool(category == "Hole" and hole_needs_review),
            "severity": severity,
            "contrast_diff": round(contrast_diff, 1),
            "delta_b_color": round(delta_b, 1),
        })
        defect_id += 1

    # Sort defects by severity (descending points, then area)
    defects.sort(key=lambda d: (d["astm_points"], d["area_px"]), reverse=True)
    return defects, clean_mask


# When two detectors report the same place, keep the more specific / more severe interpretation
CATEGORY_PRIORITY = {"Hole": 5, "Stain": 4, "ChalkMark": 3, "Crease": 3, "Weave": 1}
DUPLICATE_OVERLAP = 0.5   # |A ∩ B| / min(|A|, |B|) above which two instances are the same defect


def extract_from_component_maps(
    pil_image: Image.Image,
    component_maps: Dict[str, np.ndarray],
    threshold: float,
    **kwargs,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Extract instances separately on each detector map (colour, crease/fold, contrast, golden...) and merge
    duplicates. Fusing the maps first lets one detector's response (e.g. a fold or the fabric edge) bridge
    into another's defect (e.g. a stain): the blob then mixes both, its colour shift is averaged away and it is
    misclassified. Per-map extraction keeps each defect's own pixels for classification.
    Duplicates (pixel overlap > DUPLICATE_OVERLAP of the smaller one) keep the higher CATEGORY_PRIORITY,
    then the larger area.
    """
    candidates = []
    for source, amap in component_maps.items():
        defects, mask = extract_defect_instances(pil_image, amap, threshold=threshold, **kwargs)
        for d in defects:
            x, y, w, h = d["bbox"]
            d["source"] = source
            candidates.append((d, (x, y, w, h), mask[y:y + h, x:x + w] > 0))

    candidates.sort(key=lambda c: (CATEGORY_PRIORITY.get(c[0]["category"], 0), c[0]["area_px"]), reverse=True)
    kept = []
    for d, (x, y, w, h), m in candidates:
        duplicate = False
        for _, (kx, ky, kw, kh), km in kept:
            ix0, iy0 = max(x, kx), max(y, ky)
            ix1, iy1 = min(x + w, kx + kw), min(y + h, ky + kh)
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            inter = np.count_nonzero(m[iy0 - y:iy1 - y, ix0 - x:ix1 - x] & km[iy0 - ky:iy1 - ky, ix0 - kx:ix1 - kx])
            if inter / max(min(np.count_nonzero(m), np.count_nonzero(km)), 1) > DUPLICATE_OVERLAP:
                duplicate = True
                break
        if not duplicate:
            kept.append((d, (x, y, w, h), m))

    np_img = np.array(pil_image)
    clean_mask = np.zeros(np_img.shape[:2], np.uint8)
    defects = []
    for d, (x, y, w, h), m in kept:
        clean_mask[y:y + h, x:x + w][m] = 255
        defects.append(d)
    defects.sort(key=lambda d: (d["astm_points"], d["area_px"]), reverse=True)
    for i, d in enumerate(defects, 1):
        d["id"] = f"DEF-{i:02d}"
    return defects, clean_mask


def draw_defect_bounding_boxes(
    pil_image: Image.Image,
    defects: List[Dict[str, Any]],
    box_thickness: int = 3,
) -> Image.Image:
    """
    Renders high-visibility bounding boxes with professional industrial labels.
    """
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    color_palette = {
        "Crease": (0, 140, 255),      # Orange
        "Stain": (0, 165, 255),       # Amber / Yellow
        "Hole": (0, 0, 255),          # Bright Red
        "Weave": (255, 100, 0),       # Blue / Cyan
        "ChalkMark": (0, 220, 255),   # Golden Yellow
    }

    for d in defects:
        x, y, w, h = d["bbox"]
        cat = d.get("category", "Crease")
        box_color = color_palette.get(cat, (0, 165, 255))

        # Draw main rectangle
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), box_color, box_thickness)

        # Draw corner markers
        c_len = max(min(w, h) // 4, 12)
        cv2.line(img_bgr, (x, y), (x + c_len, y), box_color, box_thickness + 2)
        cv2.line(img_bgr, (x, y), (x, y + c_len), box_color, box_thickness + 2)
        cv2.line(img_bgr, (x + w, y), (x + w - c_len, y), box_color, box_thickness + 2)
        cv2.line(img_bgr, (x + w, y), (x + w, y + c_len), box_color, box_thickness + 2)

        # Label background pill
        review_tag = " | CAN XAC NHAN" if d.get("needs_review") else ""
        label_text = f"{d['id']}: {d['category']} | L={d.get('length_in', 0):.1f}in | {d['astm_points']}pts{review_tag}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        label_y1 = max(0, y - th - 10)
        label_y2 = max(th + 10, y)
        cv2.rectangle(img_bgr, (x, label_y1), (x + tw + 10, label_y2), box_color, -1)
        cv2.putText(
            img_bgr,
            label_text,
            (x + 5, label_y2 - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def compute_defect_statistics(
    defects: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    pixels_per_inch: float = DEFAULT_PIXELS_PER_INCH,
    grade_a_limit: float = 20.0,
    acceptance_limit: float = 40.0,
    length_axis: int = 0,
    inspected_width_px: int = 0,
    frame_only: bool = True,
) -> Dict[str, Any]:
    """
    Per-frame statistics, ASTM D5430 points and Pareto counts.

    ASTM D5430 grades a whole roll on points per 100 square yards:
        points/100yd2 = total_points * 3600 / (length_yd * width_in)
    with at most 4 points per linear yard. A single frame (~0.4 yd2) is far too small for that,
    so with `frame_only=True` (default) no A/B/C grade is issued here; use roll_session.RollSession.
    Defects flagged `needs_review` are listed but excluded from the points until confirmed.
    `length_axis` is the image axis running along the roll (0 = vertical).
    """
    ppi = max(float(pixels_per_inch), 1e-3)
    counted = [d for d in defects if not d.get("needs_review")]
    review = [d for d in defects if d.get("needs_review")]
    total_count = len(defects)
    raw_points = sum(d["astm_points"] for d in counted)
    total_defect_area = sum(d["area_px"] for d in defects)
    total_img_area = image_shape[0] * image_shape[1]
    defect_area_pct = (total_defect_area / (total_img_area + 1e-10)) * 100.0

    # Inspected geometry
    length_in = image_shape[length_axis] / ppi
    width_px = inspected_width_px if inspected_width_px > 0 else image_shape[1 - length_axis]
    width_in = width_px / ppi
    length_yd = length_in / 36.0

    # 4 points max per linear yard of fabric
    max_points = 4 * max(1, int(math.ceil(length_yd - 1e-9)))
    total_points = min(raw_points, max_points)
    points_per_100yd2 = total_points * 3600.0 / max(length_yd * width_in, 1e-9)

    # Categorization counts
    type_counts = {}
    severity_counts = {"Nghiêm Trọng (Major)": 0, "Trung Bình (Moderate)": 0, "Nhẹ (Minor)": 0}
    for d in defects:
        cat = d.get("category", "Other")
        type_counts[cat] = type_counts.get(cat, 0) + 1
        sev = d.get("severity", "Nhẹ (Minor)")
        if sev in severity_counts:
            severity_counts[sev] += 1

    if frame_only:
        grade = dict(FRAME_ONLY_GRADE)
    else:
        grade = grade_from_density(points_per_100yd2, total_points, grade_a_limit, acceptance_limit)

    return {
        "total_defects": total_count,
        "total_astm_points": total_points,
        "raw_astm_points": raw_points,
        "review_count": len(review),
        "review_points": sum(d["astm_points"] for d in review),
        "points_per_100yd2": round(points_per_100yd2, 1),
        "inspected_area_yd2": round(length_yd * width_in / 36.0, 4),
        "defect_area_pct": round(defect_area_pct, 3),
        "type_counts": type_counts,
        "severity_counts": severity_counts,
        **grade,
    }

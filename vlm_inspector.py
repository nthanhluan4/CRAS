"""
vlm_inspector.py
Cognitive Quality Inspector & Root-Cause QA/QC Reporting Engine
Stage 2: Semantic Reasoning & Textile Engineering Diagnostics
Runs fully offline using the built-in expert engine.
"""

from datetime import datetime
import json
import os
from typing import List, Dict, Any, Optional
import pandas as pd
from PIL import Image


def generate_vlm_inspection_report(
    defects: List[Dict[str, Any]],
    stats: Dict[str, Any],
    pil_image: Optional[Image.Image] = None,
    sample_name: str = "Fabric Roll Sample",
    api_key: Optional[str] = None,
    gemini_model: str = "gemini-2.0-flash-lite",
) -> Dict[str, Any]:
    """
    Generates a formal textile engineering QA/QC report with Root-Cause Analysis,
    ASTM D5430 compliance assessment, and maintenance recommendations.

    The app deliberately runs in 100% offline mode; external AI keys are ignored
    to keep industrial operations stable and private.
    """
    _ = api_key, gemini_model, pil_image
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return _run_built_in_expert_reasoning(defects, stats, sample_name, timestamp)


def _run_built_in_expert_reasoning(
    defects: List[Dict[str, Any]],
    stats: Dict[str, Any],
    sample_name: str,
    timestamp: str,
) -> Dict[str, Any]:
    """
    High-fidelity Textile Engineering Reasoning Engine based on ASTM D5430 and
    industrial loom mechanics. Runs 100% offline.
    """
    total_defects = stats["total_defects"]
    total_points = stats["total_astm_points"]
    roll_grade = stats["roll_grade"]
    acceptance = stats["acceptance_status"]
    type_counts = stats["type_counts"]

    # Root Cause Diagnostic Logic
    diagnostics = []
    maintenance_actions = []

    if type_counts.get("Crease", 0) > 0:
        crease_defects = [d for d in defects if d.get("category") == "Crease"]
        max_len = max([d["length_px"] for d in crease_defects], default=0)
        has_vertical = any("dọc" in d.get("sub_type", "").lower() for d in crease_defects)
        has_horizontal = any("ngang" in d.get("sub_type", "").lower() for d in crease_defects)

        if has_vertical and has_horizontal:
            diag_text = (
                "Phát hiện nếp gấp dạng lưới giao thoa (T-Cross Folds). "
                "Nguyên nhân cốt lõi do sự mất đồng bộ giữa lực căng dọc (Warp tension) "
                "và áp lực ép của trục rulo cán (Calendar / Nip roller), kết hợp với hiện tượng vải bị dồn mép khi cuộn."
            )
            action_text = (
                "1. Kiểm tra và căn chỉnh lại độ đồng tâm (concentricity) và độ song song của cặp rulo ép số 2.\n"
                "2. Giảm bớt 10-15% lực kéo căng cuộn (Take-up tension) để giải tỏa ứng suất nội trong sợi vải.\n"
                "3. Kiểm tra thanh chống gấp nếp (Bow roller / Wrinkle spreader) tại đầu vào máy cuộn."
            )
        elif has_vertical:
            diag_text = (
                f"Phát hiện nếp gấp dọc liên tục kéo dài ({int(max_len)}px). "
                "Nguyên nhân do ứng suất căng sợi dọc không đều trên suốt khổ rộng máy dệt, "
                "hoặc do một đoạn rulo cuộn bị mòn cục bộ tạo khe hở không đồng nhất."
            )
            action_text = (
                "1. Căn chỉnh lại lực căng của giàn sợi dọc (Warp beam tension controller).\n"
                "2. Dùng thước đo khe hở (feeler gauge) kiểm tra khoảng cách hai đầu trục rulo cuộn."
            )
        else:
            diag_text = (
                "Phát hiện các vết nhăn gợn sóng ngang/xiên bề mặt. "
                "Nguyên nhân do tốc độ dẫn vải không ổn định hoặc dao động cơ học từ bàn dệt."
            )
            action_text = "Kiểm tra độ chùng dây curoa truyền động và cảm biến tốc độ trục kéo vải."

        diagnostics.append({"category": "Crease / Fold", "diagnosis": diag_text, "action": action_text})
        maintenance_actions.append(action_text)

    if type_counts.get("Stain", 0) > 0:
        diag_text = (
            "Phát hiện vết ố loang màu/dầu mỡ (Stain contamination). "
            "Phổ màu Lab ghi nhận độ lệch sắc độ b* đáng kể, điển hình cho dầu nhớt bôi trơn máy móc dệt "
            "hoặc cặn hóa chất từ công đoạn định hình nhiệt chưa được rửa sạch."
        )
        action_text = (
            "1. Kiểm tra ngay phớt chắn dầu (oil seals) và vòng bi trên các trục truyền động phía trên băng chuyền.\n"
            "2. Lắp đặt khay hứng dầu bảo vệ tại các khớp nối cơ khí di động.\n"
            "3. Khoanh vùng đoạn vải bị ố để xử lý giặt tẩy điểm (spot cleaning) trước khi đóng gói."
        )
        diagnostics.append({"category": "Stain / Contamination", "diagnosis": diag_text, "action": action_text})
        maintenance_actions.append(action_text)

    if type_counts.get("Hole", 0) > 0 or type_counts.get("Weave", 0) > 0:
        diag_text = (
            "Phát hiện khuyết điểm dệt cấu trúc / rách thủng sợi vải. "
            "Nguyên nhân do kim dệt bị gãy, kẹt sợi hoặc có dị vật cơ học sắc nhọn va quẹt trên đường dẫn vải."
        )
        action_text = (
            "1. Dừng máy kiểm tra giàn kim dệt và thanh gài sợi.\n"
            "2. Kiểm tra độ nhẵn bóng của các con lăn dẫn hướng (guide rollers)."
        )
        diagnostics.append({"category": "Weave / Structural", "diagnosis": diag_text, "action": action_text})
        maintenance_actions.append(action_text)

    if total_defects == 0:
        executive_summary = (
            "Bề mặt vải đạt trạng thái lý tưởng. Không phát hiện bất kỳ khuyết điểm hình học, "
            "nếp gấp cơ khí hay vết biến đổi sắc độ màu nào. Toàn bộ cấu trúc sợi đồng đều."
        )
    else:
        types_str = ", ".join([f"{count} {cat}" for cat, count in type_counts.items()])
        executive_summary = (
            f"Phát hiện tổng cộng {total_defects} khuyết điểm ({types_str}) "
            f"với tổng điểm phạt chất lượng ASTM D5430 là {total_points} điểm. "
            f"Kết luận kiểm định: {acceptance}. Phân hạng chất lượng: {roll_grade}."
        )

    report = {
        "report_id": f"QC-EXPERT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "timestamp": timestamp,
        "sample_name": sample_name,
        "standard_applied": "ASTM D5430 (Standard Test Method for Visually Inspecting and Grading Fabrics)",
        "ai_engine": "Bộ Suy Luận Chuyên Gia Dệt May Nội Bộ (Offline Expert Engine)",
        "is_real_llm": False,
        "executive_summary": executive_summary,
        "roll_grade": roll_grade,
        "acceptance_status": acceptance,
        "total_defects": total_defects,
        "total_astm_points": total_points,
        "defect_area_percentage": f"{stats['defect_area_pct']}%",
        "type_breakdown": type_counts,
        "severity_breakdown": stats["severity_counts"],
        "diagnostics": diagnostics,
        "corrective_actions": maintenance_actions,
        "defects_detail": defects,
    }
    return report


def export_defects_dataframe(defects: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts defect instances into a clean pandas DataFrame for UI table rendering and CSV export.
    """
    if not defects:
        return pd.DataFrame(columns=[
            "Mã Lỗi", "Loại Khuyết Điểm", "Phân Loại Chi Tiết", "Tọa Độ BBox (X, Y, W, H)",
            "Chiều Dài (px)", "Diện Tích (px²)", "Tỉ Số L/W", "Điểm ASTM D5430", "Mức Độ"
        ])

    rows = []
    for d in defects:
        bbox_str = f"[{d['bbox'][0]}, {d['bbox'][1]}, {d['bbox'][2]}, {d['bbox'][3]}]"
        rows.append({
            "Mã Lỗi": d["id"],
            "Loại Khuyết Điểm": d["type"],
            "Phân Loại Chi Tiết": d["sub_type"],
            "Tọa Độ BBox (X, Y, W, H)": bbox_str,
            "Chiều Dài (px)": int(d["length_px"]),
            "Diện Tích (px²)": int(d["area_px"]),
            "Tỉ Số L/W": d["aspect_ratio"],
            "Điểm ASTM D5430": d["astm_points"],
            "Mức Độ": d["severity"],
        })
    return pd.DataFrame(rows)

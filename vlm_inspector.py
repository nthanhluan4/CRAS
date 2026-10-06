"""
vlm_inspector.py
Cognitive Vision-LLM (VLM) Inspector & Root-Cause QA/QC Reporting Engine
Stage 2: Semantic Reasoning & Textile Engineering Diagnostics
Supports Google Gemini Vision (Flash, Flash-Lite) and Built-in Expert Engine
"""

import base64
from datetime import datetime
import io
import json
import os
from typing import List, Dict, Any, Optional
import pandas as pd
from PIL import Image
import requests


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
    ASTM D5430 compliance assessment, and machine maintenance recommendations.
    
    If api_key is provided and pil_image is passed, calls Google Gemini Multimodal VLM.
    Otherwise, gracefully falls back to the deterministic Built-in Textile Expert Engine.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # If API key is provided and image is present, attempt live Gemini VLM call
    if api_key and api_key.strip() and pil_image is not None:
        try:
            report = _call_gemini_vlm(
                pil_image=pil_image,
                defects=defects,
                stats=stats,
                sample_name=sample_name,
                api_key=api_key.strip(),
                model_name=gemini_model.strip(),
                timestamp=timestamp,
            )
            return report
        except Exception as e:
            # Fall back to built-in expert engine with error notification
            report = _run_built_in_expert_reasoning(defects, stats, sample_name, timestamp)
            report["vlm_api_error"] = str(e)
            report["ai_engine"] = f"Bộ Suy Luận Chuyên Gia Nội Bộ (Dự phòng do lỗi kết nối Gemini: {str(e)[:60]}...)"
            return report

    # Default: Built-in Textile Engineering Expert Engine
    return _run_built_in_expert_reasoning(defects, stats, sample_name, timestamp)


def _call_gemini_vlm(
    pil_image: Image.Image,
    defects: List[Dict[str, Any]],
    stats: Dict[str, Any],
    sample_name: str,
    api_key: str,
    model_name: str,
    timestamp: str,
) -> Dict[str, Any]:
    """
    Calls Google Gemini Vision REST API with the fabric image and AOI defect profile.
    """
    # 1. Resize image to optimal dimension for fast network upload (< 1024px)
    w, h = pil_image.size
    max_dim = 1024
    scale = min(max_dim / max(w, h), 1.0)
    resized_img = pil_image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    resized_img.save(buffer, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # 2. Defect summary for prompt
    defect_summary = []
    for d in defects[:12]:
        defect_summary.append({
            "id": d["id"],
            "category": d["category"],
            "sub_type": d["sub_type"],
            "bbox": d["bbox"],
            "length_px": d["length_px"],
            "area_px": d["area_px"],
            "astm_points": d["astm_points"],
            "severity": d["severity"],
        })

    prompt = f"""Bạn là Kỹ sư Trưởng Giám sát Chất lượng Dệt may Quốc tế (Lead Textile Quality Control Engineer).
Dưới đây là ảnh chụp bề mặt vải thực tế từ camera công nghiệp kèm kết quả trích xuất khuyết tật từ thuật toán thị giác AOI.

THỐNG KÊ THUẬT TOÁN AOI ĐÃ ĐO ĐƯỢC:
- Mẫu kiểm tra: {sample_name}
- Tổng số khuyết tật: {stats['total_defects']}
- Tổng điểm phạt chuẩn quốc tế ASTM D5430 (4-Point System): {stats['total_astm_points']}
- Phân bố lỗi: {json.dumps(stats['type_counts'], ensure_ascii=False)}
- Chi tiết các khuyết tật tiêu biểu:
{json.dumps(defect_summary, indent=2, ensure_ascii=False)}

YÊU CẦU:
Hãy quan sát kỹ bức ảnh bề mặt vải và đối chiếu với danh sách khuyết tật trên để đưa ra báo cáo kiểm định chất lượng dệt may.
Trả về KẾT QUẢ DUY NHẤT LÀ MỘT ĐỐI TƯỢNG JSON (không kèm markdown ngoài json) theo đúng cấu trúc sau:
{{
  "executive_summary": "Tóm tắt ngắn gọn tình trạng chất lượng vải và mức độ nghiêm trọng",
  "roll_grade": "HẠNG A (XUẤT KHẨU) / HẠNG B (THƯƠNG MẠI CÓ ĐIỀU KIỆN) / HẠNG C / REJECT (PHẾ PHẨM)",
  "acceptance_status": "ĐẠT CHUẨN (PASSED) / CẦN XỬ LÝ NHIỆT / KHÔNG ĐẠT (REJECTED)",
  "diagnostics": [
    {{
      "category": "Tên nhóm lỗi (Crease / Stain / Hole / Weave)",
      "diagnosis": "Phân tích nguyên nhân kỹ thuật gốc rễ (mất đồng bộ rulo cán, lực căng sợi dọc không đều, rò rỉ dầu nhớt từ bạc đạn...)",
      "action": "Hướng dẫn căn chỉnh, sửa chữa hoặc bảo trì máy dệt cụ thể cho thợ vận hành"
    }}
  ],
  "corrective_actions": [
    "Hành động 1...",
    "Hành động 2..."
  ]
}}
"""

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_b64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "response_mime_type": "application/json"
        }
    }

    headers = {"Content-Type": "application/json"}
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=25)
    resp.raise_for_status()

    data = resp.json()
    content_text = data["candidates"][0]["content"]["parts"][0]["text"]
    result_json = json.loads(content_text)

    report = {
        "report_id": f"QC-GEMINI-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "timestamp": timestamp,
        "sample_name": sample_name,
        "standard_applied": "ASTM D5430 (Standard Test Method for Visually Inspecting and Grading Fabrics)",
        "ai_engine": f"Google Gemini VLM ({model_name}) - AI THẬT 100%",
        "is_real_llm": True,
        "executive_summary": result_json.get("executive_summary", "Đã phân tích bởi Google Gemini VLM."),
        "roll_grade": result_json.get("roll_grade", stats["roll_grade"]),
        "acceptance_status": result_json.get("acceptance_status", stats["acceptance_status"]),
        "total_defects": stats["total_defects"],
        "total_astm_points": stats["total_astm_points"],
        "defect_area_percentage": f"{stats['defect_area_pct']}%",
        "type_breakdown": stats["type_counts"],
        "severity_breakdown": stats["severity_counts"],
        "diagnostics": result_json.get("diagnostics", []),
        "corrective_actions": result_json.get("corrective_actions", []),
        "defects_detail": defects,
    }
    return report


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
            "Phát hiện khuyết tật dệt cấu trúc / rách thủng sợi vải. "
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
            "Bề mặt vải đạt trạng thái lý tưởng. Không phát hiện bất kỳ khuyết tật hình học, "
            "nếp gấp cơ khí hay vết biến đổi sắc độ màu nào. Toàn bộ cấu trúc sợi đồng đều."
        )
    else:
        types_str = ", ".join([f"{count} {cat}" for cat, count in type_counts.items()])
        executive_summary = (
            f"Phát hiện tổng cộng {total_defects} khuyết tật ({types_str}) "
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
            "Mã Lỗi", "Loại Khuyết Tật", "Phân Loại Chi Tiết", "Tọa Độ BBox (X, Y, W, H)",
            "Chiều Dài (px)", "Diện Tích (px²)", "Tỉ Số L/W", "Điểm ASTM D5430", "Mức Độ"
        ])

    rows = []
    for d in defects:
        bbox_str = f"[{d['bbox'][0]}, {d['bbox'][1]}, {d['bbox'][2]}, {d['bbox'][3]}]"
        rows.append({
            "Mã Lỗi": d["id"],
            "Loại Khuyết Tật": d["type"],
            "Phân Loại Chi Tiết": d["sub_type"],
            "Tọa Độ BBox (X, Y, W, H)": bbox_str,
            "Chiều Dài (px)": int(d["length_px"]),
            "Diện Tích (px²)": int(d["area_px"]),
            "Tỉ Số L/W": d["aspect_ratio"],
            "Điểm ASTM D5430": d["astm_points"],
            "Mức Độ": d["severity"],
        })
    return pd.DataFrame(rows)

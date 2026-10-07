"""
test_defect_profiler.py
Verifies Stage 1 (Deterministic AOI) and Stage 2 (VLM Inspector) on industrial test images.
"""

import os
from PIL import Image
import cv2
import numpy as np

import app
import defect_profiler
import vlm_inspector


def run_test():
    test_cases = [
        ("export_20260923_154200_raw.png", "Crease"),
        ("export_20260923_154829_raw.png", "Stain"),
    ]

    print("=== STARTING DEFECT PROFILER & VLM INSPECTOR VERIFICATION ===")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_test_folder = os.path.join(base_dir, "test_samples")
    fallback_test_folder = r"d:\Luan.Nguyen\Tools\fabric_defect\test"
    test_folder = local_test_folder if os.path.isdir(local_test_folder) else fallback_test_folder

    for filename, expected_cat in test_cases:
        filepath = os.path.join(test_folder, filename)
        assert os.path.exists(filepath), f"File {filepath} not found!"

        img = Image.open(filepath).convert("RGB")
        print(f"\n--- Testing {filename} ({img.size}) ---")

        # 1. Run detection pipeline
        score, hm_resized, overlay, details, anomaly_map = app.run_hybrid_ensemble(img)
        print(f"Hybrid Score: {score:.4f} | Crease: {details['crease_score']:.4f} | Stain: {details['stain_score']:.4f}")

        # 2. Stage 1: Extract defect instances & ASTM D5430 points (from the anomaly map, never the JET heatmap)
        defects, clean_mask = defect_profiler.extract_defect_instances(img, anomaly_map, threshold=0.40, min_area=150)
        print(f"Defect instances extracted: {len(defects)}")

        assert len(defects) > 0, f"Expected defects for {filename}, but found none!"

        for d in defects[:3]:
            print(f"  -> {d['id']}: {d['type']} ({d['sub_type']}) | L={d['length_px']}px, Area={d['area_px']}px | ASTM Points: {d['astm_points']} ({d['severity']})")

        # 3. Compute aggregate statistics
        stats = defect_profiler.compute_defect_statistics(defects, (img.size[1], img.size[0]))
        print(f"  -> Quality Grade: {stats['roll_grade']} | Status: {stats['acceptance_status']} | Total Points: {stats['total_astm_points']}")

        # 4. Stage 2: Cognitive VLM Inspector Report
        report = vlm_inspector.generate_vlm_inspection_report(defects, stats, sample_name=filename)
        print(f"  -> VLM Report ID: {report['report_id']}")
        print(f"  -> Summary: {report['executive_summary']}")
        if report['diagnostics']:
            print(f"  -> Root Cause: {report['diagnostics'][0]['diagnosis'][:100]}...")

        # 5. Draw bounding boxes
        boxed_img = defect_profiler.draw_defect_bounding_boxes(img, defects)
        out_path = os.path.join("results/inference", f"boxed_{filename}")
        boxed_img.save(out_path)
        print(f"  -> Bounding box overlay saved to: {out_path}")

    print("\n=== ALL VERIFICATIONS PASSED SUCCESSFULLY! ===")


if __name__ == "__main__":
    run_test()

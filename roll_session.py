"""
roll_session.py
Whole-roll ASTM D5430 (4-point system) accumulation for a roll inspection machine.

Per-frame detections are mapped to roll coordinates (inches along / across the roll), duplicates in the
overlap between consecutive frames are merged, the 4-points-per-linear-yard cap is applied, and the roll is
graded on points per 100 square yards against buyer limits.

Roll position comes from the frame index (frame_index * frame advance) until an encoder is wired in:
pass `frame_start_in` to add_frame() to use a measured position instead.
"""

import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from defect_profiler import astm_d5430_points, grade_from_density

ROLLS_DIR = os.path.join("results", "rolls")
YARD_IN = 36.0
MERGE_TOL_IN = 0.25


def _overlap(a0: float, a1: float, b0: float, b1: float, tol: float) -> bool:
    return a0 <= b1 + tol and b0 <= a1 + tol


class RollSession:
    def __init__(
        self,
        roll_id: str,
        usable_width_in: float,
        pixels_per_inch: float,
        frame_length_px: int,
        sku: str = "",
        frame_overlap_in: float = 0.0,
        grade_a_limit: float = 20.0,
        acceptance_limit: float = 40.0,
        y_increases_along_roll: bool = True,
    ):
        if usable_width_in <= 0:
            raise ValueError("usable_width_in must be > 0")
        self.roll_id = roll_id
        self.sku = sku
        self.usable_width_in = float(usable_width_in)
        self.ppi = max(float(pixels_per_inch), 1e-3)
        self.frame_length_in = frame_length_px / self.ppi
        self.frame_overlap_in = max(0.0, min(float(frame_overlap_in), self.frame_length_in * 0.9))
        self.grade_a_limit = float(grade_a_limit)
        self.acceptance_limit = float(acceptance_limit)
        self.y_increases_along_roll = y_increases_along_roll
        self.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.finished_at: Optional[str] = None

        self.defects: List[Dict[str, Any]] = []
        self.frames: List[Dict[str, Any]] = []
        self.inspected_length_in = 0.0
        self._next_uid = 1

    # ------------------------------------------------------------------ frames
    @property
    def frame_advance_in(self) -> float:
        return self.frame_length_in - self.frame_overlap_in

    def next_frame_index(self) -> int:
        return len(self.frames)

    def add_frame(
        self,
        defects: List[Dict[str, Any]],
        frame_index: Optional[int] = None,
        frame_start_in: Optional[float] = None,
        camera_offset_in: float = 0.0,
        frame_name: str = "",
    ) -> Dict[str, int]:
        """Register one frame's detections. Returns counts of new vs merged defects."""
        if frame_index is None:
            frame_index = self.next_frame_index()
        if frame_start_in is None:
            frame_start_in = frame_index * self.frame_advance_in
        frame_end_in = frame_start_in + self.frame_length_in

        added = merged = 0
        for d in defects:
            x, y, w, h = d["bbox"]
            across = (camera_offset_in + x / self.ppi, camera_offset_in + (x + w) / self.ppi)
            if self.y_increases_along_roll:
                along = (frame_start_in + y / self.ppi, frame_start_in + (y + h) / self.ppi)
            else:
                along = (frame_end_in - (y + h) / self.ppi, frame_end_in - y / self.ppi)

            target = self._find_duplicate(d, across, along, frame_index)
            if target is not None:
                target["across"] = [min(target["across"][0], across[0]), max(target["across"][1], across[1])]
                target["along"] = [min(target["along"][0], along[0]), max(target["along"][1], along[1])]
                target["length_in"] = round(max(target["length_in"], float(d.get("length_in", 0.0)),
                                                target["along"][1] - target["along"][0]), 2)
                target["frames"].append(frame_index)
                target["astm_points"] = astm_d5430_points(target["length_in"], is_hole=target["category"] == "Hole")
                merged += 1
            else:
                self.defects.append({
                    "uid": f"R{self._next_uid:04d}",
                    "category": d["category"],
                    "type": d.get("type", d["category"]),
                    "across": [round(across[0], 2), round(across[1], 2)],
                    "along": [round(along[0], 2), round(along[1], 2)],
                    "length_in": float(d.get("length_in", 0.0)),
                    "astm_points": int(d["astm_points"]),
                    "needs_review": bool(d.get("needs_review", False)),
                    "frames": [frame_index],
                })
                self._next_uid += 1
                added += 1

        self.frames.append({"index": frame_index, "start_in": round(frame_start_in, 2), "name": frame_name,
                            "defects": len(defects)})
        self.inspected_length_in = max(self.inspected_length_in, frame_end_in)
        return {"added": added, "merged": merged}

    def _find_duplicate(self, d, across, along, frame_index) -> Optional[Dict[str, Any]]:
        for t in self.defects:
            if frame_index in t["frames"] or t["category"] != d["category"]:
                continue
            if _overlap(t["across"][0], t["across"][1], across[0], across[1], MERGE_TOL_IN) and \
               _overlap(t["along"][0], t["along"][1], along[0], along[1], MERGE_TOL_IN):
                return t
        return None

    # ------------------------------------------------------------------ review
    def resolve_review(self, uid: str, is_defect: bool) -> None:
        """Operator decision on a suspected hole: confirm (counts) or reject (removed)."""
        for i, t in enumerate(self.defects):
            if t["uid"] == uid:
                if is_defect:
                    t["needs_review"] = False
                else:
                    self.defects.pop(i)
                return
        raise KeyError(uid)

    # ------------------------------------------------------------------ grading
    def points_per_yard(self) -> Dict[int, int]:
        """
        ASTM D5430: at most 4 points per linear yard. A defect running along the roll across several
        yards is penalised in every yard it touches (4 points each when its length earns 4 points).
        """
        per_yard: Dict[int, int] = {}
        for t in self.defects:
            if t["needs_review"]:
                continue
            y0 = int(math.floor(t["along"][0] / YARD_IN))
            y1 = int(math.floor(max(t["along"][1] - 1e-6, t["along"][0]) / YARD_IN))
            for yard in range(y0, y1 + 1):
                per_yard[yard] = min(4, per_yard.get(yard, 0) + t["astm_points"])
        return per_yard

    def summary(self) -> Dict[str, Any]:
        per_yard = self.points_per_yard()
        total_points = sum(per_yard.values())
        raw_points = sum(t["astm_points"] for t in self.defects if not t["needs_review"])
        length_yd = self.inspected_length_in / YARD_IN
        area_denominator = length_yd * self.usable_width_in
        density = total_points * 3600.0 / area_denominator if area_denominator > 0 else 0.0

        type_counts: Dict[str, int] = {}
        for t in self.defects:
            type_counts[t["category"]] = type_counts.get(t["category"], 0) + 1
        review = [t for t in self.defects if t["needs_review"]]

        grade = grade_from_density(density, total_points, self.grade_a_limit, self.acceptance_limit)
        if review:
            grade["acceptance_status"] += f" - còn {len(review)} lỗi chờ xác nhận"

        return {
            "roll_id": self.roll_id,
            "sku": self.sku,
            "frames": len(self.frames),
            "inspected_length_yd": round(length_yd, 2),
            "inspected_length_m": round(self.inspected_length_in * 0.0254, 2),
            "usable_width_in": self.usable_width_in,
            "inspected_area_yd2": round(length_yd * self.usable_width_in / YARD_IN, 2),
            "total_defects": len(self.defects),
            "total_astm_points": total_points,
            "raw_astm_points": raw_points,
            "points_per_100yd2": round(density, 1),
            "review_count": len(review),
            "type_counts": type_counts,
            "grade_a_limit": self.grade_a_limit,
            "acceptance_limit": self.acceptance_limit,
            **grade,
        }

    # ------------------------------------------------------------------ report
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "config": {
                "pixels_per_inch": self.ppi,
                "frame_length_in": round(self.frame_length_in, 2),
                "frame_overlap_in": self.frame_overlap_in,
            },
            "points_per_yard": {str(k): v for k, v in sorted(self.points_per_yard().items())},
            "defects": self.defects,
            "frames": self.frames,
        }

    def finish(self, out_dir: str = ROLLS_DIR) -> str:
        self.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(out_dir, exist_ok=True)
        safe_id = "".join(c for c in self.roll_id if c.isalnum() or c in ("_", "-")) or "roll"
        path = os.path.join(out_dir, f"{safe_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

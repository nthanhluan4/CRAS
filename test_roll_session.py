"""
test_roll_session.py
Unit checks for ASTM D5430 roll accumulation: frame -> roll mapping, overlap de-duplication,
4-points-per-yard cap, continuous defects, suspected-hole review and roll grading.
Run: python test_roll_session.py
"""

from defect_profiler import astm_d5430_points, inspection_columns
from roll_session import RollSession

PPI = 100.0
FRAME_PX = 2000          # 20 in per frame


def defect(category, x, y, w, h, needs_review=False):
    length_in = max(w, h) / PPI
    return {
        "category": category,
        "bbox": [x, y, w, h],
        "length_in": length_in,
        "astm_points": astm_d5430_points(length_in, is_hole=category == "Hole"),
        "needs_review": needs_review,
    }


def test_astm_points():
    assert [astm_d5430_points(v) for v in (0.5, 3.0, 3.01, 6.0, 8.9, 9.1)] == [1, 1, 2, 2, 3, 4]
    assert astm_d5430_points(1.0, is_hole=True) == 2 and astm_d5430_points(1.2, is_hole=True) == 4


def test_overlap_duplicate_is_merged():
    s = RollSession("T1", usable_width_in=60, pixels_per_inch=PPI, frame_length_px=FRAME_PX, frame_overlap_in=2.0)
    # Stain at the bottom of frame 0 (y 1850..1950 px = 18.5..19.5 in) ...
    s.add_frame([defect("Stain", 500, 1850, 100, 100)])
    # ... seen again at the top of frame 1, which starts at 18 in (20 - 2 overlap): y 50..150 px = 18.5..19.5 in
    r = s.add_frame([defect("Stain", 500, 50, 100, 100)])
    assert r == {"added": 0, "merged": 1}, r
    assert len(s.defects) == 1


def test_crease_split_by_frame_boundary_is_rejoined():
    s = RollSession("T2", usable_width_in=60, pixels_per_inch=PPI, frame_length_px=FRAME_PX)
    s.add_frame([defect("Crease", 1000, 1600, 20, 400)])   # 16..20 in -> 4 in -> 2 pts
    s.add_frame([defect("Crease", 1000, 0, 20, 400)])      # 20..24 in -> joined 8 in -> 3 pts
    assert len(s.defects) == 1
    assert s.defects[0]["length_in"] == 8.0 and s.defects[0]["astm_points"] == 3


def test_four_points_per_yard_cap_and_continuous_defect():
    s = RollSession("T3", usable_width_in=50, pixels_per_inch=PPI, frame_length_px=3600)  # 1 yd frames
    s.add_frame([defect("Weave", 100 + 300 * i, 500, 150, 150) for i in range(6)])  # 6 x 1 pt in yard 0
    assert s.points_per_yard() == {0: 4}
    # A 2.5 yd crease running along the roll in yards 1..3 -> 4 pts in each yard touched
    s.add_frame([defect("Crease", 2000, 100, 20, 3500)], frame_start_in=36.0)
    s.add_frame([defect("Crease", 2000, 0, 20, 3500)], frame_start_in=72.0)
    s.add_frame([], frame_start_in=108.0)
    per_yard = s.points_per_yard()
    assert per_yard[1] == 4 and per_yard[2] == 4, per_yard
    summ = s.summary()
    assert summ["inspected_length_yd"] == 4.0
    assert summ["total_astm_points"] == sum(per_yard.values())


def test_review_excluded_until_confirmed_and_grading():
    s = RollSession("T4", usable_width_in=60, pixels_per_inch=PPI, frame_length_px=FRAME_PX,
                    grade_a_limit=20, acceptance_limit=40)
    for i in range(180):                                   # 180 frames x 20 in = 100 yd
        s.add_frame([defect("Hole", 800, 800, 150, 150, needs_review=True)] if i == 90 else [])
    summ = s.summary()
    assert summ["total_astm_points"] == 0 and summ["review_count"] == 1
    assert "chờ xác nhận" in summ["acceptance_status"]
    s.resolve_review(s.defects[0]["uid"], is_defect=True)   # 1.5 in hole -> 4 pts
    summ = s.summary()
    # 4 pts over 100 yd x 60 in: 4 * 3600 / 6000 = 2.4 pts/100yd2 -> grade A
    assert summ["total_astm_points"] == 4 and summ["points_per_100yd2"] == 2.4
    assert summ["roll_grade"].startswith("HẠNG A")


def test_inspection_columns():
    assert inspection_columns(2000, 0, 0, 100) == (100, 1900)
    assert inspection_columns(2000, 200, 1800, 100) == (300, 1700)
    assert inspection_columns(2000, 0, 0, 0) == (0, 2000)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL ROLL SESSION TESTS PASSED")

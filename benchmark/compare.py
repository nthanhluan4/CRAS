"""
compare.py - diff two benchmark runs; exits with code 1 if a required defect that was correct before is not anymore.

    python benchmark/compare.py benchmark/results/<before>.json benchmark/results/<after>.json
"""

import json
import sys


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main(a_path, b_path):
    a, b = load(a_path), load(b_path)
    print(f"A = {a['tag']} ({a['created']})\nB = {b['tag']} ({b['created']})\n")
    regressions = 0
    for mode in b["summary"]:
        sa, sb = a["summary"].get(mode, {}), b["summary"][mode]
        print(f"[{mode}] found {sa.get('found')}->{sb['found']} / {sb['required']} | correct {sa.get('correct')}->{sb['correct']} | "
              f"false alarms {sa.get('false_alarms')}->{sb['false_alarms']}")
        for fname, img_b in b["results"].items():
            img_a = a["results"].get(fname)
            if not img_a:
                continue
            ga = {g["id"]: g for g in img_a["modes"][mode]["gt"]}
            for g in img_b["modes"][mode]["gt"]:
                old = ga.get(g["id"])
                if not old or old["correct"] == g["correct"] and old["category"] == g["category"]:
                    continue
                tag = "REGRESSION" if (old["correct"] and not g["correct"] and g["required"]) else (
                      "fixed" if (not old["correct"] and g["correct"]) else "changed")
                regressions += tag == "REGRESSION"
                print(f"    {tag:10} {g['id']:12} {old['category']}({old['fail_stage']}) -> {g['category']}({g['fail_stage']})")
            fa_a, fa_b = len(img_a["modes"][mode]["false_alarms"]), len(img_b["modes"][mode]["false_alarms"])
            if fa_a != fa_b:
                print(f"    false alarms {fname[-14:-8]}: {fa_a} -> {fa_b}")
    print(f"\n{'FAIL' if regressions else 'OK'}: {regressions} regression(s) on required defects")
    return 1 if regressions else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))

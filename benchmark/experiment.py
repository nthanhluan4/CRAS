"""
experiment.py - run the benchmark with an in-memory variant of one pipeline function, WITHOUT editing
production code. Use it to test a hypothesis before touching app.py / defect_profiler.py.

    python benchmark/experiment.py E1_no_soft_fold
    python benchmark/experiment.py E2_fold_valid_region
Then compare the saved result with the baseline using compare.py.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_benchmark  # noqa: E402  (imports app, sets cwd)
import numpy as np  # noqa: E402

app = run_benchmark.app
_orig_fold = app._fold_anomaly_map


def e1_no_soft_fold(gray):
    """Disable the soft-fold detector (D2b): measures its real contribution."""
    return np.zeros((8, 8), np.float32)


def e2_fold_valid_region(gray):
    """Trust the soft-fold output only where the elongated kernel lies fully inside the image."""
    out = _orig_fold(gray)
    r = int(np.ceil(3 * max(app.FOLD_S_ACROSS, app.FOLD_S_ALONG)))   # kernel half-support, thumbnail px
    out[:r, :] = 0
    out[-r:, :] = 0
    out[:, :r] = 0
    out[:, -r:] = 0
    return out


VARIANTS = {"E1_no_soft_fold": e1_no_soft_fold, "E2_fold_valid_region": e2_fold_valid_region}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in VARIANTS:
        print(__doc__, "\nvariants:", list(VARIANTS))
        sys.exit(2)
    app._fold_anomaly_map = VARIANTS[sys.argv[1]]
    sys.argv = [sys.argv[0], "--tag", sys.argv[1]]
    run_benchmark.main()

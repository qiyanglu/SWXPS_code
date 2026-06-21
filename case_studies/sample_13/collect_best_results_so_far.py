"""Conditionally promote the longer all-RC TRF fit to best_results_so_far."""

from __future__ import annotations

from pathlib import Path
import sys


CASE_DIR = Path(__file__).resolve().parent
PROMOTION_DIR = CASE_DIR / "jax_least_squares_all_rcs"
if str(PROMOTION_DIR) not in sys.path:
    sys.path.insert(0, str(PROMOTION_DIR))

from promote_long_trf_to_best_results import main  # noqa: E402


if __name__ == "__main__":
    main()

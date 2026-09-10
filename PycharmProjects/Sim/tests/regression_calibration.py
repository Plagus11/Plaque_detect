"""Calibration regression tests.

What this guards against
------------------------
The calibration pipeline is fragile: a small change in feature
computation, a re-numbered mechanism prior, or a silent numpy-behaviour
shift can move the Brier score by 0.03 without any test failing. That
kind of drift is invisible until a customer notices their P(win)
predictions have quietly diverged from the calibration curve you sold
them.

This module locks the current numbers into a golden JSON file and
fails CI if any metric drifts beyond a configurable threshold.

Commands
--------
    python3 tests/regression_calibration.py check     # run + diff (CI)
    python3 tests/regression_calibration.py freeze    # accept current as new golden
    python3 tests/regression_calibration.py show      # print current golden

Exit codes
----------
    0 = all metrics within tolerance
    1 = one or more metrics drifted beyond threshold
    2 = golden file missing / malformed

Coverage
--------
The regression suite runs:
    - Lightweight predictor: run_backtest() with defaults
    - Toxicity predictor:    a small synthetic regimen battery
    - Isotonic calibrator:   fixed-seed synthetic dataset invariance
    - Statistical tests:     canonical inputs → canonical outputs

Each produces a small metrics dict that gets diffed against the golden.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "common")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                              # noqa: E402

# --- Modules under test -------------------------------------------------
from historical_calibration import (                             # noqa: E402
    run_backtest, predict_lightweight, build_bbb_context,
    leave_one_out_cv, stratified_metrics,
)
from toxicity_predictor import ToxicityPredictor                 # noqa: E402
from isotonic_calibration import IsotonicCalibrator, fit_ood     # noqa: E402
from statistical_test_simulator import (                          # noqa: E402
    two_prop_test, logrank_test, beta_binom_posterior,
    single_arm_prop_vs_h0,
)


# ---------------------------------------------------------------------------
# Default drift tolerances per metric (absolute).  Tight on quantitative
# metrics; a little slack on stochastic ones.
# ---------------------------------------------------------------------------
DEFAULT_TOL = {
    "brier":         0.005,
    "log_loss":      0.010,
    "roc_auc":       0.010,
    "ece":           0.010,
    "accuracy":      0.010,
    "prob":          0.005,          # for individual probability outputs
    "hr":            0.020,
    "p_value":       0.001,
    "utility":       0.010,
    "counts":        0,              # integer counts must match exactly
}


# ---------------------------------------------------------------------------
# Individual test cases
# ---------------------------------------------------------------------------
def case_calibration_lightweight() -> Dict[str, Any]:
    """Full harness run with the lightweight predictor."""
    registry = os.path.join(_ROOT, "GBM/data/gbm_clinical_trials_blinded.json")
    outcomes = os.path.join(_ROOT, "GBM/data/gbm_trial_outcomes.json")
    drug_lib = os.path.join(_ROOT, "GBM/data/gbm_drug_library.json")
    bbb = build_bbb_context(drug_lib) if os.path.isfile(drug_lib) else {}
    report = run_backtest(registry, predictor=predict_lightweight,
                           context={"bbb_scores": bbb},
                           outcomes_path=outcomes,
                           drop_uncertain=True)
    strata = stratified_metrics(report.predictions, drop_uncertain=True)
    return {
        "n_total":     report.n_total,
        "n_used":      report.n_used,
        "n_positive":  report.n_positive,
        "n_negative":  report.n_negative,
        "brier":       report.brier,
        "log_loss":    report.log_loss,
        "roc_auc":     report.roc_auc if report.roc_auc is not None else -1.0,
        "ece":         report.ece,
        "accuracy":    report.accuracy_at_0p5,
        "phase_III_brier":   strata.get("III", {}).get("brier"),
        "phase_III_accuracy": strata.get("III", {}).get("accuracy"),
    }


def case_toxicity_battery() -> Dict[str, Any]:
    """Canonical regimen safety predictions."""
    tp = ToxicityPredictor()
    r_stupp = tp.predict_regimen(
        [{"drug": "temozolomide", "dose_mg": 200, "interval_hours": 24}],
        covariates={"age": 60, "crcl": 90, "ecog": 1})
    r_pcv = tp.predict_regimen(
        [{"drug": "procarbazine", "dose_mg": 100, "interval_hours": 24},
         {"drug": "lomustine",    "dose_mg": 110, "interval_hours": 1008}],
        covariates={"age": 62, "ecog": 1})
    r_bevlom = tp.predict_regimen(
        [{"drug": "bevacizumab", "dose_mg": 800, "interval_hours": 336},
         {"drug": "lomustine",   "dose_mg": 110, "interval_hours": 1008}],
        covariates={"age": 68, "crcl": 55, "ecog": 2})
    return {
        "stupp_tmz_g3plus":       r_stupp.regimen_grade3plus_rate,
        "stupp_heme":              r_stupp.by_category.get("heme"),
        "pcv_g3plus":              r_pcv.regimen_grade3plus_rate,
        "pcv_heme":                r_pcv.by_category.get("heme"),
        "bevlom_g3plus":           r_bevlom.regimen_grade3plus_rate,
    }


def case_isotonic_invariance() -> Dict[str, Any]:
    """Fixed-seed synthetic dataset; PAV output must be deterministic."""
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.30, size=100).astype(float)
    raw = np.clip(y * 0.5 + rng.normal(0.20, 0.10, size=100), 0.0, 1.0)
    cal, ood = fit_ood(raw, y, n_folds=5,
                        rng=np.random.default_rng(7))
    # Deterministic transformed values at a set of query points
    q = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    tq = cal.transform(q).tolist()
    return {
        "brier_raw":     float(np.mean((raw - y) ** 2)),
        "brier_ood":     float(np.mean((ood - y) ** 2)),
        "n_knots":       len(cal.x_knots),
        "transform_at_0.5": tq[3],
        "transform_at_0.9": tq[5],
    }


def case_statistical_tests() -> Dict[str, Any]:
    """Canonical statistical-test outputs on fixed synthetic inputs."""
    r = two_prop_test(45, 100, 20, 100, alpha=0.05)
    r_bayes = beta_binom_posterior(45, 100, threshold=0.20,
                                     rng=np.random.default_rng(11))
    r_sa = single_arm_prop_vs_h0(28, 60, 0.15, alpha=0.05)
    # Fixed-seed survival
    rng = np.random.default_rng(9)
    t_e = rng.exponential(10, 80); t_c = rng.exponential(6, 80)
    e_e = np.ones(80, int);        e_c = np.ones(80, int)
    r_lr = logrank_test(t_e, e_e, t_c, e_c, alpha=0.05)
    return {
        "twoprop_p":      r.p_value,
        "twoprop_effect": r.effect_size,
        "bayes_p_gt":     r_bayes["p_gt_threshold"],
        "bayes_mean":     r_bayes["posterior_mean"],
        "singlearm_p":    r_sa.p_value,
        "logrank_hr":     r_lr.effect_size,
        "logrank_p":      r_lr.p_value,
    }


# ---------------------------------------------------------------------------
# Metric-classification for tolerance lookup
# ---------------------------------------------------------------------------
def _tol_for(key: str) -> float:
    lk = key.lower()
    if lk.startswith("n_") or lk.endswith("_knots") or lk.endswith("_positive") \
            or lk.endswith("_negative") or lk == "n_total" or lk == "n_used":
        return DEFAULT_TOL["counts"]
    if "brier" in lk or "ece" in lk:                return DEFAULT_TOL["brier"]
    if "log_loss" in lk:                            return DEFAULT_TOL["log_loss"]
    if "auc" in lk:                                 return DEFAULT_TOL["roc_auc"]
    if "accuracy" in lk:                            return DEFAULT_TOL["accuracy"]
    if lk.endswith("_hr") or "logrank_hr" in lk:    return DEFAULT_TOL["hr"]
    if lk.endswith("_p") or lk.endswith("_p_value"):return DEFAULT_TOL["p_value"]
    if "utility" in lk:                             return DEFAULT_TOL["utility"]
    return DEFAULT_TOL["prob"]


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------
def run_suite() -> Dict[str, Dict[str, Any]]:
    return {
        "calibration_lightweight": case_calibration_lightweight(),
        "toxicity_battery":         case_toxicity_battery(),
        "isotonic_invariance":      case_isotonic_invariance(),
        "statistical_tests":        case_statistical_tests(),
    }


# ---------------------------------------------------------------------------
# Diff engine
# ---------------------------------------------------------------------------
@dataclass
class Drift:
    case: str
    metric: str
    golden: float
    current: float
    delta: float
    tol: float

    def as_line(self) -> str:
        return (f"  ✗ {self.case}.{self.metric}: "
                f"golden={self.golden:.4f}  current={self.current:.4f}  "
                f"Δ={self.delta:+.4f}  tol=±{self.tol:.4f}")


def diff_against_golden(current: Dict[str, Dict[str, Any]],
                          golden: Dict[str, Dict[str, Any]]
                          ) -> Tuple[List[Drift], List[str]]:
    drifts: List[Drift] = []
    warnings: List[str] = []
    for case, cur_metrics in current.items():
        if case not in golden:
            warnings.append(f"case '{case}' not in golden; skipping")
            continue
        gold = golden[case]
        for k, cur in cur_metrics.items():
            if k not in gold:
                warnings.append(f"{case}.{k} not in golden; will be added on freeze")
                continue
            g = gold[k]
            if g is None and cur is None:
                continue
            if g is None or cur is None:
                drifts.append(Drift(case, k, float("nan") if g is None else g,
                                      float("nan") if cur is None else cur,
                                      float("nan"), 0.0))
                continue
            try:
                delta = float(cur) - float(g)
            except (TypeError, ValueError):
                if cur != g:
                    drifts.append(Drift(case, k, g, cur, float("nan"), 0.0))
                continue
            tol = _tol_for(k)
            if abs(delta) > tol + 1e-12:
                drifts.append(Drift(case, k, float(g), float(cur),
                                     float(delta), tol))
    return drifts, warnings


# ---------------------------------------------------------------------------
# Golden I/O
# ---------------------------------------------------------------------------
GOLDEN_PATH = os.path.join(_HERE, "golden_calibration.json")


def load_golden(path: str = GOLDEN_PATH) -> Dict[str, Dict[str, Any]]:
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_golden(current: Dict[str, Dict[str, Any]],
                 path: str = GOLDEN_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(current, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_check() -> int:
    print("[regression] running calibration suite")
    current = run_suite()
    golden = load_golden()
    if not golden:
        print(f"[regression] no golden file at {GOLDEN_PATH}. "
              f"Run: python3 tests/regression_calibration.py freeze")
        print("[regression] current metrics (for reference):")
        print(json.dumps(current, indent=2, default=str))
        return 2

    drifts, warns = diff_against_golden(current, golden)
    for w in warns:
        print(f"  ⚠ {w}")
    if drifts:
        print(f"[regression] FAIL — {len(drifts)} metric(s) drifted:")
        for d in drifts:
            print(d.as_line())
        return 1
    print(f"[regression] PASS — all {sum(len(v) for v in current.values())} "
          f"metrics within tolerance across {len(current)} cases")
    return 0


def cmd_freeze() -> int:
    print("[regression] running calibration suite and freezing goldens")
    current = run_suite()
    save_golden(current)
    n = sum(len(v) for v in current.values())
    print(f"[regression] wrote {n} metrics across {len(current)} cases → {GOLDEN_PATH}")
    return 0


def cmd_show() -> int:
    golden = load_golden()
    if not golden:
        print(f"[regression] no golden file at {GOLDEN_PATH}")
        return 2
    print(json.dumps(golden, indent=2, default=str))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Calibration regression suite.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="Run suite and diff against golden.")
    sub.add_parser("freeze", help="Overwrite golden with current metrics.")
    sub.add_parser("show", help="Print the current golden JSON.")
    args = ap.parse_args()

    if args.cmd == "check":  sys.exit(cmd_check())
    if args.cmd == "freeze": sys.exit(cmd_freeze())
    if args.cmd == "show":   sys.exit(cmd_show())

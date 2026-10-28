"""
Cost model shared by train.py and app.py.

One place defines what an error costs and how a threshold turns model scores
into decisions, so the report, the saved sweep and the dashboard can never
disagree.

Unit of analysis: one WORKER OBSERVATION, meaning one worker's head seen in one
camera frame. Ground truth comes from the dataset's `head` (no hard hat) and
`helmet` (hard hat on) boxes. Positive class = `head`: a worker without a hard
hat, which is what the site wants to act on.

Each observation gets a score: the confidence of the model's `head` prediction
that matched it (IoU >= 0.5), or 0 if none did. `head` predictions that match
no worker at all (a ball, a shadow) are "background" false alarms.

Three outcomes per observation, set by two thresholds t_low <= t_high:
    score >= t_high          -> ALERT: supervisor is sent to the worker
    t_low <= score < t_high  -> REVIEW: a person looks at the frame first
    score < t_low            -> NO ACTION
With t_low == t_high there is no review band: the model decides alone.

All euro figures and volumes live in COSTS below. They are DRAFT assumptions,
justified in REPORT.md, and the dashboard reads them from here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

POSITIVE = "head"      # worker without a hard hat
NEGATIVE = "helmet"    # worker wearing one
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


@dataclass(frozen=True)
class Costs:
    # Euro cost per observation for each outcome.
    c_fn: float = 25.0       # bare head left uncorrected until the next check
    c_fp: float = 3.0        # supervisor walks over to a worker who is compliant
    c_tp: float = 3.0        # supervisor walks over and corrects a real violation
    c_review: float = 0.10   # a person checks one flagged frame
    # Production assumptions.
    base_rate: float = 0.05            # share of observations with no hard hat on site
    cases_per_day: int = 2880          # worker observations per day
    review_seconds: float = 12.0       # time to check one frame
    reviewer_hours: float = 6.0        # productive review hours per person per day
    human_miss_rate: float = 0.05      # share of real bare heads a reviewer still misses


COSTS = Costs()


# ---------------------------------------------------------------------------
# Core arithmetic
# ---------------------------------------------------------------------------

def _count_at_or_above(sorted_scores: np.ndarray, t) -> np.ndarray:
    """How many scores are >= t, for scalar or array t (scores sorted ascending)."""
    return len(sorted_scores) - np.searchsorted(sorted_scores, t, side="left")


class Scores:
    """Pre-sorted scores so any threshold pair costs O(log n)."""

    def __init__(self, workers: pd.DataFrame, background: pd.DataFrame):
        w = workers[workers["gt_class"].isin([POSITIVE, NEGATIVE])]
        self.pos = np.sort(w.loc[w["gt_class"] == POSITIVE, "score"].to_numpy(float))
        self.neg = np.sort(w.loc[w["gt_class"] == NEGATIVE, "score"].to_numpy(float))
        self.bg = np.sort(background["score"].to_numpy(float)) if len(background) else np.array([])
        self.n_pos, self.n_neg = len(self.pos), len(self.neg)
        self.n_cases = self.n_pos + self.n_neg
        self.test_base_rate = self.n_pos / self.n_cases if self.n_cases else 0.0


def counts(s: Scores, t_low, t_high) -> dict:
    """Raw test-set counts for each outcome (no reweighting)."""
    t_low, t_high = np.asarray(t_low, float), np.asarray(t_high, float)
    pos_hi, pos_lo = _count_at_or_above(s.pos, t_high), _count_at_or_above(s.pos, t_low)
    neg_hi, neg_lo = _count_at_or_above(s.neg, t_high), _count_at_or_above(s.neg, t_low)
    bg_hi, bg_lo = _count_at_or_above(s.bg, t_high), _count_at_or_above(s.bg, t_low)
    return {
        "tp_alert": pos_hi,                 # bare head, alert sent
        "tp_review": pos_lo - pos_hi,       # bare head, sent to review (person catches it)
        "fn": s.n_pos - pos_lo,             # bare head, no action
        "fp_alert_worker": neg_hi,          # helmeted worker, alert sent
        "fp_alert_bg": bg_hi,               # alert on something that is not a worker
        "neg_review_worker": neg_lo - neg_hi,  # helmeted worker sent to review, dismissed
        "bg_review": bg_lo - bg_hi,            # non-worker sent to review, dismissed
        "tn": s.n_neg - neg_lo,             # helmeted worker, no action
    }


def weights(s: Scores, c: Costs) -> tuple[float, float]:
    """Reweight test counts from the test split's base rate to production's."""
    if s.n_pos == 0 or s.n_neg == 0:
        return 1.0, 1.0
    return c.base_rate / s.test_base_rate, (1 - c.base_rate) / (1 - s.test_base_rate)


def cost_per_1000(s: Scores, t_low, t_high, c: Costs = COSTS) -> dict:
    """Expected cost per 1,000 production observations, plus the pieces."""
    k = counts(s, t_low, t_high)
    wp, wn = weights(s, c)
    caught_in_review = (1 - c.human_miss_rate) * k["tp_review"]
    tp = wp * (k["tp_alert"] + caught_in_review)
    fn = wp * (k["fn"] + k["tp_review"] - caught_in_review)
    fp = wn * k["fp_alert_worker"] + k["fp_alert_bg"]
    reviewed = wp * k["tp_review"] + wn * k["neg_review_worker"] + k["bg_review"]
    total = c.c_tp * tp + c.c_fp * fp + c.c_fn * fn + c.c_review * reviewed
    scale = 1000.0 / s.n_cases
    review_share = reviewed / s.n_cases
    return {
        "cost": total * scale,
        "fn_cost": c.c_fn * fn * scale,
        "fp_cost": c.c_fp * fp * scale,
        "tp_cost": c.c_tp * tp * scale,
        "review_cost": c.c_review * reviewed * scale,
        "review_share": review_share,
        "recall": np.divide(k["tp_alert"] + caught_in_review, s.n_pos) if s.n_pos else 0.0,
    }


def reviewers_needed(review_share: float, c: Costs = COSTS) -> float:
    per_day = review_share * c.cases_per_day
    return per_day * c.review_seconds / (3600 * c.reviewer_hours)


# ---------------------------------------------------------------------------
# Sweeps and the chosen operating point
# ---------------------------------------------------------------------------

def single_threshold_sweep(s: Scores, c: Costs = COSTS) -> pd.DataFrame:
    r = cost_per_1000(s, THRESHOLDS, THRESHOLDS, c)
    k = counts(s, THRESHOLDS, THRESHOLDS)
    df = pd.DataFrame({"threshold": THRESHOLDS, **{key: np.asarray(v) for key, v in r.items()}})
    df["tp"] = k["tp_alert"]
    df["fn"] = k["fn"]
    df["fp"] = k["fp_alert_worker"] + k["fp_alert_bg"]
    df["tn"] = k["tn"]
    df["precision"] = df["tp"] / (df["tp"] + df["fp"]).replace(0, np.nan)
    return df


def best_band(s: Scores, c: Costs = COSTS, max_reviewers: float | None = None) -> dict:
    """Grid-search (t_low, t_high) for minimum cost, optionally capped by staff."""
    lo, hi = np.meshgrid(THRESHOLDS, THRESHOLDS, indexing="ij")
    mask = lo <= hi
    lo, hi = lo[mask], hi[mask]
    r = cost_per_1000(s, lo, hi, c)
    cost = np.asarray(r["cost"], float)
    if max_reviewers is not None:
        need = reviewers_needed(np.asarray(r["review_share"]), c)
        cost = np.where(need <= max_reviewers, cost, np.inf)
    i = int(np.argmin(cost))
    return {"t_low": float(lo[i]), "t_high": float(hi[i]), "cost": float(cost[i])}


def policies(s: Scores, t_low: float, t_high: float, c: Costs = COSTS) -> pd.DataFrame:
    """The four policies the brief asks for, per 1,000 cases and per day."""
    wp, _ = weights(s, c)
    n = s.n_cases
    do_nothing = c.c_fn * wp * s.n_pos / n * 1000
    m = c.human_miss_rate
    check_all = (c.c_review * n + (c.c_tp * (1 - m) + c.c_fn * m) * wp * s.n_pos) / n * 1000
    sweep = single_threshold_sweep(s, c)
    alone = sweep.loc[sweep["cost"].idxmin()]
    band = cost_per_1000(s, t_low, t_high, c)
    rows = [
        ("Do nothing (no camera)", do_nothing, 0.0, "-"),
        ("Person checks every frame", check_all, 1.0, "-"),
        (f"Model alone, t = {alone['threshold']:.2f}", float(alone["cost"]), 0.0, f"{alone['threshold']:.2f}"),
        (f"Model + review, {t_low:.2f} to {t_high:.2f}", float(band["cost"]), float(band["review_share"]),
         f"{t_low:.2f} / {t_high:.2f}"),
    ]
    df = pd.DataFrame(rows, columns=["policy", "cost_per_1000_eur", "review_share", "thresholds"])
    df["cost_per_day_eur"] = df["cost_per_1000_eur"] * c.cases_per_day / 1000
    df["reviewers"] = [reviewers_needed(x, c) for x in df["review_share"]]
    return df


def sensitivity(s: Scores, c: Costs = COSTS) -> pd.DataFrame:
    """Where the best single threshold moves when the shakiest inputs move."""
    rows = []
    for name, field in [("c_fn", "c_fn"), ("c_fp", "c_fp"), ("base_rate", "base_rate")]:
        for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
            cc = replace(c, **{field: getattr(c, field) * mult})
            sw = single_threshold_sweep(s, cc)
            best = sw.loc[sw["cost"].idxmin()]
            band = best_band(s, cc)
            rows.append({
                "input": name, "multiplier": mult, "value": getattr(cc, field),
                "best_threshold": best["threshold"], "cost_per_1000": best["cost"],
                "recall": best["recall"], "band_low": band["t_low"], "band_high": band["t_high"],
            })
    return pd.DataFrame(rows)


def break_even_ratio(s: Scores, c: Costs = COSTS) -> pd.DataFrame:
    """Best single threshold as the FN:FP cost ratio varies (c_fp fixed)."""
    rows = []
    for ratio in [0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 100]:
        cc = replace(c, c_fn=c.c_fp * ratio)
        sw = single_threshold_sweep(s, cc)
        best = sw.loc[sw["cost"].idxmin()]
        do_nothing = cc.c_fn * weights(s, cc)[0] * s.n_pos / s.n_cases * 1000
        rows.append({"fn_fp_ratio": ratio, "c_fn": cc.c_fn, "best_threshold": best["threshold"],
                     "model_cost": best["cost"], "do_nothing_cost": do_nothing,
                     "model_beats_nothing": best["cost"] < do_nothing})
    return pd.DataFrame(rows)


def costs_dict(c: Costs = COSTS) -> dict:
    return asdict(c)

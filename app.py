"""
Assignment 2 dashboard: hard hat alerts, costed.

    uv run python main.py app

Loads weights/best.pt and the files train.py saved in artifacts/. It never
trains anything. Every number on the operating-point tab is recomputed live
from the saved test-set scores through costing.py, the same code the
training pipeline used to pick the thresholds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import costing

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
WEIGHTS = ROOT / "weights" / "best.pt"


def load_json(name: str) -> dict:
    p = ART / name
    return json.loads(p.read_text()) if p.is_file() else {}


def load_csv(name: str) -> pd.DataFrame:
    p = ART / name
    return pd.read_csv(p) if p.is_file() else pd.DataFrame()


OP = load_json("operating_point.json")
RUN = load_json("run_info.json")
STATS = load_json("dataset_stats.json")
PER_CLASS = load_csv("per_class.csv")
RESULTS = load_csv("results.csv")
CM = load_csv("confusion_matrix.csv").rename(columns={"Unnamed: 0": "predicted \\ true"})
WORKERS = load_csv("test_workers.csv")
BACKGROUND = load_csv("test_background.csv")
HAVE_SCORES = not WORKERS.empty
SCORES = costing.Scores(WORKERS, BACKGROUND if not BACKGROUND.empty else pd.DataFrame({"score": []})) \
    if HAVE_SCORES else None
C = costing.COSTS
T_LOW0 = float(OP.get("t_low", 0.3))
T_HIGH0 = float(OP.get("t_high", 0.6))
T_SINGLE = float(OP.get("single_threshold", T_HIGH0))

_model = None


def model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(str(WEIGHTS))
    return _model


def ordered(t_low: float, t_high: float) -> tuple[float, float]:
    return min(t_low, t_high), max(t_low, t_high)


# ---------------------------------------------------------------------------
# Operating point views
# ---------------------------------------------------------------------------

def outcome_table(t_low: float, t_high: float) -> pd.DataFrame:
    k = costing.counts(SCORES, t_low, t_high)
    return pd.DataFrame({
        "True situation": ["Bare head (positive)", "Hard hat on", "Not a worker (model boxed background)"],
        "Alert": [int(k["tp_alert"]), int(k["fp_alert_worker"]), int(k["fp_alert_bg"])],
        "Human review": [int(k["tp_review"]), int(k["neg_review_worker"]), int(k["bg_review"])],
        "No action": [int(k["fn"]), int(k["tn"]), "-"],
    })


def site_story(t_low: float, t_high: float) -> dict:
    """What the thresholds do to 1,000 observations on site (production base rate)."""
    k = costing.counts(SCORES, t_low, t_high)
    wp, wn = costing.weights(SCORES, C)
    scale = 1000.0 / SCORES.n_cases
    m = C.human_miss_rate
    alert_hit = wp * k["tp_alert"] * scale
    review_hit = wp * k["tp_review"] * (1 - m) * scale
    missed = wp * (k["fn"] + k["tp_review"] * m) * scale
    false_alert = (wn * k["fp_alert_worker"] + k["fp_alert_bg"]) * scale
    reviews = (wp * k["tp_review"] + wn * k["neg_review_worker"] + k["bg_review"]) * scale
    r = costing.cost_per_1000(SCORES, t_low, t_high, C)
    return {"bare": 1000 * C.base_rate, "alert_hit": alert_hit, "review_hit": review_hit,
            "missed": missed, "false_alert": false_alert, "reviews": reviews,
            "cost": float(r["cost"]), "share": float(r["review_share"]),
            "precision": alert_hit / (alert_hit + false_alert) if (alert_hit + false_alert) else 0.0}


def summary_md(t_low: float, t_high: float) -> str:
    x = site_story(t_low, t_high)
    nothing = C.c_fn * x["bare"]
    caught = x["alert_hit"] + x["review_hit"]
    walks = x["alert_hit"] + x["review_hit"] + x["false_alert"]
    staff = costing.reviewers_needed(x["share"], C)
    per_day = lambda v: v * C.cases_per_day / 1000  # noqa: E731
    return f"""
## At these thresholds: €{x['cost']:,.0f} per 1,000 observations, €{per_day(x['cost']):,.0f} a day

Of every **1,000 worker observations** on site, about **{x['bare']:.0f} are bare heads** ({C.base_rate:.0%}).

| What happens | Per 1,000 | × price | = € |
| --- | --- | --- | --- |
| Bare head, alert sent, supervisor corrects it | {x['alert_hit']:.1f} | €{C.c_tp:g} | {x['alert_hit'] * C.c_tp:,.0f} |
| Bare head, caught by a person in review | {x['review_hit']:.1f} | €{C.c_tp:g} | {x['review_hit'] * C.c_tp:,.0f} |
| Bare head, **missed** | {x['missed']:.1f} | €{C.c_fn:g} | {x['missed'] * C.c_fn:,.0f} |
| **False alert**: supervisor walks over for nothing | {x['false_alert']:.1f} | €{C.c_fp:g} | {x['false_alert'] * C.c_fp:,.0f} |
| Frames a person checks first (review band) | {x['reviews']:.0f} | €{C.c_review:g} | {x['reviews'] * C.c_review:,.0f} |
| **Total** | | | **{x['cost']:,.0f}** |

- **Without any camera** all {x['bare']:.0f} bare heads are missed: {x['bare']:.0f} × €{C.c_fn:g} = **€{nothing:,.0f}**.
  These thresholds save **€{nothing - x['cost']:,.0f} per 1,000**, about **€{per_day(nothing - x['cost']):,.0f} a day**.
- **Bare heads caught:** {caught / x['bare']:.1%}. **Alerts that are real on site:** {x['precision']:.0%}
  (supervisor walks per 1,000: {walks:.0f}).
- **Review load:** {x['share']:.1%} of observations, {x['share'] * C.cases_per_day:,.0f} checks a day,
  {staff:.2f} of one person's day.
"""


EXPLAINER = f"""

The camera sees about **{C.cases_per_day:,} worker observations a day** (one head in one frame). For each one the model
gives a confidence that the worker has **no hard hat**. Two thresholds turn that score into one of three actions:

- **Above the upper threshold → alert.** The supervisor walks over.
- **Between the two → human review.** Someone looks at the frame for about {C.review_seconds:g} seconds first.
- **Below the lower threshold → no action.**

**Why anything "costs" money.** Each outcome has a price, and the site pays it with or without AI:

| Outcome | Price | Why |
| --- | --- | --- |
| Bare head that nobody acts on (miss) | **€{C.c_fn:g}** | Risk-weighted: a small chance of a head injury or an inspection fine, times what that costs |
| Supervisor walks over (real or false alert) | **€{C.c_tp:g}** | About 5 minutes of the supervisor's time |
| A person checks one frame | **€{C.c_review:g}** | About {C.review_seconds:g} seconds of someone's time |
| Hard hat on, no action | €0 | Nothing happens |

The model saves money by **turning €{C.c_fn:g} misses into €{C.c_tp:g} walks**. A low threshold catches more but sends
the supervisor on more pointless walks; a high one avoids walks but misses more. The sliders show that trade-off, and
the saved operating point is the cheapest setting found by trying every pair.

*Assumptions: {C.base_rate:.0%} of observations are bare heads on site, reviewers miss {C.human_miss_rate:.0%} of
real bare heads. All figures are in `costing.py`.*
"""


def policy_table(t_low: float, t_high: float) -> pd.DataFrame:
    df = costing.policies(SCORES, t_low, t_high, C)
    return pd.DataFrame({
        "Policy": df["policy"],
        "€ per 1,000": df["cost_per_1000_eur"].round(0),
        "€ per day": df["cost_per_day_eur"].round(0),
        "Share reviewed": (df["review_share"] * 100).round(1).astype(str) + "%",
        "Reviewers": df["reviewers"].round(2),
    })


def cost_curve(t_low: float, t_high: float) -> go.Figure:
    sweep = costing.single_threshold_sweep(SCORES, C)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["cost"], name="Total cost", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["fn_cost"], name="Missed bare heads",
                             line=dict(dash="dash", width=1)))
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["fp_cost"], name="False alerts",
                             line=dict(dash="dash", width=1)))
    best = sweep.loc[sweep["threshold"].sub(T_SINGLE).abs().idxmin()]
    fig.add_trace(go.Scatter(x=[best["threshold"]], y=[best["cost"]], mode="markers+text",
                             marker=dict(size=12, color="crimson"), name="Chosen threshold",
                             text=[f"t = {best['threshold']:.2f}"], textposition="top right"))
    fig.add_vrect(x0=t_low, x1=t_high, fillcolor="orange", opacity=0.15, line_width=0,
                  annotation_text="review band", annotation_position="top left")
    fig.update_layout(xaxis_title="Confidence threshold for an alert (model alone)",
                      yaxis_title="Expected cost, € per 1,000 observations",
                      height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    return fig


def update(t_low: float, t_high: float):
    t_low, t_high = ordered(t_low, t_high)
    return summary_md(t_low, t_high), outcome_table(t_low, t_high), policy_table(t_low, t_high), \
        cost_curve(t_low, t_high)


# ---------------------------------------------------------------------------
# Upload and predict
# ---------------------------------------------------------------------------

COLOURS = {"alert": (220, 40, 40), "review": (255, 150, 0), "helmet": (40, 170, 70), "other": (120, 120, 120)}


def predict(image: np.ndarray | None, t_low: float, t_high: float):
    if image is None:
        return None, pd.DataFrame()
    if not WEIGHTS.is_file():
        raise gr.Error("weights/best.pt is missing. Run `uv run python main.py train` first.")
    t_low, t_high = ordered(t_low, t_high)
    imgsz = int(RUN.get("training", {}).get("imgsz", 416))
    # Gradio gives RGB; Ultralytics reads numpy arrays as BGR (OpenCV order).
    bgr = np.ascontiguousarray(image[..., ::-1])
    res = model().predict(bgr, conf=min(t_low, 0.25), imgsz=imgsz, verbose=False)[0]
    names = res.names
    out = image.copy()
    boxes = res.boxes.xyxy.cpu().numpy()
    classes = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    items = []
    for box, cls, conf in zip(boxes, classes, confs):
        name = names[cls]
        if name == costing.POSITIVE:
            if conf >= t_high:
                items.append((box, name, conf, "alert"))
            elif conf >= t_low:
                items.append((box, name, conf, "review"))
        elif conf >= 0.25:
            items.append((box, name, conf, "helmet" if name == costing.NEGATIVE else "other"))
    # The decision uses the head score alone. Hide a helmet box sitting on a
    # flagged head so the picture shows the decision, not both labels.
    flagged = np.array([b for b, _, _, d in items if d in ("alert", "review")]).reshape(-1, 4)
    shown = []
    for box, name, conf, decision in items:
        if decision in ("helmet", "other") and len(flagged):
            tl = np.maximum(flagged[:, :2], box[:2])
            br = np.minimum(flagged[:, 2:], box[2:])
            inter = np.clip(br - tl, 0, None).prod(1)
            union = (flagged[:, 2:] - flagged[:, :2]).prod(1) + (box[2:] - box[:2]).prod() - inter
            if (inter / union).max() > 0.5:
                continue
        shown.append((box, name, conf, decision))
    shown.sort(key=lambda x: x[3] in ("alert", "review"))  # flagged heads drawn last, on top
    rows = []
    for box, name, conf, decision in shown:
        x1, y1, x2, y2 = box.astype(int)
        colour = COLOURS[decision]
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        label = f"{name} {conf:.2f}" + (f" {decision.upper()}" if decision in ("alert", "review") else "")
        cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        rows.append({"class": name, "confidence": round(float(conf), 3), "decision": decision})
    return out, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def dataset_table() -> pd.DataFrame:
    rows = []
    for split, v in STATS.get("splits", {}).items():
        rows.append({"split": split, "images": v["images"], **v["instances"],
                     "images with a bare head": v["images_with_bare_head"]})
    return pd.DataFrame(rows)


def build() -> gr.Blocks:
    ds = RUN.get("dataset", {})
    title = (f"Dataset: [{ds.get('workspace')}/{ds.get('project')} v{ds.get('version')}]({ds.get('url')}) · "
             f"positive class: **{costing.POSITIVE}** (worker without a hard hat)") if ds else ""
    with gr.Blocks(title="Hard hat alerts") as demo:
        gr.Markdown("# Hard hat alerts: where to set the thresholds\n" + title)
        if not HAVE_SCORES:
            gr.Markdown("**No artifacts found.** Run `uv run python main.py train` to create them.")
            return demo
        with gr.Row():
            t_low = gr.Slider(0.01, 0.99, value=T_LOW0, step=0.01,
                              label="Lower threshold: below this, no action")
            t_high = gr.Slider(0.01, 0.99, value=T_HIGH0, step=0.01,
                               label="Upper threshold: above this, alert the supervisor")
        gr.Markdown(f"Between the two, a person reviews the frame. Saved operating point: review between "
                    f"**{T_LOW0:.2f}** and **{T_HIGH0:.2f}**. To see the model deciding alone, set both sliders to "
                    f"**{T_SINGLE:.2f}**, its cheapest single threshold.")
        with gr.Tabs():
            with gr.Tab("Operating point"):
                with gr.Accordion("How the cost works (click to fold)", open=True):
                    gr.Markdown(EXPLAINER)
                summary = gr.Markdown()
                gr.Markdown("### Compared with the alternatives")
                pol = gr.Dataframe(interactive=False)
                gr.Markdown("### Confusion matrix on the test set at these thresholds\n"
                            "Raw counts of the 706 test images, before reweighting to the site's base rate "
                            f"(the test set has {SCORES.test_base_rate:.0%} bare heads, the site {C.base_rate:.0%}).")
                outcomes = gr.Dataframe(interactive=False)
            with gr.Tab("Cost curve"):
                curve = gr.Plot()
                gr.Markdown("Total expected cost when the model decides alone at each threshold. "
                            "The red point is the minimum; the shaded band is the current review band.")
            with gr.Tab("Per class"):
                gr.Markdown("Same test split for all three. Precision and recall at confidence "
                            f"{RUN.get('report_conf', 0.25)}, IoU 0.5.")
                gr.Dataframe(RESULTS.round(3), label="Headline results", interactive=False)
                gr.Dataframe(PER_CLASS.round(3), label="Per class", interactive=False)
                gr.Dataframe(CM, label="Confusion matrix, finetuned model (rows predicted, columns true)",
                             interactive=False)
                if (ART / "confusion_matrix_normalized.png").is_file():
                    gr.Image(str(ART / "confusion_matrix_normalized.png"), label="Ultralytics confusion matrix")
            with gr.Tab("Try an image"):
                with gr.Row():
                    inp = gr.Image(type="numpy", label="Upload a site photo")
                    out = gr.Image(type="numpy", label="Red = alert, orange = review, green = hard hat")
                det = gr.Dataframe(label="Detections", interactive=False)
                gr.Button("Run").click(predict, [inp, t_low, t_high], [out, det])
            with gr.Tab("Dataset"):
                gr.Dataframe(dataset_table(), label="Images and instances per split", interactive=False)
                gr.JSON(STATS.get("box_size", {}), label="Box size relative to the image")

        outputs = [summary, outcomes, pol, curve]
        for s in (t_low, t_high):
            s.change(update, [t_low, t_high], outputs)
        demo.load(update, [t_low, t_high], outputs)
    return demo


if __name__ == "__main__":
    build().launch(server_name="127.0.0.1", server_port=int(os.environ.get("APP_PORT", "7860")))

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
CM = load_csv("confusion_matrix.csv")
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
        "True situation": ["Bare head (positive)", "Hard hat on", "Not a worker (background)"],
        "Alert": [int(k["tp_alert"]), int(k["fp_alert_worker"]), int(k["fp_alert_bg"])],
        "Human review": [int(k["tp_review"]), int(k["neg_review_worker"]), int(k["bg_review"])],
        "No action": [int(k["fn"]), int(k["tn"]), "-"],
    })


def summary_md(t_low: float, t_high: float) -> str:
    r = costing.cost_per_1000(SCORES, t_low, t_high, C)
    share = float(r["review_share"])
    per_day = share * C.cases_per_day
    staff = costing.reviewers_needed(share, C)
    cost = float(r["cost"])
    return (
        f"### €{cost:,.0f} per 1,000 observations  ·  €{cost * C.cases_per_day / 1000:,.0f} per day\n\n"
        f"| Sent to review | Reviews per day | Reviewers needed | Bare heads caught |\n"
        f"|---|---|---|---|\n"
        f"| {share:.1%} | {per_day:,.0f} | {staff:.2f} | {float(r['recall']):.1%} |\n\n"
        f"Cost split per 1,000: missed bare heads €{float(r['fn_cost']):,.0f} · false alerts "
        f"€{float(r['fp_cost']):,.0f} · interventions on real violations €{float(r['tp_cost']):,.0f} · "
        f"reviews €{float(r['review_cost']):,.0f}.\n\n"
        f"Counts in the table are raw test-set worker observations (test base rate "
        f"{SCORES.test_base_rate:.0%}). Costs are reweighted to the production base rate of "
        f"{C.base_rate:.0%} and {C.cases_per_day:,} observations a day."
    )


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
    res = model().predict(image, conf=min(t_low, 0.25), imgsz=imgsz, verbose=False)[0]
    names = res.names
    out = image.copy()
    rows = []
    for box, cls, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int),
                              res.boxes.conf.cpu().numpy()):
        name = names[cls]
        if name == costing.POSITIVE:
            if conf >= t_high:
                decision = "alert"
            elif conf >= t_low:
                decision = "review"
            else:
                continue
        elif conf >= 0.25:
            decision = "helmet" if name == costing.NEGATIVE else "other"
        else:
            continue
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
            t_low = gr.Slider(0.01, 0.99, value=T_LOW0, step=0.01, label="Lower threshold: below = no action")
            t_high = gr.Slider(0.01, 0.99, value=T_HIGH0, step=0.01, label="Upper threshold: above = alert")
        gr.Markdown(f"Saved operating point: review band {T_LOW0:.2f} to {T_HIGH0:.2f}; "
                    f"best single threshold {T_SINGLE:.2f}. Set both sliders equal to see the model alone.")
        with gr.Tabs():
            with gr.Tab("Operating point"):
                summary = gr.Markdown()
                outcomes = gr.Dataframe(label="Test-set outcomes at these thresholds", interactive=False)
                pol = gr.Dataframe(label="Against the alternatives", interactive=False)
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

"""
Evaluation helpers used by train.py.

Everything here is plain numpy over boxes, so the finetuned model, the COCO
zero-shot model and the predict-nothing baseline are all scored by exactly the
same code on exactly the same test images.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


# ---------------------------------------------------------------------------
# Labels and boxes
# ---------------------------------------------------------------------------

def read_yolo_labels(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """YOLO txt (class cx cy w h, normalised) -> (classes, xyxy pixel boxes)."""
    if not label_path.is_file():
        return np.zeros(0, int), np.zeros((0, 4))
    rows = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if len(r) == 5]  # skip polygon rows, if any
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    arr = np.array(rows, float)
    cls = arr[:, 0].astype(int)
    cx, cy, w, h = arr[:, 1] * width, arr[:, 2] * height, arr[:, 3] * width, arr[:, 4] * height
    return cls, np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (n,4) and b (m,4) in xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


# ---------------------------------------------------------------------------
# Standard detection metrics (AP50, mAP50-95, per-class P/R, confusion matrix)
# ---------------------------------------------------------------------------

def match_for_ap(pred_box, pred_cls, pred_conf, gt_box, gt_cls) -> np.ndarray:
    """tp[i, j]: prediction i is a true positive at IoU threshold j (same class)."""
    tp = np.zeros((len(pred_box), len(IOU_THRESHOLDS)), bool)
    if len(pred_box) == 0 or len(gt_box) == 0:
        return tp
    iou = box_iou(pred_box, gt_box)
    iou[pred_cls[:, None] != gt_cls[None, :]] = 0
    order = np.argsort(-pred_conf)
    for j, thr in enumerate(IOU_THRESHOLDS):
        taken = np.zeros(len(gt_box), bool)
        for i in order:
            cand = np.where((iou[i] >= thr) & ~taken)[0]
            if len(cand):
                g = cand[np.argmax(iou[i, cand])]
                taken[g] = True
                tp[i, j] = True
    return tp


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO-style 101-point interpolated AP."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def detection_metrics(images: list[dict], names: dict[int, str], conf_for_pr: float) -> pd.DataFrame:
    """Per-class AP50, AP50-95, and precision/recall at a fixed confidence."""
    rows = []
    for c, name in names.items():
        tps, confs, n_gt = [], [], 0
        tp_at, fp_at = 0, 0
        for im in images:
            gt_mask = im["gt_cls"] == c
            pr_mask = im["pred_cls"] == c
            n_gt += int(gt_mask.sum())
            if not pr_mask.any():
                continue
            tp = match_for_ap(im["pred_box"][pr_mask], im["pred_cls"][pr_mask], im["pred_conf"][pr_mask],
                              im["gt_box"][gt_mask], im["gt_cls"][gt_mask])
            tps.append(tp)
            confs.append(im["pred_conf"][pr_mask])
            # P/R at the fixed confidence, IoU 0.5, matched among confident boxes only
            keep = im["pred_conf"][pr_mask] >= conf_for_pr
            if keep.any():
                tp_k = match_for_ap(im["pred_box"][pr_mask][keep], im["pred_cls"][pr_mask][keep],
                                    im["pred_conf"][pr_mask][keep], im["gt_box"][gt_mask], im["gt_cls"][gt_mask])
                tp_at += int(tp_k[:, 0].sum())
                fp_at += int((~tp_k[:, 0]).sum())
        aps = np.zeros(len(IOU_THRESHOLDS))
        if tps and n_gt:
            tp = np.concatenate(tps)
            conf = np.concatenate(confs)
            order = np.argsort(-conf)
            tp = tp[order]
            ctp = np.cumsum(tp, 0)
            cfp = np.cumsum(~tp, 0)
            for j in range(len(IOU_THRESHOLDS)):
                rec = ctp[:, j] / n_gt
                prec = ctp[:, j] / np.maximum(ctp[:, j] + cfp[:, j], 1)
                aps[j] = average_precision(rec, prec)
        rows.append({
            "class": name, "instances": n_gt,
            "precision": tp_at / (tp_at + fp_at) if (tp_at + fp_at) else 0.0,
            "recall": tp_at / n_gt if n_gt else 0.0,
            "AP50": aps[0], "AP50_95": aps.mean(),
        })
    return pd.DataFrame(rows)


def confusion_matrix(images: list[dict], names: dict[int, str], conf: float, iou_thr: float = 0.5) -> pd.DataFrame:
    """Detection confusion matrix: rows = predicted, cols = true, plus background."""
    labels = list(names.values()) + ["background"]
    k = len(names)
    cm = np.zeros((k + 1, k + 1), int)
    for im in images:
        keep = im["pred_conf"] >= conf
        pb, pc, pconf = im["pred_box"][keep], im["pred_cls"][keep], im["pred_conf"][keep]
        # YOLO's NMS is per class, so one head can carry both a `head` and a
        # `helmet` box. Keep only the most confident label per object (agnostic
        # NMS at IoU 0.7) so the matrix shows what the model would actually say.
        order = np.argsort(-pconf)
        pb, pc, pconf = pb[order], pc[order], pconf[order]
        overlap = box_iou(pb, pb)
        alive = np.ones(len(pb), bool)
        for i in range(len(pb)):
            if alive[i]:
                alive[i + 1:] &= overlap[i, i + 1:] < 0.7
        pb, pc, pconf = pb[alive], pc[alive], pconf[alive]
        gb, gc = im["gt_box"], im["gt_cls"]
        iou = box_iou(pb, gb)
        taken_g = np.zeros(len(gb), bool)
        for i in np.argsort(-pconf):
            cand = np.where((iou[i] >= iou_thr) & ~taken_g)[0] if len(gb) else []
            if len(cand):
                g = cand[np.argmax(iou[i, cand])]
                taken_g[g] = True
                cm[pc[i], gc[g]] += 1
            else:
                cm[pc[i], k] += 1
        for g in np.where(~taken_g)[0]:
            cm[k, gc[g]] += 1
    return pd.DataFrame(cm, index=[f"pred {x}" for x in labels], columns=[f"true {x}" for x in labels])


# ---------------------------------------------------------------------------
# Worker-level scores for the cost model
# ---------------------------------------------------------------------------

def worker_scores(images: list[dict], names: dict[int, str], positive: str, negative: str,
                  iou_thr: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per true worker (head or helmet box) with its best `head` score.

    A `head` prediction is matched greedily, highest confidence first, to the
    unmatched worker box it overlaps most (IoU >= 0.5), whatever that worker's
    true class. A `head` prediction that matches no worker is a background
    false alarm.
    """
    inv = {v: k for k, v in names.items()}
    pos_id, neg_id = inv[positive], inv[negative]
    workers, background = [], []
    for im in images:
        wmask = np.isin(im["gt_cls"], [pos_id, neg_id])
        gb, gc = im["gt_box"][wmask], im["gt_cls"][wmask]
        pmask = im["pred_cls"] == pos_id
        pb, pconf = im["pred_box"][pmask], im["pred_conf"][pmask]
        score = np.zeros(len(gb))
        iou = box_iou(pb, gb)
        taken = np.zeros(len(gb), bool)
        for i in np.argsort(-pconf):
            cand = np.where((iou[i] >= iou_thr) & ~taken)[0] if len(gb) else []
            if len(cand):
                g = cand[np.argmax(iou[i, cand])]
                taken[g] = True
                score[g] = pconf[i]
            else:
                background.append({"image": im["image"], "score": float(pconf[i]),
                                   "x1": pb[i, 0], "y1": pb[i, 1], "x2": pb[i, 2], "y2": pb[i, 3]})
        for g in range(len(gb)):
            workers.append({"image": im["image"], "gt_class": names[int(gc[g])], "score": float(score[g]),
                            "x1": gb[g, 0], "y1": gb[g, 1], "x2": gb[g, 2], "y2": gb[g, 3]})
    cols = ["image", "score", "x1", "y1", "x2", "y2"]
    return pd.DataFrame(workers), pd.DataFrame(background, columns=cols)

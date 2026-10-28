"""
Assignment 2 training pipeline: hard hat compliance on construction sites.

    uv run python main.py train

Start to finish, with no manual steps:
  1. download the public Roboflow dataset (key via api_key.py)
  2. describe it (instances per class, split sizes, box sizes)
  3. finetune yolo11 from the pretrained COCO checkpoint
  4. evaluate the finetuned model, COCO zero-shot and predict-nothing on test
  5. score every test worker for the cost model
  6. sweep thresholds, pick the operating point and the review band
  7. save everything app.py needs into artifacts/, weights into weights/

Environment switches (all optional):
  SKIP_TRAIN=1   reuse weights/best.pt and only redo steps 4 to 7
  QUICK=1        smoke test: 1 epoch on 10% of the data, to check the pipeline
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import costing
import evaluation as ev
from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
WEIGHTS = ROOT / "weights" / "best.pt"
ART = ROOT / "artifacts"

# --- Dataset (public, used as-is, not forked) --------------------------------
WORKSPACE = "joseph-nelson"
PROJECT = "hard-hat-workers"
VERSION: int | None = None   # None = newest public version; the one used is recorded

# --- Training settings ---------------------------------------------------------
# Defaults are what the submitted model used; env vars override them for a test.
BASE_MODEL = os.environ.get("BASE_MODEL", "yolo11n.pt")
EPOCHS = int(os.environ.get("EPOCHS", "20"))
PATIENCE = int(os.environ.get("PATIENCE", "5"))
IMGSZ = int(os.environ.get("IMGSZ", "416"))     # the dataset's native resolution
BATCH = int(os.environ.get("BATCH", "32"))
SEED = 0
REPORT_CONF = 0.25           # confidence for the per-class P/R and confusion matrix

QUICK = os.environ.get("QUICK") == "1"
SKIP_TRAIN = os.environ.get("SKIP_TRAIN") == "1"


def log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------

def _usable(target: Path) -> tuple[dict[int, str], list[str]]:
    """Class names of a downloaded version, and anything that rules it out."""
    import yaml

    meta = yaml.safe_load((target / "data.yaml").read_text())
    names = meta["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    names = {k: str(v).strip().lower() for k, v in names.items()}
    problems = [f"no '{c}' class" for c in (costing.POSITIVE, costing.NEGATIVE) if c not in names.values()]
    problems += [f"no {d} split" for d in ("train", "valid", "test") if not (target / d / "images").is_dir()]
    return names, problems


def download() -> tuple[Path, int, dict[int, str]]:
    """Download the requested version, or the newest one with head/helmet and a test split."""
    import yaml

    if os.environ.get("DATA_YAML"):  # offline testing only: an already-downloaded copy
        candidates = [(int(os.environ.get("DATA_VERSION", "0")), Path(os.environ["DATA_YAML"]).parent, None)]
    else:
        from roboflow import Roboflow

        rf = Roboflow(api_key=load_api_key())
        project = rf.workspace(WORKSPACE).project(PROJECT)
        if VERSION is not None:
            versions = [VERSION]
        else:
            try:
                versions = sorted({int(str(v.version).split("/")[-1]) for v in project.versions()}, reverse=True)
            except Exception as error:  # noqa: BLE001 - fall back to probing numbers
                log(f"could not list versions ({type(error).__name__}: {error})")
                versions = []
            if not versions:
                # Listing versions of another workspace's public project can come back
                # empty, so probe version numbers directly, newest first.
                versions = list(range(20, 0, -1))
            log(f"version numbers to try: {versions}")
        candidates = [(v, DATA_DIR / f"{PROJECT}-v{v}", project) for v in versions]

    for version, target, project in candidates:
        if not (target / "data.yaml").is_file():
            try:
                handle = project.version(version)
            except Exception as error:  # noqa: BLE001 - that version number does not exist
                log(f"version {version} not available ({type(error).__name__})")
                continue
            log(f"downloading {WORKSPACE}/{PROJECT} version {version}")
            handle.download("yolov11", location=str(target), overwrite=True)
            if not (target / "data.yaml").is_file():
                log(f"version {version} downloaded without data.yaml, skipped")
                continue
        else:
            log(f"using cached download in {target}")
        names, problems = _usable(target)
        if problems:
            log(f"version {version} skipped: {', '.join(problems)}")
            continue
        # Roboflow's data.yaml uses ../train/images style paths; rewrite them absolute.
        clean = {"path": str(target), "train": "train/images", "val": "valid/images", "test": "test/images",
                 "names": names}
        yaml_path = target / "data_abs.yaml"
        yaml_path.write_text(yaml.safe_dump(clean, sort_keys=False))
        log(f"using version {version}: classes {list(names.values())}")
        return yaml_path, version, names
    raise SystemExit("No version of the dataset has head and helmet classes plus train/valid/test splits.")


# ---------------------------------------------------------------------------
# 2. Describe
# ---------------------------------------------------------------------------

def describe(root: Path, names: dict[int, str]) -> dict:
    stats = {"splits": {}, "box_size": {}}
    sizes = {n: [] for n in names.values()}
    for split, folder in [("train", "train"), ("valid", "valid"), ("test", "test")]:
        imgs = sorted((root / folder / "images").glob("*"))
        per_class = {n: 0 for n in names.values()}
        with_head, empty = 0, 0
        for img in imgs:
            lab = root / folder / "labels" / (img.stem + ".txt")
            rows = [r.split() for r in lab.read_text().splitlines() if r.strip()] if lab.is_file() else []
            rows = [r for r in rows if len(r) == 5]
            if not rows:
                empty += 1
            seen_head = False
            for r in rows:
                n = names[int(r[0])]
                per_class[n] += 1
                if n == costing.POSITIVE:
                    seen_head = True
                if split == "test" or split == "train":
                    sizes[n].append((float(r[3]), float(r[4])))
            with_head += seen_head
        stats["splits"][split] = {"images": len(imgs), "instances": per_class,
                                  "images_with_bare_head": with_head, "images_without_labels": empty}
    for n, wh in sizes.items():
        if wh:
            a = np.array(wh)
            area = a[:, 0] * a[:, 1]
            stats["box_size"][n] = {
                "median_width_pct": float(np.median(a[:, 0]) * 100),
                "median_area_pct": float(np.median(area) * 100),
                "share_under_1pct_area": float((area < 0.01).mean()),
            }
    test = stats["splits"]["test"]["instances"]
    p, q = test.get(costing.POSITIVE, 0), test.get(costing.NEGATIVE, 0)
    stats["test_base_rate"] = p / (p + q) if (p + q) else None
    return stats


# ---------------------------------------------------------------------------
# 3. Finetune
# ---------------------------------------------------------------------------

def finetune(yaml_path: Path, device: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(BASE_MODEL)
    args = dict(data=str(yaml_path), epochs=1 if QUICK else EPOCHS, patience=PATIENCE, imgsz=IMGSZ,
                batch=BATCH, seed=SEED, deterministic=True, device=device, project=str(ROOT / "runs"),
                name="finetune", exist_ok=True, plots=True, verbose=False)
    if QUICK:
        args["fraction"] = 0.1
    log(f"finetuning {BASE_MODEL} on {device}: {args['epochs']} epochs at {IMGSZ}px")
    started = time.time()
    model.train(**args)
    best = Path(model.trainer.best)
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS)
    log(f"saved {WEIGHTS.relative_to(ROOT)}")
    return {**{k: v for k, v in args.items() if k not in ("data", "project")},
            "train_minutes": round((time.time() - started) / 60, 1),
            "epochs_run": int(model.trainer.epoch) + 1}


# ---------------------------------------------------------------------------
# 4. Predictions on test, for any model
# ---------------------------------------------------------------------------

def predict_test(model, test_dir: Path, class_map: dict[int, int] | None, device: str) -> list[dict]:
    """Run a model on every test image at conf 0.001 and pair it with ground truth.

    class_map maps the model's class ids to dataset ids; boxes of unmapped
    classes are dropped (used for the COCO zero-shot baseline).
    """
    images = []
    paths = sorted((test_dir / "images").glob("*"))
    for res in model.predict([str(p) for p in paths], conf=0.001, iou=0.6, max_det=300, imgsz=IMGSZ,
                             device=device, stream=True, verbose=False):
        p = Path(res.path)
        h, w = res.orig_shape
        gt_cls, gt_box = ev.read_yolo_labels(test_dir / "labels" / (p.stem + ".txt"), w, h)
        b = res.boxes
        cls = b.cls.cpu().numpy().astype(int)
        box = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        if class_map is not None:
            keep = np.isin(cls, list(class_map))
            cls = np.array([class_map[c] for c in cls[keep]], int)
            box, conf = box[keep], conf[keep]
        images.append({"image": p.name, "gt_cls": gt_cls, "gt_box": gt_box,
                       "pred_cls": cls, "pred_box": box, "pred_conf": conf})
    return images


def nothing(images: list[dict]) -> list[dict]:
    return [{**im, "pred_cls": np.zeros(0, int), "pred_box": np.zeros((0, 4)), "pred_conf": np.zeros(0)}
            for im in images]


# ---------------------------------------------------------------------------
# 6. Plots and failure examples
# ---------------------------------------------------------------------------

def plot_cost_curve(sweep: pd.DataFrame, chosen: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep["threshold"], sweep["cost"], color="#1f4e79", lw=2, label="Total")
    ax.plot(sweep["threshold"], sweep["fn_cost"], color="#c0392b", lw=1, ls="--", label="Missed bare heads")
    ax.plot(sweep["threshold"], sweep["fp_cost"], color="#7f8c8d", lw=1, ls="--", label="False alerts")
    best = sweep.loc[sweep["threshold"] == chosen].iloc[0]
    ax.scatter([chosen], [best["cost"]], color="#c0392b", zorder=5)
    ax.annotate(f"t = {chosen:.2f}\n€{best['cost']:.0f} per 1,000", (chosen, best["cost"]),
                textcoords="offset points", xytext=(10, 15))
    ax.set_xlabel("Confidence threshold for an alert")
    ax.set_ylabel("Expected cost, € per 1,000 observations")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_failures(workers: pd.DataFrame, background: pd.DataFrame, test_dir: Path, t: float, out: Path) -> list[dict]:
    """Save the clearest misses and false alarms at the chosen threshold."""
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    misses = workers[(workers["gt_class"] == costing.POSITIVE) & (workers["score"] < t)].nsmallest(4, "score")
    fa_workers = workers[(workers["gt_class"] == costing.NEGATIVE) & (workers["score"] >= t)].nlargest(3, "score")
    fa_bg = background[background["score"] >= t].nlargest(3, "score")
    items = [("miss", r) for _, r in misses.iterrows()] + [("false_alert_helmet", r) for _, r in fa_workers.iterrows()] \
        + [("false_alert_background", r) for _, r in fa_bg.iterrows()]
    saved = []
    for i, (kind, r) in enumerate(items):
        img = cv2.imread(str(test_dir / "images" / r["image"]))
        if img is None:
            continue
        colour = (0, 0, 220) if kind == "miss" else (0, 140, 255)
        cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), colour, 2)
        cv2.putText(img, f"{kind} {r.score:.2f}", (int(r.x1), max(12, int(r.y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        name = f"{i:02d}_{kind}.jpg"
        cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        saved.append({"file": name, "kind": kind, "image": r["image"], "score": round(float(r["score"]), 3)})
    return saved


def md(df: pd.DataFrame, digits: int = 3) -> str:
    """DataFrame -> markdown pipe table without needing tabulate."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda x: f"{x:.{digits}f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(map(str, r)) + " |" for r in df.astype(str).values]
    return "\n".join([head, sep, *body])


def write_report_tables(stats, results, per_class, cm, sweep, pol, sens, brk, op, run_info) -> None:
    rows = []
    for split, v in stats["splits"].items():
        rows.append({"split": split, "images": v["images"], **v["instances"],
                     "images with a bare head": v["images_with_bare_head"]})
    size = pd.DataFrame([{"class": k, **v} for k, v in stats["box_size"].items()])
    picks = sweep[sweep["threshold"].isin([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, op["single_threshold"]])]
    picks = picks[["threshold", "cost", "recall", "precision", "tp", "fp", "fn", "tn"]].drop_duplicates()
    parts = [
        "# Tables generated by train.py (do not edit by hand)",
        f"Dataset: {run_info['dataset']['url']}  \nDevice: {run_info['device']} on {run_info['machine']}  \n"
        f"Training: {json.dumps(run_info['training'])}",
        "## Split sizes and instances", md(pd.DataFrame(rows)),
        "## Box size (train + test labels)", md(size, 2),
        f"Test base rate (head / (head + helmet)): {stats['test_base_rate']:.3f}",
        "## Results on the test split", md(results),
        f"Ultralytics val on the same split: {json.dumps(run_info['ultralytics_val'])}",
        f"## Per class (P/R at conf {REPORT_CONF}, IoU 0.5)", md(per_class),
        f"## Confusion matrix, finetuned model, conf {REPORT_CONF}", md(cm.reset_index().rename(columns={"index": ""}), 0),
        "## Cost sweep, selected thresholds (model alone)", md(picks, 3),
        "## Operating point", "```json\n" + json.dumps(op, indent=2) + "\n```",
        "## Policies", md(pol, 2),
        "## Sensitivity", md(sens, 3),
        "## FN:FP cost ratio vs best threshold", md(brk, 3),
    ]
    (ART / "report_tables.md").write_text("\n\n".join(parts) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from ultralytics import YOLO
    import ultralytics

    ART.mkdir(exist_ok=True)
    device = pick_device()
    yaml_path, version, names = download()
    root = yaml_path.parent

    stats = describe(root, names)
    (ART / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    log(f"dataset: { {s: v['images'] for s, v in stats['splits'].items()} } images")

    if SKIP_TRAIN and WEIGHTS.is_file():
        log("SKIP_TRAIN=1, reusing weights/best.pt")
        prev = ART / "run_info.json"
        train_info = json.loads(prev.read_text())["training"] if prev.is_file() else {}
    else:
        train_info = finetune(yaml_path, device)

    # Official ultralytics validation on test, for its confusion matrix plot.
    model = YOLO(str(WEIGHTS))
    val = model.val(data=str(yaml_path), split="test", imgsz=IMGSZ, batch=BATCH, device=device,
                    project=str(ROOT / "runs"), name="test", exist_ok=True, plots=True, verbose=False)
    for png in ("confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png"):
        src = Path(val.save_dir) / png
        if src.is_file():
            shutil.copy(src, ART / png)
    ultra = {"AP50": float(val.box.map50), "AP50_95": float(val.box.map)}

    # Same evaluator for all three contenders.
    test_dir = root / "test"
    log("predicting on test with the finetuned model")
    fine = predict_test(model, test_dir, None, device)
    log("predicting on test with COCO zero-shot")
    coco_model = YOLO(BASE_MODEL)
    coco_names = {int(k): str(v) for k, v in coco_model.names.items()}
    inv = {v: k for k, v in names.items()}
    coco_map = {cid: inv[n] for cid, n in coco_names.items() if n in inv}  # COCO has 'person' only
    coco = predict_test(coco_model, test_dir, coco_map, device)
    empty = nothing(fine)

    results, per_class = [], []
    for label, ims in [("Finetuned " + BASE_MODEL.replace(".pt", ""), fine),
                       ("COCO " + BASE_MODEL.replace(".pt", "") + " zero-shot", coco),
                       ("Predict nothing", empty)]:
        pc = ev.detection_metrics(ims, names, REPORT_CONF)
        pc.insert(0, "model", label)
        per_class.append(pc)
        head = pc[pc["class"] == costing.POSITIVE].iloc[0]
        results.append({"model": label, "AP50": pc["AP50"].mean(), "mAP50_95": pc["AP50_95"].mean(),
                        "head_AP50": head["AP50"], "head_recall": head["recall"],
                        "head_precision": head["precision"]})
    per_class = pd.concat(per_class, ignore_index=True)
    per_class.to_csv(ART / "per_class.csv", index=False)
    results = pd.DataFrame(results)
    results.to_csv(ART / "results.csv", index=False)
    cm = ev.confusion_matrix(fine, names, REPORT_CONF)
    cm.to_csv(ART / "confusion_matrix.csv")

    # Worker-level scores for the cost model.
    workers, background = ev.worker_scores(fine, names, costing.POSITIVE, costing.NEGATIVE)
    workers.to_csv(ART / "test_workers.csv", index=False)
    background.to_csv(ART / "test_background.csv", index=False)
    coco_workers, coco_bg = ev.worker_scores(coco, names, costing.POSITIVE, costing.NEGATIVE)

    s = costing.Scores(workers, background)
    c = costing.COSTS
    sweep = costing.single_threshold_sweep(s, c)
    sweep.to_csv(ART / "cost_sweep.csv", index=False)
    best_single = float(sweep.loc[sweep["cost"].idxmin(), "threshold"])
    band = costing.best_band(s, c)
    pol = costing.policies(s, band["t_low"], band["t_high"], c)
    pol.to_csv(ART / "policies.csv", index=False)
    sens = costing.sensitivity(s, c)
    sens.to_csv(ART / "sensitivity.csv", index=False)
    brk = costing.break_even_ratio(s, c)
    brk.to_csv(ART / "break_even.csv", index=False)
    coco_cost = costing.cost_per_1000(costing.Scores(coco_workers, coco_bg), 0.5, 0.5, c)["cost"]
    plot_cost_curve(sweep, best_single, ART / "cost_curve.png")
    failures = save_failures(workers, background, test_dir, best_single, ART / "failures")

    band_detail = costing.cost_per_1000(s, band["t_low"], band["t_high"], c)
    operating = {
        "single_threshold": best_single,
        "single_threshold_cost_per_1000": float(sweep["cost"].min()),
        "t_low": band["t_low"], "t_high": band["t_high"],
        "band_cost_per_1000": band["cost"],
        "band_review_share": float(band_detail["review_share"]),
        "band_reviewers": costing.reviewers_needed(float(band_detail["review_share"]), c),
        "band_recall": float(band_detail["recall"]),
        "coco_zero_shot_cost_per_1000": float(coco_cost),
        "test_workers": {"head": s.n_pos, "helmet": s.n_neg, "background_alerts_at_0.001": len(s.bg)},
        "test_base_rate": s.test_base_rate,
        "costs": costing.costs_dict(c),
    }
    (ART / "operating_point.json").write_text(json.dumps(operating, indent=2))

    run_info = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": version,
                    "url": f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}/dataset/{version}",
                    "classes": names},
        "training": train_info, "quick_mode": QUICK, "device": device,
        "machine": f"{platform.system()} {platform.machine()}",
        "ultralytics": ultralytics.__version__, "ultralytics_val": ultra,
        "report_conf": REPORT_CONF, "failures": failures,
    }
    (ART / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))
    write_report_tables(stats, results, per_class, cm, sweep, pol, sens, brk, operating, run_info)

    log("done")
    print(results.round(3).to_string(index=False))
    print(pol.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

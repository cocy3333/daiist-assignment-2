# How to run this submission

Everything runs from the repo folder in Terminal.

## 0. One-time setup (connects this folder to your GitHub repo)

```bash
cd ~/Desktop/AI/daiist-assignment-2
echo "ROBOFLOW_API_KEY=paste-your-key-here" > .env    # put your real key in; .env is never committed
git init -b main && git remote add origin https://github.com/cocy3333/daiist-assignment-2.git
git fetch origin && git reset origin/main && git checkout origin/main -- .github
git status                             # shows the new files and the deleted notebooks
```

## Install and check the key

```bash
uv sync                                # once: installs the pinned environment
uv run python api_key.py               # checks the Roboflow key in .env
```

## 1. Smoke test (about 5 to 10 minutes)

Downloads the dataset and runs every stage on 10% of it for 1 epoch, to catch
problems before the long run.

```bash
QUICK=1 uv run python main.py train
```

## 2. Real training run

```bash
caffeinate -i uv run python main.py train
```

`caffeinate -i` stops the Mac sleeping mid-run. On a MacBook Air this takes
several hours and makes the machine unusable meanwhile, so the submitted model
was trained on Google Colab instead (next section). Settings are at the top of
`train.py`: yolo11n, 416 px, up to 20 epochs, early stop after 5 without
improvement.

## 2b. Training on Google Colab (free T4 GPU, about 30 to 45 minutes)

Runtime > Change runtime type > T4 GPU, then run these cells:

```python
!git clone https://github.com/cocy3333/daiist-assignment-2.git
%cd daiist-assignment-2
!pip install -q "ultralytics>=8.4,<8.5" roboflow
import os
os.environ["ROBOFLOW_API_KEY"] = "paste-your-key-here"
!python main.py train
!zip -qr results.zip weights artifacts
from google.colab import files
files.download("results.zip")
```

Unzip `results.zip` into the repo folder on your Mac, so `weights/best.pt` and
`artifacts/` sit next to `train.py`.

When it finishes you have `weights/best.pt` and `artifacts/`, including
`artifacts/report_tables.md` with every number the report needs.

If you only change a cost figure in `costing.py`, re-run the evaluation
without retraining:

```bash
SKIP_TRAIN=1 uv run python main.py train
```

## 3. Check and look at the dashboard

```bash
uv run python check.py                 # what a grader runs
uv run python main.py app              # then open http://127.0.0.1:7860
```

## 4. Submit

```bash
git add -A
git commit -m "Finetuned hard hat model, costed operating point, dashboard"
git push
```

Then open a pull request from your fork to `ami232/daiist-assignment-2`, post
the PR link on Blackboard and upload a zip of the repo there as a backup.

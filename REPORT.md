# Assignment 2 Report

> **DRAFT written with Claude.** The course AI policy says the ideas, findings and
> conclusions in this report must be your own. Read every section, change what
> you disagree with, and rewrite the reasoning in your own words before you
> submit. Sections marked ⟦FILL AFTER TRAINING⟧ get their numbers from
> `artifacts/report_tables.md` once `main.py train` has run. Delete this box
> before submitting.

- **Name**: Constantin Hatecke
- **Student ID**: ⟦your ID⟧
- **Email**: hatecke.coco@gmail.com
- **Group**: ⟦BBADBA 5A / BBADBA 5B / PPLEDBA 5A / BDBA 3A⟧

## Business problem, and the cost of being wrong

**The decision.** On a construction site, the safety supervisor decides whether
to walk over to a worker and stop work because they are not wearing a hard hat.
Today that decision is made by looking: walking the site and glancing at CCTV.
The model watches zone cameras instead. Every 5 minutes each camera sends a
frame, the model scores every head it sees, and each worker observation ends in
one of three outcomes: an alert sends the supervisor over, a borderline case
goes to a person who checks the frame first, or nothing happens.

**Unit and positive class.** One case is one worker observation: one head in
one frame. The positive class is `head` in the dataset, meaning a worker
without a hard hat. `helmet` is the negative class.

**What each mistake costs.**

| Outcome | Cost | Why |
| --- | --- | --- |
| False negative: bare head, no action | €25 | The worker stays unprotected at least until the next frame. Expected cost = chance of a head injury in that window × its cost, plus the chance an inspector sees it × the fine. I assume a 1 in 20,000 chance of a serious head injury per uncorrected observation and €500k for a serious accident once lost time, sanctions, the employer's benefit surcharge and the site stopping are counted, which gives €25. Fines for a serious prevention breach start at €2,451 ([LISOS art. 40, Gómez-Acebo & Pombo](https://ga-p.com/publicaciones/nuevas-cuantias-de-las-sanciones-laborales/)). |
| False positive: alert on a compliant worker or on nothing | €3 | Supervisor walks over for nothing: about 5 minutes at €30/h fully loaded (€2.50) plus about a minute of the worker's time. |
| True positive: alert on a real bare head | €3 | The same walk happens. It is cheaper than the miss it prevents, not free. |
| Human review of one frame | €0.10 | About 12 seconds at €30/h. |
| True negative | €0 | Nothing happens. |

A miss costs about 8 times a false alert. That ratio, not the absolute euros,
is what moves the threshold, so it is the figure to defend. The weakest number
is the injury probability: it is an assumption, not a measured rate. Section 6
shows what happens to the threshold if it is 4 times higher or lower.

A second cost does not fit in a per-case euro figure: **alert fatigue**. If
most alerts are wrong, supervisors stop walking over, and then the system's
recall in practice falls to zero. That is a reason to care about precision at
the operating point and not only total cost.

**Production assumptions.** A mid-size site with 4 cameras, a frame every 5
minutes over a 10-hour day, and about 6 workers visible per frame gives 2,880
observations a day. I assume 5% of observations are workers without a hard hat
on a site that already enforces the rule. Reviewers spend 6 productive hours a
day on review and still miss 5% of real bare heads.

Construction is the sector where this matters most in Spain: 164 workers died
in workplace accidents in construction in 2025, up 21.5% on 2024
([Forbes España](https://forbes.es/economia/873723/mueren-735-trabajadores-en-accidente-laboral-en-2025-61-menos-que-el-ano-anterior/)).

## Dataset

⟦FILL AFTER TRAINING: exact version from `artifacts/run_info.json`⟧

- Workspace: `joseph-nelson`
- Project: `hard-hat-workers`
- Version: ⟦v⟧
- Link: https://universe.roboflow.com/joseph-nelson/hard-hat-workers
- Licence: Public Domain. Used as published, not forked.

One image is a photo of people at a work site. Every visible head is boxed and
labelled `head` (no hard hat) or `helmet` (hard hat on). Some versions also
have a `person` box around the whole body.

Why it serves the decision rather than only resembling it: the decision is
made per head, and the labels are per head, with exactly the two states the
supervisor cares about. A dataset of "construction scenes" with only `person`
boxes would look similar and be useless here.

## What the dataset's composition implies

⟦FILL AFTER TRAINING: split table and box-size table from `report_tables.md`⟧

Points to check once the numbers are in:

- **Base rate.** Test base rate = head / (head + helmet). If it is around 25%
  against my assumed 5% in production, raw test precision overstates how good
  alerts will look on site, which is why the cost model reweights the counts.
- **`person` is thin.** If `person` has only a few test instances, its AP is
  not a number to trust or report as a finding, and it plays no part in the
  decision anyway.
- **Small objects.** Heads are small relative to the image. A large share of
  boxes under 1% of the image area means misses will concentrate on workers
  far from the camera, the very situation CCTV produces.
- **Test split vs production.** These are mostly close, frontal web and phone
  photos. Site CCTV looks down from height, at lower resolution, in dust and
  rain. The test score is an upper bound on what the site would see.

## Finetuning and baselines

⟦FILL AFTER TRAINING: results table⟧

- Model: `yolo11n`, finetuned from the COCO checkpoint, object detection. ⟦epochs
  run, image size 640, batch 16, seed 0, device, training time⟧
- Where trained: ⟦your MacBook Air (Apple GPU, MPS) / Colab T4⟧
- Baseline 1, COCO zero-shot: same `yolo11n` checkpoint without finetuning. COCO
  has `person` but no `head` or `helmet`, so it scores zero on the positive class
  by construction. In cost terms it is identical to doing nothing.
- Baseline 2, predict nothing: never raises an alert. This is also the majority
  class answer, since most workers wear a hard hat.
- All three are scored by the same evaluator (`evaluation.py`) on the same test
  images. The Ultralytics validator's own AP is reported beside it as a check.

## Per-class evaluation and failure modes

⟦FILL AFTER TRAINING: per-class table, confusion matrix, failure images in
`artifacts/failures/`⟧

Things to state: which class carries the headline AP (expect `helmet`, the
largest class), how `head` compares, and what the confusion matrix says
`head` gets confused with. A bare head predicted as `helmet` is the costly
error. A shaved head, a cap or a hood are likely culprits, and the failure
images will show which.

## Cost matrix and operating point

⟦FILL AFTER TRAINING: cost curve (`artifacts/cost_curve.png`), operating point,
sensitivity table⟧

Method: for every threshold from 0.01 to 0.99, count TP, FP, FN and TN on the
test workers, reweight positives and negatives from the test base rate to the
5% production rate, apply the cost matrix and scale to 1,000 observations.
The operating point is the minimum of that curve, not a chosen value.

The assumption that would most change the answer is the FN:FP cost ratio. The
FN:FP ratio table shows the best threshold as the ratio moves from 0.5 to 100,
and the break-even point below which the model no longer beats doing nothing.

## What a person still has to check

⟦FILL AFTER TRAINING: band thresholds, share reviewed, reviews per day,
reviewers needed, four-policy table⟧

Method: search all pairs t_low ≤ t_high. Above t_high an alert goes straight
to the supervisor, between the two a person checks the frame, below t_low
nothing happens. Reviewers needed = reviews per day × 12 s / (3,600 × 6 h).

Compare against: doing nothing, a person checking every frame (2,880 a day,
about 1.6 people), and the model alone at its best single threshold.

## Limitations & next steps

- **Domain gap.** Web photos are not CCTV. Next step: label a few hundred
  frames from one real site camera and re-measure recall before trusting the
  threshold.
- **Injury probability is assumed.** Replace it with the site's own incident
  and near-miss rate, then re-run the sweep (one number in `costing.py`).
- **Observations are not independent.** The same worker appears in many frames,
  so a miss now may be caught 5 minutes later. Costing per worker-hour instead
  of per frame would lower the effective FN cost.
- **Alert fatigue is not in the euros.** A floor on precision at the operating
  point would make it explicit.
- ⟦add what the results show, e.g. a class that is too thin⟧

## Generative AI use disclosure

I used Claude (Anthropic) for this assignment. It searched Roboflow Universe
for candidate datasets, wrote the code (`train.py`, `evaluation.py`,
`costing.py`, `app.py`), tested it on a synthetic dataset, and drafted this
report including the first version of the cost figures and their reasoning.
I chose the problem and the dataset, reviewed and ⟦changed / kept⟧ the cost
assumptions, ran the training, and rewrote the analysis in my own words.
⟦Edit this to match what you actually did.⟧

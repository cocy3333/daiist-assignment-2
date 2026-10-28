# Assignment 2 Report


- **Name**: Constantin Hatecke
- **Student ID**: 20241
- **Email**: chatecke.ieu2022@student.ie.edu
- **Group**: ⟦BBADBA 5A⟧

## Business problem, and the cost of being wrong

**The decision.** The business problem here is simple but clear. Construction sites worldwide require workers to wear protective helmets. Workers without helmets can get the construction company fined or, worse, get injured. Without protection, insurance may not cover the bill, and companies and whole families can go bankrupt. Additionally, the cost of the supervisor enforcing the helmet rule costs money, as he has to enforce the rule and can't work himself. 
Today that decision is made by looking: walking the site and glancing at CCTV. The model watches zone cameras instead. Every 5 minutes each camera sends a
frame, the model scores every head it sees, and each worker observation ends in one of three outcomes: an alert sends the supervisor over, a borderline case
goes to a person who checks the frame first, or nothing happens.

**Unit and positive class.** 
There are two classes. 
For every head in frame, the model will output the positive class "head", meaning the worker is not wearing a helmet. Or "helmet", which is the negative class.

**What each mistake costs.**

| Outcome | Cost | Why |
| --- | --- | --- |
| False negative: bare head, no action | €25 | The worker stays unprotected at least until the next frame. Expected cost = chance of a head injury in that window × its cost, plus the chance an inspector sees it × the fine. In collaboration wiht claude, we established that this is the approximate real cost of not wearing a helmet in Spain. 

I assume a 1 in 20,000 chance of a serious head injury per uncorrected observation. €500k is what medical treatment and lost time on the site would add up to for a serious accident, which comes to €25 per bear head per day. 

| False positive: alert on a compliant worker or on nothing | €3 | This is due to the supervisor walking over for nothing: about 5 minutes at €30/h fully loaded (€2.50) plus about a minute of the worker's time. |

| True positive: alert on a real bare head | €3 | The same cost as a false positive but a lot cheaper than the miss it prevents. |
| Human review of one frame | €0.10 | About 12 seconds at €30/h. |
| True negative | €0 | Nothing happens. |

A miss costs about 8 times a false alert which is the number that was used to
set the threshold, so it is the figure to defend. The weakest number
is the injury probability. It is an assumption, not a measured rate and depends on the workers' experience, the overall site safety and type of construction site. The
sensitivity section shows what happens if it is 4 times higher or lower, which helps managers to differentiate their own site condition and estimate their own cost-benefit.

Another important measure taken into account is alert fatigue. If
most alerts are wrong, supervisors stop walking over to check as they are too busy for false positives. The system's real
recall falls to zero. That is a reason to care about precision at the
operating point and not only total cost.

Looking at an average construction site with 4 cameras, a frame every 5
minutes over a 10-hour day, and about 6 workers visible per frame gives 2,880
observations a day. I am assuming 5% of observations are workers without a hard hat
on while on duty and under the rule to wear one. Reviewers spend 6 productive hours a
day on review and still miss 5% of real bare heads.

Construction is where this rule enforcement matters most in Spain: 164 workers died in workplace
accidents in construction in 2025, up 21.5% on 2024. This technology could make a noticeable change.
([Forbes España](https://forbes.es/economia/873723/mueren-735-trabajadores-en-accidente-laboral-en-2025-61-menos-que-el-ano-anterior/)).

## The Dataset

- Workspace: `joseph-nelson`
- Project: `hard-hat-workers`
- Version: 13
- Link: https://universe.roboflow.com/joseph-nelson/hard-hat-workers/dataset/13
- Licence: Public Domain. Used as published, not forked.

The images show people at a work site. Every visible head is boxed and
labelled `head` (no hard hat) or `helmet` (helmet on). Other versions were skipped
because they labeled differently. For example, they would merge `helmet` into `head`, which removes the
distinction the decision depends on.

The supervisor decides one worker at a time: hard hat or no hard hat. This dataset is labelled the same way. Every
head has its own box, marked either helmet or head (no hard hat). So the
model learns exactly the question the supervisor has to answer. A dataset that
only boxed whole people on construction sites would look similar, but it could
not tell a compliant worker from a non-compliant one, which is the only thing
that matters here.

## What the dataset's composition implies

| Split | Images | `head` | `helmet` | Images with a bare head |
| --- | --- | --- | --- | --- |
| Train | 14,748 | 13,670 | 41,450 | 2,729 |
| Valid | 1,413 | 1,339 | 3,913 | 256 |
| Test | 706 | 726 | 1,915 | 140 |

| Class | Median box width | Median box area | Boxes under 1% of image area |
| --- | --- | --- | --- |
| `head` | 7.0% of image width | 0.78% | 61% |
| `helmet` | 8.4% | 1.08% | 48% |

What this implies:

- **The per-class numbers are trustworthy in size.** 726 bare heads and 1,915
  helmets in test is enough for stable precision and recall. The caveat is
  clustering: the 726 bare heads sit in only 140 images, about 5 per image, so
  they are not independent. A few crowd photos carry a lot of the `head` score. This could mean that the model has higher false positives if there are already treu positives in the image.
- **The test base rate is far from production.** 27.5% of test workers have no
  hard hat, against my assumed 5% on site. At 5%, a false alert on a worker
  weighs about 7 times more relative to a true alert than it does in the test
  set (weights 0.95/0.725 against 0.05/0.275), so
  raw test precision overstates how good alerts will look on site. The cost
  model reweights positives and negatives to 5% for this reason. This also works because sites that intend to install this will be eager to reach 100% compliance. A construction site with 27.5% of workers that don't wear helmets already doesn't have high standards, and they can reduce this level more efficiently with other methods such as incentives or individual fines.
- **Heads are small.** 61% of bare heads cover less than 1% of the image. Misses
  will concentrate on workers far from the camera, which is exactly what a site
  camera mounted at height produces. Further tests are necessary to find the optimal camera position for every site.
  
- **Test does not look like production.** These are mostly close, eye-level web
  and news photos, many from Chinese sites. Site CCTV looks down from height, at
  lower resolution, in dust and rain. The test score is probably an upper bound on what
  the site would see.

## Finetuning and baselines

`yolo11n`, finetuned from the COCO checkpoint for object detection. Trained on
Kaggle (Tesla T4): 20 epochs at 416 px (the dataset's native resolution), batch
32, seed 0, 20.9 minutes. Early stopping (patience 5) never triggered.

| Model (same 706 test images) | AP50 | mAP50-95 | `head` AP50 | `head` recall | `head` precision |
| --- | --- | --- | --- | --- | --- |
| Finetuned yolo11n | **0.959** | **0.640** | **0.948** | 0.935 | 0.897 |
| COCO yolo11n zero-shot | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Predict nothing | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

How the numbers were measured.
A box counts as correct when it overlaps the
true box by at least 50% (IoU 0.5). Precision and recall are taken at a
confidence cut-off of 0.25. As a check, I compared my evaluation code with
Ultralytics’ built-in one on the same test images: it gives AP50 0.961 and
mAP50-95 0.658, almost the same as my 0.959 and 0.640. So the numbers for all
three models can be trusted.

Why the baselines score zero. 
This is expected, not a surprise. The pretrained COCO model was never taught “head” or “helmet”, so it cannot answer
the question at all. In cost terms it behaves exactly like having no camera.
“Predict nothing” is the same as always guessing the majority answer, since
72.5% of test workers wear a hard hat. Neither baseline is a real test of the
model. The real test is whether it beats a person checking every frame, which
is compared in the review section.

Training could have run a bit longer. Validation mAP50-95 was still rising
slowly at the last epoch (0.645, 0.646, 0.647, 0.648 over the final four).
More epochs would add a little. The gain would mostly be tighter boxes, not
more bare heads found, because recall had already levelled off.

## Per-class evaluation and failure modes

| Class | Test instances | Precision | Recall | AP50 | AP50-95 |
| --- | --- | --- | --- | --- | --- |
| `head` (positive) | 726 | 0.897 | 0.935 | 0.948 | 0.648 |
| `helmet` | 1,915 | 0.913 | 0.943 | 0.970 | 0.632 |

Confusion matrix at confidence 0.25, IoU 0.5 (rows predicted, columns true):

| | True `head` | True `helmet` | True background |
| --- | --- | --- | --- |
| Predicted `head` | 677 | 6 | 71 |
| Predicted `helmet` | 13 | 1,804 | 156 |
| Missed (background) | 36 | 105 | 0 |

![Confusion matrix](artifacts/confusion_matrix_normalized.png)

'helmet' scores best (AP50 0.970) because it is the largest class and the
easiest to see. The model finds a bright hard hat stands out. head (no hard hat) is harder
at 0.948 but still very good, and it is the class the decision depends on.

The model almost never mixes the two up. Only 13 of 726 bare heads were
called helmet, and only 6 hard hats were called head. The real errors are
36 bare heads not found at all, and 71 alerts on things not labelled as workers.
Missed heads are small or partly hidden: a driver behind cab glass, a face
half covered by another worker’s hard hat, a tiny head far back in a crowd
(artifacts/failures/00 to 03).

Some “false alerts” are missing labels. The most confident alerts on
background are people clearly without a hard hat who were never labelled
(artifacts/failures/07, 08). The model is right there, so the true
precision for head is probably a bit higher than measured.

Wrong alerts on helmeted workers happen when the helmet is cut off or
only a few pixels wide (artifacts/failures/04 to 06).

## Cost matrix and operating point

For every threshold from 0.01 to 0.99, the pipeline counts TP, FP, FN and TN
over the 2,641 test workers plus the alerts on background, reweights positives
and negatives from the 27.5% test base rate to the 5% production rate, applies
the cost matrix and scales to 1,000 observations.

![Cost against threshold](artifacts/cost_curve.png)

**Operating point for the model alone: threshold 0.42, €292 per 1,000
observations.** At 0.42 the model finds 91.5% of bare heads with 94.2% of its
alerts correct on the test set. Lets have a look how this plays out.

| Threshold | € per 1,000 | Recall | Precision | FN | FP |
| --- | --- | --- | --- | --- | --- |
| 0.20 | 328 | 0.937 | 0.881 | 46 | 92 |
| 0.30 | 308 | 0.928 | 0.908 | 52 | 68 |
| **0.42** | **292** | **0.915** | **0.942** | **62** | **41** |
| 0.50 | 305 | 0.888 | 0.958 | 81 | 28 |
| 0.70 | 448 | 0.741 | 0.980 | 188 | 11 |

The curve is flat between about 0.3 and 0.5 (within €16 per 1,000) and climbs
steeply above 0.6, where misses take over. Setting the threshold a bit too low
is cheap, setting it too high is expensive, which is the cost asymmetry showing
up in the curve.

Part of the cost cannot be avoided by any threshold: at a 5% base rate there
are 50 real bare heads per 1,000 observations, and correcting each one costs
€3, so €150 of every 1,000 is the price of doing the job at all. The avoidable
part is €142 at the chosen threshold.

**Sensitivity.** The best threshold moves with the FN:FP ratio, as it should:

| Change | Best threshold | € per 1,000 |
| --- | --- | --- |
| c_FN × 0.25 (€6.25) | 0.54 | 197 |
| c_FN × 0.5 (€12.50) | 0.52 | 233 |
| **as assumed (€25)** | **0.42** | **292** |
| c_FN × 2 (€50) | 0.34 | 389 |
| c_FN × 4 (€100) | 0.14 | 552 |
| Base rate 2.5% | 0.52 | 166 |
| Base rate 10% | 0.34 | 528 |

The most important assumption is the injury probability behind c_FN, because
it decides how expensive a miss is. If a miss cost no more than a walk (€3),
the camera would be pointless: catching a bare head would cost the same as
missing it. But even at €6, twice the cost of a walk, the model is already a
third cheaper than doing nothing. And whether a miss costs €6 or €100, the best
threshold only moves between 0.14 and 0.54, so the choice holds up well even
if the €25 estimate is off.

## What a person still has to check

**Review band: alert above 0.66, review between 0.03 and 0.66, no action below
0.03.** The grid search over all threshold pairs put the lower edge very low
because a review (€0.10) is so much cheaper than a miss (€25): it is worth
having a person glance at almost any head the model is unsure about.

- Share of cases sent to review: **10.3%**
- Reviews per day: 10.3% × 2,880 ≈ **296**
- Time: 296 × 12 s ≈ **59 minutes a day**, which is **0.16 of one reviewer**
- Bare heads caught (model plus reviewer): **95.2%**

That is realistic. Nobody needs to be hired: an hour a day fits into the
existing supervisor's or site office's day, in short batches, and makes the model extremely effective.

| Policy | € per 1,000 | € per day | Share reviewed | People |
| --- | --- | --- | --- | --- |
| Do nothing | 1,250 | 3,600 | 0% | 0 |
| Person checks every frame | 305 | 878 | 100% | 1.6 |
| Model alone (t = 0.42) | 292 | 842 | 0% | 0 |
| **Model + review (0.03 / 0.66)** | **228** | **656** | **10.3%** | **0.16** |

- Every option with a camera beats doing nothing by at least €2,700 a day.
- **The model alone barely beats a person checking every frame** (€292 vs €305).
  On paper a person is competitive because reviews are cheap. In practice the
  check-everything option is weaker than it looks: 2,880 checks a day is 9.6
  hours of staring at frames, and a 5% miss rate for that is optimistic. Human error and fatigue/ distractions can become serious. Not to mention days where the person responsible is out.                                                                         
- **Model plus review is the cheapest** at €228, €186 a day less than the model
  alone and €222 less than checking everything, while using a tenth of a
  person's day. The avoidable cost falls from €142 to €78 per 1,000.
- **It also answers alert fatigue.** At the site's 5% base rate, only 74% of the
  model-alone alerts are real bare heads, so one walk in four is for nothing.
  With the review band, 89% of alerts are real, because the doubtful cases are
  filtered by a 12-second check before anyone walks.

Recommendation: ship the model with the review band, not alone.

## Limitations & next steps

- **Domain gap.** Web and news photos are not CCTV. Label a few hundred frames
  from one real site camera and re-measure recall before trusting 0.42 or the
  band.
- **Missing labels.** The top "false alerts" on background are unlabelled bare
  heads. Re-labelling the 50 highest-scoring background alerts would give a
  truer precision and would likely move the threshold down.
- **Injury probability is assumed.** Replace it with the site's own incident and
  near-miss data (one number in `costing.py`), then re-run with
  `SKIP_TRAIN=1 uv run python main.py train`. This can give any site their real 
- **Observations are not independent.** The same worker appears in many frames,
  so a miss now may be caught 5 minutes later. Costing per worker-hour instead
  of per frame would lower the effective c_FN.
- **Alert fatigue is not in the euros.** A floor on alert precision (for example
  90% at the production base rate) would make it an explicit constraint.
- **"Do nothing" is generous to the model.** Today supervisors already walk the
  site and catch some bare heads by eye. With no data on how many, the real
  saving is smaller than the €2,750 a day against no camera at all.
- **System costs are left out.** Cameras, inference hardware and maintenance are
  not in the per-case costs. At savings of this size they pay back quickly, but
  a full business case would net them off.
- **Training was cut at 20 epochs** while validation was still improving
  slightly. A longer run, or `yolo11s`, would mainly help small, distant heads.
- **Clustered positives.** 726 test bare heads come from 140 images, so
  confidence intervals should be computed by resampling images, not boxes.

## Generative AI use disclosure

I used Claude (Anthropic) for this assignment. It wrote the code (`train.py`, `evaluation.py`,
`costing.py`, `app.py`), tested it on a synthetic dataset, and drafted some parts of this report, including the cost figures and tables.
I chose the problem and the dataset, reviewed and ⟦changed / kept⟧ the cost
assumptions, ran the training on Kaggle, and wrote the analysis in my own
words. 

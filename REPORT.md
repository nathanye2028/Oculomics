# Oculomics — Retinal Age Report, 6 – 12 September 2026

Branch `disease/retinal-age`. The diabetic-retinopathy classification work that
preceded this track (BRSET → mBRSET transfer, distillation, AdaBN, the operating
point, Core ML deployment) is documented in `REPORT.md` on `main`; this document
covers only the retinal-age track, from the first design on 6 September to the
state of the experiments on 12 September.

**Headline.** A 2.8 M-parameter mobile network trained on 5,200 healthy
tabletop fundus images predicts age with **MAE 5.07 ± 0.09 years, r = 0.92**
on held-out BRSET patients — the accuracy of the published UK Biobank clocks,
on a quarter of the data. Trained in-domain on smartphone images it reaches
**MAE 4.9 years, r = 0.82** on held-out mBRSET patients, so the phone images
carry the age signal; the zero-shot transfer between cameras does not (r 0.43),
and mixed-domain training is the pending fix. The bias-corrected **retinal age
gap tracks diabetic eye disease on both cameras** in dose–response fashion
(referable DR +4.0 years and macular edema +5.1 on BRSET; referable-DR
prevalence 13 % → 56 % across gap quintiles), and three signals replicate under
two independent clocks on the same phone patients: insulin use, hypertension
(borderline) and diabetes duration. Most of the DR effect is lesion detection
rather than ageing (+2.3 → +1.0 years when the clock knows the phone domain),
systemic effects are bounded to under about a year at this sample size, and an
ungradable image alone is worth 1.5 – 2 years, so every association uses
gradable images only.

---

## 1. The question

The project's premise is that a small network running on a phone can read
clinically useful signal from an unconstrained fundus photograph. Age is the
cleanest such signal: every retina carries it, the label is free, and the
**retinal age gap** — how much older a retina looks than the patient is, after
correcting the clock's own bias — has been proposed as a biomarker of vascular
ageing (Zhu et al. 2023, Poplin et al. 2018). Two steps:

1. Train a clock on retinas that are ageing normally, and measure it on both
   cameras.
2. Test the corrected gap against every clinical label BRSET and mBRSET carry,
   at the patient level, adjusted for age and sex.

The design choices that matter: which patients count as healthy, how the split
avoids leakage, how the gap is corrected, and how the phone domain is handled.

## 2. Data

| | BRSET (tabletop, Canon CR / Nikon) | mBRSET (smartphone, Phelcom Eyer) |
|---|---|---|
| images with an age | 10,820 (5,649 patients) | 5,152 (1,288 patients) |
| healthy for training | 7,413 images, 4,061 patients | DR-0 only: 2,974 images, 747 patients (used for the ceiling) |
| excluded from training | diabetes 2,384 · ungradable 963 · DR in a non-diabetic 60 | DR 1,886 · ungradable 292 |
| age | mean 57.6, wide (979 patients under 30) | 50–79 cluster (8 patients under 30) |
| population | mixed | every patient diabetic |

The age-range asymmetry matters twice below: it makes r non-comparable across
the two sets, and it means an mBRSET-trained clock cannot extrapolate to BRSET's
young and old tails.

## 3. Method

* **Healthy cohort** (`--healthy nodm`): no diabetes, DR grade 0 on *every*
  image of the patient (one eye with retinopathy disqualifies the fellow eye),
  adequate quality. `dr0` keeps diabetics without retinopathy (used for mBRSET,
  which has no non-diabetics); `--exclude-pathology` also drops BRSET's other
  ophthalmic flags.
* **Split**: patient-grouped, age-stratified 70/10/20 drawn over *all* patients
  first, the healthy rule applied to train and val afterwards. Every diseased
  image is therefore scorable without leakage, and none was trained on.
* **Recipe**: MobileNetV4-Small at 384 px, ImageNet init, L1 on standardised
  age, warmup + cosine, checkpoint chosen on healthy-val MAE. Same loader, FOV
  crop, augmentation and AdaBN as the DR classifier, so nothing in the pipeline
  differs between cameras except the pixels.
* **Bias correction**: a regression's gap is anti-correlated with age
  (regression to the mean: young over-, old under-predicted). Following
  Beheshti et al. (2019), `gap = a + b·age` is fit on the healthy val split and
  subtracted (BRSET: a = 8.5 ± 1.0, b = −0.155 ± 0.016). Every disease analysis
  uses the corrected gap.
* **Device calibration**: the phone set is also reported after a linear
  `age ≈ c + d·pred` fit on its DR-0 patients in two patient-grouped folds,
  every row scored out of fold, with its own within-set bias correction. Age
  labels are free, so this is an honest "calibrated on the target device"
  number; it can fix scale and offset, never information.
* **Ceiling**: the same architecture trained on mBRSET's DR-0 patients
  (`--dataset mbrset --healthy dr0`), scored on held-out mBRSET patients — the
  test of whether the phone images carry the signal at all.
* **Association analysis** (`analyze_age_gap.py`): patient level (both eyes
  averaged), gradable images only, OLS adjusted for age, age², sex and (BRSET)
  camera; per exposure the adjusted difference in corrected gap with 95 % CI,
  p and Benjamini–Hochberg q, Cohen's d, disease prevalence by quintile of the
  gap, the odds ratio per +5 years of gap (logistic, same covariates), the
  DR-grade and diabetes-duration trends, and an image-quality artefact check
  (ungradable vs gradable among disease-free patients). External rows use the
  device-calibrated, within-set-corrected gap; the in-domain correction is never
  applied across devices.

Code: `train_retinal_age.py`, `run_retinal_age.sh`, `summarize_retinal_age.py`,
`analyze_age_gap.py`; tests `tests/test_retinal_age.py`, `tests/test_age_gap.py`
(suite: 99 tests, all synthetic and network-free).

## 4. The clock

### 4.1 Tabletop camera (BRSET, 3 seeds, MobileNetV4-Small at 384 px)

| BRSET held-out patients | MAE (y) | r | corrected gap (y) | n |
|---|---|---|---|---|
| healthy test | **5.07 ± 0.09** | **0.920 ± 0.004** | +0.17 ± 0.08 | 1,488 |
| test, all | 5.31 ± 0.07 | 0.917 | +0.73 ± 0.05 | 2,168 |
| test, non-healthy | 5.81 ± 0.04 | 0.913 | +1.86 ± 0.01 | 680 |
| non-healthy, never trained on | 5.77 ± 0.02 | 0.913 | +1.98 ± 0.32 | 2,727 |

MAE by age bin (healthy test):

| bin | <30 | 30–39 | 40–49 | 50–59 | 60–69 | 70–79 | 80+ |
|---|---|---|---|---|---|---|---|
| n | 135 | 133 | 219 | 298 | 319 | 253 | 131 |
| MAE (y) | 6.0 | 4.5 | 5.0 | 4.8 | 4.7 | 5.0 | 6.5 |
| raw mean gap | +4.6 | +2.2 | +2.3 | +1.0 | −1.2 | −2.3 | −6.1 |

The tails are one to two years worse — BRSET is 40–70 heavy — and the raw gap
drifts from +4.6 in the under-30s to −6.1 in the 80+, which is what the bias
correction removes. For scale: Poplin et al. (2018) and Zhu et al. (2023) report
3.3 – 3.6 years on the UK Biobank with 20 – 40 k training images, much larger
networks and a 40 – 69 age range; predicting the mean age on BRSET gives about
14 years. Excluded patients read **+1.7 ± 0.1 years** older than healthy ones
after correction, consistent between the small test slice and the 2,727
never-trained images.

### 4.2 Smartphone camera (mBRSET, n = 4,860 images, 1,282 patients)

| mBRSET | MAE (y) | r | mean gap (y) |
|---|---|---|---|
| zero-shot | 14.02 ± 2.17 | 0.434 ± 0.057 | −9.6 ± 3.8 |
| + AdaBN | 9.65 ± 0.18 | 0.416 ± 0.017 | −3.3 ± 0.4 |
| device-calibrated | 8.16 ± 0.33 | 0.431 | +0.6 |
| + AdaBN, device-calibrated | 8.11 ± 0.07 | 0.412 | +0.6 |
| **in-domain ceiling** (3 seeds, trained on mBRSET DR-0) | **4.91 ± 0.37** | **0.820 ± 0.020** | +0.4 |
| ceiling, seed ensemble | 5.03 | 0.819 | |

Zero-shot, the transferred clock is worse than predicting mBRSET's mean age
(≈ 9 y): an offset that varies by seed (SD 3.8 y) plus predictions
over-dispersed by device noise (calibration scale d = 0.40, i.e. raw
predictions spread 1.6× wider than true age). AdaBN removes the offset and its
seed variance — the same BatchNorm-statistics mechanism as in the DR work — but
r does not move, and calibration buys one more year. Two effects were then
separated:

* **Range.** mBRSET's ages cluster at 50 – 79; even a perfect in-domain clock
  scores about r = 0.8 there, not 0.92.
* **Information.** The ceiling reaches MAE 4.9 y and r = 0.82 on held-out phone
  patients (MAE by bin: 40s 5.6, 50s 4.5, 60s 4.2, 70s 4.7). The phone images
  carry the age signal; the transferred model fails to read it. **Zero-shot
  transfer of an age clock is not viable; mixed-domain training is the lever**
  (BRSET healthy + mBRSET DR-0 in training, mBRSET's held-out patients for
  scoring — the pending v3 sweep).

The reverse direction is the mirror failure for a different reason: the
mBRSET-trained clock scores r 0.57 on BRSET with 36 y MAE in the under-30 bin,
because it never saw anyone that young. BRSET must stay in the mix for the
tails.

## 5. The retinal age gap against disease

### 5.1 BRSET (3,266 patients with gradable images)

Healthy reference (held-out): mean corrected gap +0.16 y, SD 5.9, n = 1,934.
Adjusted difference in corrected gap, exposed minus reference:

| exposure | Δ (y) [95 % CI] | Cohen d | OR per +5 y gap |
|---|---|---|---|
| diabetes vs non-diabetic | **+1.87 [+1.44, +2.30]** | 0.27 | 1.31 [1.23, 1.39] |
| insulin use (within diabetics) | +1.76 [+0.85, +2.67] | — | 1.33 [1.15, 1.53] |
| any DR (within diabetics) | +3.50 [+2.82, +4.18] | 0.76 | 1.72 [1.53, 1.93] |
| referable DR (within diabetics) | **+4.03 [+3.33, +4.73]** | 0.86 | 1.89 [1.66, 2.14] |
| macular edema (within diabetics) | **+5.07 [+4.18, +5.96]** | 0.97 | 2.10 [1.81, 2.45] |
| AMD | +1.53 [+0.06, +3.01], q = 0.10 | 0.10 | — |
| drusen | +0.49 [−0.04, +1.01], n.s. | 0.07 | 1.07 |
| increased cup–disc, hypertensive retinopathy, occlusion, haemorrhage, myopia, scar, nevus | null | | |

Prevalence by quintile of the corrected gap (Q1 youngest-looking → Q5 oldest):
diabetes 36 % → 52 %, any DR 21 % → 61 %, **referable DR 13 % → 56 %**, edema
4 % → 37 %. The gap rises **+1.2 y per DR grade** (grade 0 +0.7, 1 +0.6,
2 +4.9, 3 +8.3, 4 +5.5 — treated proliferative disease reads younger than
severe non-proliferative) and **+0.76 y per decade of diabetes**. Insulin flips
sign under adjustment (unadjusted −0.89) because insulin users are younger;
quote only adjusted values. The non-diabetic ophthalmic flags show nothing
beyond a borderline AMD: the clock responds to diabetic vascular disease
specifically. Among disease-free patients an ungradable image reads
**+1.48 y [+0.92, +2.05]** older, so the table above uses gradable images only.

### 5.2 mBRSET, two clocks on the same held-out phone patients

| exposure | transferred clock, device-calibrated (r 0.43; n = 1,282) | in-domain ceiling (r 0.82; n = 904) |
|---|---|---|
| insulin use | **+1.21 [+0.74, +1.69]** | **+1.31 [+0.53, +2.08]** |
| any DR | +1.67 [+1.26, +2.09] | +0.43, n.s. |
| referable DR | +2.31 [+1.87, +2.76] | +0.97 [+0.26, +1.69] |
| macular edema | +2.79 [+2.21, +3.36] | −0.09, null |
| diabetes duration, per 10 y | — | **+0.83 [+0.46, +1.20]** |
| systemic hypertension | +0.48 [+0.03, +0.93], q 0.10 | +0.69 [−0.08, +1.45], q 0.17 |
| neuropathy (n = 43) | +0.33, null | **+2.10 [+0.53, +3.68], q 0.04** |
| nephropathy (n = 32) | +0.26, null | +1.57 [−0.24, +3.39], q 0.17 |
| smoking (n = 53) | +0.44, null | +1.36 [−0.07, +2.79], q 0.16 |
| myocardial infarction, vascular disease, diabetic foot, obesity, alcohol | null | null (obesity −1.2, q 0.16) |

With the transferred clock, referable-DR prevalence runs 11 % → 46 % and edema
5 % → 29 % across gap quintiles; the DR-grade trend is +0.03 (grade 0) → +3.7
(grade 3). With the in-domain clock the trend is +0.42 y per grade and the
duration effect +0.83 y per decade, matching BRSET's +0.76. An ungradable image
reads +2.13 y older on the phone.

### 5.3 Reading the two clocks together

* **The DR effect is mostly lesion detection.** A clock that never saw the
  phone domain is pushed by anything unfamiliar, lesions included, and reports
  it as age: +2.3 y for referable DR, +2.8 for edema. The clock that reads
  phone images well gives +1.0 and zero. The honest number for DR-associated
  retinal ageing is about a year, with +2.3 as the upper bound.
* **What survives both clocks is credible.** Insulin use (+1.2 / +1.3 y),
  hypertension (+0.5 / +0.7, borderline both times) and diabetes duration
  (+0.8 y per decade, the same on BRSET) agree across two models that share
  nothing but the patients.
* **Neuropathy and nephropathy appear only with the better clock**, and only
  just: neuropathy is the one systemic row that clears FDR, on 43 patients.
  Both are the microvascular complications, so the direction is the expected
  one; at this n they are hypotheses.
* **The systemic nulls are bounded, not absent.** With 45 – 100 exposed
  patients the intervals exclude effects above roughly one year. UK Biobank
  cardiovascular associations with the retinal age gap are a few percent per
  year and needed tens of thousands of participants; 1,282 patients cannot see
  them.

## 6. What is claimable

* **Robust:** a 2.8 M-parameter mobile clock at MAE 5.1 y / r 0.92 on the
  tabletop camera and, trained in-domain, MAE 4.9 y / r 0.82 on smartphone
  images. The phone images carry the signal.
* **Robust:** on both cameras the corrected gap rises with diabetic eye disease
  in dose–response fashion (grade trend, duration trend, quintile gradients),
  with effect sizes up to d ≈ 1 on the tabletop camera.
* **Robust, interpret carefully:** the size of the DR effect depends on the
  clock; most of it is lesion detection, the biological part is about a year.
* **Replicated across two clocks on the phone:** insulin use, hypertension
  (borderline), diabetes duration.
* **Exploratory:** neuropathy +2.1 y (q 0.04, n = 43); nephropathy and smoking
  point the same way without clearing FDR.
* **Do not claim:** zero-shot transfer of the clock (r 0.43 is information
  loss, not fixable by calibration); any systemic effect from a single clock;
  any effect under about a year without ruling out image quality.

## 7. Infrastructure and levers

The lab box's GPUs are 11.6 GiB (a 512 px ConvNeXt-S teacher at batch 32
overflowed). The trainer now probes one training step at start-up and halves the
batch with gradient accumulation until it fits, tolerates a bounded number of
mid-run OOMs from a neighbouring process, and records the batch used and the
peak memory in its results JSON. Levers for a sharper clock, all knobs of
`run_retinal_age.sh` and off by default: label-distribution head (`HEAD=ldl`),
test-time flips (`TTA=1`), smartphone-capture augmentation (`PHONE_AUG=1`),
age-balanced sampling (`AGE_BALANCE=1`), 512 px (`SIZE=512`), a large teacher
with regression distillation (`TEACHER=…`, `kd_seed<n>`), mixed-domain training
(`MIX=1`, conditions named `*_mix`, external numbers on mBRSET's held-out rows),
and a seed-ensemble line in the summary. The **v3 sweep** (MIX + LDL + TTA +
age balance + EMA at 512 px, then teacher + KD) is the pending experiment: its
`student_mix` rows on mBRSET's held-out patients are the phone clock of record,
and its association report is a third model on the same patients.

## 8. Open items

1. **v3 results:** mixed-domain MAE / r on held-out mBRSET (expected ≈ 5 y,
   r ≈ 0.8), and whether insulin, hypertension and neuropathy hold under a
   third clock.
2. **Pre-lesion signal:** add the "diabetes without DR vs non-diabetic" row to
   the BRSET table (grade-0 diabetics already sit at +0.67 y).
3. **Deployment:** `export_coreml.py` has no regression path; the clock shares
   the classifier's trunk, so its on-device latency should match the 0.7 ms
   ANE figure, but it has not been measured.
4. **Distillation and resolution:** the teacher / KD arms of v3 quantify what
   capacity and 512 px buy on BRSET.

## 9. Reproduction

```bash
# cohort report only (no images touched)
python train_retinal_age.py --root <BRSET> --external-test-root <mBRSET> --inspect

# 3 seeds + the in-domain mBRSET ceiling, then summary, pooled predictions and the association report
B=<BRSET> M=<mBRSET> CEILING=1 OUT=exp_retinal_age CK=ck_retinal_age bash run_retinal_age.sh 0 1 2

# the two association reports (transferred clock; in-domain ceiling)
python analyze_age_gap.py --predictions exp_retinal_age/predictions_pooled.csv --brset-csv <BRSET>/labels_brset.csv
python analyze_age_gap.py --predictions exp_retinal_age/predictions_pooled.csv --condition ceiling --out exp_retinal_age/associations_ceiling

# the pending v3 sweep: mixed-domain + levers, then teacher + distillation (one job per 12 GB GPU)
MIX=1 SIZE=512 HEAD=ldl TTA=1 AGE_BALANCE=1 EXTRA="--ema-decay 0.999" OUT=exp_retinal_age_v3 CK=ck_retinal_age_v3 bash run_retinal_age.sh 0 1 2
SIZE=512 HEAD=ldl TTA=1 AGE_BALANCE=1 EXTRA="--ema-decay 0.999" TEACHER=timm:convnext_small.fb_in22k_ft_in1k OUT=exp_retinal_age_v3 CK=ck_retinal_age_v3 bash run_retinal_age.sh 0 1 2
```

Data: BRSET and mBRSET (PhysioNet, credentialed), passed as `B=` / `M=`; on the
lab box they live under `/data/users4/nshaik3/Datasets/{BRSET,mBRSET}`.

References: Poplin et al., Nat. Biomed. Eng. 2018; Zhu et al., Br. J.
Ophthalmol. 2023 (retinal age gap and mortality); Beheshti et al., NeuroImage:
Clinical 2019 (bias correction of brain-age gaps); Gao et al., 2018 (label
distribution learning for age estimation).

# Oculomics — Retinal Age Report, 6 – 16 September 2026

Branch `disease/retinal-age`. The diabetic-retinopathy classification work that
preceded this track (BRSET → mBRSET transfer, distillation, AdaBN, the operating
point, Core ML deployment) is documented in `REPORT.md` on `main`; this document
covers only the retinal-age track, from the first design on 6 September to the
state of the experiments on 16 September.

**Headline.** A MobileNetV4-Medium clock (8.8 M parameters, 2.1 ms at 512 px on
the Apple Neural Engine, 17 MB at fp16) trained on BRSET's healthy cohort with mBRSET's
retinopathy-free patients mixed in predicts age with **MAE 4.72 ± 0.05 years
(4.46 per patient, both eyes), r = 0.93** on held-out BRSET patients and
**4.57 ± 0.19 (4.17 per patient), r = 0.85** on held-out smartphone patients —
below the 4.9-year in-domain ceiling measured a week earlier, and approaching
the published UK Biobank clocks (3.3 – 3.6 y) with a quarter of their data and a
phone-sized network. Mixing the phone domain into training solved transfer
(zero-shot 14 y → 4.6); a bigger model in the same latency class moved the floor
(Small → Medium −0.28 y on BRSET, −0.1 to −0.3 on the phone); more labelled-age
data from ODIR-5K moved nothing, and distillation from a 50 M-parameter teacher
bought 0.07 y. Under this clock the bias-corrected **retinal age gap** is +4.2 y
in BRSET diabetics (adjusted) and rises with DR grade, diabetes duration and
insulin use on both cameras; diabetics with **no visible retinopathy** still read
+2.1 y older than non-diabetics, an effect concentrated under age 70 and without
a duration gradient of its own (section 5.4), so it is an association with
diabetic status, not yet evidence of a progressive pre-lesion process; on the phone smoking (+1.9 y) now clears FDR alongside
insulin, referable DR and edema, and hypertension stays borderline. An
ungradable image is worth +1.1 y on the tabletop camera and +2.3 y on the phone,
so every association uses gradable images only and phone effects under about
two years are not separable from image quality.

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

**ODIR-5K as an auxiliary training set (built 12 September; tried in v4 and
found to add nothing — section 4.3).** The Kaggle
mirror of ODIR-5K (`andrewmvd/ocular-disease-recognition-odir5k`: 6,392 eyes of
3,358 patients, several tabletop cameras, mean age 57.9) carries an age and
per-eye diagnostic keywords. Under the same patient-level rule as BRSET — every
eye of the patient reads "normal fundus" and nothing else, adequate quality — it
contributes **2151 training-eligible images from 1149 patients** (of 6364
with a usable age; excluded: 4103 abnormal, 110 ungradable; verified on the
real files with `--inspect`). Its 70/10/20 patient split gives 1483 training
and 234 validation images, roughly 30 % more healthy training retinas on top
of BRSET's 5,200, and a 434-image / 233-patient test partition that is
scored as a third-camera sanity check. It joins training only (`AUX=1`): own
split, own bias correction, never the external set. Its healthy ages are 40–70
heavy like BRSET's (26 under-30 training images), so it does not fix the tails.

*Provenance and audit (13 September).* The data are the training set of the
ODIR-2019 challenge (Peking University's health-data and AI institutes with
Shanggong Medical Technology; fundus photos from several Chinese hospitals on
Canon, Zeiss and Kowa cameras; "annotations labeled by trained human readers
with quality control management"). The Kaggle mirror was posted in 2020 by
Larxel (Andrew Maranhão, senior data scientist at Hospital Israelita Albert
Einstein, São Paulo; Kaggle Datasets Grandmaster, 104 datasets), has 566
upvotes, 66 k downloads and 265 public notebooks, and adds no labels of its
own. The checks run here on the downloaded files: the mirror's `full_df.csv`
matches the original annotation sheet `data.xlsx` on every field for all 6,392
rows; the mirror drops 608 of the 7,000 original images (142 patients; mostly
lens-dust and normal eyes, the originals are still in the package); its 512 px
`preprocessed_images` are a field-of-view crop squashed to a square, i.e. the
same operation our loader applies to BRSET (correlation 0.97–0.999 against our
own crop); 16 records carry the placeholder age 1 (all female, mostly
pathological myopia) and are now read as unknown, so the usable age range is
14–91; two patients are filed twice under different IDs (both eyes
identical, same age and sex; one of the pairs is in the healthy cohort, so it
can straddle ODIR's own train/test split — the only leakage, and only into
ODIR's own test number); a third-party audit that reports 42 % "patient
leakage" refers to naive image-level splitting, which this trainer never does;
and 139 patients flagged normal at the patient level have lens dust on an eye,
which our quality-required rule excludes. No licence is stated at the source
or on Kaggle ("license was not specified on source"), so the set is used for
research only and never redistributed, and the challenge organisers are the
citation, not the mirror.

## 3. Method

* **Healthy cohort** (`--healthy nodm`): no diabetes, DR grade 0 on *every*
  image of the patient (one eye with retinopathy disqualifies the fellow eye),
  adequate quality. `dr0` keeps diabetics without retinopathy (used for mBRSET,
  which has no non-diabetics); `normal` keeps only patients whose every eye is
  read as "normal fundus" (ODIR-5K's keywords); `--exclude-pathology` also
  drops BRSET's other ophthalmic flags.
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
`analyze_age_gap.py`, `public_fundus.py` (the ODIR-5K adapter),
`export_coreml.py --model age`; tests `tests/test_retinal_age.py`,
`tests/test_age_gap.py`, `tests/test_odir_age.py`, `tests/test_export_age.py`
(suite: 106 tests, all synthetic and network-free).

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

The MAEs above are per image. A phone exam captures both eyes, so the number
the deployment delivers is the **patient-level MAE**, both eyes averaged before
the error is taken. The trainer has always computed it (`patient_mae` in every
results JSON) and the summariser now tables it beside the image-level MAE for
every set and in the seed-ensemble lines; section 4.3 quotes it for the v3 – v5
sweeps, and it is the number to headline.

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

### 4.3 What moved the floor: the v3 – v5 sweeps (10 – 16 September)

Three sweeps at 512 px with the label-distribution head, flip TTA, age-balanced
sampling and EMA; 3 seeds each; image-level MAE with the patient-level number
in brackets. "Phone DR-0" is mBRSET's held-out retinopathy-free patients, the
in-domain accuracy number; "phone calibrated" is every held-out phone patient
after device calibration, diseased retinas included.

| sweep | condition | params | BRSET healthy test | phone DR-0 held-out | phone calibrated |
|---|---|---|---|---|---|
| v1 | Small, 384 px | 2.8 M | 5.07 ± 0.09 | zero-shot 14.0; ceiling 4.91 | 8.16 |
| v3 | Small + levers | 2.8 M | 4.86 ± 0.15 | — | 8.21 |
| v3 | Small + levers, mixed (`student_mix`) | 2.8 M | 4.96 ± 0.08 | 4.88 ± 0.18 | 5.74 |
| v3 | ConvNeXt-S teacher | 49.7 M | 4.49 ± 0.12 | — | 7.52 |
| v4 | Small, mixed + ODIR | 2.8 M | 5.00 ± 0.10 (4.79) | 4.85 ± 0.22 (4.33) | 5.77 (5.24) |
| v4 | Small, mixed + ODIR, distilled | 2.8 M | 4.93 ± 0.09 (4.70) | 4.67 ± 0.20 (4.20) | 5.69 (5.20) |
| v4 | Medium, mixed + ODIR | 8.8 M | 4.73 ± 0.05 (4.46) | 4.52 ± 0.08 (4.09) | 5.53 (5.08) |
| v4 | ConvNeXt-S teacher, mixed + ODIR | 49.7 M | 4.56 ± 0.08 (4.32) | 4.30 ± 0.10 (3.94) | 5.26 (4.88) |
| **v5** | **Medium, mixed (`medium_mix`) — clock of record** | 8.8 M | **4.72 ± 0.05 (4.46)** | **4.57 ± 0.19 (4.17)** | 5.61 (5.15) |
| v5 | Small, mixed (paired control) | 2.8 M | 5.00 ± 0.00 (4.76) | 4.72 ± 0.12 (4.19) | 5.72 (5.23) |

What each lever bought:

* **The v3 recipe** (512 px, LDL, TTA, age balance, EMA): 5.07 → 4.86 on
  BRSET — tenths of a year, as expected.
* **Mixed-domain training** took the phone from 14 y zero-shot to the 4.9-y
  ceiling in one step, at a cost of 0.1 y on BRSET. Zero-shot transfer is
  closed as a question.
* **ODIR-5K as an auxiliary set bought nothing.** With and without it the Small
  and Medium clocks agree to within seed noise on both cameras, and every model
  reads ODIR's own healthy test badly (MAE 7.2 – 7.4, r ≈ 0.5), so its age
  labels or images are a weak source. It is dropped from the recipe, which also
  removes the one dataset with no stated licence.
* **Capacity is what moved the floor.** MobileNetV4-Medium beats Small by
  0.28 y on BRSET in both sweeps that contain the pair (paired over seeds,
  significant both times) and by 0.11 – 0.28 y on the phone (significant in one
  of two; the phone healthy test holds 150 patients). At 2.1 ms on the Neural
  Engine at its 512 px input (section 7) it is still a phone model, and it is the clock of record.
  The seed ensemble of the three Medium runs adds about 0.05 y.
* **Distillation** into the Small model bought 0.07 y — significant on BRSET,
  not worth the teacher's cost. The ConvNeXt-S teacher itself is a further
  0.2 y better (4.56 / 4.30) at 50 M parameters.
* **Patient-level MAE** (both eyes averaged) runs 0.25 – 0.45 y below the
  image-level number on every set: **4.46 y on the tabletop camera and 4.17 y
  on the phone** are the deployment numbers.

The phone "calibrated" column stays near 5.6 y because it includes every
diseased patient, whose retinas read older by construction (referable minus
DR-0: +3.0 y, section 5.4); it is not clock error.

### 4.4 What the clock looks at (Grad-CAM on the v5 Medium clock)

`explain_retinal_age.py` on `medium_mix_seed0`, 21 BRSET images (three typical
ones per age bin, |gap| mostly under 2 y), positive Grad-CAM ("evidence for
older") on the deepest trunk activation, a 16 × 16 map. Averaged over the 21
images 39 % of the map's mass falls in the central quarter of the image (25 %
would be uniform), but that average hides a shift with age:

| age of the retina | images | mass in the central quarter | where the map fires |
|---|---|---|---|
| under 30 | 3 | 0.47 | fovea, disc rim and along the major arcades |
| 35 – 40 | 4 | **0.77** | one tight spot on the fovea / papillomacular bundle |
| 45 – 69 | 8 | **0.18** | disc margin and peripapillary zone, mid-periphery and the inferior-temporal edge where the choroid shows through |
| 70 and over | 6 | 0.37 | mixed: papillomacular region, the disc, peripapillary atrophy; on the two images with macular or disc lesions, the lesions |

Read descriptively (21 images, a coarse map, one seed): the clock reads a young
adult's age off the macula — the foveal reflex and sheen that fade through the
thirties — and a middle-aged or older one off the disc margin and the periphery,
where tessellation, choroidal visibility and peripapillary change accumulate.
That is the anatomy a clinician would name, and it matches the published clocks,
which find diffuse vascular and peripapillary features rather than one landmark.
Two reassuring details: on the image with a bright light-leak along its lower
edge the map ignores the artefact and fires on the disc, and the one large miss
in the set (age 81 read as 88) is an eye with marked peripapillary atrophy, on
which the map sits. The grid is `exp_retinal_age_v5/explain/explain_grid.png`.

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

### 5.4 Under the clock of record (v5 Medium, mixed-domain; both cameras)

The Medium clock is sharper (gap SD 5.9 y on BRSET, 4.8 on the phone) and the
disease signal grows with it. Adjusted differences in corrected gap, exposed
minus reference, gradable images only; the grade-0 rows are mean corrected gaps
against a healthy reference at ≈ 0:

| exposure | BRSET (n = 3,266 patients) | mBRSET phone (n = 904) |
|---|---|---|
| diabetes vs non-diabetic | **+4.19 [+3.74, +4.64]**, d 0.65 | every patient diabetic |
| diabetics with DR grade 0, mean gap | **+2.91 [+2.49, +3.34]** | +0.60 [+0.17, +1.03] |
| insulin use (within diabetics) | **+3.19 [+2.18, +4.20]** | **+2.24 [+1.49, +2.98]** |
| any DR | +4.32 [+3.60, +5.05] | +1.68 [+1.03, +2.33] |
| referable DR | **+4.67 [+3.92, +5.43]** | **+2.55 [+1.87, +3.22]** |
| macular edema | **+4.90 [+3.93, +5.88]** | **+2.35 [+1.49, +3.20]** |
| DR-grade trend | +1.48 y per grade | +1.07 y per grade |
| diabetes duration | +1.35 y per decade | +1.08 y per decade |
| systemic hypertension | — | +0.77 [+0.03, +1.51], q 0.09 |
| smoking (n = 53) | — | **+1.88 [+0.49, +3.26], q 0.02** |
| neuropathy (n = 43) / nephropathy (n = 32) | — | +1.49, q 0.10 / +1.61, q 0.12 |
| AMD (n = 70) | +1.99 [+0.38, +3.60], q 0.035, unstable logistic | — |
| drusen, cup–disc, hypertensive retinopathy, occlusion, haemorrhage, myopia, scar, nevus | null | — |
| infarction, vascular disease, diabetic foot, obesity, alcohol | — | null |
| ungradable image, disease-free patients | +1.10 [+0.54, +1.66] | +2.31 [+1.39, +3.23] |

Prevalence across gap quintiles (youngest- to oldest-looking retinas): on
BRSET diabetes 23 % → 64 %, referable DR 13 % → 47 %, edema 4 % → 30 %, insulin
32 % → 86 %; on the phone referable DR 16 % → 51 %, edema 7 % → 27 %, insulin
16 % → 38 %. The phone DR-grade curve is +0.6 (grade 0), 0.0 (1), +2.2 (2),
+3.1 (3), +6.1 (4).

What changed against the first clocks (5.1 – 5.3):

* **Diabetes without retinopathy: present, but not yet a pre-lesion biomarker.**
  The overall diabetes effect more than doubled against the v1 clock (+1.9 →
  +4.2), and grade-0 diabetics have a mean gap of +2.9 y (was +0.7). The
  dedicated check (`analyze_age_gap.py`, "diabetes before retinopathy": worst eye
  grade 0, no edema, diabetics vs non-diabetics, n = 890 vs 1,929, both never
  trained on) puts the **adjusted difference at +2.05 y [+1.58, +2.52], d 0.32**.
  Three things keep it from being a stronger claim. It is strongly
  age-dependent: +4.4 y under 50, +1.1 at 50 – 59, +1.8 at 60 – 69 and a
  non-significant +0.5 at 70 and over. It has **no duration gradient of its
  own** (+0.30 y per decade [−0.28, +0.88]; tertiles +1.8 → +2.1 → +2.2), so the
  +1.35 y per decade seen in all diabetics runs through retinopathy; insulin use
  inside the group is +1.0 y, not significant. And the camera check is
  uninformative because 96 % of these patients were photographed on the Canon.
  What argues for it: the training bias points the other way (the clock learned
  mBRSET diabetics at their true ages, so it discounts diabetic appearance).
  Open explanations besides subclinical microvascular change: lesions below the
  graders' threshold, which a clock this sensitive to lesions would pick up, and
  an older non-diabetic reference that carries other eye disease (tested by the
  "no other ophthalmic flag" rows and the diabetes × age term the script now
  prints). On the phone the grade-0 effect is +0.6 y because there the grade-0
  group *is* the reference population.
* **The DR effect is larger than "about a year" again** (+4.7 BRSET, +2.6
  phone). The lesion-detection reading in 5.3 was based on the transferred
  clock; with a clock that reads both cameras well the phone number sits
  between the two earlier estimates. The honest statement is that referable DR
  adds 2.5 – 4.7 y of apparent age and the split between lesion detection and
  biology cannot be made with these labels.
* **Insulin, referable DR, edema and the duration trend replicate** across all
  three phone clocks; **smoking** (+1.9 y, q 0.02) clears FDR for the first
  time, neuropathy and nephropathy point the same way at q ≈ 0.1, and
  hypertension stays borderline (+0.8 y, q 0.09).
* **Image quality bounds the phone claims.** An ungradable image is worth
  +2.3 y there, the size of the disease effects. Every table uses gradable
  images only, but residual quality variation among gradable images cannot be
  excluded for any phone effect under about two years.

## 6. What is claimable

* **Robust:** an 8.8 M-parameter mobile clock (2.1 ms on the Neural Engine) at
  MAE 4.7 y image-level / 4.5 y per patient, r 0.93, on the tabletop camera and
  4.6 / 4.2 y, r 0.85, on held-out smartphone patients, trained once for both
  cameras. The 2.8 M-parameter version trails it by a quarter of a year.
* **Robust:** zero-shot transfer of an age clock between cameras fails (r 0.43)
  and mixed-domain training fixes it completely; more data from a third public
  set does not help, more capacity does.
* **Robust:** on both cameras the corrected gap rises with diabetic eye disease
  in dose–response fashion — grade trend, duration trend, quintile gradients —
  with d 0.6 – 0.75 on the tabletop camera.
* **Present, interpret carefully:** diabetics with no visible retinopathy read
  +2.1 y older on the tabletop camera (d 0.32), concentrated under age 70 and
  with no duration gradient inside the lesion-free group; an association with
  diabetic status, not a demonstrated pre-lesion process.
* **Robust, interpret carefully:** the size of the DR effect depends on the
  clock (+2.6 to +4.7 y for referable DR); lesion detection and biology cannot
  be separated with these labels.
* **Replicated across three phone clocks:** insulin use, referable DR, edema,
  diabetes duration; hypertension borderline each time.
* **Exploratory:** smoking +1.9 y (q 0.02, n = 53); neuropathy and nephropathy
  +1.5 y at q ≈ 0.1; AMD +2.0 y on 70 BRSET patients.
* **Do not claim:** any phone effect under about two years without ruling out
  image quality (an ungradable image is worth +2.3 y there); any systemic
  effect from a single clock; that ODIR-5K or distillation improve the clock
  (both measured, neither did).

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

Three more levers were built on 12 September, each verified on real files or
real hardware:

* **More labelled-age data — `AUX=1`.** ODIR-5K's normal-fundus patients
  (section 2) join training as an auxiliary set (`--aux-train-root`,
  repeatable: own split, own bias correction, its val in the checkpoint-
  selection pool, its test partition reported under `aux`, never external).
  `O=kaggle:andrewmvd/ocular-disease-recognition-odir5k` fetches the mirror with
  kagglehub on the lab box; conditions get `_odir` appended; the ceiling arm
  stays ODIR-free.
* **Capacity inside the phone budget — `MEDIUM=1`, judged on Core ML latency.**
  `export_coreml.py` now exports a `train_retinal_age.py` checkpoint (kind
  `age`: the head is folded into the graph so the output is years, and
  `--verify-images` compares Core ML with PyTorch in years) and, without a
  checkpoint, the bare architecture for a latency probe. Measured on this Mac's
  M2 (fp16, 10 warm-up + 60 timed runs; an optimistic floor for an iPhone,
  valid for the relative cost):

| clock trunk | params | 384 px ANE | 512 px ANE | 384 px CPU |
|---|---|---|---|---|
| MobileNetV3-Small (GCG trunk) | 1.09 M | 0.57 ms | – | 2.4 ms |
| MobileNetV4-Small (student) | 2.82 M | 0.67 ms | 0.95 ms | 2.6 ms |
| MobileNetV4-Medium (`MEDIUM=1`) | 8.76 M | 1.43 ms | 1.96 ms | 8.3 ms |

  The LDL head (100 bins) adds nothing measurable (Small at 384 px: 0.69 ms;
  its expectation is exported as multiply-and-sum because Core ML cannot type a
  matmul against a 1-D bin vector). Every graph stays on the ANE (`ALL` matches
  `CPU_AND_NE`). The Medium student
  costs about twice the Small one and 512 px about 1.4×; all of it is far inside
  the budget, so the arm is affordable and only its MAE decides.

  **The trained clock of record, exported (17 September).** `medium_mix_seed0`
  from v5 at 512 px, fp16, checked on 32 real fundus photographs: Core ML and
  PyTorch agree to within **0.07 y** when both see the same bytes (0.00002 y at
  fp32), so the conversion is faithful; latency is **2.1 ms on the Neural
  Engine** (2.3 ms with the framework choosing, 12 ms CPU-only) in a 17 MB
  package. The fp32 export falls off the ANE (24 ms), so fp16 is the artefact to
  ship. One caveat belongs to the app, not the export: handing Core ML a uint8
  image instead of the float tensor the trainer saw moves a prediction by
  0.1 y on average and 0.65 y in the worst of the 32 cases, identical at fp16
  and fp32. The verifier therefore gates the age clock on the same-bytes error
  and reports the input-rounding gap separately.
* **The v4 sweep** combines them with the v3 levers (section 9): mixed-domain +
  ODIR-5K + LDL + TTA + age balance + EMA at 512 px, the Medium student, then
  the ConvNeXt-S teacher with distillation. Run 13 – 14 September, followed by
  v5 (Medium, mixed-domain, no ODIR) on 15 – 16 September; results in section
  4.3: the levers bought tenths, the Medium arm a quarter of a year, ODIR
  nothing.

## 8. Open items

1. **Poster numbers:** section 4.3's v5 row and section 5.4 are the results of
   record; sections 4.1 – 4.2 and 5.1 – 5.3 are the history that motivated them.
2. **On-device latency:** the trained clock passes the Core ML fidelity check
   and runs in 2.1 ms on an M2's Neural Engine (section 7), which is an
   optimistic floor; the quotable phone number needs an Xcode Core ML
   performance report on an iPhone.
3. **The grade-0 diabetes signal** (+2.05 y adjusted; section 5.4) is checked for
   camera, age band, duration and insulin. Still open: whether the fade with age
   survives restricting both groups to patients with no other ophthalmic flag
   (rows now printed by the script), and whether sub-threshold lesions explain
   it — the repository's lesion segmenter run on grade-0 diabetic vs
   non-diabetic images would test that directly.
4. **Sample size on the phone:** the healthy test partition is 150 patients; a
   5-fold patient-grouped cross-fit on mBRSET would tighten the phone MAE and
   the Medium-vs-Small contrast without new data.
5. **ODIR-5K** stays in the code as an adapter and a negative result; do not put
   it back in the recipe.

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

# v4: + ODIR-5K auxiliary set (fetched with kagglehub) + the Medium student; then the teacher + KD
# on the same mixed/auxiliary data (already-finished conditions are skipped, so the two lines resume each other)
O=kaggle:andrewmvd/ocular-disease-recognition-odir5k MIX=1 AUX=1 SIZE=512 HEAD=ldl TTA=1 AGE_BALANCE=1 MEDIUM=1 \
  EXTRA="--ema-decay 0.999" OUT=exp_retinal_age_v4 CK=ck_retinal_age_v4 bash run_retinal_age.sh 0 1 2
O=kaggle:andrewmvd/ocular-disease-recognition-odir5k MIX=1 AUX=1 SIZE=512 HEAD=ldl TTA=1 AGE_BALANCE=1 \
  EXTRA="--ema-decay 0.999" TEACHER=timm:convnext_small.fb_in22k_ft_in1k TEACHER_EXTRA="--batch-size 8" \
  OUT=exp_retinal_age_v4 CK=ck_retinal_age_v4 bash run_retinal_age.sh 0 1 2
# (or through the launcher: RA_ENV="MIX=1 AUX=1 O=kaggle:... SIZE=512 HEAD=ldl TTA=1 AGE_BALANCE=1 MEDIUM=1 OUT=... CK=..." bash launch_disease_runs.sh retinalage)

# v5, the clock of record: Medium + mixed-domain, no ODIR; then its association report
MIX=1 MEDIUM=1 SIZE=512 HEAD=ldl TTA=1 AGE_BALANCE=1 EXTRA="--ema-decay 0.999" OUT=exp_retinal_age_v5 CK=ck_retinal_age_v5 bash run_retinal_age.sh 0 1 2
python analyze_age_gap.py --predictions exp_retinal_age_v5/predictions_pooled.csv --condition medium_mix --brset-csv <BRSET>/labels_brset.csv --datasets brset mbrset --out exp_retinal_age_v5/associations

# ODIR-5K cohort report alone (no images touched)
python train_retinal_age.py --dataset odir --root <ODIR-5K> --healthy normal --inspect

# Core ML: export a trained clock with the real-image fidelity check; time an untrained architecture
python export_coreml.py --checkpoint ck_retinal_age_v4/student_mix_odir_seed0.pt --verify-images <mBRSET>/images
python export_coreml.py --model age --backbone timm:mobilenetv4_conv_medium.e500_r256_in1k --image-size 512
```

Data: BRSET and mBRSET (PhysioNet, credentialed), passed as `B=` / `M=`; on the
lab box they live under `/data/users4/nshaik3/Datasets/{BRSET,mBRSET}`.

References: Poplin et al., Nat. Biomed. Eng. 2018; Zhu et al., Br. J.
Ophthalmol. 2023 (retinal age gap and mortality); Beheshti et al., NeuroImage:
Clinical 2019 (bias correction of brain-age gaps); Gao et al., 2018 (label
distribution learning for age estimation).

"""
aireadi_dataset.py
==================
AI-READI (FAIRhub, type 2 diabetes flagship) as a measured-label dataset: blood
pressure, urine albumin / creatinine and the MoCA score, so a systemic model is
judged against numbers measured at the visit instead of a free-text history.

Three steps, each on the box, in order:

  inspect   prints what the download really holds before anything is trusted:
            the folder tree, the retinal-photography manifest's columns, the
            participants file's columns, and every clinical measurement /
            observation name that looks like blood pressure, albumin, creatinine
            or MoCA (with units and counts). The public docs leave several of
            these names unverified; ``targets`` refuses to guess.
  convert   every COLOUR fundus DICOM (3 samples per pixel, one frame) under the
            retinal-photography folder -> ``<out>/images/*.jpg`` (YBR -> RGB,
            longer side capped at ``--max-side``) and ``<out>/labels_aireadi.csv``:
            file, patient (DICOM Patient ID), camera (manufacturer + model),
            device_class (handheld = Optomed Aurora, else tabletop), laterality.
  targets   the measured targets per participant, merged into labels_aireadi.csv:
              sbp, dbp           mean of the visit's readings (mmHg)
              htn_measured       sbp >= 130 or dbp >= 80
              uacr, log_uacr     urine albumin / creatinine x 1000 (mg/g), and its log
              albuminuria        uacr >= 30 mg/g
              moca, moca_impaired  total score, and < 26
              age, diabetes_stage (0 none, 1 pre-diabetes / lifestyle, 2 oral / non-insulin
              injectable, 3 insulin) and its indicators dm_pre / dm_oral / dm_insulin (the
              covariates to use), diabetes (stage >= 2), clinical_site, recommended_split
            A '<x' (below-detection) albumin counts as x/2; albumin 0 is albuminuria 0 and
            log_uacr is floored at half the smallest positive ratio; htn_measured is 0 only
            when both readings are known and below their cut-offs.

What the public release does NOT have (docs.aireadi.org, v3 "controlled variables"):
sex, race / ethnicity and medications. So ``htn_measured`` is the visit's blood
pressure alone (a treated, controlled hypertensive reads 0), and the covariate
comparator is age + diabetes-stage indicators, not age + sex. ``convert`` reads
the ``cfp`` folder when there is one and skips mosaics / peripheral fields
(``--exclude-field``); it prints the fields per camera so one field per device can
be chosen for AutoMorph and the clock.

    python aireadi_dataset.py inspect --root <FAIRhub download>
    python aireadi_dataset.py convert --root <download> --out <AIREADI>
    python aireadi_dataset.py targets --root <download> --out <AIREADI>

Then ``<AIREADI>`` works as ``--dataset aireadi --root <AIREADI>`` in
extract_retfound.py, probe_systemic.py, train_mbrset.py (binary flags) and
score_external.py. Data use is limited to type 2 diabetes research under the
AI-READI licence; keep the converted images on the box.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# measurement_source_value strings: SBP / DBP verified on docs.aireadi.org (v3
# physical assessment page); the others are patterns, checked by ``inspect`` and
# pinned with --*-source when they match more than one name.
DEFAULT_SOURCES = {
    "sbp": r"^Systolic \(mmHg\)$",
    "dbp": r"^Diastolic \(mmHg\)$",
    "albumin": r"(?i)urine.*albumin|albumin.*urine",
    "creatinine": r"(?i)urine.*creatinine|creatinine.*urine",
    "moca": r"(?i)^(moca.*total.*|total score|moca_total.*)$",
}
INSPECT_PATTERNS = r"(?i)systolic|diastolic|albumin|creatinine|moca|total score|blood pressure"
MOCA_TOTAL_CONCEPT = 37174522           # "Total Score" row of the docs' MoCA page
# per-participant means outside these ranges mean a wrong name or unit, not a patient
PLAUSIBLE = {"sbp": (60, 260), "dbp": (30, 160), "moca": (0, 30)}


# --------------------------------------------------------------------------- #
# Finding files in the download
# --------------------------------------------------------------------------- #
def find_files(root: str, names: Tuple[str, ...], max_depth: int = 6) -> List[str]:
    out = []
    base = root.rstrip(os.sep).count(os.sep)
    for d, dirs, files in os.walk(root):
        if d.count(os.sep) - base >= max_depth:
            dirs[:] = []
        out += [os.path.join(d, f) for f in files if f.lower() in names]
    return sorted(out)


def photo_dir(root: str) -> str:
    for d, dirs, _ in os.walk(root):
        for x in dirs:
            if x.lower() == "retinal_photography":
                return os.path.join(d, x)
    raise SystemExit(f"[fatal] no retinal_photography folder under {root}; run `inspect`")


def _read_table(path: str):
    import pandas as pd
    return pd.read_csv(path, sep="\t" if path.lower().endswith(".tsv") else ",", low_memory=False)


# --------------------------------------------------------------------------- #
# inspect
# --------------------------------------------------------------------------- #
def inspect(root: str) -> int:
    print(f"== folder tree (3 levels) of {root}")
    base = root.rstrip(os.sep).count(os.sep)
    for d, dirs, files in os.walk(root):
        depth = d.count(os.sep) - base
        if depth >= 3:
            dirs[:] = []
            continue
        dirs.sort()
        print(f"{'  ' * depth}{os.path.basename(d) or d}/  ({len(files)} files)")
    for path in find_files(root, ("manifest.tsv", "manifest.csv")):
        t = _read_table(path)
        print(f"\n== {path}: {len(t)} rows\n   columns: {list(t.columns)}")
        for c in t.columns:
            if re.search(r"(?i)manufact|model|device|modality|imaging|laterality|anatomic|protocol", c):
                print(f"   {c}: {dict(Counter(t[c].astype(str)).most_common(12))}")
    for path in find_files(root, ("participants.tsv", "participants.csv")):
        t = _read_table(path)
        print(f"\n== {path}: {len(t)} rows\n   columns: {list(t.columns)}")
        for c in t.columns:
            if t[c].nunique() <= 12:
                print(f"   {c}: {dict(Counter(t[c].astype(str)).most_common(12))}")
    for fname in ("measurement.csv", "observation.csv"):
        for path in find_files(root, (fname,)):
            t = _read_table(path)
            src = next((c for c in t.columns if c.endswith("source_value") and fname[:-4] in c), None)
            print(f"\n== {path}: {len(t)} rows; name column {src!r}\n   columns: {list(t.columns)}")
            if src is None:
                continue
            hit = t[t[src].astype(str).str.contains(INSPECT_PATTERNS, regex=True)]
            unit = next((c for c in t.columns if c.startswith("unit_source")), None)
            for name, g in hit.groupby(src):
                units = dict(Counter(g[unit].astype(str)).most_common(3)) if unit else "-"
                val = next((c for c in t.columns if c == "value_as_number"), None)
                rng = (f"values {g[val].min():.4g}..{g[val].max():.4g}" if val and g[val].notna().any()
                       else "no value_as_number")
                print(f"   {name!r:<55} n={len(g):6d} persons={g['person_id'].nunique():5d}  units {units}  {rng}")
    print("\nNext: `convert`, then `targets` (pin any ambiguous name with --albumin-source etc.)")
    return 0


# --------------------------------------------------------------------------- #
# convert
# --------------------------------------------------------------------------- #
def _rgb(ds) -> np.ndarray:
    """Pixel data as uint8 RGB. pydicom >= 3 converts YBR to RGB itself; 2.x does not."""
    import pydicom
    arr = ds.pixel_array
    pi = str(getattr(ds, "PhotometricInterpretation", "RGB"))
    if int(pydicom.__version__.split(".")[0]) < 3 and pi.startswith("YBR"):
        from pydicom.pixel_data_handlers.util import convert_color_space
        arr = convert_color_space(arr, pi, "RGB")
    if arr.dtype != np.uint8:
        arr = np.clip(arr.astype(np.float64) / max(float(arr.max()), 1.0) * 255.0, 0, 255).astype(np.uint8)
    return arr


def device_class(manufacturer: str, model: str) -> str:
    t = f"{manufacturer} {model}".lower()
    return "handheld" if ("aurora" in t or "optomed" in t) else "tabletop"


DEFAULT_EXCLUDE_FIELD = r"(?i)mosaic|nasal|temporal|periph"


def _field(ds) -> str:
    """What the photograph shows, from the DICOM tags that carry it: protocol, series
    description and anatomic region (the manifest's names are not verified)."""
    parts = [str(getattr(ds, k, "")) for k in ("ProtocolName", "SeriesDescription")]
    try:
        parts.append(str(ds.AnatomicRegionSequence[0].CodeMeaning))
    except (AttributeError, IndexError):
        pass
    return " | ".join(p for p in parts if p and p != "None")


def convert_one(job) -> Optional[dict]:
    path, out_dir, max_side, fmt, quality, exclude = job
    import pydicom
    from PIL import Image
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception as e:                                       # not a DICOM, truncated
        return {"skip": f"unreadable: {type(e).__name__}"}
    if int(getattr(ds, "SamplesPerPixel", 1)) != 3:
        return {"skip": "not colour (IR / autofluorescence)"}
    if int(getattr(ds, "NumberOfFrames", 1) or 1) != 1:
        return {"skip": "multi-frame"}
    field = _field(ds)
    if exclude and re.search(exclude, field):
        return {"skip": "excluded field (--exclude-field)"}
    man, model = str(getattr(ds, "Manufacturer", "")), str(getattr(ds, "ManufacturerModelName", ""))
    uid = str(getattr(ds, "SOPInstanceUID", path))
    pid = str(getattr(ds, "PatientID", "")) or os.path.basename(os.path.dirname(path))
    lat = str(getattr(ds, "ImageLaterality", getattr(ds, "Laterality", ""))).strip().upper()
    fname = f"{re.sub(r'[^A-Za-z0-9_-]', '_', pid)}_{lat or 'X'}_{hashlib.sha1(uid.encode()).hexdigest()[:12]}.{fmt}"
    dst = os.path.join(out_dir, "images", fname)
    row = {"file": fname, "patient": pid, "camera": f"{man} {model}".strip(),
           "device_class": device_class(man, model), "laterality": lat, "field": field,
           "rows": int(getattr(ds, "Rows", 0)), "cols": int(getattr(ds, "Columns", 0)),
           "sop_instance_uid": uid, "source": path}
    if os.path.exists(dst):
        return row
    try:
        img = Image.fromarray(_rgb(pydicom.dcmread(path)))
    except Exception as e:
        return {"skip": f"pixel decode failed: {type(e).__name__}: {e}"}
    if max_side and max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.LANCZOS)
    tmp = dst + ".tmp"
    img.save(tmp, format="JPEG" if fmt == "jpg" else "PNG", **({"quality": quality} if fmt == "jpg" else {}))
    os.replace(tmp, dst)
    return row


def convert(root: str, out: str, max_side: int, fmt: str, quality: int, workers: int, limit: int,
            exclude: str = DEFAULT_EXCLUDE_FIELD) -> int:
    import pandas as pd
    pdir = photo_dir(root)
    # the colour-fundus modality folder when the download has one (ir/, faf/ hold other imaging)
    cfp = [os.path.join(d, x) for d, dirs, _ in os.walk(pdir) for x in dirs if x.lower() == "cfp"]
    roots = cfp or [pdir]
    files = sorted(os.path.join(d, f) for r in roots for d, _, fs in os.walk(r) for f in fs
                   if f.lower().endswith(".dcm"))
    print(f"[info] reading {'the cfp folder(s) ' + str(cfp) if cfp else pdir + ' (no cfp folder found)'}")
    if limit:
        files = files[:limit]
    print(f"[info] {len(files)} DICOM files under {pdir}")
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    jobs = [(f, out, max_side, fmt, quality, exclude) for f in files]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            res = list(pool.imap(convert_one, jobs, chunksize=16))
    else:
        res = [convert_one(j) for j in jobs]
    rows = [r for r in res if r and "skip" not in r]
    skips = Counter(r["skip"] for r in res if r and "skip" in r)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"[fatal] no colour fundus photograph converted; skipped: {dict(skips)}")
    dst = os.path.join(out, "labels_aireadi.csv")
    df["patient"] = df["patient"].astype(str)
    if os.path.exists(dst):                                      # keep targets already merged
        old = pd.read_csv(dst)
        old["patient"] = old["patient"].astype(str)              # read back as int: ids are text
        extra = [c for c in old.columns if c not in df.columns and c != "patient"]
        if extra:
            df = df.merge(old.drop_duplicates("patient")[["patient"] + extra], on="patient", how="left")
    df.to_csv(dst, index=False)
    print(f"[info] {len(df)} colour photographs from {df['patient'].nunique()} participants -> {dst}")
    print(f"       by camera: {dict(Counter(df['camera']))}")
    print(f"       by class:  {dict(Counter(df['device_class']))}")
    print(f"       skipped:   {dict(skips)}")
    print("       by camera and field (choose one field per device for AutoMorph / the clock):")
    for (cam, fld), n in df.groupby(["camera", "field"]).size().items():
        print(f"         {cam:<24} {fld!r:<60} {n}")
    return 0


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #
def pick_source(names, pattern: str, what: str, exact: Optional[str]) -> Optional[str]:
    names = sorted(set(str(n) for n in names))
    if exact is not None:
        if exact not in names:
            raise SystemExit(f"[fatal] --{what}-source {exact!r} is not a recorded name; "
                             f"close ones: {[n for n in names if re.search(INSPECT_PATTERNS, n)][:20]}")
        return exact
    hit = [n for n in names if re.search(pattern, n)]
    if len(hit) == 1:
        return hit[0]
    if not hit:
        return None
    raise SystemExit(f"[fatal] {len(hit)} names match {what} ({pattern}): {hit[:12]}. "
                     f"Pin one with --{what}-source \"<name>\" (see `inspect`).")


def norm_id(x) -> str:
    """Participant ids as one text form: '1001', 1001, 1001.0 and '01001' -> '1001'
    (DICOM Patient ID, OMOP person_id and participants.tsv may each store it differently)."""
    t = str(x).strip()
    try:
        f = float(t)
        if np.isfinite(f) and f == int(f):
            return str(int(f))
    except ValueError:
        pass
    return t


def _censored(text) -> float:
    """'<0.3' (below the assay's reporting limit) -> 0.15, half the limit; else NaN."""
    m = re.match(r"^\s*<\s*=?\s*([0-9]*\.?[0-9]+)", str(text))
    return float(m.group(1)) / 2.0 if m else np.nan


def per_person(table, src_col: str, name: str, censored_low: bool = False):
    """Per participant, the mean of ``value_as_number`` over the rows named ``name``;
    with ``censored_low``, a row whose number is missing but whose source text reads
    '<x' counts as x/2 (a below-detection urine albumin is a low value, not missing).
    Returns (series, rows, n_censored)."""
    import pandas as pd
    t = table[table[src_col].astype(str) == name]
    v = pd.to_numeric(t["value_as_number"], errors="coerce")
    n_cens = 0
    if censored_low:
        txt = next((c for c in ("value_source_value", "value_as_string") if c in t.columns), None)
        if txt is not None:
            cens = t[txt].map(_censored)
            fill = v.isna() & cens.notna()
            n_cens = int(fill.sum())
            v = v.where(~fill, cens)
    return v.groupby(t["person_id"].map(norm_id)).mean(), t, n_cens


def diabetes_stage(group) -> float:
    t = str(group).lower()
    if not t or t == "nan":
        return np.nan
    # "oral_medication_and_or_non_insulin_injectable..." contains "insulin": test it first
    if "oral" in t or "non_insulin" in t or "non-insulin" in t:
        return 2.0
    if "insulin" in t:
        return 3.0
    if "pre" in t or "lifestyle" in t:
        return 1.0
    if "healthy" in t or "no_diabetes" in t or "no diabetes" in t or t == "control":
        return 0.0
    return np.nan


def build_targets(root: str, out: str, sources: Dict[str, Optional[str]], allow_units: bool,
                  moca_concept: int = 37174522, moca_concept_check_off: bool = False) -> int:
    import pandas as pd
    dst = os.path.join(out, "labels_aireadi.csv")
    if not os.path.isfile(dst):
        raise SystemExit(f"[fatal] {dst} not found: run `convert` first")
    labels = pd.read_csv(dst)
    meas = [p for p in find_files(root, ("measurement.csv",))]
    obs = [p for p in find_files(root, ("observation.csv",))]
    if not meas:
        raise SystemExit(f"[fatal] no measurement.csv under {root}")
    M = _read_table(meas[0])
    O = _read_table(obs[0]) if obs else None
    msrc = "measurement_source_value"
    osrc = "observation_source_value"
    found, per = {}, {}
    for key in ("sbp", "dbp", "albumin", "creatinine"):
        name = pick_source(M[msrc].dropna().unique(), DEFAULT_SOURCES[key], key, sources.get(key))
        if name is None:
            print(f"[warn] no measurement name matches {key} ({DEFAULT_SOURCES[key]}); that target stays empty")
            continue
        found[key] = name
        per[key], rows, n_cens = per_person(M, msrc, name, censored_low=key in ("albumin", "creatinine"))
        if n_cens:
            print(f"[info] {key}: {n_cens} below-detection readings ('<x') counted as x/2")
        if key in ("albumin", "creatinine") and "unit_source_value" in rows.columns:
            units = set(rows["unit_source_value"].dropna().astype(str).str.lower().str.replace(" ", ""))
            if units and not units <= {"mg/dl"} and not allow_units:
                raise SystemExit(f"[fatal] {key} units are {units}, expected mg/dL (AI-READI 2.0.0 changelog); "
                                 f"the ratio below assumes both in the same unit. --allow-units to override.")
    tables = [(t, src) for t, src in ((M, msrc), (O, osrc)) if t is not None]
    pinned = sources.get("moca")
    if pinned is not None and not any(pinned in set(t[src].dropna().astype(str)) for t, src in tables):
        raise SystemExit(f"[fatal] --moca-source {pinned!r} is in neither measurement.csv nor observation.csv")
    for table, src in tables:
        if "moca" in found:
            break
        names = table[src].dropna().astype(str).unique()
        name = (pinned if pinned in set(names) else None) if pinned is not None else \
            pick_source(names, DEFAULT_SOURCES["moca"], "moca", None)
        if name is None:
            continue
        found["moca"] = name
        sub = table[table[src].astype(str) == name]
        if "moca" not in name.lower() and not moca_concept_check_off:
            # a generic "Total Score" may belong to any questionnaire: keep only the
            # MoCA total's concept (37174522 on docs.aireadi.org), or refuse
            ok = pd.Series(False, index=sub.index)
            for cc in (src.replace("_source_value", "_concept_id"), src.replace("_value", "_concept_id")):
                if cc in sub.columns:
                    ok |= pd.to_numeric(sub[cc], errors="coerce") == moca_concept
            if not ok.any():
                raise SystemExit(f"[fatal] the MoCA total matched the generic name {name!r} but no row carries "
                                 f"concept {moca_concept}; it may be another questionnaire's total. Check "
                                 f"`inspect`, then pass --moca-concept <id> (or --moca-no-concept-check).")
            sub = sub[ok]
        per["moca"], _, _ = per_person(sub, src, name)
    for k, (lo, hi) in PLAUSIBLE.items():
        if k in per and len(per[k].dropna()) and not per[k].dropna().between(lo, hi).all():
            bad = per[k][~per[k].between(lo, hi)].dropna()
            raise SystemExit(f"[fatal] {k} ({found.get(k)!r}): {len(bad)} participants outside {lo}-{hi}, e.g. "
                             f"{bad.head(3).round(2).to_dict()}: wrong name or unit; check `inspect`.")
    print(f"[info] measurement names used: {found}")
    P = pd.DataFrame(index=sorted(set().union(*[set(s.index) for s in per.values()]) if per else []))
    for k, s in per.items():
        P[k] = s
    measured = [c for c in ("sbp", "dbp", "albumin", "creatinine", "moca") if c in P.columns]
    if not measured or P[measured].isna().all().all():
        raise SystemExit("[fatal] no target could be built: every measurement column is empty")
    if {"albumin", "creatinine"} <= set(P.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            P["uacr"] = P["albumin"] / P["creatinine"] * 1000.0          # mg/dL over mg/dL -> mg/g
        P.loc[~np.isfinite(P["uacr"]) | (P["uacr"] < 0), "uacr"] = np.nan  # 0 albumin stays: normal
        P["albuminuria"] = np.where(P["uacr"].notna(), (P["uacr"] >= 30).astype(float), np.nan)
        pos = P["uacr"][P["uacr"] > 0]
        floor = float(pos.min()) / 2.0 if len(pos) else 1e-3
        P["log_uacr"] = np.log(P["uacr"].clip(lower=floor))
        n0 = int((P["uacr"] == 0).sum())
        if n0:
            print(f"[info] uacr: {n0} participants with albumin 0; log_uacr floored at {floor:.4g} mg/g")
    if {"sbp", "dbp"} <= set(P.columns):
        # 1 if either reading meets its cut-off; 0 only when both are known and below
        hi = (P["sbp"] >= 130) | (P["dbp"] >= 80)
        P["htn_measured"] = np.where(hi, 1.0, np.where(P["sbp"].notna() & P["dbp"].notna(), 0.0, np.nan))
    if "moca" in P.columns:
        P["moca_impaired"] = np.where(P["moca"].notna(), (P["moca"] < 26).astype(float), np.nan)
    # the guard is against participants with a MEASURED target, before participants.tsv
    # adds rows for everyone enrolled
    pids = set(labels["patient"].map(norm_id))
    overlap = len(pids & set(P.index)) / max(len(pids), 1)
    print(f"[info] {overlap:.0%} of photographed participants have a measured target")
    if overlap < 0.9:
        raise SystemExit(f"[fatal] only {overlap:.0%} of DICOM Patient IDs match a clinical person_id with a "
                         f"measurement; e.g. DICOM {sorted(pids)[:3]} vs clinical {list(P.index[:3])}. Check `inspect`.")
    parts = find_files(root, ("participants.tsv", "participants.csv"))
    if parts:
        T = _read_table(parts[0])
        idc = next((c for c in ("participant_id", "person_id") if c in T.columns), None)
        missing = [c for c in ("age", "study_group") if c not in T.columns]
        if idc is None or missing:
            print(f"[warn] {parts[0]}: no id column or no {missing}; columns are {list(T.columns)}. The "
                  f"covariates (age, diabetes stage) will be missing: a probe or vessel run with them will refuse.")
        if idc:
            T = T.set_index(T[idc].map(norm_id))
            if "age" in T.columns:
                P = P.reindex(P.index.union(T.index))
                P["age"] = pd.to_numeric(T["age"], errors="coerce").reindex(P.index)
            if "study_group" in T.columns:
                P["diabetes_stage"] = T["study_group"].map(diabetes_stage).reindex(P.index)
                P["diabetes"] = np.where(P["diabetes_stage"].notna(), (P["diabetes_stage"] >= 2).astype(float), np.nan)
                # indicators (none = reference): the stage as one linear term would under-adjust
                for k, v in (("dm_pre", 1), ("dm_oral", 2), ("dm_insulin", 3)):
                    P[k] = np.where(P["diabetes_stage"].notna(), (P["diabetes_stage"] == v).astype(float), np.nan)
            for c in ("clinical_site", "recommended_split"):
                if c in T.columns:
                    P[c] = T[c].reindex(P.index)
    else:
        print("[warn] no participants file: no age / diabetes stage covariates")
    P = P.drop(columns=[c for c in ("albumin", "creatinine") if c in P.columns])
    cols = [c for c in P.columns]
    labels = labels.drop(columns=[c for c in cols if c in labels.columns])
    labels["patient"] = labels["patient"].map(norm_id)
    P = P.rename_axis("patient").reset_index()
    P["patient"] = P["patient"].astype(str)
    labels = labels.merge(P, on="patient", how="left")
    labels.to_csv(dst, index=False)
    per_pat = labels.drop_duplicates("patient")
    print(f"[info] wrote targets into {dst}")
    for c in ("sbp", "htn_measured", "log_uacr", "albuminuria", "moca", "moca_impaired", "age", "diabetes_stage"):
        if c in per_pat.columns:
            v = per_pat[c]
            extra = f", positive {int(v.sum())}" if c in ("htn_measured", "albuminuria", "moca_impaired") else \
                f", mean {v.mean():.2f}"
            print(f"   {c:<16} {int(v.notna().sum()):5d} participants{extra}")
    have = [c for c in ("age", "dm_pre", "dm_oral", "dm_insulin") if c in per_pat.columns]
    print("   NB: no medication data in the public release -> htn_measured ignores treatment; no sex -> "
          f"covariates available: {' '.join(have) or 'NONE'}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="AI-READI: inspect, convert DICOM photographs, build measured targets.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inspect")
    a.add_argument("--root", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--root", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--max-side", type=int, default=2048, help="cap on the longer side (0 = full size)")
    c.add_argument("--format", default="jpg", choices=["jpg", "png"])
    c.add_argument("--quality", type=int, default=95)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--exclude-field", default=DEFAULT_EXCLUDE_FIELD,
                   help="regex on protocol / series / anatomic region; matching photographs are skipped "
                        "(default drops mosaics and peripheral fields; '' keeps everything)")
    t = sub.add_parser("targets")
    t.add_argument("--root", required=True)
    t.add_argument("--out", required=True)
    for k in DEFAULT_SOURCES:
        t.add_argument(f"--{k}-source", default=None, help=f"exact recorded name for {k} (see inspect)")
    t.add_argument("--allow-units", action="store_true")
    t.add_argument("--moca-concept", type=int, default=MOCA_TOTAL_CONCEPT,
                   help="concept id of the MoCA total, used when its name is generic (e.g. 'Total Score')")
    t.add_argument("--moca-no-concept-check", action="store_true",
                   help="trust a generic MoCA name without the concept filter (only after reading `inspect`)")
    args = p.parse_args(argv)
    if args.cmd == "inspect":
        return inspect(args.root)
    if args.cmd == "convert":
        return convert(args.root, args.out, args.max_side, args.format, args.quality, args.workers, args.limit,
                       args.exclude_field)
    return build_targets(args.root, args.out, {k: getattr(args, f"{k}_source") for k in DEFAULT_SOURCES},
                         args.allow_units, args.moca_concept, args.moca_no_concept_check)


if __name__ == "__main__":
    raise SystemExit(main())

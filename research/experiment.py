"""Rare-event evaluation. See PROTOCOL.md before interpreting results."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, precision_recall_curve, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
VARIANTS = [f"{family}_{method}" for family in ("lr", "rf")
            for method in ("none", "balanced", "smote")]
PREVALENCES = [0.0005, 0.005, 0.01]


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare_frame(df):
    """Validate schema and remove exact duplicate observations before splitting."""
    cols = ["Time"] + FEATURES + ["Class"]
    if set(df.columns) != set(cols):
        raise ValueError("Expected Time, V1-V28, Amount, Class only")
    if not np.isfinite(df[cols].to_numpy(dtype=float)).all():
        raise ValueError("Data must be finite and complete")
    if not set(df.Class.unique()) == {0, 1} or (df.Amount < 0).any():
        raise ValueError("Expected both binary classes and nonnegative amounts")
    return df.drop_duplicates(cols).sort_values("Time", kind="stable")


def split_frame(df):
    """Chronological 55/15/15/15, with timestamp ties kept together."""
    times = df.Time.to_numpy()
    boundaries = [times[int(len(df) * q)] for q in (0.55, 0.70, 0.85)]
    masks = [times < boundaries[0],
             (times >= boundaries[0]) & (times < boundaries[1]),
             (times >= boundaries[1]) & (times < boundaries[2]),
             times >= boundaries[2]]
    parts = dict(zip(("train", "calibration", "validation", "test"),
                     [df.loc[mask].copy() for mask in masks]))
    for name, frame in parts.items():
        if frame.Class.value_counts().get(1, 0) < 6 or frame.Class.nunique() != 2:
            raise ValueError(f"Insufficient examples in {name}")
    return parts


def features(df):
    """Stateless transform; no Time or full-hour transaction counts."""
    x = df[FEATURES].copy()
    x["Amount"] = np.log1p(x.Amount)
    return x


def make_model(variant, seed, trees=100, jobs=4):
    family, method = variant.split("_")
    weight = "balanced" if method == "balanced" else None
    if family == "lr":
        model = LogisticRegression(C=1.0, max_iter=2000, tol=1e-6,
                                   class_weight=weight, random_state=seed)
    elif family == "rf":
        model = RandomForestClassifier(n_estimators=trees, max_depth=12,
                                       min_samples_leaf=2, max_features="sqrt",
                                       class_weight=weight, random_state=seed,
                                       n_jobs=jobs)
    else:
        raise ValueError(variant)
    if method not in ("none", "balanced", "smote"):
        raise ValueError(variant)
    sampler = SMOTE(sampling_strategy=0.1, random_state=seed) if method == "smote" else "passthrough"
    return Pipeline([("scale", StandardScaler()), ("sample", sampler), ("model", model)])


def logits(p):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrate(p, y):
    """Fit a regularized sigmoid to held-out scores at natural prevalence."""
    return LogisticRegression(C=1000, max_iter=2000, tol=1e-8).fit(logits(p), y)


def choose_threshold(y, p):
    """Max validation F1; break ties using the highest threshold."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    score = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    best = np.flatnonzero(np.isclose(score, score.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best])


def prevalence_weights(y, target):
    y = np.asarray(y)
    if not 0 < target < 1 or np.unique(y).size != 2:
        raise ValueError("Prevalence must be between zero and one, with both classes present")
    source = y.mean()
    return np.where(y == 1, target / source, (1 - target) / (1 - source))


def reliability(y, p, weights=None):
    y, p = np.asarray(y), np.asarray(p)
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    bins = np.minimum((p * 10).astype(int), 9)
    records = []
    for i in range(10):
        mask = bins == i
        if mask.any():
            records.append({"bin": i, "n": int(mask.sum()), "mass": float(w[mask].sum() / w.sum()),
                            "mean_probability": float(np.average(p[mask], weights=w[mask])),
                            "fraction_fraud": float(np.average(y[mask], weights=w[mask]))})
    return records


def metrics(y, p, threshold, weights=None):
    y, p = np.asarray(y), np.asarray(p)
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    positive = p >= threshold
    tp = w[(y == 1) & positive].sum()
    fp = w[(y == 0) & positive].sum()
    fn = w[(y == 1) & ~positive].sum()
    tn = w[(y == 0) & ~positive].sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    bins = reliability(y, p, w)
    return {"average_precision": float(average_precision_score(y, p, sample_weight=w)),
            "roc_auc": float(roc_auc_score(y, p, sample_weight=w)),
            "precision": float(precision), "recall": float(recall),
            "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
            "brier": float(brier_score_loss(y, p, sample_weight=w)),
            "log_loss": float(log_loss(y, p, sample_weight=w, labels=[0, 1])),
            "ece_10": float(sum(b["mass"] * abs(b["mean_probability"] - b["fraction_fraud"]) for b in bins)),
            "prevalence": float(np.average(y, weights=w)), "threshold": float(threshold),
            "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn)}


def bootstrap(y, p, threshold, repetitions, seed):
    """Class-stratified bootstrap, conditional on fixed model and prevalence."""
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(y == label) for label in (0, 1)]
    keys = ["average_precision", "precision", "recall", "brier"]
    records = []
    for _ in range(repetitions):
        ids = np.concatenate([rng.choice(c, len(c), replace=True) for c in classes])
        records.append(metrics(y[ids], p[ids], threshold))
    return {k: np.quantile([r[k] for r in records], [0.025, 0.975]).tolist() for k in keys}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(data, output, seeds=(42, 43, 44), trees=100, jobs=4, bootstraps=200):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    raw = pd.read_csv(data)
    df = prepare_frame(raw)
    parts = split_frame(df)
    xy = {k: (features(v), v.Class.to_numpy()) for k, v in parts.items()}
    split_info = {k: {"rows": len(v), "frauds": int(v.Class.sum()),
                      "time_min": float(v.Time.min()), "time_max": float(v.Time.max()),
                      "row_ids_sha256": hashlib.sha256(v.index.to_numpy(dtype="<i8").tobytes()).hexdigest()}
                  for k, v in parts.items()}
    manifest = {"dataset_sha256": digest(data), "raw_rows": len(raw), "raw_frauds": int(raw.Class.sum()),
                "duplicate_rows_removed": len(raw) - len(df), "splits": split_info,
                "seeds": list(seeds), "trees": trees, "jobs": jobs, "bootstraps": bootstraps,
                "source_sha256": digest(__file__), "protocol_sha256": digest(Path(__file__).with_name("PROTOCOL.md")),
                "python": platform.python_version(), "platform": platform.platform(),
                "versions": {p: importlib.metadata.version(p) for p in
                             ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "scipy", "joblib")}}
    try:
        manifest["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        manifest["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        manifest["git_head"] = None
    write_json(output / "manifest.json", manifest)
    # All candidate and threshold decisions finish before any test prediction.
    fitted, validation = {}, []
    for seed in seeds:
        for variant in VARIANTS:
            started = time.perf_counter()
            model = make_model(variant, seed, trees, jobs)
            model.fit(*xy["train"])
            calibrator = calibrate(model.predict_proba(xy["calibration"][0])[:, 1], xy["calibration"][1])
            raw_p = model.predict_proba(xy["validation"][0])[:, 1]
            thresholds = {}
            for mode, p in (("raw", raw_p), ("sigmoid", calibrator.predict_proba(logits(raw_p))[:, 1])):
                thresholds[mode] = choose_threshold(xy["validation"][1], p)
                validation.append({"seed": seed, "variant": variant, "mode": mode,
                                   **metrics(xy["validation"][1], p, thresholds[mode])})
            fitted[seed, variant] = (model, calibrator, thresholds)
            print(f"Fit {seed} {variant}: {time.perf_counter() - started:.1f}s", flush=True)
    val = pd.DataFrame(validation)
    # Primary family/imbalance choice uses raw ranking AP, averaged across fixed seeds.
    ap = val[val["mode"] == "raw"].groupby("variant").average_precision.mean()
    selected = sorted(VARIANTS, key=lambda v: (-ap[v], v))[0]
    # Probability mapping selected by mean validation log loss for that variant.
    losses = val[val.variant == selected].groupby("mode").log_loss.mean()
    mode = sorted(("raw", "sigmoid"), key=lambda m: (losses[m], m))[0]
    selection = {"variant": selected, "mode": mode, "representative_seed": seeds[0],
                 "variant_criterion": "mean validation raw average precision",
                 "mode_criterion": "mean validation log loss for selected variant"}
    write_json(output / "validation.json", validation)
    write_json(output / "selection.json", selection)
    print(f"Frozen selection: {selection}", flush=True)
    records, curves, bins, primary = [], [], [], None
    x, y = xy["test"]
    for (seed, variant), (model, calibrator, thresholds) in fitted.items():
        raw_p = model.predict_proba(x)[:, 1]
        for mapping, p in (("raw", raw_p), ("sigmoid", calibrator.predict_proba(logits(raw_p))[:, 1])):
            for scenario, target in [("chronological", None)] + [(f"prior_{p:g}", p) for p in PREVALENCES]:
                w = None if target is None else prevalence_weights(y, target)
                records.append({"seed": seed, "variant": variant, "mode": mapping, "scenario": scenario,
                                **metrics(y, p, thresholds[mapping], w)})
            if seed == seeds[0]:
                bins.extend({"variant": variant, "mode": mapping, **b} for b in reliability(y, p))
                pr, rc, _ = precision_recall_curve(y, p)
                # Retain representative curve points; full scores remain local only.
                ids = np.unique(np.linspace(0, len(pr) - 1, min(2000, len(pr))).astype(int))
                curves.append({"variant": variant, "mode": mapping, "precision": pr[ids].tolist(), "recall": rc[ids].tolist()})
            if seed == seeds[0] and variant == selected and mapping == mode:
                primary = (p, thresholds[mapping])
    write_json(output / "results.json", records)
    write_json(output / "reliability.json", bins)
    write_json(output / "curves.json", curves)
    p, threshold = primary
    if bootstraps:
        write_json(output / "bootstrap.json", {"method": "class-stratified percentile, fixed model and class counts",
                                               "seed": 20261105, "repetitions": bootstraps,
                                               "intervals_95": bootstrap(y, p, threshold, bootstraps, 20261105)})
    print("Experiment complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--bootstraps", type=int, default=200)
    args = parser.parse_args()
    if args.trees < 1 or args.jobs < 1 or args.bootstraps < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Require positive trees/jobs, nonnegative bootstraps and distinct seeds")
    with threadpool_limits(limits=args.jobs):
        run(args.data, args.output, args.seeds, args.trees, args.jobs, args.bootstraps)

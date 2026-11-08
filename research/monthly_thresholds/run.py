"""Run from the repository root with python -m research.monthly_thresholds.run."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.experiment import calibrate, logits, make_model

URL = "https://www.kaggle.com/api/v1/datasets/download/kartik2112/fraud-detection"
SHA256 = "4e32829b9ba5a6b17af707c513c15204011044df0e8971e431cedca3d8c0a8a1"
OUT = Path(__file__).resolve().parent / "results"
COLS = ["trans_date_trans_time", "trans_num", "is_fraud", "amt", "category",
        "lat", "long", "merch_lat", "merch_long"]


def read_data(path):
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != SHA256:
            raise ValueError("Unexpected archive checksum. Use the documented dataset release.")
    with ZipFile(path) as source:
        parts = [pd.read_csv(source.open(name), usecols=COLS, dtype={"trans_num": str})
                 for name in ("fraudTrain.csv", "fraudTest.csv")]
    df = pd.concat(parts, ignore_index=True)
    if df.isna().any().any() or not df.trans_num.is_unique or set(df.is_fraud) != {0, 1}:
        raise ValueError("Missing values, duplicate transaction IDs or invalid labels")
    df["time"] = pd.to_datetime(df.pop("trans_date_trans_time"), format="%Y-%m-%d %H:%M:%S")
    df = df.sort_values(["time", "trans_num"], ignore_index=True)
    df["month"] = df.time.dt.strftime("%Y-%m")
    counts = df.groupby("month").is_fraud.agg(transactions="size", frauds="sum")
    expected = pd.period_range("2019-01", "2020-12", freq="M").astype(str).tolist()
    test = counts.tail(6)
    if counts.index.tolist() != expected or (test.frauds < 100).any() or (test.transactions < 35000).any():
        raise ValueError("Calendar data gate failed")
    counts.to_csv(OUT / "data_audit.csv")
    return df


def features(df, categories):
    hour = df.time.dt.hour + df.time.dt.minute / 60 + df.time.dt.second / 3600
    weekday = df.time.dt.dayofweek
    lat1, lat2 = np.radians(df.lat), np.radians(df.merch_lat)
    delta = np.sin((lat2 - lat1) / 2)**2
    delta += np.cos(lat1) * np.cos(lat2) * np.sin(np.radians(df.merch_long - df.long) / 2)**2
    km = 12742 * np.arcsin(np.sqrt(np.clip(delta, 0, 1)))
    x = pd.DataFrame({"amount": np.log1p(df.amt), "km": km,
                      "hour_sin": np.sin(2*np.pi*hour/24), "hour_cos": np.cos(2*np.pi*hour/24),
                      "weekday_sin": np.sin(2*np.pi*weekday/7), "weekday_cos": np.cos(2*np.pi*weekday/7)})
    for cat in categories:
        x["cat_" + cat] = (df.category == cat).astype(float)
    return x.to_numpy(dtype=np.float32)


def cutoff(data):
    k = len(data) // 1000
    ranked = data.sort_values(["score", "trans_num"], ascending=[False, True])
    boundary = ranked.iloc[k - 1]
    return float(boundary.score), str(boundary.trans_num)


def decisions(data, boundary):
    score, key = boundary
    candidates = ((data.score > score) | ((data.score == score) & (data.trans_num <= key))).to_numpy()
    reviewed = np.zeros(len(data), dtype=bool)
    used = 0
    for pos in np.flatnonzero(candidates):
        if used < (pos + 1) // 1000:
            reviewed[pos] = True
            used += 1
    return candidates, reviewed


def intervals(daily):
    rng = np.random.default_rng(42)
    reps = 5000
    fields = ["transactions", "frauds", "fixed_caught", "monthly_caught", "fixed_reviews", "monthly_reviews"]
    sums = np.zeros((reps, len(fields)))
    for _, group in daily.groupby(daily.index.astype(str).str[:7]):
        values = group[fields].to_numpy()
        n = len(values)
        starts = rng.integers(n, size=(reps, (n + 6) // 7))
        idx = ((starts[:, :, None] + np.arange(7)) % n).reshape(reps, -1)[:, :n]
        sums += values[idx].sum(axis=1)
    tx, fraud, f, m, fr, mr = sums.T
    values = [100*f/fraud, 100*m/fraud, 100*(m-f)/fraud, 10000*fr/tx, 10000*mr/tx]
    names = ["fixed_recall_pct", "monthly_recall_pct", "delta_pp", "fixed_reviews_per_10k", "monthly_reviews_per_10k"]
    return {key: np.percentile(v, [2.5, 97.5]).tolist() for key, v in zip(names, values)}


def chart(results):
    months = [s[-2:] for s in results.month]
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.8), sharex=True, constrained_layout=True)
    for policy, color, marker in [("fixed", "#234b72", "o"), ("monthly", "#aa4927", "s")]:
        for ax, metric, denom, scale in zip(axes, ["caught", "reviews"], ["frauds", "transactions"], [100, 10000]):
            ax.plot(months, scale * results[policy+"_"+metric] / results[denom],
                    label=policy.title(), color=color, marker=marker, markersize=5,
                    linestyle="-" if policy == "fixed" else "--")
    axes[0].set_ylabel("Fraud caught (%)")
    axes[1].set_ylabel("Reviews per 10,000")
    axes[1].axhline(10, ls=":", lw=1, color="black", label="Budget")
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncols=2)
    axes[1].set_xlabel("Held-out month in 2020")
    fig.savefig(OUT/"monthly.png", dpi=200)
    plt.close(fig)


def run(path):
    OUT.mkdir(exist_ok=True)
    df = read_data(path)
    print("Data audit passed", flush=True)
    train = df[df.month < "2020-01"]
    calib = df[df.month.between("2020-01", "2020-05")]
    cats = sorted(train.category.unique())
    model = make_model("rf_smote", 42, trees=100, jobs=4)
    model.fit(features(train, cats), train.is_fraud)
    model.named_steps["model"].n_jobs = 1  # Stable summation order when scoring.
    raw = model.predict_proba(features(calib, cats))[:, 1]
    sigmoid = calibrate(raw, calib.is_fraud)
    def score_month(month):
        part = df.loc[df.month == month].copy()
        raw = model.predict_proba(features(part, cats))[:, 1]
        part["score"] = sigmoid.predict_proba(logits(raw))[:, 1]
        return part
    previous = fixed = cutoff(score_month("2020-06"))
    rows, days = [], []
    for month in sorted(df.loc[df.month > "2020-06", "month"].unique()):
        group = score_month(month)
        labels = group.is_fraud.to_numpy()
        n = len(labels)
        row = dict(month=month, transactions=n, frauds=int(labels.sum()), cap=n//1000)
        day = pd.DataFrame(dict(day=group.time.dt.normalize().to_numpy(), transactions=np.ones(n, dtype=int), frauds=labels))
        selected = {}
        for policy, threshold in [("fixed", fixed), ("monthly", previous)]:
            candidates, reviews = decisions(group, threshold)
            hit = labels * reviews
            assert np.all(np.cumsum(reviews) <= np.arange(1, n+1)//1000)
            selected[policy] = labels[reviews]
            row.update({policy+"_cutoff": threshold[0], policy+"_tie_id": threshold[1],
                        policy+"_alerts": int(candidates.sum()), policy+"_reviews": int(reviews.sum()),
                        policy+"_caught": int(hit.sum()), policy+"_over_budget": bool(candidates.sum() > n//1000)})
            day[policy+"_caught"] = hit
            day[policy+"_reviews"] = reviews
        equal = min(map(len, selected.values()))
        row["matched_reviews"] = equal
        for policy, selected_labels in selected.items():
            row[policy+"_matched_caught"] = int(selected_labels[:equal].sum())
        rows.append(row)
        days.append(day.groupby("day").sum())
        previous = cutoff(group)
    results = pd.DataFrame(rows)
    daily = pd.concat(days)
    results.to_csv(OUT/"monthly.csv", index=False)
    daily.to_csv(OUT/"daily.csv")
    (OUT/"intervals.json").write_text(json.dumps(intervals(daily), indent=2)+"\n", encoding="utf-8")
    chart(results)
    print(results.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a local checksum-verified Sparkov zip")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.archive is not None:
            run(args.archive)
        else:
            with TemporaryDirectory() as temp:
                path = Path(temp)/"sparkov.zip"
                urlretrieve(URL, path)
                run(path)


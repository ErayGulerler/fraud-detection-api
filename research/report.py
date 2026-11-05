"""Render aggregate experiment artifacts without refitting models."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load(folder, name):
    return json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def render(folder):
    manifest, choice = load(folder, "manifest"), load(folder, "selection")
    result = pd.DataFrame(load(folder, "results"))
    natural = result[result.scenario == "chronological"]
    selected = result[(result.variant == choice["variant"]) & (result["mode"] == choice["mode"])
                      & (result.seed == choice["representative_seed"])]
    main = selected[selected.scenario == "chronological"].iloc[0]
    low = selected[selected.scenario == "prior_0.0005"].iloc[0]
    selected_losses = natural[natural.variant == choice["variant"]].groupby("mode").log_loss.mean()
    top_test = natural[natural["mode"] == "raw"].groupby("variant").average_precision.mean().idxmax()
    intervals = load(folder, "bootstrap")["intervals_95"] if (folder / "bootstrap.json").exists() else {}
    summary = []
    for (variant, mode), rows in natural.groupby(["variant", "mode"]):
        summary.append([variant, mode] + [f"{rows[k].mean():.4f} ({rows[k].std(ddof=0):.4f})"
                       for k in ["average_precision", "precision", "recall", "brier", "log_loss"]])
    split_rows = [[k, v["rows"], v["frauds"], f"{100 * v['frauds'] / v['rows']:.4f}%",
                   f"{v['time_min']:g}-{v['time_max']:g}"] for k, v in manifest["splits"].items()]
    shifts = [[r.scenario, f"{100*r.prevalence:.4f}%", f"{r.average_precision:.4f}",
               f"{r.roc_auc:.4f}", f"{r.precision:.4f}", f"{r.recall:.4f}", f"{r.brier:.6f}",
               f"{r.log_loss:.6f}"] for r in selected.itertuples()]
    ci = [[k, f"{main[k]:.6f}", f"[{v[0]:.6f}, {v[1]:.6f}]"] for k, v in intervals.items()]
    report = f"""# Rare-event classifiers under distribution shift

## Findings

The validation-selected configuration was **{choice['variant']} / {choice['mode']}**.
For the predeclared representative seed {choice['representative_seed']}, the
chronological test average precision was **{main.average_precision:.4f}**,
precision **{main.precision:.4f}**, and recall **{main.recall:.4f}** at the
validation-selected threshold {main.threshold:.6f}.

At a controlled fraud prevalence of 0.05%, with identical class-conditional
observations and fixed decisions, precision was **{low.precision:.4f}** while
recall remained **{low.recall:.4f}**. This isolates sensitivity to the class prior.
It is not evidence of a change observed at a bank or a robust deployment.

## Design

The public ULB/Worldline dataset contained {manifest['raw_rows']:,} rows and
{manifest['raw_frauds']} frauds. Exact deduplication removed
{manifest['duplicate_rows_removed']:,} rows before a chronological 55/15/15/15
split, keeping timestamp ties together. No full-dataset fitted scaling or
full-hour transaction-count feature was used. V1-V28 plus log1p(Amount) were
scaled on the training partition only. The upstream PCA provenance is unknown.

{table(['Partition', 'Rows', 'Frauds', 'Prevalence', 'Time range (seconds)'], split_rows)}

Six fixed model/imbalance variants were trained with seeds 42,43,44. Calibration
used a separate partition at natural prevalence. Variant selection maximized
mean raw validation AP; probability mapping minimized validation log loss for
that variant. Thresholds maximized validation F1. All decisions were frozen
before test prediction. F1 does not encode real fraud-loss or investigation costs.
See [the complete protocol](PROTOCOL.md) and [implementation](experiment.py).

## All predeclared comparisons

Natural chronological holdout; mean (population standard deviation) over three
training seeds on the SAME split. These deviations are not confidence intervals.
AP means sklearn average precision, not trapezoidal area under the PR curve.

{table(['Variant', 'Probabilities', 'AP', 'Precision', 'Recall', 'Brier', 'Log loss'], summary)}

Do not interpret the best test-table cell as the selected model. Selection used
validation data only. Calibration is assessed through Brier/log loss and the
reliability diagram, while recognizing that the proper scores also reflect
discrimination. ECE and full confusion values are in the aggregate JSON.
The sigmoid mapping uses clipped log-odds. Clipping and floating-point
saturation can merge score ties, so AP can change even though the mapping is
monotone. In particular, the calibrated Logistic Regression AP change is not
evidence that calibration learned better feature discrimination.

For the selected variant, mean test log loss was {selected_losses['raw']:.4f}
raw and {selected_losses['sigmoid']:.4f} after calibration. The highest mean
raw test AP belonged to {top_test}; this is a descriptive comparison and does
not justify changing the frozen selection or claiming universal superiority.

## Prior-shift sensitivity

Selected configuration, representative seed. Exact class reweighting preserves
empirical P(X|Y), with no model, calibration or threshold adaptation. These are
four views of the same test observations, not four independent datasets.

{table(['Scenario', 'Fraud rate', 'AP', 'ROC-AUC', 'Precision', 'Recall', 'Brier', 'Log loss'], shifts)}

## Conditional uncertainty

{table(['Metric', 'Point estimate', '95% bootstrap interval'], ci)}

The 200 class-stratified bootstrap intervals hold the fitted model, chosen
threshold and class counts fixed. They exclude model-selection uncertainty,
training-data uncertainty, changing prevalence and dependence between
transactions. They should not be read as evidence of significant differences
between candidates. The representative seed was fixed before evaluation.

## Figures

![Ranking and probability diagnostics](results/{folder.name}/diagnostics.png)

Left: subsampled PR curves for all six variants at seed42; dashed line is
test prevalence. Right: selected variant, raw and sigmoid reliability, with
10 equal-width probability bins. Counts per bin are in reliability.json;
sparsely occupied high-probability bins are especially uncertain.

## Limitations and next experiments

- Only two days of anonymized transactions and relatively few positive cases.
  No card IDs, raw feature meanings, external validation or deployment cost data.
- A chronological holdout measures later-period performance but does not by
  itself establish or identify distribution shift. Prior reweighting covers
  only label shift, not changing P(X|Y), label errors, or adversarial behavior.
- Small calibration/validation fraud counts can produce unstable mappings and
  thresholds. Low aggregate Brier or ECE alone is insufficient for rare events.
- The legacy Flask path has separate preprocessing and serving issues and is
  not validated by this study. No trained research model is deployed here.
- Future work should use additional temporal windows and an independent dataset,
  a business-cost threshold, and calibration uncertainty. Such changes need a
  fresh evaluation protocol and holdout; do not tune to the test results above.

## Reproduction

```bash
pip install -r research/requirements-lock.txt
python -m pytest tests/ -q
python -m research.experiment --data data/creditcard.csv --output research/results/new-run
python -m research.report --run research/results/new-run
```

Python {manifest['python']}; dependency versions, seeds, source/protocol hashes,
Git state, timestamp ranges and row-ID hashes are in manifest.json. Input SHA256:
`{manifest['dataset_sha256']}`.
The dataset and any personal academic/application records are excluded from Git.

Dataset attribution: ULB Machine Learning Group and Worldline;
[original Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
Download mirror verified in the
[TensorFlow tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data).
"""
    destination = folder.parents[1] / "REPORT.md"
    destination.write_text(report, encoding="utf-8")
    curves = load(folder, "curves")
    bins = pd.DataFrame(load(folder, "reliability"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for curve in curves:
        if curve["mode"] == "raw":
            axes[0].plot(curve["recall"], curve["precision"], label=curve["variant"], linewidth=1.3)
    axes[0].axhline(main.prevalence, color="gray", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Chronological holdout: seed 42", xlim=(0, 1), ylim=(0, 1.03))
    axes[0].legend(fontsize=8, loc="lower left")
    for mode in ("raw", "sigmoid"):
        b = bins[(bins.variant == choice["variant"]) & (bins["mode"] == mode)]
        axes[1].plot(b.mean_probability, b.fraction_fraud, "o-", label=mode, linewidth=1.3)
    axes[1].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[1].set(xlabel="Mean predicted probability", ylabel="Observed fraud fraction",
                title=f"Reliability: {choice['variant']}, seed 42", xlim=(0, 1), ylim=(0, 1.03))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.savefig(folder / "diagnostics.png", dpi=180)
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    render(parser.parse_args().run)

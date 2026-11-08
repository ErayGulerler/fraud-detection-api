# Monthly fraud-alert thresholds

This technical report extends the [original chronological fraud study](../REPORT.md) with a publicly downloadable two-year synthetic transaction stream. The [experiment plan](PROTOCOL.md) was committed before model scoring. The two policies share one frozen scorer and a strict review allowance. Candidate alerts, actual reviews and fraud caught are separate outcomes.

[Four-page technical report](technical-report.pdf) · [One-page plan](experiment-plan.pdf) · [All 24 monthly counts](results/data_audit.csv) · [Held-out results and alert flags](results/monthly.csv) · [Interval estimates](results/intervals.json)

In July through December 2020, monthly updating caught 330 of 2,012 frauds (16.40%) in 362 reviews. The fixed cutoff caught 302 (15.01%) in 331 reviews. The paired difference was +1.39 percentage points (95% seven-day block-bootstrap interval +0.74 to +2.08). At 330 reviews per policy, matched within each month, both caught 301 frauds. September was the only month in which candidate alerts exceeded the allowance for either method. Both stayed within the review limit throughout.

## Reproduce

From the root of this research branch in a Python 3.12 environment, install the versions in `research/requirements-lock.txt` using `python -m pip install -r research/requirements-lock.txt`. Then run:

```sh
python -m research.monthly_thresholds.run
```

The command downloads the [Sparkov dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection), checks the archive SHA-256, verifies the calendar gate, fits once and regenerates all CSVs, intervals and the figure. For an offline run add `--archive PATH_TO_ZIP`. The input archive contains both original CSVs and is not redistributed. Research results may differ in the final digits across platforms. The live API is independent of this offline experiment.

The primary comparison uses a monthly allowance accrued in chronological order. A candidate can expire before capacity becomes available. The two methods can therefore consume different numbers of reviews. The published equal-volume sensitivity analysis truncates each ordered set of completed reviews to the smaller observed monthly count. The daily totals permit independently reconstructing the reported intervals.

# Fixed versus monthly fraud-alert thresholds
Experiment plan | Eray Gulerler | 8 November 2026

**Question.** With a frozen transaction scorer, does resetting its cutoff each month catch more fraud within 10 reviews per 10,000 incoming transactions?

**Data gate.** Pool both public Sparkov CSVs by timestamp. Require unique transaction IDs, all 24 consecutive calendar months, at least 100 frauds and 35 review slots in each of the final six months. Counts were inspected before modeling. The six months have 258 to 415 fraud cases and 69 to 139 slots each. This supports a descriptive comparison, with limited precision for small differences.

**Chronology.** Fit the baseline study's seed-42 random forest and SMOTE recipe on January to December 2019. Fit its sigmoid calibrator on January to May 2020. June 2020 sets the initial cutoff. Reserve July to December 2020 for sequential evaluation. Fit neither model nor calibrator again. Use log amount, category (training vocabulary), cyclic hour and weekday, and distance between cardholder and merchant coordinates. Exclude identifiers as predictors. Break timestamp and score ties by transaction ID.

**Thresholds.** Use the cutoff pair (score, ID) at rank floor(reference transactions / 1,000), highest score first and smallest ID first. Fixed always uses June. Monthly uses the immediately preceding month, including its scores but no labels. Both policies therefore coincide in July. Never rank within the current month to select alerts.

**Budget and estimand.** Reset the allowance each month. At transaction i, review a threshold crossing only if reviews so far are below floor(i / 1,000). Missed candidates expire. Report all candidate alerts, excess demand and capped reviews. Pool caught frauds divided by all held-out frauds for primary recall. Report rates per 10,000, monthly recall, and pooled monthly-minus-fixed difference in percentage points. Also compare the first min(review counts) reviews per method in each month as a retrospective equal-volume sensitivity analysis.

**Uncertainty.** Sample seven-day circular blocks of realized daily totals within each held-out month, truncating to its length, with 5,000 paired replicates and seed 42. Use percentile 95% intervals for pooled recall, its difference and review rates. Condition on fitted policies and this calendar. Treat small or uncertain differences as inconclusive. Publish counts, thresholds, figure and code alongside the baseline.

import numpy as np
import pandas as pd

from research.monthly_thresholds.run import cutoff, decisions


def test_equal_scores_have_a_repeatable_reference_quota():
    reference = pd.DataFrame({"score": np.ones(2500), "trans_num": [f"{i:04d}" for i in range(2500)]})
    threshold = cutoff(reference.sample(frac=1, random_state=3))
    assert threshold == (1.0, "0001")
    candidates, _ = decisions(reference, threshold)
    assert candidates.sum() == 2


def test_capacity_is_earned_online_and_missed_candidates_expire():
    score = np.zeros(2200)
    score[[10, 999, 1000, 1999, 2000]] = 1.0
    month = pd.DataFrame({"score": score, "trans_num": ["x"] * len(score)})
    candidates, reviewed = decisions(month, (0.5, "x"))
    assert candidates.sum() == 5
    assert np.flatnonzero(reviewed).tolist() == [999, 1999]
    _, prefix = decisions(month.iloc[:1600], (0.5, "x"))
    np.testing.assert_array_equal(prefix, reviewed[:1600])


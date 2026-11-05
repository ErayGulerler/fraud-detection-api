"""Evaluation safeguards and an end-to-end synthetic smoke test."""
import json

import numpy as np
import pandas as pd
import pytest

from research.experiment import (FEATURES, bootstrap, choose_threshold, features,
                                 make_model, metrics, prepare_frame,
                                 prevalence_weights, run, split_frame)


@pytest.fixture
def frame():
    rng = np.random.default_rng(19)
    df = pd.DataFrame(rng.normal(size=(2000, 28)), columns=FEATURES[:-1])
    df["Amount"] = rng.exponential(30, len(df))
    df["Time"] = np.arange(len(df)) // 2
    df["Class"] = (np.arange(len(df)) % 20 == 0).astype(int)
    df["V1"] += 2 * df.Class
    return df


def test_temporal_splits_disjoint_exhaustive_and_ties_together(frame):
    df = prepare_frame(pd.concat([frame, frame.iloc[:3]]))
    parts = split_frame(df)
    assert len(df) == len(frame)
    ids = [set(p.index) for p in parts.values()]
    assert sum(map(len, ids)) == len(set.union(*ids)) == len(df)
    for a, b in zip(list(parts.values())[:-1], list(parts.values())[1:]):
        assert a.Time.max() < b.Time.min()


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_reject_nonfinite_data(frame, bad):
    frame.loc[0, "V1"] = bad
    with pytest.raises(ValueError, match="finite"):
        prepare_frame(frame)


def test_reject_single_class(frame):
    frame["Class"] = 0
    with pytest.raises(ValueError, match="binary"):
        prepare_frame(frame)


def test_scaler_and_sampler_see_training_only(frame):
    parts = split_frame(prepare_frame(frame))
    train = parts["train"]
    model = make_model("lr_smote", 42)
    model.fit(features(train), train.Class)
    scaler = model.named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, features(train).mean())
    assert scaler.n_samples_seen_ == len(train)
    before = scaler.mean_.copy()
    poisoned = parts["test"].copy()
    poisoned[FEATURES[:-1]] = 1e9
    assert model.predict_proba(features(poisoned)).shape == (len(poisoned), 2)
    np.testing.assert_array_equal(scaler.mean_, before)
    assert model.named_steps["sample"].sampling_strategy_[1] > 0


def test_threshold_uses_best_validation_f1_and_consistent_comparison():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.4, 0.6, 0.8])
    threshold = choose_threshold(y, p)
    assert threshold == 0.4
    assert metrics(y, p, threshold)["f1"] == pytest.approx(0.8)


def test_prior_shift_matches_bayes_precision_and_preserves_roc_recall():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.7, 0.8, 0.2, 0.1, 0.05])
    natural = metrics(y, p, 0.75)
    weights = prevalence_weights(y, 0.01)
    shifted = metrics(y, p, 0.75, weights)
    assert np.average(y, weights=weights) == pytest.approx(0.01)
    assert shifted["recall"] == pytest.approx(natural["recall"])
    assert shifted["roc_auc"] == pytest.approx(natural["roc_auc"])
    assert shifted["precision"] == pytest.approx(0.01 * 0.5 / (0.01 * 0.5 + 0.99 * 0.25))
    assert shifted["precision"] < natural["precision"]


def test_bootstrap_reproducible_and_bounded():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.2, 0.8, 0.4])
    result = bootstrap(y, p, 0.5, 10, 42)
    assert result == bootstrap(y, p, 0.5, 10, 42)
    assert all(0 <= a <= b <= 1 for a, b in result.values())


def test_end_to_end_freezes_selection_before_test_and_preserves_input(frame, tmp_path, monkeypatch):
    import research.experiment as experiment
    source = tmp_path / "fixture.csv"
    frame.to_csv(source, index=False)
    output = tmp_path / "run"
    original_predict = experiment.Pipeline.predict_proba
    parts = split_frame(prepare_frame(frame))
    test_ids = set(parts["test"].index)

    def checked_predict(self, x, **kwargs):
        if set(x.index) == test_ids:
            assert (output / "selection.json").exists()
            assert (output / "validation.json").exists()
        return original_predict(self, x, **kwargs)

    monkeypatch.setattr(experiment.Pipeline, "predict_proba", checked_predict)
    run(source, output, seeds=[42], trees=3, jobs=1, bootstraps=3)
    result = json.loads((output / "results.json").read_text())
    assert len(result) == 6 * 2 * 4
    assert all(0 <= r["average_precision"] <= 1 for r in result)
    assert (output / "bootstrap.json").exists()
    with pytest.raises(FileExistsError):
        run(source, output, seeds=[42], trees=3, jobs=1, bootstraps=0)

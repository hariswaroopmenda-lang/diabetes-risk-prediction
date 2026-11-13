"""Run with python -m unittest test_project -v after training."""
import unittest
import joblib
import numpy as np
import pandas as pd
from diabetes_prediction import (ROOT, FEATURES, capacity_flags, choose_threshold,
                                 prepare_features, rank_patients)


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT / "data" / "diabetes.csv")
        cls.saved = joblib.load(ROOT / "outputs" / "model.joblib")

    def test_capacity_never_exceeds_thirty_percent(self):
        for size in range(101):
            flags = capacity_flags(np.linspace(0, 1, size))
            self.assertLessEqual(int(flags.sum()), size * 0.30)
            self.assertEqual(int(flags.sum()), int(size * 0.30))

    def test_ties_keep_input_order(self):
        np.testing.assert_array_equal(capacity_flags(np.ones(10)), [1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

    def test_missing_measurements_keep_zero_pregnancies(self):
        sample = self.raw.iloc[:1].copy()
        sample.loc[:, ["Pregnancies", "Glucose", "BMI"]] = 0
        X = prepare_features(sample)
        self.assertEqual(X.iloc[0].Pregnancies, 0)
        self.assertTrue(pd.isna(X.iloc[0].Glucose))
        self.assertTrue(pd.isna(X.iloc[0].BMI))
        self.assertEqual(sample.iloc[0].Glucose, 0)

    def test_bad_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            prepare_features(self.raw.drop(columns="BMI"))
        sample = self.raw.iloc[:1].copy()
        sample.loc[:, "Glucose"] = -1
        with self.assertRaises(ValueError):
            prepare_features(sample)

    def test_threshold_tuning_respects_validation_capacity(self):
        y = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1])
        scores = np.array([.1, .1, .1, .1, .2, .3, .4, .7, .8, .9])
        threshold = choose_threshold(y, scores)
        flags = scores >= threshold
        self.assertEqual(int(flags.sum()), 3)
        self.assertEqual(int(y[flags].sum()), 3)

    def test_tied_threshold_does_not_break_capacity(self):
        threshold = choose_threshold(np.array([0, 1] * 5), np.ones(10))
        self.assertGreater(threshold, 1)

    def test_split_and_training_statistics(self):
        assignments = pd.read_csv(ROOT / "outputs" / "split_assignments.csv")
        self.assertEqual(len(assignments), len(self.raw))
        self.assertFalse(assignments.dataset_row.duplicated().any())
        self.assertEqual(set(assignments.split), {"train", "validation", "test"})
        train_rows = assignments.loc[assignments.split == "train", "dataset_row"] - 1
        train = prepare_features(self.raw.iloc[train_rows])
        np.testing.assert_allclose(self.saved["model"].named_steps["imputer"].statistics_,
                                   train.median().to_numpy())

    def test_batch_scores_ignore_outcome(self):
        batch = self.raw.iloc[:10].copy()
        model, meta = self.saved["model"], self.saved["metadata"]
        before = rank_patients(batch, model, meta["threshold"])
        batch["Outcome"] = 1 - batch["Outcome"]
        after = rank_patients(batch, model, meta["threshold"])
        np.testing.assert_array_equal(before.model_score, after.model_score)
        self.assertEqual(int(before.selected_for_test.sum()), 3)
        self.assertTrue(before.model_score.is_monotonic_decreasing)

    def test_saved_test_results_reconcile(self):
        ranked = pd.read_csv(ROOT / "outputs" / "test_screening_report.csv")
        result = self.saved["metadata"]["test_capacity_results"]
        selected = ranked.selected_for_test
        self.assertEqual(int(selected.sum()), result["selected"])
        self.assertEqual(int(ranked.loc[selected, "Outcome"].sum()), result["tp"])
        self.assertAlmostEqual(result["recall"], result["tp"] / ranked.Outcome.sum())


if __name__ == "__main__":
    unittest.main()

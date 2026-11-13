"""Compare three diabetes classifiers and rank a batch of patients.

Run ``python diabetes_prediction.py`` to train and compare the models.
Add ``--batch data/sample_patients.csv`` to try a sample patient batch.
This project is for learning, not for making medical decisions.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
SEED = 42
CAPACITY = 0.30
SOURCE = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"


def prepare_features(data):
    """Check the eight inputs and mark zero measurements that mean “missing.”"""
    missing = set(FEATURES) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if data.empty:
        raise ValueError("The input has no patient rows.")
    if data.columns.duplicated().any():
        raise ValueError("Column names must be unique.")
    X = data[FEATURES].copy()
    for column in FEATURES:
        X[column] = pd.to_numeric(X[column], errors="raise")
        if np.isinf(X[column].to_numpy(dtype=float)).any():
            raise ValueError(f"{column} contains an infinite value.")
        if (X[column].dropna() < 0).any():
            raise ValueError(f"{column} cannot contain negative values.")
    for column in ["Pregnancies", "Age"]:
        if (X[column].dropna() % 1 != 0).any():
            raise ValueError(f"{column} must use whole numbers.")
    if (X["Age"].dropna() == 0).any():
        raise ValueError("Age must be positive when provided.")
    # Zero pregnancies is possible, so only the five listed measurements change.
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)
    return X


def load_data(path):
    """Read the training CSV and check its Outcome labels."""
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    prepare_features(data)
    if "Outcome" not in data:
        raise ValueError("Training data needs an Outcome column.")
    if data["Outcome"].isna().any() or not data["Outcome"].isin([0, 1]).all():
        raise ValueError("Outcome must be 0 or 1, with no missing labels.")
    # Drop repeated rows so the same record cannot appear in more than one split.
    data = data[FEATURES + ["Outcome"]].drop_duplicates().reset_index(drop=True)
    if len(data) < 50 or data["Outcome"].value_counts().min() < 10:
        raise ValueError("Use at least 50 rows with at least 10 in each class.")
    if data["Outcome"].nunique() != 2:
        raise ValueError("The dataset must contain both outcome classes.")
    return data


def build_models():
    """Set up the three classifiers and the steps each one needs."""
    models = {}
    classifiers = {
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=10, random_state=SEED),
    }
    for name, classifier in classifiers.items():
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if name != "Decision Tree":
            steps.append(("scaler", StandardScaler()))
        steps.append(("classifier", classifier))
        models[name] = Pipeline(steps)
    return models


def capacity_flags(scores, capacity=CAPACITY):
    """Choose the highest scores that fit the batch's testing limit."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite one-dimensional array.")
    if not 0 <= capacity <= 1:
        raise ValueError("Capacity must be between 0 and 1.")
    count = math.floor(len(scores) * capacity)
    order = np.argsort(-scores, kind="stable")
    selected = np.zeros(len(scores), dtype=int)
    selected[order[:count]] = 1
    return selected


def score_predictions(y, predictions):
    """Count correct and incorrect results, then calculate the main scores."""
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_threshold(y, scores):
    """Choose a validation cutoff that catches positives without filling too many slots."""
    # This extra cutoff lets the function select nobody if tied scores fill the slots.
    candidates = np.r_[np.unique(scores), np.nextafter(1.0, 2.0)]
    limit = math.floor(len(scores) * CAPACITY)
    best_key, best_threshold = None, None
    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        if predictions.sum() > limit:
            continue
        metrics = score_predictions(y, predictions)
        key = (metrics["recall"], metrics["precision"], float(threshold))
        if best_key is None or key > best_key:
            best_key, best_threshold = key, float(threshold)
    return best_threshold


def evaluate_model(name, split, y, scores, threshold):
    """Measure the default cutoff, tuned cutoff and fixed-capacity ranking."""
    rows = []
    for policy, flags in {
        "threshold_0.5": (scores >= 0.5).astype(int),
        "validation_threshold": (scores >= threshold).astype(int),
        "capacity_30_percent": capacity_flags(scores),
    }.items():
        rows.append({
            "model": name, "split": split, "policy": policy,
            "threshold": threshold if policy == "validation_threshold" else
                         (0.5 if policy == "threshold_0.5" else np.nan),
            **score_predictions(y, flags),
            "roc_auc": float(roc_auc_score(y, scores)),
            "average_precision": float(average_precision_score(y, scores)),
            "patients": len(y), "selected": int(flags.sum()),
            "selection_rate": float(flags.mean()),
        })
    return rows


def rank_patients(data, model, threshold):
    """Score a patient batch and mark the highest-scoring rows for testing."""
    reserved = {"source_row", "priority_rank", "model_score", "threshold_flag", "selected_for_test"}
    if reserved.intersection(data.columns):
        raise ValueError("Upload original patient data, not a previously generated ranking report.")
    X = prepare_features(data)
    scores = model.predict_proba(X)[:, 1]
    result = data.copy().reset_index(drop=True)
    result.insert(0, "source_row", np.arange(1, len(result) + 1))
    result["model_score"] = scores
    result["threshold_flag"] = scores >= threshold
    result["selected_for_test"] = capacity_flags(scores).astype(bool)
    result = result.sort_values("model_score", ascending=False, kind="stable")
    result.insert(0, "priority_rank", np.arange(1, len(result) + 1))
    return result.reset_index(drop=True)


def train_project(csv_path=ROOT / "data" / "diabetes.csv", output=ROOT / "outputs"):
    """Train the models, choose a winner on validation data and check it on test data."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(csv_path)
    X, y = prepare_features(data), data["Outcome"].astype(int)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=SEED)
    if X_train.isna().all().any():
        raise ValueError("A training feature has no usable measurements.")

    models, thresholds, rows = build_models(), {}, []
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_val)[:, 1]
        thresholds[name] = choose_threshold(y_val, scores)
        rows.extend(evaluate_model(name, "validation", y_val, scores, thresholds[name]))

    validation = pd.DataFrame(rows)
    ranked = validation[validation.policy == "capacity_30_percent"].sort_values(
        ["recall", "precision", "average_precision", "model"],
        ascending=[False, False, False, True], kind="stable")
    winner = str(ranked.iloc[0]["model"])
    # Make the choices before using the held-back test results.
    test_scores = {}
    for name, model in models.items():
        scores = model.predict_proba(X_test)[:, 1]
        test_scores[name] = scores
        rows.extend(evaluate_model(name, "test", y_test, scores, thresholds[name]))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    selected_model = models[winner]
    test_batch = data.loc[X_test.index].copy()
    test_batch.insert(0, "dataset_row", X_test.index + 1)
    screening = rank_patients(test_batch, selected_model, thresholds[winner])
    screening.to_csv(output / "test_screening_report.csv", index=False)

    # Save each row's group so the split can be checked later.
    assignments = pd.DataFrame({"dataset_row": np.arange(1, len(data) + 1),
                                "split": "", "Outcome": y})
    for label, frame in [("train", X_train), ("validation", X_val), ("test", X_test)]:
        assignments.loc[frame.index, "split"] = label
    assignments.to_csv(output / "split_assignments.csv", index=False)
    audit = pd.DataFrame({
        "feature": FEATURES,
        "dtype": [str(data[c].dtype) for c in FEATURES],
        "raw_missing": data[FEATURES].isna().sum().to_numpy(),
        "zeros_marked_missing": [int((data[c] == 0).sum()) if c in ZERO_AS_MISSING
                                 else 0 for c in FEATURES],
        "missing_after_cleaning": X.isna().sum().to_numpy(),
        "training_median": selected_model.named_steps["imputer"].statistics_,
    })
    audit.to_csv(output / "data_audit.csv", index=False)
    final = comparison[(comparison.model == winner) & (comparison.split == "test") &
                       (comparison.policy == "capacity_30_percent")].iloc[0].to_dict()
    metadata = {
        "selected_model": winner, "threshold": thresholds[winner],
        "capacity": CAPACITY, "random_state": SEED,
        "dataset_rows": len(data), "train_rows": len(X_train),
        "validation_rows": len(X_val), "test_rows": len(X_test),
        "positive_rows": int(y.sum()), "negative_rows": int((y == 0).sum()),
        "dataset_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "source_url": SOURCE, "test_capacity_results": final,
        "selection_rule": "Validation recall@30%, then precision@30%, then average precision",
    }
    # The capacity rule has no single cutoff, so save its threshold as null.
    metadata["test_capacity_results"]["threshold"] = None
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    joblib.dump({"model": selected_model, "metadata": metadata}, output / "model.joblib")
    from project_report import save_reports
    save_reports(data, comparison, audit, metadata, y_test, test_scores, output)
    return {"model": selected_model, "metadata": metadata, "comparison": comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "diabetes.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--batch", type=Path, help="Optional CSV of patients to rank")
    args = parser.parse_args()
    try:
        result = train_project(args.data, args.output)
        print((args.output / "evaluation_report.txt").read_text(encoding="utf-8"))
        test_results = result["comparison"].query("split == 'test'")
        print("How the models did on the test set")
        print(test_results[["model", "policy", "accuracy", "precision", "recall", "f1",
                            "selected"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\nConfusion matrices at 30% capacity (actual rows; selected columns)")
        for _, row in test_results.query("policy == 'capacity_30_percent'").iterrows():
            print(f"\n{row['model']}:\n{np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)}")
        if args.batch:
            batch = pd.read_csv(args.batch)
            ranked = rank_patients(batch, result["model"], result["metadata"]["threshold"])
            ranked.to_csv(args.output / "batch_screening_report.csv", index=False)
            print(f"Batch: {len(ranked)} patients; {ranked.selected_for_test.sum()} selected.")
        print(f"Results saved in {args.output.resolve()}")
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f"Could not complete the project: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

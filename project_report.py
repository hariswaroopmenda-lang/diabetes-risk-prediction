"""Save readable results and figures from the actual experiment."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, precision_recall_curve


def save_reports(data, comparison, audit, meta, y_test, scores, output):
    output = Path(output)
    figures = output / "figures"
    matrices = output / "confusion_matrices"
    figures.mkdir(exist_ok=True)
    matrices.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    test = comparison[comparison.split == "test"]
    capacity = test[test.policy == "capacity_30_percent"].set_index("model")
    ax = capacity[["recall", "precision", "f1"]].plot.bar(
        figsize=(8, 4.5), color=["#167d9a", "#e59c39", "#6b77ba"], rot=0)
    ax.set(title="Untouched test set: screening at 30% capacity",
           xlabel="", ylabel="Score", ylim=(0, 1))
    ax.legend(["Recall", "Precision", "F1"], loc="upper right")
    plt.tight_layout()
    plt.savefig(figures / "model_comparison.png")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for name, values in scores.items():
        fpr, tpr, _ = roc_curve(y_test, values)
        axes[0].plot(fpr, tpr, label=name)
        precision, recall, _ = precision_recall_curve(y_test, values)
        axes[1].plot(recall, precision, label=name)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].axhline(float(np.mean(y_test)), linestyle="--", color="gray")
    axes[0].set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Precision-recall curves", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(figures / "roc_pr_curves.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    data.Outcome.value_counts().sort_index().plot.bar(
        ax=axes[0], color=["#167d9a", "#e59c39"], rot=0)
    axes[0].set(title="Dataset class balance", xlabel="Outcome (0 / 1)", ylabel="Patients")
    audit.set_index("feature").missing_after_cleaning.plot.barh(ax=axes[1], color="#167d9a")
    axes[1].set(title="Missing measurements after zero handling", xlabel="Patients", ylabel="")
    fig.tight_layout()
    fig.savefig(figures / "data_overview.png")
    plt.close(fig)

    for _, row in test.iterrows():
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        name = row.model.lower().replace(" ", "_") + "_" + row.policy
        np.savetxt(matrices / f"{name}.csv", matrix, fmt="%d", delimiter=",")
        fig, ax = plt.subplots(figsize=(4.8, 4.4))
        ConfusionMatrixDisplay(matrix, display_labels=["Outcome 0", "Outcome 1"]).plot(
            ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{row.model}\n{row.policy.replace('_', ' ')}", fontsize=11)
        fig.tight_layout()
        fig.savefig(matrices / f"{name}.png")
        plt.close(fig)

    from report_summary import write_summary
    write_summary(output)

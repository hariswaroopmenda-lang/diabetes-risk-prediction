"""Write the short project report from the saved results."""

import csv
import html
import json
from pathlib import Path


def write_summary(output):
    output = Path(output)
    meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    with (output / "model_comparison.csv").open(newline="", encoding="utf-8") as source:
        rows = [r for r in csv.DictReader(source)
                if r["split"] == "test" and r["policy"] == "capacity_30_percent"]
    result = meta["test_capacity_results"]
    table = "Model                 Accuracy  Precision  Recall   F1\n"
    for row in rows:
        table += (f"{row['model']:<21} {float(row['accuracy']):>7.1%}  "
                  f"{float(row['precision']):>8.1%}  {float(row['recall']):>6.1%}  "
                  f"{float(row['f1']):.3f}\n")
    report = f"""DIABETES RISK PREDICTION

Aim
Build an app that scores patient records and ranks a batch for diabetes
screening, with no more than 30% selected for further testing.

Dataset and preparation
The Pima diabetes dataset contains {meta['dataset_rows']} records and eight input features.
Zero values in Glucose, BloodPressure, SkinThickness, Insulin and BMI were
treated as missing and filled with training medians. Zero pregnancies was kept.

Method
The data was split into {meta['train_rows']} training, {meta['validation_rows']} validation and {meta['test_rows']} test records.
KNN, Logistic Regression and Decision Tree were compared. KNN and Logistic
Regression used standardised features. Preprocessing was fitted on training
data only. The validation set was used to choose the model and its threshold.

{meta['selected_model']} was selected because it had the highest validation recall
at 30% capacity. Its threshold was {meta['threshold']:.6f}. For batch screening, the app
ranks all scores and selects the highest 30%, rounded down to a whole number.

Test results at 30% capacity
{table}
The selected model flagged {result['selected']} of {result['patients']} test records. It caught
{result['tp']} of {result['tp'] + result['fn']} positive cases, giving {result['recall']:.1%} recall.

Confusion matrix for the selected model:
                       Not selected   Selected
Actual negative        {result['tn']:<14} {result['fp']}
Actual positive        {result['fn']:<14} {result['tp']}

Conclusion
The app supports individual predictions, model comparison and batch screening.
The Decision Tree performed better on this test batch, but the model choice
was made earlier using validation data. The selected model missed {result['fn']}
positive test cases, so the results leave clear room for improvement.

This is a coursework prototype and cannot diagnose patients. It needs validation
on relevant clinical data; glucose and insulin also require lab measurements.

References
Plotly. Pima diabetes CSV used in this project.
https://github.com/plotly/datasets/blob/master/diabetes.csv

Scikit-learn. Common pitfalls and recommended practices.
https://scikit-learn.org/stable/common_pitfalls.html
"""
    (output / "evaluation_report.txt").write_text(report, encoding="utf-8")
    sections = report.split("\n\n")
    body = ""
    for i, section in enumerate(sections):
        if i == 0:
            body += f"<h1>{html.escape(section)}</h1>"
        elif section.startswith("Confusion matrix") or section.startswith("Test results"):
            body += f"<pre>{html.escape(section)}</pre>"
        else:
            lines = section.splitlines()
            if lines[0] in {"Aim", "Dataset and preparation", "Method", "Conclusion", "References"}:
                body += f"<h2>{html.escape(lines.pop(0))}</h2>"
            body += "<p>" + "<br>".join(html.escape(line) for line in lines) + "</p>"
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Diabetes Risk Prediction</title><style>
body{{font:16px/1.65 Arial,sans-serif;color:#222;max-width:850px;margin:40px auto;padding:0 24px}}
h1{{font-size:27px}}h2{{font-size:19px;margin-top:24px}}pre{{font:14px/1.7 monospace;overflow:auto;padding:15px;background:#f4f6f7}}
p{{overflow-wrap:anywhere}}@media print{{body{{margin:0}}pre{{break-inside:avoid}}}}
</style></head><body>{body}</body></html>"""
    (output / "Project_Report.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    write_summary(Path(__file__).resolve().parent / "outputs")

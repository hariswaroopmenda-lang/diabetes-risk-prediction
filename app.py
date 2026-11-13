"""Small dashboard for the diabetes prediction coursework project."""

import joblib
import pandas as pd
import streamlit as st

from diabetes_prediction import ROOT, FEATURES, prepare_features, rank_patients, train_project

st.set_page_config(page_title="Diabetes Screening Project", page_icon="🩺", layout="wide")
st.title("Diabetes risk project")
st.caption("Enter a patient record or rank a batch under the clinic's 30% testing limit.")
st.info("Coursework demonstration: this score is not a diagnosis or a calibrated medical risk.")


@st.cache_resource
def load_project():
    """Open the model created by the training script, training it if needed."""
    path = ROOT / "outputs" / "model.joblib"
    if not path.exists():
        train_project()
    return joblib.load(path)


try:
    project = load_project()
except Exception as error:
    st.error(f"Could not load the project: {error}")
    st.stop()

model, meta = project["model"], project["metadata"]
page = st.sidebar.radio("Choose a screen", ["Dashboard", "Patient score", "Model performance", "Batch screening"])
st.sidebar.markdown(f"**Selected model:** {meta['selected_model']}")
st.sidebar.caption("The validation data was used to choose the model.")

if page == "Dashboard":
    st.subheader("Project overview")
    columns = st.columns(4)
    columns[0].metric("Dataset records", meta["dataset_rows"])
    columns[1].metric("Testing capacity", "30%")
    columns[2].metric("Test recall at capacity", f"{meta['test_capacity_results']['recall']:.1%}")
    columns[3].metric("Test patients selected", meta["test_capacity_results"]["selected"])
    st.write(f"Split: {meta['train_rows']} training, {meta['validation_rows']} validation, "
             f"and {meta['test_rows']} testing records.")
    st.image(str(ROOT / "outputs" / "figures" / "data_overview.png"))
    st.markdown("**How it works:** missing measurements are filled using training medians. "
                "Three models are compared, then the selected model ranks a whole batch. "
                "At most the highest-scoring 30% are marked for follow-up testing.")
    st.warning("Glucose and insulin already require lab measurements. This project does not "
               "establish a clinically validated way to screen patients before blood tests.")
    st.download_button("Download project report", (ROOT / "outputs" / "Project_Report.html").read_bytes(),
                       file_name="Project_Report.html", mime="text/html")

elif page == "Patient score":
    st.subheader("Enter one patient record")
    st.write("The starting values are examples. Use Batch screening to allocate the limited test slots.")
    st.caption("A zero for glucose, blood pressure, skin thickness, insulin or BMI is treated as missing.")
    with st.form("patient_form"):
        left, right = st.columns(2)
        with left:
            pregnancies = st.number_input("Pregnancies", min_value=0, value=1, step=1)
            glucose = st.number_input("Glucose (2-hour plasma measurement)", min_value=0.0, value=120.0)
            pressure = st.number_input("Diastolic blood pressure (mm Hg)", min_value=0.0, value=70.0)
            skin = st.number_input("Skin thickness (mm)", min_value=0.0, value=20.0)
        with right:
            insulin = st.number_input("Insulin (2-hour serum measurement)", min_value=0.0, value=0.0)
            bmi = st.number_input("BMI", min_value=0.0, value=28.0)
            pedigree = st.number_input("Diabetes pedigree function", min_value=0.0, value=0.5, step=0.01)
            age = st.number_input("Age (years)", min_value=1, value=35, step=1)
        submitted = st.form_submit_button("Calculate model score")
    if submitted:
        values = [pregnancies, glucose, pressure, skin, insulin, bmi, pedigree, age]
        patient = pd.DataFrame([values], columns=FEATURES)
        score = float(model.predict_proba(prepare_features(patient))[0, 1])
        st.metric("Model score", f"{score:.3f} out of 1")
        relation = "meets or exceeds" if score >= meta["threshold"] else "is below"
        st.write(f"The score {relation} the validation cutoff ({meta['threshold']:.3f}).")
        st.caption("This is not a diagnosis. A patient's place in line depends on the rest of the batch.")

elif page == "Model performance":
    st.subheader("Model comparison")
    results = pd.read_csv(ROOT / "outputs" / "model_comparison.csv")
    split = st.selectbox("Data split", ["test", "validation"])
    policy = st.selectbox("Decision policy", ["capacity_30_percent", "validation_threshold", "threshold_0.5"])
    table = results[(results.split == split) & (results.policy == policy)]
    st.dataframe(table[["model", "accuracy", "precision", "recall", "f1", "roc_auc",
                        "average_precision", "selected", "tp", "fp", "fn", "tn"]], hide_index=True)
    st.caption("A cutoff alone cannot guarantee the capacity limit on a new batch. The ranking rule enforces it.")
    if split == "test":
        st.image(str(ROOT / "outputs" / "figures" / "model_comparison.png"))
        chosen = st.selectbox("Confusion matrix model", list(table.model))
        filename = chosen.lower().replace(" ", "_") + "_" + policy + ".png"
        st.image(str(ROOT / "outputs" / "confusion_matrices" / filename))
        st.image(str(ROOT / "outputs" / "figures" / "roc_pr_curves.png"))
    st.download_button("Download all metrics", (ROOT / "outputs" / "model_comparison.csv").read_bytes(),
                       file_name="model_comparison.csv", mime="text/csv")

else:
    st.subheader("Rank a batch of patients")
    st.write("Upload a CSV with the eight health measurements. Outcome is optional and is not used to score patients.")
    st.caption("The example has ten records from the dataset's test group. They are for demonstration only.")
    st.download_button("Download example CSV", (ROOT / "data" / "sample_patients.csv").read_bytes(),
                       file_name="sample_patients.csv", mime="text/csv")
    uploaded = st.file_uploader("Patient CSV", type=["csv"])
    demo = st.checkbox("Use the example batch")
    if uploaded is not None or demo:
        try:
            batch = pd.read_csv(uploaded if uploaded is not None else ROOT / "data" / "sample_patients.csv")
            ranked = rank_patients(batch, model, meta["threshold"])
            left, right = st.columns(2)
            left.metric("Patients in batch", len(ranked))
            right.metric("Selected for testing", int(ranked.selected_for_test.sum()))
            if len(ranked) < 4:
                st.info("This batch has fewer than four patients, so the strict 30% limit allows no slots.")
            st.dataframe(ranked, hide_index=True)
            st.caption("The list is ordered from highest score to lowest. Tied scores follow the CSV row order. "
                       "Selected for test applies the 30% limit; the cutoff flag is shown separately.")
            st.download_button("Download screening priorities", ranked.to_csv(index=False).encode("utf-8"),
                               file_name="screening_priorities.csv", mime="text/csv")
        except (ValueError, KeyError) as error:
            st.error(f"Check the CSV: {error}")

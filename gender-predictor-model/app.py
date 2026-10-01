from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


FEATURES = [
    "Favorite Color",
    "Favorite Music Genre",
    "Favorite Beverage",
    "Favorite Soft Drink",
]

HERE = Path(__file__).parent
MODEL_PATH = HERE / "gender_model.pkl"
COLUMNS_PATH = HERE / "feature_columns.pkl"
CATEGORIES_PATH = HERE / "feature_categories.pkl"
DATA_PATH = HERE / "Transformed Data Set - Sheet1.csv"

st.set_page_config(page_title="Gender Predictor")
st.title("Gender Predictor")
st.caption("A logistic-regression demo using four preference features.")

st.warning(
    "This model uses a small dataset and had variable cross-validation results. "
    "Treat it as a learning demo, not a reliable way to infer gender."
)

required_files = [MODEL_PATH, COLUMNS_PATH, CATEGORIES_PATH]
missing_files = [path.name for path in required_files if not path.exists()]

if missing_files:
    st.error(
        "Put these downloaded Colab files beside app.py: "
        + ", ".join(missing_files)
    )
    st.stop()

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(COLUMNS_PATH)
feature_categories = joblib.load(CATEGORIES_PATH)

# Show the EDA count charts if the CSV is beside this app.
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)

    st.subheader("Explore the data")

    for feature in FEATURES:
        if feature in df.columns and "Gender" in df.columns:
            st.markdown(f"**{feature}**")
            counts = pd.crosstab(df[feature], df["Gender"]).sort_index()
            st.bar_chart(counts)

st.subheader("Try a prediction")

with st.form("prediction_form"):
    selected = {
        feature: st.selectbox(feature, feature_categories[feature])
        for feature in FEATURES
    }

    submitted = st.form_submit_button("Predict gender")

if submitted:
    # Make the input columns match the one-hot columns used in Colab.
    encoded = {column: 0 for column in feature_columns}

    for feature, value in selected.items():
        baseline = feature_categories[feature][0]

        if value != baseline:
            dummy_column = f"{feature}_{value}"

            if dummy_column in encoded:
                encoded[dummy_column] = 1

    input_row = pd.DataFrame([encoded], columns=feature_columns)

    prediction = int(model.predict(input_row)[0])
    gender = {0: "Female", 1: "Male"}[prediction]

    st.success(f"Model prediction: **{gender}**")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_row)[0]
        class_index = list(model.classes_).index(prediction)
        st.caption(
            f"Model probability for this prediction: "
            f"{probabilities[class_index]:.1%}"
        )
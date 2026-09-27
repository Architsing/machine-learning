import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(
    page_title="Life Expectancy Predictor",
    page_icon="🌍",
    layout="wide"
)

@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model = joblib.load(os.path.join(BASE_DIR, "linear_regression_model.pkl"))
    country_encoder = joblib.load(os.path.join(BASE_DIR, "country_encoder.pkl"))
    feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

    return model, country_encoder, feature_columns

model, country_encoder, feature_columns = load_model()

st.title("🌍 Life Expectancy Predictor")
st.write("Neeche saari details bharo, aur Predict button dabao.")

st.subheader("Enter Feature Values")

user_inputs = {}

for feature in feature_columns:

    if feature == "Country_encoded":
        country_name = st.selectbox("Country", country_encoder.classes_)
        user_inputs[feature] = country_encoder.transform([country_name])[0]

    elif feature == "Developing_Status":
        status = st.selectbox("Status", ["Developing", "Developed"])
        user_inputs[feature] = 1 if status == "Developing" else 0

    else:
        user_inputs[feature] = st.number_input(f"{feature}", value=0.0, format="%.2f")

if st.button("🚀 Predict Life Expectancy"):
    input_array = np.array([[user_inputs[feature] for feature in feature_columns]])
    prediction = model.predict(input_array)

    st.markdown("## 📈 Prediction Result")
    st.success(f"Predicted Life Expectancy: **{prediction[0]:.2f} years**")

st.markdown("---")
st.markdown("## 🧠 About This Model")

st.write("""
This application uses Multiple Linear Regression to predict
a country's Life Expectancy based on health, economic, and
social indicators such as Adult Mortality, GDP, Schooling,
BMI, and more.
""")

st.info(
    "Machine Learning Model: Multiple Linear Regression | "
    "Prediction Target: Life Expectancy"
)

with st.expander("🔍 View Model Features"):
    st.write("The model expects these features:")
    for feature in feature_columns:
        st.write(f"• {feature}")

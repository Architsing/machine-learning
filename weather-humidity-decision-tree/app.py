from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).parent / "humidity_tree.joblib"

FEATURES = [
    "air_pressure_9am",
    "air_temp_9am",
    "avg_wind_direction_9am",
    "avg_wind_speed_9am",
    "max_wind_direction_9am",
    "max_wind_speed_9am",
    "rain_accumulation_9am",
    "rain_duration_9am",
    "relative_humidity_9am",
]


st.set_page_config(
    page_title="Afternoon humidity predictor",
    page_icon="💧",
    layout="centered",
)

st.title("Afternoon humidity predictor")
st.write(
    "Enter the weather measurements from 9 a.m. "
    "to predict whether humidity will be high at 3 p.m."
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Put humidity_tree.joblib "
        "in the same folder as app.py."
    )
    st.stop()


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


with st.form("weather_inputs"):
    st.subheader("9 a.m. measurements")

    left, right = st.columns(2)

    with left:
        air_pressure = st.number_input(
            "Air pressure",
            value=918.9,
            min_value=850.0,
            max_value=1100.0,
            step=0.1,
        )
        air_temp = st.number_input(
            "Air temperature",
            value=64.9,
            min_value=-50.0,
            max_value=150.0,
            step=0.1,
        )
        avg_wind_direction = st.number_input(
            "Average wind direction",
            value=142.2,
            min_value=0.0,
            max_value=360.0,
            step=1.0,
        )
        avg_wind_speed = st.number_input(
            "Average wind speed",
            value=5.5,
            min_value=0.0,
            max_value=200.0,
            step=0.1,
        )
        max_wind_direction = st.number_input(
            "Maximum wind direction",
            value=149.0,
            min_value=0.0,
            max_value=360.0,
            step=1.0,
        )

    with right:
        max_wind_speed = st.number_input(
            "Maximum wind speed",
            value=7.0,
            min_value=0.0,
            max_value=200.0,
            step=0.1,
        )
        rain_accumulation = st.number_input(
            "Rain accumulation",
            value=0.2,
            min_value=0.0,
            max_value=1000.0,
            step=0.1,
        )
        rain_duration = st.number_input(
            "Rain duration",
            value=294.1,
            min_value=0.0,
            max_value=100000.0,
            step=1.0,
        )
        relative_humidity = st.number_input(
            "Relative humidity",
            value=34.2,
            min_value=0.0,
            max_value=100.0,
            step=0.1,
        )

    submitted = st.form_submit_button("Predict humidity")


if submitted:
    input_data = pd.DataFrame(
        [[
            air_pressure,
            air_temp,
            avg_wind_direction,
            avg_wind_speed,
            max_wind_direction,
            max_wind_speed,
            rain_accumulation,
            rain_duration,
            relative_humidity,
        ]],
        columns=FEATURES,
    )

    prediction = int(model.predict(input_data)[0])

    if prediction == 1:
        st.success("Prediction: high humidity at 3 p.m.")
    else:
        st.info("Prediction: humidity is not expected to be high at 3 p.m.")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        class_probabilities = dict(zip(model.classes_, probabilities))
        confidence = float(class_probabilities[prediction])
        st.metric("Model confidence", f"{confidence:.1%}")

    with st.expander("See the input values"):
        st.dataframe(input_data, hide_index=True)
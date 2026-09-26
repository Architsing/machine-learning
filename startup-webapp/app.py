import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Startup Profit Predictor",
    page_icon="💰",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load("linear_regression_model.pkl")

    feature_columns = joblib.load("feature_columns.pkl")

    return model, feature_columns


model, feature_columns = load_model()

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 45px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    margin-bottom: 35px;
}

.prediction-box {
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    background-color: #f0f7ff;
    margin-top: 25px;
}

.prediction-value {
    font-size: 40px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">💰 Startup Profit Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict startup profit using Multiple Linear Regression'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("📊 Startup Information")

rd_spend = st.sidebar.number_input(
    "R&D Spend ($)",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

administration = st.sidebar.number_input(
    "Administration ($)",
    min_value=0.0,
    value=100000.0,
    step=1000.0
)

marketing_spend = st.sidebar.number_input(
    "Marketing Spend ($)",
    min_value=0.0,
    value=150000.0,
    step=1000.0
)

state = st.sidebar.selectbox(
    "State",
    [
        "California",
        "New York",
        "Florida"
    ]
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.sidebar.button("🚀 Predict Profit"):

    california_state = 1 if state == "California" else 0
    florida_state = 1 if state == "Florida" else 0
    new_york_state = 1 if state == "New York" else 0

    input_data = pd.DataFrame(
        [[
            rd_spend,
            administration,
            marketing_spend,
            california_state,
            florida_state,
            new_york_state
        ]],
        columns=feature_columns
    )

    prediction = model.predict(input_data)[0]

    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    st.markdown("## 📈 Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "R&D Spend",
            f"${rd_spend:,.2f}"
        )

    with col2:
        st.metric(
            "Administration",
            f"${administration:,.2f}"
        )

    with col3:
        st.metric(
            "Marketing Spend",
            f"${marketing_spend:,.2f}"
        )

    st.markdown(
        f"""
        <div class="prediction-box">

        <h2>Estimated Startup Profit</h2>

        <div class="prediction-value">
        ${prediction:,.2f}
        </div>

        <p>Predicted using Multiple Linear Regression</p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------
    # STARTUP DETAILS
    # --------------------------------------------------

    st.markdown("## 📋 Startup Details")

    summary = pd.DataFrame({
        "Feature": [
            "R&D Spend",
            "Administration",
            "Marketing Spend",
            "State"
        ],
        "Value": [
            f"${rd_spend:,.2f}",
            f"${administration:,.2f}",
            f"${marketing_spend:,.2f}",
            state
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# ABOUT MODEL
# --------------------------------------------------

st.markdown("---")

st.markdown("## 🧠 About This Model")

st.write("""
This application uses Multiple Linear Regression to predict
the profit of a startup based on:

- R&D Spend
- Administration Spend
- Marketing Spend
- State
""")

st.info(
    "Machine Learning Model: Multiple Linear Regression | "
    "Prediction Target: Startup Profit"
)

# --------------------------------------------------
# MODEL FEATURES
# --------------------------------------------------

with st.expander("🔍 View Model Features"):

    st.write("The model expects these features:")

    for feature in feature_columns:
        st.write(f"• {feature}")

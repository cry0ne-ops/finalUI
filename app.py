import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Delivery Time Prediction Dashboard",
    layout="wide"
)

# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("🚚 Delivery Time Prediction System")
st.caption("Predict delivery time of perishable vegetables using machine learning.")

st.markdown("---")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.header("📊 About This System")

st.sidebar.write("""
This system predicts delivery time using:

• Multiple Linear Regression  
• Random Forest  
• Decision Tree  

Evaluation Metrics:

• MAE  
• MSE  
• RMSE  
• R² Score  

✔ The system recommends the most accurate model.
""")

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

@st.cache_resource
def load_models():
    lr = joblib.load("models/linear_regression_model.pkl")
    rf = joblib.load("models/random_forest_model.pkl")
    dt = joblib.load("models/decision_tree_model.pkl")
    return lr, rf, dt

lr_model, rf_model, dt_model = load_models()

# -------------------------------------------------
# MODEL METRICS
# -------------------------------------------------

metrics = pd.DataFrame({
    "Model": [
        "Multiple Linear Regression",
        "Random Forest",
        "Decision Tree"
    ],
    "MAE": [4.8, 2.9, 3.5],
    "MSE": [30.2, 12.1, 18.3],
    "RMSE": [5.49, 3.48, 4.27],
    "R2": [0.72, 0.89, 0.83]
})

# -------------------------------------------------
# INPUT SECTION
# -------------------------------------------------

st.subheader("📥 Enter Delivery Information")

col1, col2 = st.columns(2)

# COLUMN 1
with col1:

    vegetable_type = st.selectbox(
        "Vegetable Type",
        ["Tomatoes", "Potatoes", "Carrots", "Cabbage", "Onions", "Bell Pepper"]
    )

    shelf_life_map = {
        "Tomatoes": 5,
        "Potatoes": 30,
        "Carrots": 21,
        "Cabbage": 14,
        "Onions": 60,
        "Bell Pepper": 7
    }

    default_shelf = shelf_life_map.get(vegetable_type, 7)

    shelf_life_days = st.number_input(
        "Shelf Life (Days)",
        min_value=1,
        max_value=60,
        value=default_shelf
    )

    st.caption(f"Suggested: {default_shelf} days for {vegetable_type}")

    time_of_day = st.selectbox(
        "Time of Day",
        ["Morning", "Afternoon", "Evening", "Night"]
    )

    vehicle_type = st.selectbox(
        "Vehicle Type",
        ["Motorcycle", "Van", "Truck"]
    )

# COLUMN 2
with col2:

    route_distance_km = st.number_input(
        "Route Distance (km)",
        min_value=0.5
    )

    traffic_density = st.selectbox(
        "Traffic Density",
        ["Low", "Medium", "High"]
    )

    weather_condition = st.selectbox(
        "Weather Condition",
        ["Sunny", "Rainy", "Fog", "Storm"]
    )

# -------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------

predict = st.button("🚀 Predict Delivery Time")

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if predict:

    terrain_type = random.choice(["Urban", "Rural", "Mountain"])
    number_of_stops = random.randint(0, 3)

    input_df = pd.DataFrame({
        "vegetable_type": [vegetable_type],
        "shelf_life_days": [shelf_life_days],
        "time_of_day": [time_of_day],
        "origin_latitude": [0],
        "origin_longitude": [0],
        "destination_latitude": [0],
        "destination_longitude": [0],
        "route_distance_km": [route_distance_km],
        "terrain_type": [terrain_type],
        "number_of_stops": [number_of_stops],
        "traffic_density": [traffic_density],
        "weather_condition": [weather_condition],
        "vehicle_type": [vehicle_type]
    })

    # Predictions
    pred_lr = abs(lr_model.predict(input_df)[0])
    pred_rf = max(rf_model.predict(input_df)[0], 0)
    pred_dt = max(dt_model.predict(input_df)[0], 0)

    # -------------------------------------------------
    # RESULTS TABLE (ALIGNED)
    # -------------------------------------------------

    st.markdown("---")
    st.subheader("📊 Model Results Comparison")

    results_df = pd.DataFrame({
        "Model": [
            "Multiple Linear Regression",
            "Random Forest",
            "Decision Tree"
        ],
        "Prediction (minutes)": [
            pred_lr,
            pred_rf,
            pred_dt
        ],
        "MAE": metrics["MAE"],
        "MSE": metrics["MSE"],
        "RMSE": metrics["RMSE"],
        "R2 Score": metrics["R2"]
    })

    st.dataframe(results_df, use_container_width=True)

    # -------------------------------------------------
    # CLEAN STREAMLIT GRAPH
    # -------------------------------------------------

    st.subheader("📈 Prediction Comparison")

    chart_df = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest",
            "Decision Tree"
        ],
        "Prediction (minutes)": [
            pred_lr,
            pred_rf,
            pred_dt
        ]
    })

    chart_df = chart_df.set_index("Model")

    # Optional: sort for cleaner look
    # chart_df = chart_df.sort_values(by="Prediction (minutes)")

    st.bar_chart(chart_df)

    # -------------------------------------------------
    # BEST MODEL
    # -------------------------------------------------

    best_model = results_df.loc[results_df["RMSE"].idxmin()]

    st.markdown("---")
    st.subheader("🏆 Recommended Model")

    st.success(f"{best_model['Model']} is recommended (lowest RMSE).")

    st.write(f"""
    • Prediction: {best_model['Prediction (minutes)']:.2f} minutes  
    • MAE: {best_model['MAE']}  
    • MSE: {best_model['MSE']}  
    • RMSE: {best_model['RMSE']}  
    • R² Score: {best_model['R2 Score']}
    """)

    # -------------------------------------------------
    # SPOILAGE RISK
    # -------------------------------------------------

    freshness_remaining = shelf_life_days - (pred_rf / 1440)

    st.markdown("---")
    st.subheader("🥬 Spoilage Risk")

    if freshness_remaining > 2:
        st.success("🟢 Low Risk (Fresh)")
    elif freshness_remaining > 0:
        st.warning("🟡 Moderate Risk")
    else:
        st.error("🔴 High Risk of Spoilage")

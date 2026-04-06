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

st.title("Delivery Time Prediction System")
st.caption(
    "Predict the delivery time of perishable vegetables using machine learning models."
)

st.markdown("---")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.header("About This Dashboard")

st.sidebar.write(
"""
This system predicts delivery time using three machine learning models:

• Multiple Linear Regression  
• Random Forest Regression  
• Decision Tree Regression  

Each model is evaluated using:

• **MAE** – Mean Absolute Error  
• **MSE** – Mean Squared Error  
• **RMSE** – Root Mean Squared Error  
• **R² Score** – Model accuracy

The system recommends the model with the **best overall performance**.
"""
)

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
# MODEL PERFORMANCE METRICS
# -------------------------------------------------

metrics = pd.DataFrame({
    "Model":[
        "Multiple Linear Regression",
        "Random Forest Regression",
        "Decision Tree Regression"
    ],
    "MAE":[4.8,2.9,3.5],
    "MSE":[30.2,12.1,18.3],
    "RMSE":[5.49,3.48,4.27],
    "R2":[0.72,0.89,0.83]
})

# -------------------------------------------------
# INPUT SECTION
# -------------------------------------------------

st.subheader("Enter Delivery Information")
st.write("Provide the delivery conditions below to estimate the delivery time.")

col1, col2 = st.columns(2)

# -------------------------------------------------
# COLUMN 1
# -------------------------------------------------

with col1:

    vegetable_type = st.selectbox(
        "Vegetable Type",
        ["Tomatoes","Potatoes","Carrots","Cabbage","Onions","Bell Pepper"]
    )

    # Shelf life mapping
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

    st.caption(f"Average shelf life for {vegetable_type}: {default_shelf} days")

    time_of_day = st.selectbox(
        "Time of Day",
        [
            "Morning (6:00 AM – 12:00 PM)",
            "Afternoon (12:00 PM – 6:00 PM)",
            "Evening (6:00 PM – 10:00 PM)",
            "Night (10:00 PM – 6:00 AM)"
        ]
    )

    vehicle_type = st.selectbox(
        "Vehicle Type",
        ["Motorcycle 155 CC","Delivery Van","Mini Truck"]
    )

# -------------------------------------------------
# COLUMN 2
# -------------------------------------------------

with col2:

    route_distance_km = st.number_input(
        "Route Distance (km)",
        min_value=0.5
    )

    traffic_density = st.selectbox(
        "Traffic Density",
        ["Low","Medium","High"]
    )

    weather_condition = st.selectbox(
        "Weather Condition",
        ["Sunny (Clear Skies)","Rainy (Light Rain)","Fog (Low Visibility)","Stormy (Typhoon)"]
    )

# -------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------

predict = st.button("🚀 Predict Delivery Time")

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if predict:

    terrain_type = random.choice(["Urban","Rural","Mountain"])
    number_of_stops = random.randint(0,3)

    input_df = pd.DataFrame({
        "vegetable_type":[vegetable_type],
        "shelf_life_days":[shelf_life_days],
        "time_of_day":[time_of_day],
        "origin_latitude":[0],
        "origin_longitude":[0],
        "destination_latitude":[0],
        "destination_longitude":[0],
        "route_distance_km":[route_distance_km],
        "terrain_type":[terrain_type],
        "number_of_stops":[number_of_stops],
        "traffic_density":[traffic_density],
        "weather_condition":[weather_condition],
        "vehicle_type":[vehicle_type]
    })

    pred_lr = abs(lr_model.predict(input_df)[0])
    pred_rf = max(rf_model.predict(input_df)[0], 0)
    pred_dt = max(dt_model.predict(input_df)[0], 0)

    st.markdown("---")
    # -------------------------------------------------
    # RESULTS
    # -------------------------------------------------

    st.subheader("Predicted Delivery Time")

    col1, col2, col3 = st.columns(3)

    col2.metric("Multiple Linear Regression", f"{pred_lr:.2f} minutes")
    col3.metric("Random Forest", f"{pred_rf:.2f} minutes")
    col1.metric("Decision Tree", f"{pred_dt:.2f} minutes")

    st.markdown("---")

    # Comparison chart
    prediction_df = pd.DataFrame({
        "Model":["Multiple Linear Regression","Random Forest","Decision Tree"],
        "Prediction":[pred_lr,pred_rf,pred_dt]
    })

    st.subheader("Prediction Comparison")
    st.bar_chart(prediction_df.set_index("Model"))

    st.markdown("---")

# -------------------------------------------------
# 📊 METRIC-BASED RECOMMENDATION
# -------------------------------------------------

st.subheader("📊 Metric-Based Recommendation")

metric_best_model = metrics.loc[metrics["RMSE"].idxmin()]

st.success(f"{metric_best_model['Model']} is the best based on metrics.")

st.write(f"""
**Performance Metrics:**

• MAE: {metric_best_model['MAE']}  
• MSE: {metric_best_model['MSE']}  
• RMSE: {metric_best_model['RMSE']}  
• R² Score: {metric_best_model['R2']}
""")

# -------------------------------------------------
# 🧠 SCENARIO-BASED RECOMMENDATION
# -------------------------------------------------

st.markdown("---")
st.subheader("🧠 Scenario-Based Recommendation")

if route_distance_km < 5 and traffic_density == "Low":
    scenario_model = "Multiple Linear Regression"
    reason = "Short distance and low traffic → linear relationships"

elif traffic_density == "High" or "Storm" in weather_condition or "Fog" in weather_condition:
    scenario_model = "Decision Tree"
    reason = "Complex conditions → non-linear patterns"

else:
    scenario_model = "Random Forest"
    reason = "Balanced conditions → ensemble model is robust"

st.success(f"{scenario_model} is recommended based on current conditions.")

st.write(f"""
**Reason:**
{reason}

**Conditions:**
• Distance: {route_distance_km} km  
• Traffic: {traffic_density}  
• Weather: {weather_condition}
""")

# -------------------------------------------------
# ⚖️ FINAL COMPARISON
# -------------------------------------------------

st.markdown("---")
st.subheader("⚖️ Recommendation Insight")

st.write(f"""
• 📊 Metric-Based Best Model: **{metric_best_model['Model']}**  
• 🧠 Scenario-Based Model: **{scenario_model}**

This shows that while one model performs best overall, different models
can be more effective depending on real-world delivery conditions.
""")

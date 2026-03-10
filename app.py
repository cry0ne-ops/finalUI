import streamlit as st
import pandas as pd
import joblib
import numpy as np

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
# Replace with your actual results
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
# SYSTEM OVERVIEW
# -------------------------------------------------

st.subheader("System Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Models Used", "3")
col2.metric("Prediction Type", "Regression")
col3.metric("Dataset Features", "12")

st.markdown("---")

# -------------------------------------------------
# INPUT SECTION
# -------------------------------------------------

st.subheader("Enter Delivery Information")

st.write(
"Provide the delivery conditions below to estimate the delivery time."
)

col1, col2 = st.columns(2)

with col1:

    vegetable_type = st.selectbox(
        "Vegetable Type",
        ["Tomatoes","Potatoes","Carrots","Cabbage","Onions","Bell Pepper"]
    )

    shelf_life_days = st.number_input(
        "Shelf Life (Days)",1,60,7
    )

    time_of_day = st.selectbox(
        "Time of Day",
        ["Morning","Afternoon","Evening","Night"]
    )

    vehicle_type = st.selectbox(
        "Vehicle Type",
        ["Motorcycle","Van","Truck"]
    )

with col2:

    route_distance_km = st.number_input(
        "Route Distance (km)",0.0
    )

    number_of_stops = st.number_input(
        "Number of Stops",0
    )

    terrain_type = st.selectbox(
        "Terrain Type",
        ["Urban","Rural","Mountain"]
    )

    traffic_density = st.selectbox(
        "Traffic Density",
        ["Low","Medium","High"]
    )

    weather_condition = st.selectbox(
        "Weather Condition",
        ["Sunny","Rainy","Fog","Storm"]
    )

predict = st.button("🚀 Predict Delivery Time")

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if predict:

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
    pred_rf = max(rf_model.predict(input_df)[0],0)
    pred_dt = max(dt_model.predict(input_df)[0],0)

    st.markdown("---")

    # -------------------------------------------------
    # PREDICTION RESULTS
    # -------------------------------------------------

    st.subheader("Predicted Delivery Time")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Linear Regression",
        f"{pred_lr:.2f} minutes"
    )

    col2.metric(
        "Random Forest",
        f"{pred_rf:.2f} minutes"
    )

    col3.metric(
        "Decision Tree",
        f"{pred_dt:.2f} minutes"
    )

    st.markdown("---")

    # -------------------------------------------------
    # PREDICTION COMPARISON
    # -------------------------------------------------

    prediction_df = pd.DataFrame({
        "Model":["Linear Regression","Random Forest","Decision Tree"],
        "Prediction":[pred_lr,pred_rf,pred_dt]
    })

    st.subheader("Prediction Comparison")

    st.bar_chart(prediction_df.set_index("Model"))

    st.markdown("---")

    # -------------------------------------------------
    # MODEL PERFORMANCE METRICS
    # -------------------------------------------------

    st.subheader("Model Performance Metrics")

    st.write(
    "These metrics show how accurate each model performed during training."
    )

    st.dataframe(metrics)

    st.bar_chart(metrics.set_index("Model")[["RMSE"]])

    st.markdown("---")

    # -------------------------------------------------
    # RECOMMENDATION BASED ON METRICS
    # -------------------------------------------------

    best_model = metrics.loc[metrics["RMSE"].idxmin()]

    st.subheader("Recommended Model")

    st.success(
        f"**{best_model['Model']}** is recommended."
    )

    st.write(
    f"""
The recommendation is based on the model evaluation metrics.

The recommended model is Random Forest Regression based on its overall
performance across multiple evaluation metrics.

• **MAE:** {best_model['MAE']}  
• **MSE:** {best_model['MSE']}  
• **RMSE:** {best_model['RMSE']}  
• **R² Score:** {best_model['R2']}

This indicates that the model provides the most accurate and reliable
predictions among the evaluated models.
"""
    )

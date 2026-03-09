import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Delivery Time Prediction",
    layout="centered"
)

# ------------------------------
# Title
# ------------------------------

st.title("Delivery Time Prediction System")
st.caption("Optimizing delivery for perishable vegetables using machine learning")

# ------------------------------
# Load Models
# ------------------------------

lr_model = joblib.load("models/linear_regression_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")
dt_model = joblib.load("models/decision_tree_model.pkl")

# ------------------------------
# Input Section
# ------------------------------

st.header("Delivery Information")

col1, col2 = st.columns(2)

with col1:

    vegetable = st.selectbox(
        "Vegetable Type",
        ["Tomatoes","Potatoes","Carrots","Cabbage","Onions"]
    )

    shelf_life = st.number_input(
        "Shelf Life (Days)",
        min_value=1,
        max_value=60,
        value=7
    )

    time_day = st.selectbox(
        "Time of Day",
        ["Morning","Afternoon","Evening","Night"]
    )

    vehicle = st.selectbox(
        "Vehicle Type",
        ["Motorcycle","Van","Truck"]
    )

with col2:

    distance = st.number_input(
        "Route Distance (km)",
        min_value=0.0,
        value=5.0
    )

    stops = st.number_input(
        "Number of Stops",
        min_value=0,
        value=1
    )

    terrain = st.selectbox(
        "Terrain Type",
        ["Urban","Rural","Mountain"]
    )

    traffic = st.selectbox(
        "Traffic Density",
        ["Low","Medium","High"]
    )

    weather = st.selectbox(
        "Weather Condition",
        ["Sunny","Rainy","Fog","Storm"]
    )

# ------------------------------
# Prediction Button
# ------------------------------

if st.button("Predict Delivery Time"):

    input_df = pd.DataFrame({

        "vegetable_type":[vegetable],
        "shelf_life_days":[shelf_life],
        "time_of_day":[time_day],
        "origin_latitude":[0],
        "origin_longitude":[0],
        "destination_latitude":[0],
        "destination_longitude":[0],
        "route_distance_km":[distance],
        "terrain_type":[terrain],
        "number_of_stops":[stops],
        "traffic_density":[traffic],
        "weather_condition":[weather],
        "vehicle_type":[vehicle]

    })

    pred_lr = lr_model.predict(input_df)[0]
    pred_rf = rf_model.predict(input_df)[0]
    pred_dt = dt_model.predict(input_df)[0]

    st.header("Prediction Results")

    results = pd.DataFrame({
        "Model":[
            "Multiple Linear Regression",
            "Random Forest Regression",
            "Decision Tree Regression"
        ],
        "Predicted Delivery Time (minutes)":[
            pred_lr,
            pred_rf,
            pred_dt
        ]
    })

    st.table(results)

    best_model = results.loc[
        results["Predicted Delivery Time (minutes)"].idxmin()
    ]

    st.success(
        f"Recommended Prediction: {best_model['Model']}"
    )

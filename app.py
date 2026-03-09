import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(
    page_title="Delivery Time Prediction System",
    layout="centered"
)

st.title("Delivery Time Prediction System")
st.caption("Predict delivery time for perishable vegetables using machine learning")

st.markdown("---")

# -----------------------------
# Load models
# -----------------------------

lr_model = joblib.load("models/linear_regression_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")
dt_model = joblib.load("models/decision_tree_model.pkl")

# -----------------------------
# Input Section
# -----------------------------

st.header("Enter Delivery Information")

col1, col2 = st.columns(2)

with col1:

    vegetable_type = st.selectbox(
        "Vegetable Type",
        ["Tomatoes","Bell Pepper","Potatoes","Cabbage","Carrots","Onions"]
    )

    shelf_life_days = st.number_input(
        "Shelf Life (Days)",
        min_value=1,
        max_value=60,
        value=7
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
        "Route Distance (km)",
        min_value=0.0,
        value=5.0
    )

    number_of_stops = st.number_input(
        "Number of Stops",
        min_value=0,
        value=1
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

st.markdown("---")

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Delivery Time"):

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

    pred_lr = lr_model.predict(input_df)[0]
    pred_lr = max(pred_lr, 0)
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
            round(pred_lr,2),
            round(pred_rf,2),
            round(pred_dt,2)
        ]

    })

    st.table(results)

    st.markdown("---")

    best_model = results.loc[
        results["Predicted Delivery Time (minutes)"].idxmin()
    ]

    st.success(
        f"Recommended Model: **{best_model['Model']}**"
    )

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------

st.set_page_config(
    page_title="Delivery Time Prediction Dashboard",
    layout="wide"
)

# ---------------------------------------
# SIDEBAR
# ---------------------------------------

st.sidebar.title("Delivery Prediction System")
st.sidebar.info(
"""
This system predicts delivery time for perishable goods using
three regression models:

• Multiple Linear Regression  
• Random Forest Regression  
• Decision Tree Regression
"""
)

st.sidebar.markdown("---")

st.sidebar.write("Developed for Thesis Project")

# ---------------------------------------
# TITLE
# ---------------------------------------

st.title("🚚 Delivery Time Prediction Dashboard")
st.caption("Machine Learning System for Predicting Delivery Time of Perishable Vegetables")

st.markdown("---")

# ---------------------------------------
# LOAD MODELS (cached)
# ---------------------------------------

@st.cache_resource
def load_models():
    lr = joblib.load("models/linear_regression_model.pkl")
    rf = joblib.load("models/random_forest_model.pkl")
    dt = joblib.load("models/decision_tree_model.pkl")
    return lr, rf, dt

lr_model, rf_model, dt_model = load_models()

# ---------------------------------------
# SYSTEM OVERVIEW
# ---------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Models Used", "3")
col2.metric("Prediction Type", "Regression")
col3.metric("Dataset Features", "12")

st.markdown("---")

# ---------------------------------------
# INPUT SECTION
# ---------------------------------------

st.header("Enter Delivery Information")

col1, col2 = st.columns(2)

with col1:

    vegetable_type = st.selectbox(
        "Vegetable Type",
        ["Tomatoes","Potatoes","Carrots","Cabbage","Onions","Bell Pepper"]
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

st.markdown("")

predict = st.button("🚀 Predict Delivery Time")

# ---------------------------------------
# PREDICTION
# ---------------------------------------

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

    # predictions
    pred_lr = max(lr_model.predict(input_df)[0],0)
    pred_rf = max(rf_model.predict(input_df)[0],0)
    pred_dt = max(dt_model.predict(input_df)[0],0)

    st.markdown("---")

    st.header("Prediction Results")

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

    # ---------------------------------------
    # COMPARISON CHART
    # ---------------------------------------

    results = pd.DataFrame({

        "Model":[
            "Linear Regression",
            "Random Forest",
            "Decision Tree"
        ],

        "Prediction":[
            pred_lr,
            pred_rf,
            pred_dt
        ]

    })

    st.markdown("---")

    st.subheader("Model Prediction Comparison")

    st.bar_chart(
        results.set_index("Model")
    )

    # ---------------------------------------
    # RECOMMENDATION
    # ---------------------------------------

    best_model = results.loc[
        results["Prediction"].idxmin()
    ]

    st.success(
        f"Recommended Model: {best_model['Model']}"
    )

    st.info(
        "This model produced the lowest predicted delivery time for the given conditions."
    )

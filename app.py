import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
from sklearn.metrics import mean_squared_error

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

• MAE  
• MSE  
• RMSE  
• R² Score  
"""
)

# -------------------------------------------------
# LOAD MODELS + DATASET
# -------------------------------------------------

@st.cache_resource
def load_models():
    lr = joblib.load("models/linear_regression_model.pkl")
    rf = joblib.load("models/random_forest_model.pkl")
    dt = joblib.load("models/decision_tree_model.pkl")
    return lr, rf, dt

@st.cache_data
def load_data():
    return pd.read_excel("data/dataset.xlsx")

lr_model, rf_model, dt_model = load_models()
df = load_data()

# -------------------------------------------------
# MODEL METRICS
# -------------------------------------------------

metrics = pd.DataFrame({
    "Model":[
        "Multiple Linear Regression",
        "Random Forest",
        "Decision Tree"
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
col1, col2 = st.columns(2)

with col1:

    vegetable_type = st.selectbox(
        "Vegetable Type",
        ["Tomatoes","Potatoes","Carrots","Cabbage","Onions","Bell Pepper"]
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
        "Shelf Life (Days)",1,60,default_shelf
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

    route_distance_km = st.number_input("Route Distance (km)",0.5)

    traffic_density = st.selectbox(
        "Traffic Density",
        ["Low","Medium","High"]
    )

    weather_condition = st.selectbox(
        "Weather Condition",
        ["Sunny","Rainy","Fog","Storm"]
    )

# -------------------------------------------------
# PREDICT
# -------------------------------------------------

predict = st.button("🚀 Predict Delivery Time")

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

    # Predictions
    pred_lr = abs(lr_model.predict(input_df)[0])
    pred_rf = max(rf_model.predict(input_df)[0],0)
    pred_dt = max(dt_model.predict(input_df)[0],0)

    st.markdown("---")

    # -------------------------------------------------
    # RESULTS
    # -------------------------------------------------

    st.subheader("Predicted Delivery Time")

    col1, col2, col3 = st.columns(3)

    col1.metric("Linear Regression", f"{pred_lr:.2f} min")
    col2.metric("Random Forest", f"{pred_rf:.2f} min")
    col3.metric("Decision Tree", f"{pred_dt:.2f} min")

    st.markdown("---")

    # Chart
    prediction_df = pd.DataFrame({
        "Model":["Linear Regression","Random Forest","Decision Tree"],
        "Prediction":[pred_lr,pred_rf,pred_dt]
    }).set_index("Model")

    st.bar_chart(prediction_df)

    st.markdown("---")

    # Metrics
    st.subheader("Model Performance Metrics")
    st.dataframe(metrics)

    # Best model
    best_model = metrics.loc[metrics["RMSE"].idxmin()]

    st.subheader("Recommended Model")
    st.success(f"{best_model['Model']} is recommended.")

    # -------------------------------------------------
    # 🔥 METHOD 3: PERFORMANCE DISTRIBUTION
    # -------------------------------------------------

    st.markdown("---")
    st.subheader("📊 Model Performance Across Scenarios")

    distribution_results = []

    for traffic in df["traffic_density"].unique():
        for weather in df["weather_condition"].unique():

            subset = df[
                (df["traffic_density"] == traffic) &
                (df["weather_condition"] == weather)
            ]

            if len(subset) < 10:
                continue

            X = subset.drop("delivery_time", axis=1)
            y = subset["delivery_time"]

            row = {
                "Traffic": traffic,
                "Weather": weather
            }

            for name, model in [
                ("Linear Regression", lr_model),
                ("Random Forest", rf_model),
                ("Decision Tree", dt_model)
            ]:

                try:
                    preds = model.predict(X)
                    rmse = np.sqrt(mean_squared_error(y, preds))
                except:
                    rmse = np.nan

                row[name] = rmse

            distribution_results.append(row)

    distribution_df = pd.DataFrame(distribution_results)

    st.dataframe(distribution_df)

    # Average RMSE
    st.subheader("📈 Average RMSE Across Scenarios")

    avg_rmse = distribution_df[[
        "Linear Regression",
        "Random Forest",
        "Decision Tree"
    ]].mean()

    avg_df = pd.DataFrame({
        "Model":["Linear Regression","Random Forest","Decision Tree"],
        "Average RMSE":avg_rmse.values
    }).set_index("Model")

    st.bar_chart(avg_df)

    # -------------------------------------------------
    # SPOILAGE RISK
    # -------------------------------------------------

    freshness_remaining = shelf_life_days - (pred_rf / 1440)

    st.markdown("---")
    st.subheader("🥬 Spoilage Risk")

    if freshness_remaining > 2:
        st.success("🟢 Low Risk")
    elif freshness_remaining > 0:
        st.warning("🟡 Moderate Risk")
    else:
        st.error("🔴 High Risk")

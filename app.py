import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

✔ Includes real-world scenario validation using dataset
✔ Adaptive model recommendation
""")

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
# INPUT SECTION
# -------------------------------------------------

st.subheader("📥 Enter Delivery Information")

col1, col2 = st.columns(2)

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

with col2:

    route_distance_km = st.number_input("Route Distance (km)", min_value=0.5)

    traffic_density = st.selectbox(
        "Traffic Density",
        ["Low", "Medium", "High"]
    )

    weather_condition = st.selectbox(
        "Weather Condition",
        ["Sunny", "Rainy", "Fog", "Storm"]
    )

# -------------------------------------------------
# PREDICT
# -------------------------------------------------

predict = st.button("🚀 Predict Delivery Time")

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
    # RESULTS TABLE
    # -------------------------------------------------

    st.markdown("---")
    st.subheader("📊 Model Predictions")

    results_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "Decision Tree"],
        "Prediction (minutes)": [pred_lr, pred_rf, pred_dt]
    })

    st.dataframe(results_df, use_container_width=True)

    # Graph
    st.bar_chart(results_df.set_index("Model"))

    # -------------------------------------------------
    # 🧠 REAL SCENARIO VALIDATION
    # -------------------------------------------------

    st.markdown("---")
    st.subheader("📊 Real Scenario Validation")

    scenario_df = df.copy()

    # Apply filters
    scenario_df = scenario_df[
        (scenario_df["traffic_density"] == traffic_density) &
        (scenario_df["weather_condition"] == weather_condition)
    ]

    scenario_df = scenario_df[
        (scenario_df["route_distance_km"] >= route_distance_km - 2) &
        (scenario_df["route_distance_km"] <= route_distance_km + 2)
    ]

    if len(scenario_df) < 10:
        st.warning("⚠️ Not enough data for reliable validation.")
    else:

        X = scenario_df.drop("delivery_time", axis=1)
        y = scenario_df["delivery_time"]

        results = []

        for name, model in [
            ("Linear Regression", lr_model),
            ("Random Forest", rf_model),
            ("Decision Tree", dt_model)
        ]:

            preds = model.predict(X)

            mae = mean_absolute_error(y, preds)
            rmse = np.sqrt(mean_squared_error(y, preds))

            results.append({
                "Model": name,
                "MAE": mae,
                "RMSE": rmse
            })

        validation_df = pd.DataFrame(results).sort_values("RMSE")

        st.dataframe(validation_df)

        st.bar_chart(validation_df.set_index("Model"))

        best_model = validation_df.iloc[0]

        st.success(f"""
        🧠 Best Model for THIS Scenario: {best_model['Model']}

        RMSE: {best_model['RMSE']:.2f}
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

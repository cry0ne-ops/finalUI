

# Load trained models
lr_model = joblib.load("linear_regression_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

st.title("Vegetable Delivery Time Prediction System")

st.write("Enter delivery details to estimate delivery time.")

# User Inputs
vegetable = st.selectbox(
    "Vegetable Type",
    ["Carrots","Lettuce","Potatoes","Cabbage","Broccoli",
     "Cauliflower","Bell Pepper","Tomatoes","Baguio Beans","Chayote"]
)

distance = st.number_input("Route Distance (km)",0.1,50.0)

terrain = st.selectbox(
    "Terrain Type",
    ["Flat","Moderate_Slope","Steep_Slope"]
)

traffic = st.selectbox(
    "Traffic Density",
    ["Low","Medium","High"]
)

weather = st.selectbox(
    "Weather Condition",
    ["Clear","Cloudy","Rainy","Foggy"]
)

stops = st.slider("Number of Stops",0,5)

vehicle = st.selectbox(
    "Vehicle Type",
    ["Motorcycle","Truck"]
)

time_of_day = st.selectbox(
    "Time of Day",
    ["Morning","Afternoon","Evening","Night"]
)

# Prediction
if st.button("Predict Delivery Time"):

    input_data = pd.DataFrame({
        "vegetable_type":[vegetable],
        "route_distance_km":[distance],
        "terrain_type":[terrain],
        "number_of_stops":[stops],
        "traffic_density":[traffic],
        "weather_condition":[weather],
        "vehicle_type":[vehicle],
        "time_of_day":[time_of_day]
    })

    lr_pred = lr_model.predict(input_data)[0]
    dt_pred = dt_model.predict(input_data)[0]
    rf_pred = rf_model.predict(input_data)[0]

    st.subheader("Model Predictions")

    st.write(f"Multiple Linear Regression: {lr_pred:.2f} minutes")
    st.write(f"Decision Tree: {dt_pred:.2f} minutes")
    st.write(f"Random Forest: {rf_pred:.2f} minutes")

    avg_pred = (lr_pred + dt_pred + rf_pred) / 3

    st.subheader("Estimated Delivery Time")

    st.success(f"Estimated Delivery Time: {avg_pred:.2f} minutes")

    # Simple explanation for normal users
    explanation = f"""
    The system analyzed the delivery route using three machine learning models.

    Factors affecting the delivery time include:
    • Distance of {distance} km  
    • Traffic condition: {traffic}  
    • Terrain type: {terrain}  
    • Number of stops: {stops}

    Based on these factors, the estimated delivery time is around **{avg_pred:.1f} minutes**.
    """

    st.write(explanation)

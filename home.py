import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from data_clean_utils import perform_data_cleaning

def get_distance_type(distance):
    if distance < 5:
        return "short"
    elif distance < 10:
        return "medium"
    elif distance < 15:
        return "long"
    else:
        return "very_long"

st.set_page_config(page_title="Delivery Time Prediction", layout="wide")

@st.cache_resource
def load_models():
    model = joblib.load('saved_models/model.pkl')
    preprocessor = joblib.load('saved_models/preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_models()

st.title("🚚 Food Delivery Time Prediction")

col0_1, col0_2 = st.columns(2)
with col0_1:
    order_id = st.text_input("Order ID", value="0x4607")
with col0_2:
    rider_id = st.text_input("Delivery Person ID", value="INDORES13DEL02")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🚴 Rider Details")
    age = st.number_input("Age", min_value=18.0, max_value=70.0, value=25.0)
    ratings = st.number_input("Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
    multiple_deliveries = st.selectbox("Multiple Deliveries", [0.0, 1.0])
    vehicle_condition = st.number_input("Vehicle Condition", min_value=0, max_value=5, value=2)

with col2:
    st.subheader("📦 Order Details")
    weather = st.selectbox("Weather", ["sunny", "cloudy", "fog", "stormy", "sandstorms", "windy"])
    traffic = st.selectbox("Traffic Density", ["low", "medium", "high", "jam"])
    order_type = st.selectbox("Order Type", ["snack", "meal", "drinks", "buffet"])
    type_of_vehicle = st.selectbox("Vehicle Type", ["motorcycle", "scooter", "bicycle"])
    
with col3:
    st.subheader("📍 Location & Other")
    festival = st.selectbox("Festival", ["no", "yes"])
    city_type = st.selectbox("City Type", ["metropolitan", "urban", "semi-urban"])
    distance = st.number_input("Distance (km)", min_value=0.0, max_value=50.0, value=5.0)
    pickup_time_minutes = st.number_input("Pickup Time (minutes)", min_value=0, max_value=120, value=15)

st.subheader("🕒 Time Details")
col4, col5 = st.columns(2)
with col4:
    order_date = st.date_input("Order Date")
with col5:
    order_time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "evening", "night", "after_midnight"])

if st.button("Predict Delivery Time", type="primary", use_container_width=True):
    try:
        if not order_id or not rider_id:
            st.error("Please enter both Order ID and Delivery Person ID")
            st.stop()
            
        input_data = pd.DataFrame({
            'age': [age],
            'ratings': [ratings],
            'pickup_time_minutes': [pickup_time_minutes],
            'distance': [distance],
            'weather': [weather],
            'type_of_order': [order_type],
            'type_of_vehicle': [type_of_vehicle],
            'festival': [festival],
            'city_type': [city_type],
            'is_weekend': [1 if order_date.strftime('%A') in ['Saturday', 'Sunday'] else 0],
            'order_time_of_day': [order_time_of_day],
            'traffic': [traffic],
            'distance_type': [get_distance_type(distance)],
            'vehicle_condition': [vehicle_condition],
            'multiple_deliveries': [multiple_deliveries]
        })

        with st.spinner('🔄 Calculating delivery time...'):
            prediction = model.predict(preprocessor.transform(input_data))[0]
            st.success(f"⏱️ Estimated Delivery Time: {int(prediction)} minutes")
            
            st.subheader("📊 Delivery Insights")
            col6, col7, col8 = st.columns(3)
            
            with col6:
                st.metric("🛣️ Distance", f"{distance:.2f} km")
            with col7:
                st.metric("⏳ Pickup Time", f"{pickup_time_minutes} min")
            with col8:
                st.metric("📅 Weekend Delivery", "Yes" if input_data['is_weekend'].values[0] else "No")

    except Exception as e:
        st.error(f"Error: {str(e)}")

with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app predicts food delivery time using ML models trained on historical delivery data.")
    st.divider()
    st.caption("Created with Streamlit & Python")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="🚚",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        color: #FF4B4B;
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        padding: 20px;
    }
    .subtitle {
        color: #666666;
        font-size: 1.5em;
        text-align: center;
        margin-bottom: 40px;
    }
    .section-header {
        color: #333333;
        font-size: 1.8em;
        margin-top: 30px;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stRadio [role=radiogroup]{
        padding: 10px;
        border-radius: 4px;
        background-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

def get_distance_type(distance):
    """Determine the distance category."""
    if distance < 5:
        return "short"
    elif distance < 10:
        return "medium"
    elif distance < 15:
        return "long"
    else:
        return "very_long"

@st.cache_resource
def load_models():
    """Load the ML model and preprocessor."""
    try:
        model = joblib.load('saved_models/model.pkl')
        preprocessor = joblib.load('saved_models/preprocessor.pkl')
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def show_prediction_page():
    """Display the prediction interface."""
    st.title("🚚 Food Delivery Time Prediction")
    
    model, preprocessor = load_models()
    if model is None or preprocessor is None:
        st.error("Could not load the required models.")
        return

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
        order_time_of_day = st.selectbox("Time of Day", 
                                       ["morning", "afternoon", "evening", "night", "after_midnight"])

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
                    st.metric("📅 Weekend Delivery", 
                            "Yes" if input_data['is_weekend'].values[0] else "No")

        except Exception as e:
            st.error(f"Error: {str(e)}")

def show_home_page():
    """Display the home page."""
    st.markdown("<h1 class='main-title'>Food Delivery Time Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>ML-Powered Delivery Time Estimation System</p>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='section-header'>🎯 Project Overview</h2>", unsafe_allow_html=True)
    st.write("""
    This project aims to predict food delivery times accurately using machine learning. 
    By analyzing various factors such as distance, traffic conditions, and historical delivery data, 
    our system provides reliable estimates for delivery duration.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='feature-box'>
        <h3>🎯 Prediction Module</h3>
        <ul>
            <li>Real-time delivery time predictions</li>
            <li>Multiple input parameters support</li>
            <li>Interactive prediction interface</li>
            <li>Instant results visualization</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class='feature-box'>
        <h3>📊 EDA Module</h3>
        <ul>
            <li>Comprehensive data analysis</li>
            <li>Interactive visualizations</li>
            <li>Pattern discovery</li>
            <li>Statistical insights</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h2 class='section-header'>🚀 How to Use</h2>", unsafe_allow_html=True)
    st.markdown("""
    1. **Prediction Section:**
       - Select "Prediction" from the sidebar
       - Enter delivery details
       - Get instant time predictions
    
    2. **EDA Section:**
       - Select "EDA" from the sidebar
       - Choose analysis type
       - Explore interactive visualizations
       - Gain data insights
    """)

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    selected = st.selectbox(
        "Choose Section",
        ["Home", "Prediction", "EDA"],
        format_func=lambda x: "🏠 Home" if x == "Home" else 
                            "🎯 Prediction" if x == "Prediction" else 
                            "📊 EDA"
    )
    
    # Show EDA options only when EDA is selected
    if selected == "EDA":
        st.markdown("---")
        st.subheader("EDA Sections")
        eda_selection = st.radio(
            "",
            [
                "Univariate Numerical Analysis",
                "Univariate Categorical Analysis",
                "Multivariate Numerical Analysis",
                "Multivariate Categorical Analysis",
                "Mixed Multivariate Analysis"
            ],
            format_func=lambda x: "📊 " + x
        )

# Main content
if selected == "Home":
    show_home_page()
elif selected == "Prediction":
    show_prediction_page()
elif selected == "EDA":
    if 'eda_selection' in locals():
        try:
            if eda_selection == "Univariate Numerical Analysis":
                from pages.page1_univariate_numerical import app
                app()
            elif eda_selection == "Univariate Categorical Analysis":
                from pages.page2_univariate_categorical import app
                app()
            elif eda_selection == "Multivariate Numerical Analysis":
                from pages.page3_multivariate_numerical import app
                app()
            elif eda_selection == "Multivariate Categorical Analysis":
                from pages.page4_multivariate_categorical import app
                app()
            elif eda_selection == "Mixed Multivariate Analysis":
                from pages.page5_mixed_multivariate import app
                app()
        except ImportError as e:
            st.error(f"Could not load the selected analysis module. Error: {str(e)}")
    else:
        st.write("Please select an EDA section from the sidebar")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666666;'>
        Made with ❤️ using Python & Streamlit
    </div>
""", unsafe_allow_html=True)
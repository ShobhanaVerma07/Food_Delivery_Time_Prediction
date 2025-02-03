import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm

st.set_page_config(
    page_title="Food Delivery EDA",
    page_icon="🛵",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cleaned_data.csv')
        return df
    except FileNotFoundError:
        st.error("Error: cleaned_data.csv not found. Please ensure the file exists in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Main title with custom CSS
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 3em;
            font-weight: bold;
            padding: 20px;
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            font-size: 3em;
            color: #666;
            margin-bottom: 30px;
        }
    </style>
    <h1 class="main-title">Food Delivery Analysis Dashboard</h1>
    <p class="subtitle">Comprehensive Exploratory Data Analysis</p>
    """, unsafe_allow_html=True)

# Load the data
df = load_data()

if df is not None:
    # Quick overview section
    st.header("📊 Dataset Quick Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Orders", f"{len(df):,}")
    with col2:
        st.metric("Average Delivery Time", f"{df['time_taken'].mean():.1f} mins")
    with col3:
        st.metric("Total Delivery Partners", f"{df['vehicle_condition'].nunique():,}")

    # Dataset information
    st.header("🎯 About the Dataset")
    
    st.markdown("""
    This dataset contains information about food delivery orders including:
    - Customer demographics
    - Order details
    - Delivery information
    - Geographic data
    - Time-related features
    """)

    # Show data distribution
    st.header("📈 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Time taken distribution
        fig1 = px.histogram(df, x='time_taken', 
                          title='Distribution of Delivery Times',
                          labels={'time_taken': 'Delivery Time (minutes)'},
                          color_discrete_sequence=['#FF6B6B'])
        st.plotly_chart(fig1)

    with col2:
        # Orders by day
        daily_orders = df['order_day'].value_counts().sort_index()
        fig2 = px.line(x=daily_orders.index, y=daily_orders.values,
                      title='Orders by Day of Month',
                      labels={'x': 'Day of Month', 'y': 'Number of Orders'},
                      color_discrete_sequence=['#4ECDC4'])
        st.plotly_chart(fig2)

    # Data quality information
    st.header("📋 Data Quality")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Shape")
        st.write(f"Rows: {df.shape[0]:,}")
        st.write(f"Columns: {df.shape[1]:,}")
        
    with col2:
        st.subheader("Missing Values")
        missing_data = df.isnull().sum()
        if missing_data.sum() == 0:
            st.write("No missing values found in the dataset! 🎉")
        else:
            st.write(missing_data[missing_data > 0])

    # Navigation guide
    st.header("🧭 Navigation Guide")
    st.markdown("""
    Use the sidebar to navigate to different analysis pages:
    1. **Univariate Numerical Analysis**: Explore individual numerical variables
    2. **Univariate Categorical Analysis**: Analyze categorical variables
    3. **Multivariate Numerical Analysis**: Study relationships between numerical variables
    4. **Multivariate Categorical Analysis**: Examine categorical variable relationships
    5. **Mixed Multivariate Analysis**: Investigate relationships between different types of variables
    """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Made with ❤️ using Streamlit and Python | Data Science Project
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("Unable to load the dataset. Please check if the data file exists and is accessible.")
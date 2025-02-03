import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.helper_functions import load_data, get_column_types, numerical_analysis
import plotly.express as px

def app():
    # Page config
    # st.set_page_config(page_title="Univariate Numerical Analysis", page_icon="📊", layout="wide")

    # Title and description
    st.title("📊 Univariate Numerical Analysis")
    st.markdown("""
    This page allows you to analyze individual numerical variables in the dataset.
    Explore distributions, central tendencies, and outliers through various visualizations.
    """)

    # Load data
    df = load_data()

    if df is not None:
        # Get column types
        num_cols, _ = get_column_types(df)
        
        # Column selection
        st.sidebar.header("Select Variables")
        selected_col = st.sidebar.selectbox("Choose a numerical column", num_cols)
        bins = st.sidebar.slider("Number of bins for histogram", 5, 50, 20)
        
        # Main analysis
        st.header(f"Analysis of {selected_col}")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📈 Visualizations", "📊 Statistics"])
        
        with tab1:
            # Display plots
            fig = numerical_analysis(df, selected_col, bins=bins)
            st.pyplot(fig)
            
            # Additional interactive plot using plotly
            st.subheader("Interactive Distribution Plot")
            fig = px.histogram(df, x=selected_col, marginal="box")
            st.plotly_chart(fig)
        
        with tab2:
            # Display statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Basic Statistics")
                stats = df[selected_col].describe()
                st.write(stats)
            
            with col2:
                st.subheader("Additional Metrics")
                metrics = {
                    "Skewness": df[selected_col].skew(),
                    "Kurtosis": df[selected_col].kurtosis(),
                    "Missing Values": df[selected_col].isnull().sum(),
                    "Missing Percentage": f"{(df[selected_col].isnull().sum() / len(df)) * 100:.2f}%"
                }
                st.write(pd.Series(metrics))
        
        # Add insights section
        st.markdown("---")
        st.header("💡 Key Insights")
        
        # Calculate and display insights
        mean_val = df[selected_col].mean()
        median_val = df[selected_col].median()
        skew = df[selected_col].skew()
        
        st.markdown(f"""
        - The average {selected_col} is {mean_val:.2f}
        - 50% of values are below {median_val:.2f}
        - The distribution is {'positively' if skew > 0 else 'negatively'} skewed (skewness: {skew:.2f})
        """)
    else:
        st.error("Please ensure the dataset is available and properly loaded.")
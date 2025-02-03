import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils.helper_functions import load_data, get_column_types

def app():
    # Page config
    # st.set_page_config(page_title="Univariate Categorical Analysis", page_icon="📊", layout="wide")

    # Title and description
    st.title("📊 Univariate Categorical Analysis")
    st.markdown("""
    This page allows you to analyze individual categorical variables in the dataset.
    Explore distributions, frequencies, and proportions through various visualizations.
    """)

    # Load data
    df = load_data()

    if df is not None:
        # Get column types
        _, cat_cols = get_column_types(df)
        
        # Column selection
        st.sidebar.header("Select Variables")
        selected_col = st.sidebar.selectbox("Choose a categorical column", cat_cols)
        
        # Main analysis
        st.header(f"Analysis of {selected_col}")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📈 Visualizations", "📊 Statistics"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar plot
                st.subheader("Bar Plot")
                fig = plt.figure(figsize=(10, 6))
                sns.countplot(data=df, x=selected_col)
                plt.xticks(rotation=45)
                plt.title(f"Distribution of {selected_col}")
                st.pyplot(fig)
            
            with col2:
                # Pie chart
                st.subheader("Pie Chart")
                value_counts = df[selected_col].value_counts()
                fig = px.pie(values=value_counts.values, 
                            names=value_counts.index,
                            title=f"Distribution of {selected_col}")
                st.plotly_chart(fig)
        
        with tab2:
            # Display frequency statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Frequency Table")
                freq_df = pd.DataFrame({
                    'Count': df[selected_col].value_counts(),
                    'Percentage': df[selected_col].value_counts(normalize=True).mul(100).round(2)
                })
                freq_df['Percentage'] = freq_df['Percentage'].astype(str) + '%'
                st.write(freq_df)
            
            with col2:
                st.subheader("Summary Statistics")
                stats = {
                    "Total Categories": df[selected_col].nunique(),
                    "Most Common": df[selected_col].mode()[0],
                    "Most Common Count": df[selected_col].value_counts().iloc[0],
                    "Missing Values": df[selected_col].isnull().sum(),
                    "Missing Percentage": f"{(df[selected_col].isnull().sum() / len(df)) * 100:.2f}%"
                }
                st.write(pd.Series(stats))
        
        # Add insights section
        st.markdown("---")
        st.header("💡 Key Insights")
        
        # Calculate insights
        most_common = df[selected_col].mode()[0]
        most_common_pct = (df[selected_col].value_counts().iloc[0] / len(df)) * 100
        least_common = df[selected_col].value_counts().index[-1]
        least_common_pct = (df[selected_col].value_counts().iloc[-1] / len(df)) * 100
        
        st.markdown(f"""
        - Most common category is "{most_common}" ({most_common_pct:.1f}% of all values)
        - Least common category is "{least_common}" ({least_common_pct:.1f}% of all values)
        - There are {df[selected_col].nunique()} unique categories
        """)

    else:
        st.error("Please ensure the dataset is available and properly loaded.")
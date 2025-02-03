import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils.helper_functions import load_data, get_column_types

def app():
    # Page config
    # st.set_page_config(page_title="Multivariate Numerical Analysis", page_icon="📈", layout="wide")

    # Title and description
    st.title("📈 Multivariate Numerical Analysis")
    st.markdown("""
    This page allows you to analyze relationships between multiple numerical variables.
    Explore correlations, patterns, and dependencies through various visualizations.
    """)

    # Load data
    df = load_data()

    if df is not None:
        # Get column types
        num_cols, _ = get_column_types(df)
        
        # Column selection
        st.sidebar.header("Select Variables")
        selected_cols = st.sidebar.multiselect(
            "Choose numerical columns (2-5 recommended)",
            num_cols,
            default=num_cols[:3]
        )
        
        if len(selected_cols) < 2:
            st.warning("Please select at least 2 columns for multivariate analysis.")
        else:
            # Create tabs for different analyses
            tab1, tab2, tab3 = st.tabs(["📊 Correlation Analysis", "📈 Scatter Plots", "🔄 Advanced Visualizations"])
            
            with tab1:
                st.header("Correlation Analysis")
                
                # Correlation matrix
                corr_matrix = df[selected_cols].corr()
                
                # Heatmap
                fig = plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
                plt.title('Correlation Heatmap')
                st.pyplot(fig)
                
                # Detailed correlations
                st.subheader("Detailed Correlations")
                st.write("Strong correlations (|r| > 0.5):")
                strong_corr = []
                for i in range(len(selected_cols)):
                    for j in range(i+1, len(selected_cols)):
                        corr = corr_matrix.iloc[i,j]
                        if abs(corr) > 0.5:
                            strong_corr.append(f"{selected_cols[i]} vs {selected_cols[j]}: {corr:.3f}")
                
                if strong_corr:
                    for corr in strong_corr:
                        st.write(f"- {corr}")
                else:
                    st.write("No strong correlations found.")
            
            with tab2:
                st.header("Scatter Plot Matrix")
                
                if len(selected_cols) <= 4:  # Limit to 4 variables for readability
                    fig = px.scatter_matrix(
                        df[selected_cols],
                        dimensions=selected_cols,
                        title="Scatter Matrix"
                    )
                    st.plotly_chart(fig)
                else:
                    st.warning("Please select 4 or fewer columns for the scatter plot matrix.")
                
                # Pairwise scatter plots with regression lines
                if len(selected_cols) >= 2:
                    st.subheader("Detailed Pairwise Plots")
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("Select X variable", selected_cols)
                    with col2:
                        y_col = st.selectbox("Select Y variable", 
                                        [col for col in selected_cols if col != x_col])
                    
                    fig = px.scatter(df, x=x_col, y=y_col, 
                                trendline="ols",
                                title=f"Scatter Plot: {x_col} vs {y_col}")
                    st.plotly_chart(fig)
            
            with tab3:
                st.header("Advanced Visualizations")
                
                # Parallel coordinates plot
                st.subheader("Parallel Coordinates Plot")
                fig = px.parallel_coordinates(
                    df[selected_cols],
                    title="Parallel Coordinates Plot"
                )
                st.plotly_chart(fig)
                
                # 3D scatter plot (if 3 columns selected)
                if len(selected_cols) >= 3:
                    st.subheader("3D Scatter Plot")
                    x_col = st.selectbox("Select X axis", selected_cols, key='3d_x')
                    y_col = st.selectbox("Select Y axis", 
                                    [col for col in selected_cols if col != x_col],
                                    key='3d_y')
                    z_col = st.selectbox("Select Z axis", 
                                    [col for col in selected_cols if col not in [x_col, y_col]],
                                    key='3d_z')
                    
                    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col,
                                    title="3D Scatter Plot")
                    st.plotly_chart(fig)
            
            # Statistical Summary
            st.markdown("---")
            st.header("📊 Statistical Summary")
            
            # Display basic statistics
            stats_df = df[selected_cols].describe()
            st.write(stats_df)

    else:
        st.error("Please ensure the dataset is available and properly loaded.")
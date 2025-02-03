import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import f_oneway
from utils.helper_functions import load_data, get_column_types

def app():
    # Page config
    # st.set_page_config(page_title="Mixed Multivariate Analysis", page_icon="📊", layout="wide")

    # Title and description
    st.title("📊 Mixed Multivariate Analysis")
    st.markdown("""
    This page allows you to analyze relationships between numerical and categorical variables.
    Explore how numerical variables vary across different categories and identify patterns.
    """)

    def perform_anova(df, num_col, cat_col):
        """Perform one-way ANOVA test."""
        categories = df[cat_col].unique()
        groups = [df[df[cat_col] == cat][num_col].values for cat in categories]
        f_stat, p_val = f_oneway(*groups)
        return f_stat, p_val

    def calculate_eta_squared(df, num_col, cat_col):
        """Calculate eta-squared (effect size) for ANOVA."""
        categories = df[cat_col].unique()
        grand_mean = df[num_col].mean()
        n = len(df)
        
        # Calculate SS_between
        ss_between = sum(len(df[df[cat_col] == cat]) * 
                        (df[df[cat_col] == cat][num_col].mean() - grand_mean) ** 2 
                        for cat in categories)
        
        # Calculate SS_total
        ss_total = sum((df[num_col] - grand_mean) ** 2)
        
        # Calculate eta-squared
        return ss_between / ss_total

    # Load data
    df = load_data()

    if df is not None:
        # Get column types
        num_cols, cat_cols = get_column_types(df)
        
        # Column selection
        st.sidebar.header("Select Variables")
        
        selected_num_cols = st.sidebar.multiselect(
            "Choose numerical columns",
            num_cols,
            default=num_cols[:2]
        )
        
        selected_cat_cols = st.sidebar.multiselect(
            "Choose categorical columns",
            cat_cols,
            default=cat_cols[:1]
        )
        
        if not selected_num_cols or not selected_cat_cols:
            st.warning("Please select at least one numerical and one categorical column.")
        else:
            # Create tabs for different analyses
            tab1, tab2, tab3 = st.tabs(["📊 Distribution Analysis", "📈 Advanced Visualizations", "📑 Statistical Tests"])
            
            with tab1:
                st.header("Distribution Analysis")
                
                for cat_col in selected_cat_cols:
                    for num_col in selected_num_cols:
                        st.subheader(f"{num_col} by {cat_col}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Box plot
                            fig = plt.figure(figsize=(10, 6))
                            sns.boxplot(data=df, x=cat_col, y=num_col)
                            plt.xticks(rotation=45)
                            plt.title("Box Plot")
                            st.pyplot(fig)
                        
                        with col2:
                            # Violin plot
                            fig = plt.figure(figsize=(10, 6))
                            sns.violinplot(data=df, x=cat_col, y=num_col)
                            plt.xticks(rotation=45)
                            plt.title("Violin Plot")
                            st.pyplot(fig)
                        
                        # Summary statistics
                        st.write("Summary Statistics:")
                        summary_stats = df.groupby(cat_col)[num_col].agg([
                            'count', 'mean', 'std', 'min', 'max'
                        ]).round(2)
                        st.write(summary_stats)
                        st.markdown("---")
            
            with tab2:
                st.header("Advanced Visualizations")
                
                viz_type = st.radio(
                    "Select Visualization Type",
                    ["Strip Plots", "Swarm Plots", "Point Plots", "Bar Plots", "Interactive Plots"]
                )
                
                for cat_col in selected_cat_cols:
                    for num_col in selected_num_cols:
                        st.subheader(f"{num_col} by {cat_col}")
                        
                        if viz_type != "Interactive Plots":
                            fig = plt.figure(figsize=(12, 6))
                            
                            if viz_type == "Strip Plots":
                                sns.stripplot(data=df, x=cat_col, y=num_col)
                            elif viz_type == "Swarm Plots":
                                sns.swarmplot(data=df, x=cat_col, y=num_col)
                            elif viz_type == "Point Plots":
                                sns.pointplot(data=df, x=cat_col, y=num_col)
                            else:  # Bar Plots
                                sns.barplot(data=df, x=cat_col, y=num_col)
                            
                            plt.xticks(rotation=45)
                            plt.title(f"{viz_type[:-1]} Plot")
                            st.pyplot(fig)
                        else:
                            # Interactive Plotly plots
                            fig = px.box(df, x=cat_col, y=num_col, 
                                    title=f"Interactive Box Plot: {num_col} by {cat_col}")
                            st.plotly_chart(fig)
                            
                            fig = px.violin(df, x=cat_col, y=num_col, 
                                        title=f"Interactive Violin Plot: {num_col} by {cat_col}")
                            st.plotly_chart(fig)
                        
                        # Optional: Add second categorical variable for hue
                        if len(selected_cat_cols) > 1 and viz_type != "Interactive Plots":
                            other_cat_cols = [c for c in selected_cat_cols if c != cat_col]
                            hue_col = st.selectbox(
                                f"Add color grouping for {cat_col} vs {num_col}",
                                ["None"] + other_cat_cols,
                                key=f"hue_{cat_col}_{num_col}"
                            )
                            
                            if hue_col != "None":
                                fig = plt.figure(figsize=(12, 6))
                                if viz_type == "Strip Plots":
                                    sns.stripplot(data=df, x=cat_col, y=num_col, hue=hue_col)
                                elif viz_type == "Swarm Plots":
                                    sns.swarmplot(data=df, x=cat_col, y=num_col, hue=hue_col)
                                elif viz_type == "Point Plots":
                                    sns.pointplot(data=df, x=cat_col, y=num_col, hue=hue_col)
                                else:  # Bar Plots
                                    sns.barplot(data=df, x=cat_col, y=num_col, hue=hue_col)
                                
                                plt.xticks(rotation=45)
                                plt.title(f"{viz_type[:-1]} Plot with {hue_col}")
                                st.pyplot(fig)
            
            with tab3:
                st.header("Statistical Tests")
                
                # ANOVA tests
                st.subheader("One-way ANOVA Tests")
                
                results = []
                for cat_col in selected_cat_cols:
                    for num_col in selected_num_cols:
                        # Perform ANOVA
                        f_stat, p_val = perform_anova(df, num_col, cat_col)
                        
                        # Calculate effect size
                        eta_squared = calculate_eta_squared(df, num_col, cat_col)
                        
                        results.append({
                            'Variables': f"{num_col} by {cat_col}",
                            'F-statistic': round(f_stat, 3),
                            'p-value': round(p_val, 4),
                            'Effect Size (η²)': round(eta_squared, 3)
                        })
                
                # Display results
                results_df = pd.DataFrame(results)
                st.write(results_df)
                
                # Interpretation guide
                st.subheader("Interpretation Guide")
                st.markdown("""
                **P-value interpretation:**
                - p < 0.05: Statistically significant difference between groups
                - p ≥ 0.05: No statistically significant difference between groups
                
                **Effect Size (η²) interpretation:**
                - 0.01: Small effect
                - 0.06: Medium effect
                - 0.14: Large effect
                """)
                
                # Additional insights
                st.subheader("Key Insights")
                for result in results:
                    variables = result['Variables']
                    p_val = result['p-value']
                    effect = result['Effect Size (η²)']
                    
                    st.write(f"**{variables}**:")
                    if p_val < 0.05:
                        st.write("- Significant difference found between groups")
                        if effect < 0.01:
                            st.write("- Very small effect size")
                        elif effect < 0.06:
                            st.write("- Small effect size")
                        elif effect < 0.14:
                            st.write("- Medium effect size")
                        else:
                            st.write("- Large effect size")
                    else:
                        st.write("- No significant difference between groups")
                    st.write("---")

    else:
        st.error("Please ensure the dataset is available and properly loaded.")
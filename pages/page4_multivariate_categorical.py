import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import chi2_contingency
from utils.helper_functions import load_data, get_column_types

def app():
    # Page config
    # st.set_page_config(page_title="Multivariate Categorical Analysis", page_icon="📊", layout="wide")

    # Title and description
    st.title("📊 Multivariate Categorical Analysis")
    st.markdown("""
    This page allows you to analyze relationships between multiple categorical variables.
    Explore associations, patterns, and dependencies through various visualizations.
    """)

    def cramers_v(confusion_matrix):
        """Calculate Cramér's V statistic for categorical correlation."""
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
        rcorr = r-((r-1)**2)/(n-1)
        kcorr = k-((k-1)**2)/(n-1)
        return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

    # Load data
    df = load_data()

    if df is not None:
        # Get column types
        _, cat_cols = get_column_types(df)
        
        # Column selection
        st.sidebar.header("Select Variables")
        selected_cols = st.sidebar.multiselect(
            "Choose categorical columns (2-4 recommended)",
            cat_cols,
            default=cat_cols[:2] if len(cat_cols) >= 2 else cat_cols
        )
        
        if len(selected_cols) < 2:
            st.warning("Please select at least 2 columns for multivariate analysis.")
        else:
            # Create tabs for different analyses
            tab1, tab2, tab3 = st.tabs(["📊 Contingency Analysis", "📈 Visualizations", "📑 Statistical Tests"])
            
            with tab1:
                st.header("Contingency Analysis")
                
                # Pairwise contingency tables
                for i in range(len(selected_cols)):
                    for j in range(i+1, len(selected_cols)):
                        st.subheader(f"{selected_cols[i]} vs {selected_cols[j]}")
                        
                        # Create and display contingency table
                        contingency_table = pd.crosstab(
                            df[selected_cols[i]], 
                            df[selected_cols[j]], 
                            normalize='index'
                        ).round(4)
                        
                        # Convert to percentage for display
                        display_table = contingency_table.multiply(100).round(2)
                        st.write("Contingency Table (Row Percentages):")
                        st.dataframe(display_table.style.format("{:.2f}%"))
                        
                        # Heatmap visualization
                        fig = plt.figure(figsize=(10, 6))
                        sns.heatmap(contingency_table, annot=True, fmt='.2%', cmap='YlOrRd')
                        plt.title(f"Contingency Table Heatmap")
                        plt.xlabel(selected_cols[j])
                        plt.ylabel(selected_cols[i])
                        st.pyplot(fig)
                        
                        # Calculate Cramér's V
                        cv = cramers_v(pd.crosstab(df[selected_cols[i]], df[selected_cols[j]]))
                        st.write(f"Cramér's V: {cv:.3f} (strength of association)")
                        st.markdown("---")
            
            with tab2:
                st.header("Visualizations")
                
                viz_type = st.radio(
                    "Select Visualization Type",
                    ["Stacked Bar Charts", "Grouped Bar Charts", "Heat Maps"]
                )
                
                for i in range(len(selected_cols)):
                    for j in range(i+1, len(selected_cols)):
                        st.subheader(f"{selected_cols[i]} vs {selected_cols[j]}")
                        
                        if viz_type == "Stacked Bar Charts":
                            # Calculate percentages
                            df_pct = pd.crosstab(df[selected_cols[i]], df[selected_cols[j]], normalize='index') * 100
                            
                            fig = plt.figure(figsize=(12, 6))
                            df_pct.plot(kind='bar', stacked=True)
                            plt.title("Stacked Bar Chart (Percentage)")
                            plt.xlabel(selected_cols[i])
                            plt.ylabel("Percentage")
                            plt.legend(title=selected_cols[j], bbox_to_anchor=(1.05, 1))
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                        elif viz_type == "Grouped Bar Charts":
                            # Use raw counts for grouped bars
                            df_counts = pd.crosstab(df[selected_cols[i]], df[selected_cols[j]])
                            
                            fig = plt.figure(figsize=(12, 6))
                            df_counts.plot(kind='bar')
                            plt.title("Grouped Bar Chart (Counts)")
                            plt.xlabel(selected_cols[i])
                            plt.ylabel("Count")
                            plt.legend(title=selected_cols[j], bbox_to_anchor=(1.05, 1))
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                        else:  # Heat Maps
                            # Create interactive heatmap using plotly
                            counts = pd.crosstab(df[selected_cols[i]], df[selected_cols[j]])
                            fig = px.imshow(counts,
                                        labels=dict(x=selected_cols[j], y=selected_cols[i]),
                                        title="Interactive Heat Map")
                            st.plotly_chart(fig)
            
            with tab3:
                st.header("Statistical Tests")
                
                # Chi-square tests for independence
                st.subheader("Chi-square Tests of Independence")
                
                results = []
                for i in range(len(selected_cols)):
                    for j in range(i+1, len(selected_cols)):
                        # Create contingency table
                        cont_table = pd.crosstab(df[selected_cols[i]], df[selected_cols[j]])
                        
                        # Perform chi-square test
                        chi2, p_val, dof, expected = chi2_contingency(cont_table)
                        
                        # Calculate Cramér's V
                        cv = cramers_v(cont_table)
                        
                        results.append({
                            'Variables': f"{selected_cols[i]} vs {selected_cols[j]}",
                            'Chi-square': f"{chi2:.2f}",
                            'p-value': f"{p_val:.4f}",
                            'Degrees of Freedom': dof,
                            'Cramér\'s V': f"{cv:.3f}"
                        })
                
                # Display results as a formatted DataFrame
                results_df = pd.DataFrame(results)
                st.dataframe(results_df.style.format({'Chi-square': '{:.2f}', 'p-value': '{:.4f}', 'Cramér\'s V': '{:.3f}'}))
                
                # Interpretation guide
                st.subheader("Interpretation Guide")
                st.markdown("""
                - **p-value interpretation**:
                    - p < 0.05: Variables are significantly associated
                    - p ≥ 0.05: No significant association found
                    
                - **Cramér's V interpretation**:
                    - 0.0 to 0.1: Negligible association
                    - 0.1 to 0.3: Weak association
                    - 0.3 to 0.5: Moderate association
                    - 0.5 to 1.0: Strong association
                """)

    else:
        st.error("Please ensure the dataset is available and properly loaded.")
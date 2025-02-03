import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.express as px
from scipy.stats import chi2_contingency, f_oneway, jarque_bera, probplot

# Set default plotting style
plt.style.use('seaborn-v0_8')
sns.set_theme()

@st.cache_data
def load_data():
    """
    Load and cache the dataset.
    Returns:
        pandas.DataFrame or None: The loaded dataset or None if loading fails
    """
    try:
        df = pd.read_csv('cleaned_data.csv')
        return df
    except FileNotFoundError:
        st.error("Error: cleaned_data.csv not found. Please ensure the file exists in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_column_types(df):
    """
    Identify numerical and categorical columns in the dataset.
    Args:
        df (pandas.DataFrame): Input dataset
    Returns:
        tuple: Lists of numerical and categorical column names
    """
    num_cols = ['age', 'ratings', 'restaurant_latitude', 'restaurant_longitude', 
                'delivery_latitude', 'delivery_longitude', 'vehicle_condition',
                'multiple_deliveries', 'time_taken', 'order_day', 'order_month',
                'is_weekend', 'pickup_time_minutes', 'order_time_hour', 'distance']
    
    cat_cols = [col for col in df.columns if col not in num_cols]
    return num_cols, cat_cols

def create_correlation_heatmap(dataframe, columns):
    """
    Create a correlation heatmap for selected numerical columns.
    Args:
        dataframe (pandas.DataFrame): Input dataset
        columns (list): List of numerical column names
    Returns:
        matplotlib.figure.Figure: The correlation heatmap figure
    """
    corr_matrix = dataframe[columns].corr()
    
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Heatmap')
    return fig

def numerical_analysis(dataframe, column_name, cat_col=None, bins="auto"):
    """
    Generate numerical analysis plots (KDE, box plot, histogram).
    Args:
        dataframe (pandas.DataFrame): Input dataset
        column_name (str): Name of the numerical column
        cat_col (str, optional): Name of categorical column for grouping
        bins (int or str): Number of bins for histogram
    Returns:
        matplotlib.figure.Figure: The combined figure with all plots
    """
    fig = plt.figure(figsize=(15, 10))
    grid = GridSpec(nrows=2, ncols=2, figure=fig)
    
    # Create subplots
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, :])
    
    # Generate plots
    sns.kdeplot(data=dataframe, x=column_name, hue=cat_col, ax=ax1)
    ax1.set_title('KDE Plot')
    
    sns.boxplot(data=dataframe, x=column_name, hue=cat_col, ax=ax2)
    ax2.set_title('Box Plot')
    
    sns.histplot(data=dataframe, x=column_name, bins=bins, hue=cat_col, kde=True, ax=ax3)
    ax3.set_title('Histogram with KDE')
    
    plt.tight_layout()
    return fig

def numerical_categorical_analysis(dataframe, cat_column_1, num_column):
    """
    Generate numerical-categorical analysis plots.
    Args:
        dataframe (pandas.DataFrame): Input dataset
        cat_column_1 (str): Name of categorical column
        num_column (str): Name of numerical column
    Returns:
        matplotlib.figure.Figure: The combined figure with all plots
    """
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(15, 7.5))
    
    sns.barplot(data=dataframe, x=cat_column_1, y=num_column, ax=ax1[0])
    ax1[0].set_title('Bar Plot')
    ax1[0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=dataframe, x=cat_column_1, y=num_column, ax=ax1[1])
    ax1[1].set_title('Box Plot')
    ax1[1].tick_params(axis='x', rotation=45)
    
    sns.violinplot(data=dataframe, x=cat_column_1, y=num_column, ax=ax2[0])
    ax2[0].set_title('Violin Plot')
    ax2[0].tick_params(axis='x', rotation=45)
    
    sns.stripplot(data=dataframe, x=cat_column_1, y=num_column, ax=ax2[1])
    ax2[1].set_title('Strip Plot')
    ax2[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def chi_square_test(dataframe, col1, col2, alpha=0.05):
    """
    Perform chi-square test of independence.
    Args:
        dataframe (pandas.DataFrame): Input dataset
        col1 (str): First categorical column name
        col2 (str): Second categorical column name
        alpha (float): Significance level
    Returns:
        tuple: Chi-square statistic, p-value, and interpretation
    """
    contingency_table = pd.crosstab(dataframe[col1], dataframe[col2])
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    
    interpretation = "significant" if p_val < alpha else "not significant"
    return chi2, p_val, interpretation

def calculate_cramers_v(confusion_matrix):
    """
    Calculate Cramér's V statistic for categorical correlation.
    Args:
        confusion_matrix (pandas.DataFrame): Contingency table
    Returns:
        float: Cramér's V statistic
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))

def perform_anova(dataframe, num_col, cat_col):
    """
    Perform one-way ANOVA test.
    Args:
        dataframe (pandas.DataFrame): Input dataset
        num_col (str): Numerical column name
        cat_col (str): Categorical column name
    Returns:
        tuple: F-statistic, p-value
    """
    categories = dataframe[cat_col].unique()
    groups = [dataframe[dataframe[cat_col] == cat][num_col].values for cat in categories]
    return f_oneway(*groups)

def calculate_effect_size(dataframe, num_col, cat_col):
    """
    Calculate eta-squared effect size for ANOVA.
    Args:
        dataframe (pandas.DataFrame): Input dataset
        num_col (str): Numerical column name
        cat_col (str): Categorical column name
    Returns:
        float: Eta-squared effect size
    """
    categories = dataframe[cat_col].unique()
    grand_mean = dataframe[num_col].mean()
    
    ss_between = sum(len(dataframe[dataframe[cat_col] == cat]) * 
                    (dataframe[dataframe[cat_col] == cat][num_col].mean() - grand_mean) ** 2 
                    for cat in categories)
    
    ss_total = sum((dataframe[num_col] - grand_mean) ** 2)
    
    return ss_between / ss_total

def test_normality(dataframe, column):
    """
    Test for normality using Jarque-Bera test.
    Args:
        dataframe (pandas.DataFrame): Input dataset
        column (str): Column name to test
    Returns:
        tuple: Test statistic, p-value
    """
    return jarque_bera(dataframe[column].dropna())

def create_qq_plot(dataframe, column):
    """
    Create a Q-Q plot for normality check.
    Args:
        dataframe (pandas.DataFrame): Input dataset
        column (str): Column name to plot
    Returns:
        matplotlib.figure.Figure: The Q-Q plot figure
    """
    fig = plt.figure(figsize=(8, 6))
    probplot(dataframe[column].dropna(), dist="norm", plot=plt)
    plt.title(f"Q-Q Plot for {column}")
    return fig

def create_residual_plot(dataframe, x, y):
    """
    Create residual plot for regression analysis.
    Args:
        dataframe (pandas.DataFrame): Input dataset
        x (str): Independent variable column name
        y (str): Dependent variable column name
    Returns:
        matplotlib.figure.Figure: The residual plot figure
    """
    # Fit a simple linear regression
    coefficients = np.polyfit(dataframe[x], dataframe[y], 1)
    predicted = np.polyval(coefficients, dataframe[x])
    residuals = dataframe[y] - predicted
    
    # Create the plot
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(predicted, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    return fig

def format_large_number(num):
    """
    Format large numbers with K, M, B suffixes.
    Args:
        num (int/float): Number to format
    Returns:
        str: Formatted number string
    """
    for suffix in ['', 'K', 'M', 'B']:
        if num < 1000:
            return f"{num:.1f}{suffix}"
        num /= 1000
    return f"{num:.1f}T"
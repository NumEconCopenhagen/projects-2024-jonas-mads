from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# FRED API called with API Key
fred = Fred(api_key='e8b9ae83016cbd8575ae24ab46c990e1')

# Economic indicators to be prompted for in the FRED API
indicators = {
    'GDP': 'GDP',
    'Inflation': 'CPIAUCNS',  
    'Unemployment': 'UNRATE',
    'Interest Rate': 'FEDFUNDS' 
}

# Gets data for each indicator
data = {}
for key, value in indicators.items():
    data[key] = fred.get_series(value)

# Filters data for the past 32 years
for key in data:
    data[key] = data[key].loc['1992-01-01':'2024-01-01']  

# Converts GDP to percentage change
data['GDP'] = data['GDP'].pct_change() * 100

# Calculates year-over-year percentage change for inflation
data['Inflation'] = data['Inflation'].pct_change(12) * 100

# Combines into a single DataFrame for descriptive statistics usage
df = pd.DataFrame(data)
df = df.dropna()

def analyze_data():
    # Descriptive statistics
    descriptive_stats = df.describe()
    print("Descriptive Statistics:")
    print(descriptive_stats)
    
    # Calculate and print correlation matrix
    correlation_matrix = df.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

def plot_gdp(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['GDP'].index, data['GDP'].values, color='blue', linestyle='-')
    plt.title('GDP Percentage Change')
    plt.xlabel('Year')
    plt.ylabel('Percentage Change')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_inflation(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Inflation'].index, data['Inflation'].values, color='red', linestyle='-')
    plt.title('Inflation Year-over-Year Percentage Change')
    plt.xlabel('Year')
    plt.ylabel('Percentage Change')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_unemployment(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Unemployment'].index, data['Unemployment'].values, color='green', linestyle='-')
    plt.title('Unemployment Rate')
    plt.xlabel('Year')
    plt.ylabel('Unemployment Rate (%)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_interest_rate(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Interest Rate'].index, data['Interest Rate'].values, color='purple', linestyle='-')
    plt.title('Federal Funds Rate')
    plt.xlabel('Year')
    plt.ylabel('Interest Rate (%)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_inflation_interest(df):

    # Creates a figure with Plotly
    fig = go.Figure()

    # Adds Inflation Rate trace
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['Inflation'], 
            name='Inflation Rate', 
            mode='lines', 
            line=dict(color='red'),
            hovertemplate='Inflation Rate: %{y:.3f}%<extra></extra>'  # Formats hover data to three decimal places
        )
    )

    # Adds Interest Rate trace
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['Interest Rate'], 
            name='Interest Rate', 
            mode='lines', 
            line=dict(color='blue', dash='dash'),
            hovertemplate='Interest Rate: %{y:.3f}%<extra></extra>'  # Formats hover data to three decimal places
        )
    )

    # Sets x-axis title
    fig.update_xaxes(title_text='Year')

    # Sets y-axis title
    fig.update_yaxes(title_text='Rate (%)', color='black')

    # Add figure title and layout adjustments
    fig.update_layout(
        title_text='Interactive Analysis of Inflation Rate vs Interest Rate Over Time',
        legend_title_text='Indicator',
        template='plotly_white', 
        hovermode='x unified'  # This sets the hover information to be unified and appear on the x-axis
    )

    # Shows plot
    fig.show()

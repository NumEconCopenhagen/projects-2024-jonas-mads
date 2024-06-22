# Data Project

## Group Members
- **Jonas Kristensen** (TVP211)
- **Mads Rye** (BNC277)

## Introduction
This project conducts a detailed analysis of four key economic indicators using data from the FRED API: Gross Domestic Product (GDP), Inflation (CPI), Unemployment Rate, and Federal Funds Rate. The objective is to understand the interrelationships among these indicators and examine how they impact the US economy.

## Code Structure
The code is structured to perform a series of analyses and illustrations on the economic data retrieved:

- **Data Retrieval:** Retrieves historical data for each indicator from the FRED API.
- **Data Processing:** Processes the data by calculating percentage changes and filtering based on date ranges.
- **Descriptive Statistics:** Generates descriptive statistics to provide a foundational understanding of the data.
- **Correlation Analysis:** Examines the correlation matrix to determine relationships between the indicators.
- **Visualization of Individual Variables:** Plots each indicator to observe trends over time.
- **Interactive Visualization:** Focuses on the relationship between Inflation Rate and the Federal Funds Rate using interactive Plotly graphs.
  - **Note: This graph cannot be loaded in github, as plotly plots are not supported by github.**

## Usage
To run the code, ensure you have a valid FRED API key set up. This is a free API key, which can be generated here: https://fred.stlouisfed.org/docs/api/api_key.html. I have pasted my API key in the code, so that it runs and this can be used too.

## Dependencies
The project relies on several external libraries, which need to be installed for the script to function correctly:
- **fredapi:** To retrieve data from the FRED database. This one is included in the repository. 
- **pandas:** For data manipulation and analysis.
- **matplotlib:** For creating static plots of the data.
- **plotly:** For generating interactive visualizations.

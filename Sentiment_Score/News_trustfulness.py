import pandas as pd

# 1. Load your project dataset
ticker = input("Enter a ticker:")
input_file = f"{ticker}_processed.csv"
output_file = f"{ticker}_trustfulness.csv"
df = pd.read_csv(input_file)

# 2. Define the Trustworthiness Weights (x3 Factor)
# Weights are based on professional editorial rigor vs. crowdsourced content
source_weights = {
    'DowJones' : 0.90, #Institutional gold standard
    'MarketWatch': 0.90,    # Professional newsroom, high standards
    'Yahoo': 0.80,          # Large-scale aggregator
    'Finnhub': 0.4,        # Data/News API aggregator
    'SeekingAlpha': 0.3,   # Crowdsourced/User-generated (Higher bias risk)
}

# 3. Map the source to the factor
# Default to 0.50 if a new, unknown source appears
df['x3_factor'] = df['source'].map(source_weights).fillna(0.50)

# 4. Calculate the average of x3_factor and create a new column
# This assigns the global mean value to every row in the new column
df['x3_factor_avg'] = df['x3_factor'].mean()

# 5. Save the updated data for your backtesting
df.to_csv(output_file, index=False)

# Preview results
print(df[['source', 'x3_factor', 'x3_factor_avg']].head())
print(f"\nOverall Average x3_factor: {df['x3_factor'].mean():.4f}")

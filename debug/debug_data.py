from generate_scenario import generate_scenario
import numpy as np
import pandas as pd

# Generate data
print("Generating scenario data...")
data = generate_scenario(10, 42)

print('\nData Shape:', data.shape)

# Print first few rows
print("\nFirst 5 rows of market data:")
print(data.iloc[:5, 1:11])

# Extract returns
stock_returns = data.iloc[:5, [1, 5, 9, 13, 17, 21, 25, 29, 33, 37]].values
print('\nStock Returns (first 5 rows):\n', stock_returns)

print('\nReturn ranges:')
for i in range(10):
    idx = i*4+1  # Return is the first feature for each stock
    returns = data.iloc[:, idx].values
    print(f'Stock {i+1} return range: {np.min(returns):.4f} to {np.max(returns):.4f}, Mean: {np.mean(returns):.4f}')

# Check how data is processed in train.py
print("\nSimulating PortfolioEnv step for returns...")
returns_indices = np.arange(0, 40, 4)  # Like in train.py
raw_returns = data.iloc[0, returns_indices + 1].values  # +1 because iloc is 0-based but data columns start at 1
print("Raw returns from data:", raw_returns)
scaled_returns = raw_returns / 100.0
print("Scaled returns (divide by 100):", scaled_returns)

# Calculate portfolio return as in train.py
weights = np.array([0.1] * 10)  # Equal weights
portfolio_return = np.sum(weights * scaled_returns)
print("\nSimulated portfolio return calculation:")
print(f"Portfolio return with equal weights: {portfolio_return:.6f}")
print(f"Annualized return: {portfolio_return * 252 * 100:.2f}%")

# Check data scaling
print("\nChecking data scaling in generate_scenario.py...")
# The scaling happens in generate_scenario.py lines 93-101
print("Original returns are multiplied by 100, so original data is in decimal form (e.g., 0.01 = 1%)")
print("When we divide by 100 in train.py, we're converting back to decimal form")
import numpy as np
import pandas as pd
from generate_scenario import generate_scenario

# Generate sample data with a fixed seed for reproducibility
print("Generating sample data...")
data = generate_scenario(10, seed=42)
print(f"Data shape: {data.shape}")

# Examine the first few rows to understand structure
print("\nFirst 3 rows of data:")
print(data.iloc[:3, :10])  # First 10 columns

# Define portfolio weights (equal weighting for simplicity)
initial_weights = np.array([0.1] * 10)
print("\nInitial portfolio weights:", initial_weights)

# Simulate portfolio returns calculation similar to train.py
print("\nSimulating daily returns calculation...")

# Store results
days = data.shape[0] - 1  # Skip first row as we need return for t+1
daily_returns = []
portfolio_values = [10000]  # Start with $10,000

# Extract return indices (every 4th column starting from column 1)
returns_indices = np.arange(0, 40, 4)

for day in range(days):
    # Get stock returns for the day (in percent form, need to convert to decimal)
    stock_returns = data.iloc[day+1, returns_indices+1].values / 100.0
    
    # Calculate portfolio return
    portfolio_return = np.sum(initial_weights * stock_returns)
    
    # Store the return
    daily_returns.append(portfolio_return)
    
    # Update portfolio value
    portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
    
    # Print details for first 5 days
    if day < 5:
        print(f"Day {day+1}:")
        print(f"  Stock returns: {[round(r, 6) for r in stock_returns]}")
        print(f"  Portfolio return: {portfolio_return:.6f} ({portfolio_return*100:.4f}%)")
        print(f"  Portfolio value: ${portfolio_values[-1]:.2f}")

# Calculate performance metrics
total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
trading_days_per_year = 252
annualized_return = ((1 + total_return/100) ** (trading_days_per_year/days) - 1) * 100
daily_std = np.std(daily_returns) * 100
annualized_std = daily_std * np.sqrt(trading_days_per_year)
sharpe_ratio = annualized_return / annualized_std  # Simplified, ignoring risk-free rate

print("\n===== Portfolio Performance =====")
print(f"Total days: {days}")
print(f"Total return: {total_return:.2f}%")
print(f"Annualized return: {annualized_return:.2f}%")
print(f"Daily standard deviation: {daily_std:.4f}%")
print(f"Annualized standard deviation: {annualized_std:.2f}%")
print(f"Sharpe ratio (simplified): {sharpe_ratio:.4f}")

# Verify data scaling in generate_scenario.py
print("\n===== Checking Data Scaling =====")
raw_returns = []
for i in range(10):
    idx = i*4+1  # Return column for each stock
    raw_returns.append(data.iloc[1:, idx].values)  # Skip first row

raw_returns = np.array(raw_returns).T  # Transpose to get days in rows, stocks in columns

# Range check
returns_min = np.min(raw_returns, axis=0)
returns_max = np.max(raw_returns, axis=0)
returns_mean = np.mean(raw_returns, axis=0)

print("Return statistics by stock:")
for i in range(10):
    print(f"Stock {i+1}: Min={returns_min[i]:.4f}%, Max={returns_max[i]:.4f}%, Mean={returns_mean[i]:.4f}%")

# Overall return distribution
all_returns = raw_returns.flatten()
print(f"\nOverall return distribution: Min={np.min(all_returns):.4f}%, Max={np.max(all_returns):.4f}%, Mean={np.mean(all_returns):.4f}%")

# Annual return simulation (to check if annualized numbers make sense)
annual_stock_returns = np.sum(raw_returns, axis=0)
print("\nSimulated annual returns for each stock:")
for i in range(10):
    print(f"Stock {i+1} annual return: {annual_stock_returns[i]:.2f}%")

print(f"\nEqual-weighted portfolio annual return: {np.mean(annual_stock_returns):.2f}%")
print(f"This should be similar to but not exactly the same as the compounded total return of {total_return:.2f}%")
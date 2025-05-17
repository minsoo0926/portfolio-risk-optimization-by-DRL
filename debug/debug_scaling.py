import numpy as np
import pandas as pd
from generate_scenario import generate_scenario
import matplotlib.pyplot as plt

# Generate sample data
print("Generating sample data...")
data = generate_scenario(10, seed=42)

# Extract returns data (1st column for each stock)
returns_indices = np.arange(0, 40, 4)

# Three approaches to portfolio returns
print("\nComparing three approaches to handling returns:")
print("1. Using raw data (as stored in CSV) / 100")
print("2. Using raw data without dividing by 100")
print("3. Using log returns instead of simple returns")

# Set up portfolio
weights = np.array([0.1] * 10)  # Equal weighting
initial_capital = 10000
days = len(data) - 1  # Skip first day

# Store results for each approach
portfolio_values1 = [initial_capital]  # Approach 1: Dividing by 100
portfolio_values2 = [initial_capital]  # Approach 2: Not dividing by 100
portfolio_values3 = [initial_capital]  # Approach 3: Using log returns

# Simulation
for day in range(days):
    # Get returns for the day
    stock_returns_raw = data.iloc[day+1, returns_indices+1].values
    
    # Approach 1: Divide by 100 to convert percentages to decimals
    stock_returns1 = stock_returns_raw / 100.0
    portfolio_return1 = np.sum(weights * stock_returns1)
    portfolio_values1.append(portfolio_values1[-1] * (1 + portfolio_return1))
    
    # Approach 2: Use raw percentages without dividing
    stock_returns2 = stock_returns_raw
    portfolio_return2 = np.sum(weights * stock_returns2)
    portfolio_values2.append(portfolio_values2[-1] * (1 + portfolio_return2))
    
    # Approach 3: Convert to log returns (log(1+r))
    log_returns = np.zeros(len(stock_returns_raw))
    for i, r in enumerate(stock_returns_raw):
        log_returns[i] = np.log(1 + r/100.0)
    portfolio_return3 = np.sum(weights * log_returns)
    # Convert back to simple return for updating portfolio value
    simple_return3 = np.exp(portfolio_return3) - 1
    portfolio_values3.append(portfolio_values3[-1] * (1 + simple_return3))
    
    # Print first few days
    if day < 5:
        print(f"\nDay {day+1}:")
        print(f"  Raw stock returns: {[round(r, 4) for r in stock_returns_raw]}")
        print(f"  Approach 1 - Portfolio return: {portfolio_return1:.6f} ({portfolio_return1*100:.4f}%)")
        print(f"  Approach 2 - Portfolio return: {portfolio_return2:.6f} ({portfolio_return2:.4f}%)")
        print(f"  Approach 3 - Log return: {portfolio_return3:.6f}, Simple return: {simple_return3:.6f} ({simple_return3*100:.4f}%)")

# Calculate performance metrics
total_return1 = (portfolio_values1[-1] / initial_capital - 1) * 100
total_return2 = (portfolio_values2[-1] / initial_capital - 1) * 100
total_return3 = (portfolio_values3[-1] / initial_capital - 1) * 100

# Annualized returns
trading_days_per_year = 252
annualized_return1 = ((1 + total_return1/100) ** (trading_days_per_year/days) - 1) * 100
annualized_return2 = ((1 + total_return2/100) ** (trading_days_per_year/days) - 1) * 100
annualized_return3 = ((1 + total_return3/100) ** (trading_days_per_year/days) - 1) * 100

print("\n===== Performance Comparison =====")
print(f"Approach 1 (Divide by 100):")
print(f"  Total return: {total_return1:.2f}%")
print(f"  Annualized return: {annualized_return1:.2f}%")
print(f"  Final portfolio value: ${portfolio_values1[-1]:.2f}")

print(f"\nApproach 2 (Raw percentages):")
print(f"  Total return: {total_return2:.2f}%")
print(f"  Annualized return: {annualized_return2:.2f}%")
print(f"  Final portfolio value: ${portfolio_values2[-1]:.2f}")

print(f"\nApproach 3 (Log returns):")
print(f"  Total return: {total_return3:.2f}%")
print(f"  Annualized return: {annualized_return3:.2f}%") 
print(f"  Final portfolio value: ${portfolio_values3[-1]:.2f}")

# Create a comparison chart
plt.figure(figsize=(10, 6))
plt.plot(portfolio_values1, label='Approach 1: Divide by 100')
plt.plot(portfolio_values2, label='Approach 2: Raw Percentages', linestyle='--')
plt.plot(portfolio_values3, label='Approach 3: Log Returns', linestyle=':')
plt.title('Portfolio Value Comparison')
plt.xlabel('Trading Day')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.savefig('scaling_comparison.png')
print("\nComparison chart saved as 'scaling_comparison.png'")

# Analysis of correct scaling
print("\n===== Analysis of Correct Approach =====")
print("Approach 1 (Divide by 100) is correct because:")
print("1. The data is stored in percentage form (e.g., 1.5 means 1.5%)")
print("2. For portfolio calculations, we need returns in decimal form (e.g., 0.015)")
print("3. generate_scenario.py multiplies returns by 100 (lines 96-97)")
print("4. train.py correctly divides by 100 to convert back to decimal (lines 84, 88)")

# Verify by looking at raw returns
mean_raw_returns = []
for i in range(10):
    col_idx = returns_indices[i] + 1  # +1 because our indices are 0-based
    mean_raw_returns.append(np.mean(data.iloc[1:, col_idx].values))

mean_raw_return = np.mean(mean_raw_returns)
print(f"\nMean daily return in raw data: {mean_raw_return:.4f}%")
print(f"This converts to a decimal return of: {mean_raw_return/100:.6f}")
print(f"Annualized return from mean daily: {(1 + mean_raw_return/100)**252 * 100 - 100:.2f}%")

# Verify with evaluation.py logic
print("\nIn evaluation.py, the daily_return value is already in decimal form")
print("This is correct because train.py's step() method returns stock_returns / 100.0")
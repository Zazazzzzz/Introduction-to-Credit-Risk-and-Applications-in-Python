import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

### Plot: Introduction_stock_returns
# Description of the Figure
# -------------------------
#
# This figure displays simulated daily returns for four stocks (A, B, C, D)
# over 5,000 periods in a single 2x2 layout.
#
# Design of the simulation:
#
# - Stock A and Stock B:
#   Very highly correlated (ρ ≈ 0.9). Their return paths move closely
#   together, illustrating strong dependence.
#
# - Stock C:
#   Weakly correlated with A and B (ρ ≈ 0.2). Its return dynamics are
#   largely independent relative to A and B.
#
# - Stock D:
#   Moderately correlated with A and B, but subject to rare, large negative
#   shocks (2% probability). This creates pronounced downside tail events.
#
# Narrative purpose:
#
# The figure illustrates three key ideas:
#
# 1. High correlation (A & B) leads to visibly similar movements.
# 2. Low correlation (C) reduces co-movement.
# 3. Tail amplification (D) can dominate extreme risk even if average
#    volatility appears similar.
#
# This helps distinguish:
# - Linear correlation
# - Dependence structure
# - Tail risk amplification

# Number of observations (days)
T = 5000

# Correlation design:
# A & B: very correlated (0.9)
# C: weakly correlated with A/B (0.2)
# D: moderately correlated with A/B (0.5), weakly with C (0.3)
corr_matrix = np.array([
    [1.0, 0.9, 0.2, 0.5],
    [0.9, 1.0, 0.2, 0.5],
    [0.2, 0.2, 1.0, 0.3],
    [0.5, 0.5, 0.3, 1.0]
])

# Volatilities (std devs)
std_devs = np.array([0.02, 0.02, 0.02, 0.025])
cov_matrix = np.outer(std_devs, std_devs) * corr_matrix

# Simulate baseline (multivariate normal) returns
mu = np.array([0.0005, 0.0005, 0.0005, 0.0005])
returns = np.random.multivariate_normal(mean=mu, cov=cov_matrix, size=T)
df = pd.DataFrame(returns, columns=["A", "B", "C", "D"])

# Tail amplification: rare large negative shocks to D (e.g., 2% of days)
shock_prob = 0.02
shock = np.random.binomial(1, shock_prob, size=T)
df["D"] += shock * (-0.15)

# ---- Plot: one figure with 4 panels (A, B, C, D) ----
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
axes = axes.flatten()

for ax, col in zip(axes, ["A", "B", "C", "D"]):
    ax.plot(df[col].values)
    ax.set_title(f"Stock {col} Returns")
    ax.set_xlabel("Time")
    ax.set_ylabel("Return")

plt.tight_layout()
plt.show()

# Optional: check the empirical correlation matrix
print("Empirical correlation matrix:")
print(df.corr())
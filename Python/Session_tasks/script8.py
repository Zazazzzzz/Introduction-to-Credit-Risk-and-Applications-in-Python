# ---------------------------------------------------------------
# Script Name: Task12
# Author: Hongyi Shen
# Description: Section 8
# ----------------------------------------------------------------

##### Task12: quantile as credit risk measurement
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bond_value = pd.read_excel('Bond_value.xlsx')
ratings = bond_value['Rating']
probabilities = bond_value['Probability (%)']  # in %
bond_values = bond_value['Bond Value']

# Convert probabilities to decimals
probabilities = np.array(probabilities)

# Calculate the cumulative distribution function (CDF)
sorted_indices = np.argsort(bond_values)  # Sort bond values in ascending order
sorted_values = np.array(bond_values)[sorted_indices]
sorted_probs = np.array(probabilities)[sorted_indices]
cdf = np.cumsum(sorted_probs)

# Plot the CDF
plt.figure(figsize=(10, 6))
plt.plot(sorted_values, cdf, marker='o', linestyle='-', color='purple', label='CDF')
plt.xlabel('Bond Value + Coupon ($)')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Distribution Function (CDF) of Bond Values')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Highlight a specific quantile level (e.g., 95%)
quantile_level = 0.01
quantile_value = np.interp(quantile_level, cdf, sorted_values)
plt.axvline(x=quantile_value, color='grey', linestyle='-', label=f'{quantile_level:.0%} Quantile: {quantile_value:.2f}')

plt.legend()
plt.tight_layout()
plt.show()




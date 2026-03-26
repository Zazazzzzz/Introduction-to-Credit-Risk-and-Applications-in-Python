# ---------------------------------------------------------------
# Script Name: Task8, Task9, Task10
# Author: Hongyi Shen
# Description: Section 6
# ----------------------------------------------------------------

### Task8
# Parameters
num_obligors_group1 = 50  # Number of obligors in group 1
num_obligors_group2 = 50  # Number of obligors in group 2
pd_group1 = 0.005          # Probability of default for group 1
pd_group2 = 0.002          # Probability of default for group 2
lgd_group1 = 0.004          # Loss given default for group 1 * w_i
lgd_group2 = 0.003          # Loss given default for group 2 * w_i

# expected loss
expected_loss = num_obligors_group1 * pd_group1 * lgd_group1 + num_obligors_group2 * pd_group2 * lgd_group2

# Loop through each obligor and calculate expected loss
expected_loss = 0
for i in range(num_obligors_group1 + num_obligors_group2):
    if i < num_obligors_group1:  # First group
        expected_loss += pd_group1 * lgd_group1
    else:  # Second group
        expected_loss += pd_group2 * lgd_group2

# Output the result
print(f"The expected loss of the portfolio is: {expected_loss:.4f}")

# Simulation for 2000 times
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

num_simulations = 2000  # Number of simulations

non_default_color = 'silver'
default_color = 'thistle'

# Set random seed for reproducibility
np.random.seed(42)

# Simulate the defaults and calculate total loss for each trial
total_losses = []

for _ in range(num_simulations):
    # Simulate defaults for group 1 and group 2
    defaults_group1 = np.random.binomial(1, pd_group1, num_obligors_group1)
    defaults_group2 = np.random.binomial(1, pd_group2, num_obligors_group2)

    # Calculate the total loss for this trial
    loss_group1 = np.sum(defaults_group1) * lgd_group1
    loss_group2 = np.sum(defaults_group2) * lgd_group2
    total_loss = loss_group1 + loss_group2

    # Append the total loss to the list of total losses
    total_losses.append(total_loss)

# Plot the density of total losses across the 1000 simulations
plt.figure(figsize=(10, 6))
plt.hist(total_losses, bins=30, edgecolor=non_default_color, alpha=0.90, density=True, color=default_color, label="Histogram")
sns.kdeplot(total_losses, color=non_default_color, linewidth=2, label="KDE Curve")
# Seaborn does not create a new figure by default. Instead, it adds the KDE curve to the existing Matplotlib figure.
# KDE stands for Kernel Density Estimate. A KDE curve is a smoothed estimate of the probability density function (PDF) of a continuous variable.
# A histogram shows the frequency of data points in bins.
# A KDE curve takes that idea and smooths it into a continuous line that represents how your data is distributed.

# plt.figure() creates a new figure (canvas), ensuring that the next plot is drawn on a separate window rather than overlapping with the previous one.
# plt.clf() clears the current figure without creating a new one.
# plt.close() completely closes the current figure, freeing up memory and preventing figure accumulation,
# especially when generating multiple plots inside a loop.
# If you don’t close figures in such cases, excessive memory usage or too many open figure windows can cause issues.

# density = count/(total samples * bin width)
# If the bin width is small, the y-axis values become larger to keep the total area equal to 1.

# Add title and labels
plt.title('Density Plot of Total Loss Across 2000 Simulations')
plt.xlabel('Total Loss')
plt.ylabel('Density (Probability)')

# Display the grid
plt.grid(True, axis='y', linestyle='--', color=non_default_color, linewidth=0.5)
plt.legend()

# Show the plot
plt.show()

# Output some statistics
print(f"Mean total loss: {np.mean(total_losses):.2f}")
print(f"Standard deviation of total loss: {np.std(total_losses):.2f}")

###### Task9: VaR with historical data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Read the file
loss = pd.read_excel(os.path.join('credit risk modeling', 'loss_data.xlsx'))
loss['Loss Percentage'] = loss['Loss']/loss['Total']

# Compute VaR at 5% and 1% confidence levels
var_95 = np.percentile(loss["Loss Percentage"], 95)
var_99 = np.percentile(loss["Loss Percentage"], 99)

# Compute expected loss (mean)
expected_loss = loss["Loss Percentage"].mean()

# Plot the density distribution
plt.figure(figsize=(8, 5))
sns.kdeplot(loss["Loss Percentage"], fill=True, color=non_default_color, alpha=0.5)

# VaR vertical lines
plt.axvline(var_95, color="black", linestyle="--", label=f"VaR 95%: {var_95:.2f}")
plt.axvline(var_99, color="red", linestyle="--", label=f"VaR 99%: {var_99:.2f}")
plt.axvline(expected_loss, color='purple', linestyle="--", label=f"Expected Loss (Mean): {expected_loss:.2f}")

# Labels and title
plt.xlabel("Loss")
plt.ylabel("Density")
plt.title("Density Distribution of Losses")
plt.grid(True, axis='y', linestyle='--', color=default_color, linewidth=0.5)
plt.legend(loc='upper left')
plt.show()

#### Task10:  VaR simulation under one factor model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Set random seed for reproducibility
np.random.seed(42)

# Model parameters
n_obligors = 100  # Number of obligors in the portfolio
default_prob = 0.03  # Default probability for each obligor (3%)
asset_corr = 0.5  # Asset correlation
LGD = 0.004  # Loss Given Default * w
EAD = 1  # Exposure at Default
n_simulations = 2000  # Number of simulations

# Function to simulate a single portfolio loss
def simulate_loss(num_trials, num_obligors, default_prob, correlation, LGD, EAD):
    # Generate common factor F
    F = np.random.normal(0, 1, num_trials)
    # Generate idiosyncratic factors epsilon_i
    epsilon = np.random.normal(0, 1, (num_trials, num_obligors))
    # Compute latent variables X_i
    X = correlation * F[:, np.newaxis] + np.sqrt(1 - correlation ** 2) * epsilon
    # Determine defaults
    threshold = norm.ppf(default_prob)
    defaults = X < threshold
    # Calculate the losses
    total_loss = (defaults * LGD * EAD).sum(axis=1)
    return total_loss

# Run simulations and store the losses
losses = simulate_loss(n_simulations, n_obligors, default_prob, asset_corr, LGD, EAD)

# Calculate VaR at 99.9% percentile
VaR_99_9 = np.percentile(losses, 99.9)

# Plot the loss distribution
sns.kdeplot(losses, fill=True, color=default_color, alpha=0.5)
plt.axvline(VaR_99_9, color=non_default_color, linestyle='--', label=f'VaR (99.9%) = {VaR_99_9:.2f}')
plt.title("Simulated Loss Distribution")
plt.xlabel("Total Loss")
plt.ylabel("Density")
plt.grid(True, axis='y', linestyle='--', color=non_default_color, linewidth=0.5)
plt.legend()
plt.show()

# Optionally, display some statistics
print(f"VaR at 99.9%: {VaR_99_9}")

# Calculate
p_99_9 = norm.cdf ((norm.ppf(default_prob) - asset_corr * norm.ppf(1-0.999))/ np.sqrt(1 - asset_corr ** 2))
VaR_99_9_cal = n_obligors * LGD * EAD * p_99_9
print(f'Calculated VaR at 99.9%: {VaR_99_9_cal}')
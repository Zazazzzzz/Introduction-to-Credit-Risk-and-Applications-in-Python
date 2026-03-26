# ---------------------------------------------------------------
# Script Name: Task1, Task2
# Author: Hongyi Shen
# Description: Section 3
# ----------------------------------------------------------------

#### Fig 2
import numpy as np
import matplotlib.pyplot as plt

# Define colors using RGB values (normalized between 0 and 1)
non_default_color = (0.58, 0.58, 0.58)
default_color = (0.82, 0.58, 0.58)      # Example: x/255 normalized

# Parameters
default_prob = 0.01  # 1% default probability
times = 1000  # Number of periods to simulate

# For reproducibility
np.random.seed(42)

# Simulation: 1 = default, 0 = no default
simulations = np.random.binomial(1, default_prob, times)

# Bernoulli Trial: A single experiment or trial with only two possible outcomes — usually called "success" and "failure".
# Example: flipping a coin (heads = success, tails = failure)
# Binomial Distribution answers:
# If I repeat this trial n times, and each time the probability of success is p
# what is the probability of getting exactly k successes?
#####
# Parameters:
# * **n**: number of trials
# * **p**: probability of success in a single trial
# * **k**: number of successes (the outcome we're interested in)
# ### Probability Mass Function (PMF):
# The probability of getting exactly $k$ successes in $n$ trials:
# P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
# Where:
# * $\binom{n}{k}$ is the number of ways to choose $k$ successes from $n$ trials (combinations)
# * $p^k$ is the probability of $k$ successes
# * $(1-p)^{n-k}$ is the probability of $n-k$ failures

# Count occurrences
default_count = np.sum(simulations)
non_default_count = times - default_count

# Plotting
plt.figure(figsize=(10, 6))

# Bar Chart of Defaults and Non-Defaults
plt.bar(['Non-Default', 'Default'], [non_default_count, default_count], color=[non_default_color, default_color], width=0.5)
# | Argument    | Description                                                               |
# | ----------- | ------------------------------------------------------------------------- |
# | `x`         | Positions of the bars on the x-axis (e.g. categories or numerical bins)   |
# | `height`    | Heights of the bars (e.g. values or frequencies)                          |
# | `width`     | Width of each bar (default is 0.8)                                        |
# | `align`     | `'center'` (default) or `'edge'` — controls bar alignment relative to `x` |
# | `color`     | Fill color of the bars (can be a single color or list)                    |
# | `edgecolor` | Color of the bar borders                                                  |
# | `label`     | Label for the legend                                                      |
# A width of 1.0 means the bar spans the entire distance between adjacent positions on the x-axis.
# A width of 0.5 means the bar takes up half the space between adjacent x-axis positions.
plt.ylabel('Frequency')
plt.title('Simulation of Default Outcomes')
plt.show()
# plt.savefig('default simulation_one obligor.png')
# plt.close()

#### Fig 3
# Parameters
num_obligors = 100       # Number of obligors in the portfolio
default_prob = 0.01      # Default probability of each obligor
num_trials = 1000        # Number of simulation trials

# Set random seed for reproducibility
np.random.seed(42)

# simulating the total number of defaults out of num_obligors for each of num_trials.
# The output will be a 1D array of length num_trials,
# where each element is the number of defaults in that trial (but not which obligors defaulted).
defaults = np.random.binomial(n=num_obligors, p=default_prob, size=num_trials)


# Simulate defaults for each trial: each obligor has a 1% chance of default
# Each row corresponds to a trial, and each column corresponds to an obligor
defaults = np.random.binomial(1, default_prob, size=(num_trials, num_obligors))
# using n = 1, meaning each trial is a single Bernoulli trial (i.e., the result is either 0 or 1 — default or no default).
# repeat this for each obligor and each trial.
# The resulting shape is a 2D array with:
# num_trials rows (e.g. simulations or time steps),
# num_obligors columns (each obligor gets a Bernoulli outcome per trial).

# Count the number of defaults in each trial
default_counts = np.sum(defaults, axis=1)


# Plot histogram of the number of defaults as a density function across the 1000 trials
plt.figure(figsize=(10, 6))
plt.hist(default_counts, bins=range(0, num_obligors+1), color=default_color, edgecolor='#CCCCCC', alpha=0.75, density=True)
# | Argument    | Description                                                              |
# | ----------- | ------------------------------------------------------------------------ |
# | `data`      | The numeric data you want to plot                                        |
# | `bins`      | Number of bins (intervals) or a sequence defining bin edges              |
# | `range`     | Tuple `(min, max)` defining the range of values to include               |
# | `density`   | If `True`, scales the histogram to form a probability density (area = 1) |
# | `color`     | Fill color of the bars                                                   |
# | `edgecolor` | Color of the edges of the bars                                           |
# | `label`     | Label used in the legend                                                 |
# range(0, num_obligors+1) creates a sequence of integers from 0 to num_obligors (which is 100 in this case),
# representing the possible number of defaults in a trial (from 0 to 100).
# By default, plt.hist() shows the raw frequency (count) of data points in each bin.
# Setting density=True scales the height of the bars so that the total area under the histogram equals 1.
# This makes the y-axis represent the probability (i.e., the density) of a given number of defaults occurring in a trial.
# alpha=0.9 means the bars will be 90% opaque (i.e., 10% transparent).
# A higher value (e.g., alpha=1.0) would make the bars completely opaque.
plt.xlim(-3, 30)
plt.title('Density Function of the Number of Defaults Across 1000 Trials')
plt.xlabel('Number of Defaults')
plt.ylabel('Density (Probability)')
plt.grid(True, axis='y', linestyle='--', color=non_default_color, linewidth=0.5)
plt.show()
# plt.savefig('Simulation of Default Outcomes of a Portfolio With No Interdependence.png')
# plt.close()
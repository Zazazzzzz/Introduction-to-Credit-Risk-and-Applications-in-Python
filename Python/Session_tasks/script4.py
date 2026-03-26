# ---------------------------------------------------------------
# Script Name: Task4, Copula
# Author: Hongyi Shen
# Description: Section 4
# ----------------------------------------------------------------


import numpy as np
from scipy.stats import norm, multivariate_normal

# === Normal Distribution ===

# pdf(x): Probability Density Function at point x
# cdf(x): Cumulative Distribution Function at point x
# ppf(q): Percent-Point Function (inverse of CDF) at quantile q
# np.random.norm(): Simulation

# Task 4
# Given parameters
rho = 0.5  # Asset correlation
p1 = 0.005  # Default probability for obligor 1
p2 = 0.01   # Default probability for obligor 2

# Step 1: Compute the inverse standard normal CDF (quantile function)
d1 = norm.ppf(p1)  # Inverse CDF for p1
d2 = norm.ppf(p2)  # Inverse CDF for p2

# Step 2: Compute the joint probability of default using the bivariate normal CDF
# The bivariate normal CDF is computed using the multivariate_normal.cdf function
mean = [0, 0]  # Mean vector for standard normal distribution
cov = [[1, rho], [rho, 1]]  # Covariance matrix with correlation rho

# Compute P(Y1 = 1, Y2 = 1) = Phi2(d1, d2; rho)
joint_prob_default = multivariate_normal.cdf([d1, d2], mean=mean, cov=cov)

# Step 3: Compute the default correlation
# Numerator: P(Y1 = 1, Y2 = 1) - p1 * p2
numerator = joint_prob_default - (p1 * p2)

# Denominator: sqrt(p1 * (1 - p1) * p2 * (1 - p2))
denominator = np.sqrt(p1 * (1 - p1) * p2 * (1 - p2))

# Default correlation
default_correlation = numerator / denominator

# Output the results
print(f"Joint Probability of Default (P(Y1 = 1, Y2 = 1)): {joint_prob_default:.6f}")
print(f"Default Correlation: {default_correlation:.6f}")


### Extension 1: Norm Margins with Gaussian copula and t copula
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import invgamma

np.random.seed(42)

# Parameters
rho = 0.7
df = 4  # degrees of freedom for t-copula
n = 5000

# Correlation matrix
corr = np.array([[1, rho],
                 [rho, 1]])

# 1. Multivariate normal
mean = [0, 0]
mvnorm = stats.multivariate_normal(mean=mean, cov=corr)
x_normal = mvnorm.rvs(size=n)

# 2. t-Copula with normal marginals
# Step 1: Generate samples from multivariate t
z = np.random.multivariate_normal(mean=[0, 0], cov=corr, size=n)
chi2 = np.random.chisquare(df, size=(n, 1))
t_samples = z / np.sqrt(chi2 / df)  # shape: (n, 2)

# Step 2: Convert to uniform using t CDF
u = stats.t.cdf(t_samples, df=df)

# Step 3: Apply inverse normal CDF (normal marginals)
x_tc_normal = stats.norm.ppf(u)

# 3. Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(x_normal[:, 0], x_normal[:, 1], alpha=0.3, color='silver', edgecolor='silver', s=10)
axes[0].set_title('Multivariate Normal (ρ = 0.7)')
axes[0].axis('equal')
axes[1].scatter(x_tc_normal[:, 0], x_tc_normal[:, 1], alpha=0.3, color='thistle', edgecolor='thistle', s=10)
axes[1].set_title('t-Copula with Normal Marginals (df = 4)')
axes[1].axis('equal')
plt.tight_layout()
plt.show()

### Extension 2: t-copula and Gaussian copula

# Gaussian copula
def gaussian_copula(rho, size=2000):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    normal_samples = np.random.multivariate_normal(mean, cov, size)
    uniform_samples = stats.norm.cdf(normal_samples)  # Convert to U(0,1)
    return uniform_samples

# t-Copula
def t_copula(rho, df, size=2000):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    normal_samples = np.random.multivariate_normal(mean, cov, size)
    W_samples = invgamma.rvs(a=df / 2, scale=df / 2, size=size)
    t_samples = np.sqrt(W_samples[:, np.newaxis]) * normal_samples
    uniform_samples = stats.t.cdf(t_samples, df)  # Convert to U(0,1)
    return uniform_samples

# Parameters
rho = 0.7  # Correlation for both copulas
df = 3     # Degrees of freedom for t-copula

# Generate samples
gaussian_samples = gaussian_copula(rho)
t_samples = t_copula(rho, df)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot for Gaussian copula
axes[0].scatter(gaussian_samples[:, 0], gaussian_samples[:, 1], alpha=0.5, color='thistle')
axes[0].set_title('Gaussian Copula (rho = 0.7)')
axes[0].set_xlabel('$Phi(X_1)$')
axes[0].set_ylabel('$Phi(X_2)$')

# Scatter plot for t-copula
axes[1].scatter(t_samples[:, 0], t_samples[:, 1], alpha=0.5, color='silver')
axes[1].set_title('t-Copula (rho = 0.7, df = 3)')
axes[1].set_xlabel('$t(X_1)$')
axes[1].set_ylabel('$t(X_2)$')

plt.tight_layout()
plt.show()


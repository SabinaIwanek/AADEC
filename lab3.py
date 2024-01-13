import numpy as np
import matplotlib.pyplot as plt

# Variant
num_samples = 2000  # N
A_values = 300.25  # A
B_values = 299.75  # B
f_values = 300  # Frequency

# Function to generate random signals
def generate_signal(k):
    return A_values * np.cos(2 * np.pi * f_values / k) + B_values * np.random.normal(0, 1, 100)

# Estimate the linear mean as ensemble average
linear_mean = np.mean(np.array([generate_signal(k) for k in range(1, num_samples + 1)]), axis=0)

# Estimate the linear mean and squared linear mean
linear_mean_squared = np.mean(linear_mean ** 2)

# Estimate the quadratic mean and variance
quadratic_mean = np.sqrt(np.mean(np.array([generate_signal(k) ** 2 for k in range(1, num_samples + 1)]), axis=0))
variance = np.var(linear_mean)

# Plot 1-4 graphically
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(linear_mean)
plt.title('Linear Mean')

plt.subplot(2, 2, 2)
plt.plot(linear_mean_squared)
plt.title('Squared Linear Mean')

plt.subplot(2, 2, 3)
plt.plot(quadratic_mean)
plt.title('Quadratic Mean')

plt.subplot(2, 2, 4)
plt.plot(variance)
plt.title('Variance')

plt.tight_layout()
plt.show()

# Estimate and plot the auto-correlation function (ACF)
acf = np.correlate(linear_mean, linear_mean, mode='full')
plt.figure(figsize=(8, 6))
plt.plot(acf[len(acf)//2:])
plt.title('Auto-correlation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.show()
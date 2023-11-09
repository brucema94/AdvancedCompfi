import numpy as np
import matplotlib.pyplot as plt

# Hull-White model parameters
a = 0.1  # Mean reversion speed
sigma = 0.02  # Volatility
theta = 0.03  # Long-term mean interest rate

# IRS parameters
notional = 1000000  # Notional amount in the IRS
fixed_rate = 0.035  # Fixed interest rate in the IRS
maturity = 1  # Maturity of the IRS (10 years)
payment_frequency = 0.25  # 3M IRS

# Simulation parameters
T = 1  # 10 years IRS
num_steps = 252  # Frequency is set to daily to make sure instantaneous interest rate has a sufficiently small interval
num_paths = 10  # Number of simulation paths

# Generate random normal samples
np.random.seed(123)  # Set a seed for reproducibility
dt = T / num_steps  # Time step
dW = np.random.normal(0, np.sqrt(dt), (num_paths, num_steps))

# Initialize arrays to store interest rate paths and PE values
interest_rate_paths = np.zeros((num_paths, num_steps + 1))
interest_rate_paths[:, 0] = theta  # Initial interest rate
pe_values = np.zeros(num_steps + 1)

# Simulate interest rate paths using Euler discretization and calculate PE
for i in range(1, num_steps + 1):
    dR = (theta - a * interest_rate_paths[:, i - 1]) * dt + sigma * dW[:, i - 1]
    interest_rate_paths[:, i] = interest_rate_paths[:, i - 1] + dR

    t = i * dt
    expected_rate = np.mean(interest_rate_paths[:, i])
    npv = notional * (fixed_rate * payment_frequency * np.exp(-expected_rate * t) - expected_rate * payment_frequency * np.exp(-expected_rate * t))

    # Calculate Positive Exposure (PE) at each time step
    pe_values[i] = max(0, npv)

# Plot Positive Exposure (PE) over time
plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_steps + 1) * dt, pe_values, label="Positive Exposure (PE)")
plt.xlabel("Time")
plt.ylabel("Positive Exposure")
plt.title("Positive Exposure (PE) Over Time")
plt.legend()
plt.grid(True)
plt.show()

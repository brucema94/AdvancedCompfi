import numpy as np
import matplotlib.pyplot as plt


# Hull-White model parameters
a = 0.1  # Mean reversion speed
sigma = 0.02  # Volatility
theta = 0.03  # Long-term mean interest rate

# IRS parameters
notional = 1000000  # Notional amount in the IRS
fixed_rate = 0.035  # Fixed interest rate in the IRS
maturity = 10  # Maturity of the IRS (10 year)
payment_frequency = 0.25  # 3M IRS

# Simulation parameters
T = 1  # 10y IRS 
num_steps = 252  # Frequency is set to daily to make sure instantanous interest rate has sufficiently small interval
num_paths = 10  # Number of simulation paths

# Generate random normal samples
np.random.seed(123)  # Set a seed for reproducibility
dt = T/num_steps  # Time step
dW = np.random.normal(0, np.sqrt(dt), (num_paths, num_steps))

# Initialize arrays to store interest rate paths
interest_rate_paths = np.zeros((num_paths, num_steps + 1))
interest_rate_paths[:, 0] = theta  # Initial interest rate
fixed_leg_present_values = np.zeros(num_steps+1)
floating_leg_present_values = np.zeros(num_steps+1)
npv_values = np.zeros(num_steps+1)

# Simulate interest rate paths using Euler discretization
for i in range(1, num_steps + 1):
    dR = (theta - a * interest_rate_paths[:, i - 1]) * dt + sigma * dW[:, i - 1]
    interest_rate_paths[:, i] = interest_rate_paths[:, i - 1] + dR
    # Calculate the IRS price for each path
    t = i * dt
    if t % payment_frequency == 0:  # Check if it's a coupon payment date
        expected_rate = np.mean(interest_rate_paths[:, i])
        fixed_leg_present_values[i] = fixed_leg_present_values[i-1] + notional * fixed_rate * payment_frequency * np.exp(-expected_rate * t)
        floating_leg_present_values[i] = floating_leg_present_values[i-1] + notional * expected_rate * payment_frequency * np.exp(-expected_rate * t)
        npv_values[i] = floating_leg_present_values[i] - fixed_leg_present_values[i]

positive_exposure = np.maximum(npv_values, 0)
epe = np.mean(positive_exposure)

print(f"EPE (Expected Positive Exposure): {epe}")

plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_steps + 1) * dt, positive_exposure, label="Positive Exposure (PE)")
plt.xlabel("Time")
plt.ylabel("Positive Exposure")
plt.title("Positive Exposure (PE) Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Plot all interest rate paths together
#plt.figure(figsize=(10, 6))
#for i in range(num_paths):
#    plt.plot(np.linspace(0, T, num_steps + 1), interest_rate_paths[i], label=f'Path {i + 1}')

#plt.xlabel('Time')
#plt.ylabel('Interest Rate')
#plt.title('Simulated Interest Rate Paths (Hull-White Model)')
#plt.legend()
#plt.grid(True)
#plt.show()
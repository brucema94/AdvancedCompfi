import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate


# Parameters
dt = 0.001
num_paths = 1
monte_carlo_amount = 1000
maturity=2
vol=1
# initial interest rate is set to 0 for now. 
r0 = 0.1
a=0.2
notional=100
Coupon_frequency = 0.25


# We obtain historical data through interpolation

###############################################################################
#Here we load in the data. The filepath is of course different per user
loaded_data_discount = np.load('C:/Users/Trist/Master/Jaar 2/Semester 1/CompFi/CompFi/std_tenor_eonia.npy', allow_pickle=True)
loaded_data_forward = np.load('C:/Users/Trist/Master/Jaar 2/Semester 1/CompFi/CompFi/std_tenor_eur3m.npy', allow_pickle=True)

# Define a function for linear interpolation
def linear_interpolation(x, x_values, y_values):
    f = interpolate.interp1d(x_values, y_values, kind='linear')
    return f(x)


num_intervals = int(maturity / dt)

Intervals = np.arange(0.0, maturity+dt, dt)

# taking a single day entry as our t0
Test_discount = loaded_data_discount[:20]
Test_forward = loaded_data_forward[:20]

# split columns for interpolation 
dates = Test_discount[:, 0]
tenors = Test_discount[:, 1]
interest_rates_discount = Test_discount[:, 2]
interest_rates_forward = Test_forward[:, 2]
#tenors = loaded_data_forward[:, 1]

# discount rate interpolation, final product contains all P(0,t)
interpolated_rates_discount = [linear_interpolation(interval, tenors, interest_rates_discount) for interval in Intervals]
result_list = list(zip(Intervals, interpolated_rates_discount))
result_list_float = [(interval, float(rate)) for interval, rate in result_list]
Discount_factor_M = [(t, np.exp(-r * t)) for t, r in result_list_float]

# Compute forward rates, final product contains f(0,t)
interpolated_rates_forward = [linear_interpolation(interval, tenors, interest_rates_forward) for interval in Intervals]
result_list_forward = list(zip(Intervals, interpolated_rates_forward))
forward_rates = [(interval, float(rate)) for interval, rate in result_list_forward]


# Discount_factor_M contains tuples of (t, P(0,t)) for all t between 0 and maturity,
# through linear interpolation
#print(Discount_factor_M[-5:]) #Test

# Zip the data in neat one-dimensional arrays.
p_grid = np.zeros(num_intervals+1)
f_grid = np.zeros(num_intervals+1)
for i in range(num_intervals+1):
    p_grid[i] = Discount_factor_M[i][1]
    f_grid[i] = forward_rates[i][1]

#This is the end of the interpolation chunk!
###############################################################################


# These are possible values of T0, T1, ... ,Tm 
# Note: THIS INCLUDES T0
maturity_grid = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

# [0,0.25] to compute the fixed rate
timeGrid = np.linspace(0, maturity_grid[0], num = 1+int(maturity_grid[0]/dt))

# Functions
###############################################################################

# Theta(t) in the Hull-White model, We are under the impression that 
# this needs to be calibrated, we skipped for now
def theta(t):
    return -0.1

def B(s,t):
    return (1/a)*(1-math.e**(-1*a*(s-t)))

# input would be P(0,t), P(o,s) and instantanous forward rate at s
def A(s,t, p_s, p_t, fM):
    return (p_t/p_s)*math.e**(B(s,t)*fM - ((vol**2)/(4*a))*((B(s,t))**2)*(1-math.e**(-2*a*s)))

# Function to simulate N paths of Hull-White and obtain short rate path
# Here note that the expected rate is the sum over all path
def hull_white(num_paths, vol, maturity, a, dt):
    num_steps = int(maturity/dt)
    interest_rate_paths = np.zeros((num_paths, num_steps + 1))
    interest_rate_paths[:, 0] = r0  # Initial interest rate
    expected_shrotrate = []
    dW = np.random.normal(0, np.sqrt(dt), (num_paths, num_steps))
    for i in range(1,num_steps+1):
        dR = (theta(i*dt) - a*interest_rate_paths[:,i-1])*dt + vol*math.sqrt(dt)*dW[:,i-1]
        interest_rate_paths[:,i] = interest_rate_paths[:,i-1] + dR
        expected_shrotrate.append(np.mean(interest_rate_paths[:,i]))
    return interest_rate_paths, expected_shrotrate

# Retrieve the value of P(s,t) based on r(s) from Hull-White
def P(s,t, short_rate, p_s, p_t, fM):
    return A(s,t, p_s, p_t, fM)*math.e**(-1*B(s,t)*(short_rate[int(s/dt)]))

# Determine the instantanous forward rate f(0,s) = - d/dt log(p(0,s))
# via a numerical scheme of the derivative
# this is to be used in calculation of A hence P
def f(s, p_grid):
    return f_grid[int(s/dt)]

# Function to determine the value of a swap at time t, where we make a 
# distintion between t <= T0 and t > T0 
def V(t, maturity_grid, notional):
    sample_paths, short_rate = hull_white(num_paths, vol, maturity_grid[-1], a, dt)
    fixed_grid = []
    # If t < T0 we simply use the formula given in the notes
    if (t < maturity_grid[0]):
        p_t_T_grid = []
        for T in maturity_grid: 
            p_t_T_grid.append(P(t, T, short_rate, p_grid[int(t/dt)], p_grid[int(T/dt)], f(t, p_grid)))
            fixed_grid.append(P(0, T, short_rate, p_grid[int(0/dt)], p_grid[int(T/dt)], f(0, p_grid)))
        fixed = (fixed_grid[0] - fixed_grid[-1])/(Coupon_frequency*(sum(fixed_grid) - fixed_grid[0]))
        temp = fixed*Coupon_frequency*(sum(p_t_T_grid) - p_t_T_grid[0])
        return notional*(p_t_T_grid[0] - p_t_T_grid[-1] - temp), fixed, sample_paths, short_rate
    # Otherwise (t > T0) we make a small change to the formula
    else:
        for T in maturity_grid:
            fixed_grid.append(P(0, T, short_rate, p_grid[int(0/dt)], p_grid[int(T/dt)], f(0, p_grid)))
        fixed = (fixed_grid[0] - fixed_grid[-1])/(Coupon_frequency*(sum(fixed_grid) - fixed_grid[0]))
        return 0, fixed, sample_paths, short_rate
    
# Function to compute net present value
def NPV(t, maturity_grid, notional):
    v, K, sample_paths, short_rate = V(t, maturity_grid, notional) #Fixed rate
    PV_fixed = 0
    p_t_T_grid = []
    forward_grid = []
    for T in [T for T in maturity_grid if t <= T]:
        p_t_T_grid.append(P(t, T, short_rate, p_grid[int(t/dt)], p_grid[int(T/dt)], f(t, p_grid)))
    forward_grid.append((1/Coupon_frequency)*p_t_T_grid[0]*(1/p_t_T_grid[0] - 1))
    for i in range(1,len(p_t_T_grid)):  
        forward_grid.append((1/Coupon_frequency)*p_t_T_grid[i-1]*(p_t_T_grid[i-1]/p_t_T_grid[i] - 1))
    PV_fixed = Coupon_frequency*K*sum(p_t_T_grid)
    PV_float = Coupon_frequency*sum(forward_grid)
    return PV_float - PV_fixed

# function to compute the EPE based on NPV
def EPE(t, maturity_grid, notional):
    return max(NPV(t, maturity_grid, notional), 0)


# Function to determine the value of the option of a swap
# for all t < T0
def swaption_value(maturity_grid, notional):
    # First compute the "par rate" and the stochastic short rate
    t_grid = np.linspace(0, maturity_grid[0]-dt, num = int(Coupon_frequency/dt))
    sample_paths, short_rate = hull_white(num_paths, vol, maturity_grid[-1], a, dt)
    new_short_rate = [short_rate[i] for i in range(len(t_grid))]
    fixed_grid = []
    p_t_T_grid = np.zeros((len(t_grid), len(maturity_grid)))
    value_grid = np.zeros(len(t_grid))
    for i in range(len(maturity_grid)): 
        fixed_grid.append(P(0, maturity_grid[i], short_rate, p_grid[int(0/dt)], p_grid[int(maturity_grid[i]/dt)], f(0, p_grid)))
        for j in range(len(t_grid)):
            p_t_T_grid[j, i] = P(t_grid[j], maturity_grid[i], short_rate, p_grid[int(t_grid[j]/dt)], p_grid[int(maturity_grid[i]/dt)], f(t_grid[j], p_grid))
    K = (fixed_grid[0] - fixed_grid[-1])/(Coupon_frequency*(sum(fixed_grid) - fixed_grid[0]))
    for i in range(len(t_grid)):
        s = (p_t_T_grid[i,0] - p_t_T_grid[i,-1])/(Coupon_frequency*(sum(p_t_T_grid[i]) - p_t_T_grid[i,0]))
        annuity = Coupon_frequency*(sum(p_t_T_grid[i]) - p_t_T_grid[i,0])
        value_grid[i] = notional*annuity*max(s-K, 0)
        #value_grid[i] = notional*annuity*(s-K)
    return value_grid, short_rate, new_short_rate

def monte_carlo_swaption(maturity_grid, notional, monte_carlo_amount):
    t_grid = np.linspace(0, maturity_grid[0]-dt, num = int(Coupon_frequency/dt))
    M = len(t_grid)
    value = np.zeros((monte_carlo_amount, M))
    interest_paths = np.zeros((monte_carlo_amount, M))
    for n in range(monte_carlo_amount):
        value[n], x, interest_paths[n] = swaption_value(maturity_grid, notional)
    # Now take an empirical average
    mean_value = np.zeros(M)
    mean_interest = np.zeros(M)
    for i in range(M):
        mean_value[i] = np.mean(value[:, i])
        mean_interest[i] = np.mean(interest_paths[:, i])
    return value, mean_value, t_grid, interest_paths, mean_interest
        
        
    

###############################################################################
#Set up a time grid
t_begin = 0
t_end = 1
t_grid = np.linspace(t_begin, t_end, num = int(1+(t_end - t_begin)/dt))
show_interest_rate = True # This determines which graph we show

#Now we can visualise the fair price of a swaption at t < T0,
# or we can simulate the interest paths
value, mean, t_grid_swaption, interest_paths, mean_interest = monte_carlo_swaption(maturity_grid, notional, monte_carlo_amount)

#This is the option for showing the swaption value
if not show_interest_rate:
    for n in range(monte_carlo_amount):
        plt.plot(t_grid_swaption, value[n, :])
    plt.plot(t_grid_swaption, mean, label='Empirical average', color='0')
    plt.xlabel('Time (t)')
    plt.ylabel('Value of the swaption')

    plt.xlim(0, maturity_grid[0])
    plt.title('Simulation of swaption values')
    plt.legend()
    plt.show()
#And this the option for the interest rate paths
if show_interest_rate:
    for n in range(monte_carlo_amount):
        plt.plot(t_grid_swaption, interest_paths[n, :])
    plt.plot(t_grid_swaption, mean_interest, label='Empirical average', color='0')
    
    plt.xlabel('Time (t)')
    plt.ylabel('Interest rate')
    plt.xlim(0, maturity_grid[0])
    
    plt.title('Simulation of interest rate values using Hull-White')
    plt.legend()
    plt.show()





        
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import interpolate


# Parameters
dt = 0.01
num_paths = 1
monte_carlo_amount = 1000
maturity=2
vol=0.5
# initial interest rate is set to 0 for now. 
r0 = 0.01
a=1
notional=100
Coupon_frequency = 0.25


# These are possible values of T0, T1, ... ,Tm where Tm = maturity.
# Note: THIS INCLUDES T0
maturity_grid = [k*Coupon_frequency for k in range(1, 1+int(maturity/Coupon_frequency))]


# [0,0.25] to compute the fixed rate
timeGrid = np.linspace(0, maturity_grid[0], num = 1+int(maturity_grid[0]/dt))


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



# Functions
###############################################################################

# Theta(t) in the Hull-White model, We are under the impression that 
# this needs to be calibrated, we skipped for now
def theta(t):
    return 0.01*(t+1)

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


# Now we have functions to determine SWAP value/NPV/EPE


# Function to compute net present value for any t
# given an interest path
def NPV(t, interest_path):
    # First, simulate the short rate and compute the fixed rate
    fixed_grid = []
    p_t_T_grid = []
    sample_paths, short_rate = hull_white(num_paths, vol, maturity_grid[-1], a, dt)
    for T in maturity_grid:
        fixed_grid.append(P(0, T, short_rate, p_grid[int(0/dt)], p_grid[int(T/dt)], f(0, p_grid)))
    fixed = (fixed_grid[0] - fixed_grid[-1])/(Coupon_frequency*(sum(fixed_grid) - fixed_grid[0]))
    # We make a case distinction
    upcoming_maturities = [T for T in maturity_grid if T > t]
    # This is the case t < t0
    if t < maturity_grid[0]:
        for T in maturity_grid: 
            p_t_T_grid.append(P(t, T, short_rate, p_grid[int(t/dt)], p_grid[int(T/dt)], f(t, p_grid)))
        temp = fixed*Coupon_frequency*(sum(p_t_T_grid) - p_t_T_grid[0])
        return notional*(p_t_T_grid[0] - p_t_T_grid[-1] - temp)
    # If we are at T0 or beyond, we make a slight change to the formula
    else:
        # If only Tm is coming up, all payouts are deterministic
        # and our exposure is known
        if len(upcoming_maturities) == 1:
            return 0
        # There are multiple maturities coming up, so there is some
        # undeterministic exposure. The payout at the upcoming maturity is
        # known, so we only need to consider the other payouts. For this,
        # we can use the formula from the syllabus again
        else: 
            for T in upcoming_maturities:
                p_t_T_grid.append(P(t, T, short_rate, p_grid[int(t/dt)], p_grid[int(T/dt)], f(t, p_grid)))
            temp = fixed*Coupon_frequency*(sum(p_t_T_grid) - p_t_T_grid[0])
            return notional*(p_t_T_grid[0] - p_t_T_grid[-1] - temp)
            

            
# Graph the value of a swap until time T
# N is the amount
def graph_swap_value(N, see_all_paths):
    T = maturity_grid[-1]-dt
    t_grid = np.linspace(0, T, num=int(1+(T/dt)))
    swap_grid = np.zeros((N, len(t_grid)))
    mean_grid = np.zeros(len(t_grid))
    # First compute N paths from 0 to T
    # where per path we use one interest rate path
    for i in range(N):
        interest_path, short_rate = hull_white(num_paths, vol, maturity, a, dt)
        for j in range(len(t_grid)):
            swap_grid[i,j] = NPV(t_grid[j], interest_path)
    # Next compute Monte Carlo average
    for k in range(len(t_grid)):
        mean_grid[k] = np.mean(swap_grid[:, k])
    # If you want to see all paths
    if see_all_paths:
        for i in range(N):
            plt.plot(t_grid, swap_grid[i, :])
    # In any case we graph the average
    plt.plot(t_grid, mean_grid, label='Empirical average', color='0')
    plt.xlabel('Time (t)')
    plt.ylabel('Net Present Value (NPV)')
    plt.legend()
    plt.xlim(0, maturity)
    plt.show()
    


# And here we have functions to determine SWAPTION value

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



# A function to graph the swaption value
def graph_swaption_value(show_interest_rate):
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


###############################################################################

# Now we can call on functions to e.g. graph the value of a swap(tion)

graph_swap_value(monte_carlo_amount, True)
#graph_swaption_value(False)





        
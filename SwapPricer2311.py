import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd

# Parameters
dt = 0.01
monte_carlo_amount = 100
maturity=5
final_time = 5
vol=0.05
r0 = 0.01
a=0.01
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

# Load the changed csv file for hazard rates, this part is for PD 
csv_file_path = 'C:/Users/Trist/Master/Jaar 2/Semester 1/CompFi/CompFi/allHazardRates.csv'
df = pd.read_csv(csv_file_path,sep=';')

df['MTM_DATE'] = pd.to_datetime(df['MTM_DATE'], errors='coerce')
filtered_data = df[(df['MTM_DATE'] == '2016-08-30') & (df['KEY1'] == 'NETHRS')].copy()
filtered_data['X'] = pd.to_numeric(filtered_data['X'].str.replace(',', '.'), errors='coerce')
filtered_data['Y'] = pd.to_numeric(filtered_data['Y'].str.replace(',', '.'), errors='coerce')

X_values = filtered_data['X'].to_numpy()
Y_values = filtered_data['Y'].to_numpy()
# this is needed as the original data given doesn't contain the initial time 0, and it is trivial that firm is not default 
# as observed in the market
Plus_initial_X = np.insert(X_values, 0, 0.0)
Plus_initial_Y = np.insert(Y_values, 0, 0.0)

loaded_data_discount = np.load('C:/Users/Trist/Master/Jaar 2/Semester 1/CompFi/CompFi/std_tenor_eonia.npy', allow_pickle=True)
loaded_data_forward = np.load('C:/Users/Trist/Master/Jaar 2/Semester 1/CompFi/CompFi/std_tenor_eur3m.npy', allow_pickle=True)

# Define a function for linear interpolation
def linear_interpolation(x, x_values, y_values):
    f = interpolate.interp1d(x_values, y_values, kind='linear',fill_value='extrapolate')
    return f(x)

Intervals = np.arange(0.0, maturity+dt, dt)

# taking a single day entry as our t0
Test_discount = loaded_data_discount[:20]
Test_forward = loaded_data_forward[:20]

# split columns for interpolation 
dates = Test_discount[:, 0]
tenors = Test_discount[:, 1]
interest_rates_discount = Test_discount[:, 2]
interest_rates_forward = Test_forward[:, 2]

# discount rate interpolation, final product contains all P(0,t)
interpolated_rates_discount = [linear_interpolation(interval, tenors, interest_rates_discount) for interval in Intervals]
p_grid = [float(np.exp(-r * t)) for r, t in zip(interpolated_rates_discount, Intervals)]

# Compute forward rates, final product contains f(0,t)
f_grid = [float(linear_interpolation(interval, tenors, interest_rates_forward)) for interval in Intervals]
interpolated_rates_XY = [linear_interpolation(interval, X_values, Y_values) for interval in Intervals]
PD_list = [1-np.exp(-rate) for rate in interpolated_rates_XY]

#This is the end of the interpolation chunk!
###############################################################################



# Functions
###############################################################################

# Theta(t) in the Hull-White model. We calibrate this using a formula
# from the syllabus. We estimate the differential with the central
# difference method
def theta(t):
    # Compute the required part
    i = int(t/dt)
    temp = a*f_grid[i] + ((vol**2)/(2*a))*(1-np.exp(-2*a*t))
    if t < maturity and t > 0:
        return (f_grid[i+1] - f_grid[i-1])/(2*dt) + temp
    elif t == 0:
        return (f_grid[1] - f_grid[0])/dt + temp
    elif t == maturity:
        return (f_grid[-2] - f_grid[-1])/dt + temp
    #return 0.01*(t+1)
        
    

def B(s, t, a):
    return (1/a)*(1-np.exp(-1*a*(s-t)))

# input would be P(0,t), P(o,s) and instantanous forward rate at s
def A(s,t, p_s, p_t, fM, vol, a):
    b_st = B(s, t, a)
    return (p_t/p_s)*np.exp(b_st*fM - ((vol**2)/(4*a))*((b_st)**2)*(1-np.exp(-2*a*s)))

# Function to simulate 1 path of Hull-White and obtain short rate path
# Here note that the expected rate is the sum over all path
def hull_white(vol, maturity, a, dt):
    num_steps = int(maturity/dt)
    short_rate = np.zeros(num_steps + 1)
    short_rate[0] = r0  # Initial interest rate
    
    dW = np.random.normal(0, np.sqrt(dt),num_steps)
    for i in range(1,num_steps+1):
        dR = (theta(i*dt) - a*short_rate[i-1])*dt + vol*np.sqrt(dt)*dW[i-1]
        short_rate[i] = short_rate[i-1] + dR
    return short_rate


# Retrieve the value of P(s,t) based on r(s) from Hull-White
def P(s, t, short_rate, p_s, p_t, fM, vol, a):
    b_st = B(s, t, a)
    return A(s, t, p_s, p_t, fM, vol, a)*np.exp(-1*b_st*(short_rate[int(s/dt)]))

# Determine the instantanous forward rate f(0,s) = - d/dt log(p(0,s))
# via a numerical scheme of the derivative
# this is to be used in calculation of A hence P
def f(s):
    return f_grid[int(s/dt)]


# Now we have functions to determine SWAP value/NPV/EPE

def find_closest_greater_index(t, T):
    # Get the index of the closest value in T that is greater than t
    closest_greater_index = min(range(len(T)), key=lambda i: T[i] if T[i] > t else float('inf'))
    return closest_greater_index

# Function to compute net present value for any t
# given an interest path
def NPV(t, short_rate):
    if t >= maturity:
        return 0
    # First, simulate the short rate and compute the fixed rate
    fixed_grid = []
    p_t_T_grid = []
    for T in maturity_grid:
        fixed_grid.append(P(0, T, short_rate, p_grid[int(0/dt)], p_grid[int(T/dt)], f(0), vol, a))
    fixed = (fixed_grid[0] - fixed_grid[-1])/(Coupon_frequency*(sum(fixed_grid) - fixed_grid[0]))
    upcoming_maturities = [T for T in maturity_grid if T > t]
    if t < maturity_grid[0]:
        for T in maturity_grid: 
            p_t_T_grid.append(P(t, T, short_rate, p_grid[int(t/dt)], p_grid[int(T/dt)], f(t), vol, a))
        temp = fixed*Coupon_frequency*(sum(p_t_T_grid) - p_t_T_grid[0])
        return max(notional*(p_t_T_grid[0] - p_t_T_grid[-1] - temp),0)
    else:
        if len(upcoming_maturities) == 1:
            T = upcoming_maturities[0]
            p_t_T_grid.append(P(t, T, short_rate, p_grid[int(t/dt)], p_grid[int(T/dt)], f(t), vol, a))
            J = upcoming_maturities[0]-Coupon_frequency
            first_p = P(t, J, short_rate,
                        p_grid[int(t/dt)], p_grid[int(J/dt)], f(t), vol, a)
            next_leg = p_t_T_grid[0]*Coupon_frequency*notional*(-fixed + (1/Coupon_frequency)*(1-(p_t_T_grid[0]/first_p))/(p_t_T_grid[0]/first_p))
            return max(next_leg,0)
        else: 
            J = upcoming_maturities[0]-Coupon_frequency
            for T in upcoming_maturities:
                p_t_T_grid.append(P(t, T, short_rate, p_grid[int(t/dt)], p_grid[int(T/dt)], f(t), vol, a))
            temp = fixed*Coupon_frequency*(sum(p_t_T_grid) - p_t_T_grid[0])
            first_p = P(t, J, short_rate,
                        p_grid[int(t/dt)], p_grid[int(J/dt)], f(t), vol, a)
            next_leg = p_t_T_grid[0]*Coupon_frequency*notional*(-fixed + (1/Coupon_frequency)*(1-(p_t_T_grid[0]/first_p))/(p_t_T_grid[0]/first_p))
            return max(notional*(p_t_T_grid[0] - p_t_T_grid[-1] - temp) + next_leg,0)

            
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
        short_rate = hull_white(vol, maturity, a, dt)
        #print(short_rate)
        for j in range(len(t_grid)):
            swap_grid[i,j] = NPV(t_grid[j], short_rate)
        # If you want to see all paths
        if see_all_paths:
            plt.plot(t_grid, swap_grid[i, :])
    # Next compute Monte Carlo average
    for k in range(len(t_grid)):
        mean_grid[k] = np.mean(swap_grid[:, k])
        
    # In any case we graph the average
    plt.plot(t_grid, mean_grid  , label='Empirical average', color='0')
    plt.xlabel('Time (t)')
    plt.ylabel('Expected positive exposure (EPE), N = '+str(N))
    plt.legend()
    plt.xlim(0, maturity)
    plt.show()


def graph_CVA(N, final_time, see_all_paths):
    #T = maturity_grid[-1]-dt
    t_grid = np.linspace(0, final_time, num=int(1+(final_time/dt)))
    swap_grid = np.zeros((N, len(t_grid)))
    mean_grid = np.zeros(len(t_grid))
    # First compute N paths from 0 to T
    # where per path we use one interest rate path
    for i in range(N):
        short_rate = hull_white(vol, maturity, a, dt)
        for j in range(len(t_grid)):
            swap_grid[i,j] = NPV(t_grid[j], short_rate)
        # If you want to see all paths
        if see_all_paths:
            plt.plot(t_grid, swap_grid[i, :])
    # Next compute Monte Carlo average
    for k in range(len(t_grid)):
        if t_grid[k] < maturity:
            mean_grid[k] = np.mean(swap_grid[:, k]) * (PD_list[k])
        else:
            mean_grid[k] = 0
    
    CVA = np.mean(mean_grid)
    print(CVA)
    # In any case we graph the average
    plt.plot(t_grid, mean_grid  , label='Empirical average', color='0')
    plt.xlabel('Time (t)')
    plt.ylabel('Expected positive exposure (EPE), N = '+str(N))
    plt.legend()
    plt.xlim(0, final_time)
    plt.show()
    

# Now we can call on functions to e.g. graph the value of a swap(tion)

graph_CVA(monte_carlo_amount, final_time, False)
#graph_swaption_value(False)





        
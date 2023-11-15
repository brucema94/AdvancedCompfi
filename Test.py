import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd

loaded_data_discount = np.load('C:/Users/bruce/Desktop/Compfi/std_tenor_eonia.npy', allow_pickle=True)
loaded_data_forward = np.load('C:/Users/bruce/Desktop/Compfi/std_tenor_eur3m.npy', allow_pickle=True)

# Define a function for linear interpolation
def linear_interpolation(x, x_values, y_values):
    f = interpolate.interp1d(x_values, y_values, kind='linear')
    return f(x)

dt = 0.001
maturity=10
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

# discount rate interpolation 
interpolated_rates_discount = [linear_interpolation(interval, tenors, interest_rates_discount) for interval in Intervals]
result_list = list(zip(Intervals, interpolated_rates_discount))
result_list_float = [(interval, float(rate)) for interval, rate in result_list]
Discount_factor_M = [(t, np.exp(-r * t)) for t, r in result_list_float]

interpolated_rates_forward = [linear_interpolation(interval, tenors, interest_rates_forward) for interval in Intervals]
result_list_forward = list(zip(Intervals, interpolated_rates_forward))
forward_rates = [(interval, float(rate)) for interval, rate in result_list_forward]


#print(Discount_factor_M)
print(Discount_factor_M[-5:])
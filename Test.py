import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd

loaded_data = np.load('C:/Users/bruce/Desktop/Compfi/std_tenor_eur3m.npy', allow_pickle=True)

Test = loaded_data[:20]
tenors = Test[:, 1]
interest_rates = Test[:, 2]
Discount_factors = np.zeros(len(interest_rates))

for i in range(len(tenors)):
    Discount_factors[i] = 1/ (1+interest_rates[i] * tenors[i])

# 0.0023111894406597226 = 0.23111894406597226 Annualize fixed rate
Par_Swap_rate = (1- Discount_factors[-1])/ sum(Discount_factors)

print(Par_Swap_rate)


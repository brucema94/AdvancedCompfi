import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd

loaded_data = np.load('C:/Users/bruce/Desktop/Compfi/std_tenor_eur3m.npy', allow_pickle=True)

#print("Shape:", loaded_data1.shape)
#print("Data Type:", loaded_data1.dtype)
#print("First few elements:", loaded_data1[:5])

#csv_file_path = 'C:/Users/bruce/Desktop/Compfi/check.csv'
#np.savetxt(csv_file_path, loaded_data1, delimiter=',', fmt='%s')


# Define a function for linear interpolation
def linear_interpolation(x, x_values, y_values):
    f = interpolate.interp1d(x_values, y_values, kind='linear')
    return f(x)

# Extract dates and interest rates
dates = loaded_data[:, 0]
tenors = loaded_data[:, 1]
interest_rates = loaded_data[:, 2]

monthly_intervals = np.arange(0.0, 10.0, 1/12)
mask = tenors <= 10.0
tenors = tenors[mask]
interest_rates = interest_rates[mask]
dates = dates[mask]

interpolated_rates = []

for date in np.unique(dates):
    rates_at_date = interest_rates[dates == date]
    tenors_at_date = tenors[dates == date]
    #print(len(rates_at_date))
    #print(len(tenors))
    interpolated_rates_at_date = [linear_interpolation(interval, tenors_at_date, rates_at_date) for interval in monthly_intervals]
    interpolated_rates.append(interpolated_rates_at_date)

df = pd.DataFrame(interpolated_rates, columns=[f"Month {i+1}" for i in range(len(monthly_intervals))])
df.insert(0, "Date", np.unique(dates))

# Save the DataFrame to a CSV file
df.to_csv('C:/Users/bruce/Desktop/Compfi/interpolated_data.csv', index=False)


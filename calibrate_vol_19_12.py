import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
from tqdm import tqdm
import cProfile
import pstats
import math

def main():
    np.random.seed(123)
    dt = 0.01
    monte_carlo_amount = 10
    MAT = 30
    #vol=0.05
    r0 = 0.01
    a=0.01
    notional=100
    Coupon_frequency = 0.25
  

    # These are possible values of T0, T1, ... ,Tm where Tm = maturity.
    # Note: THIS INCLUDES T0
    #maturity_grid = [k*Coupon_frequency for k in range(1, 1+int(MAT/Coupon_frequency))]

    # We obtain historical data through interpolation

    ###############################################################################
    #Here we load in the data. The filepath is of course different per user

    # DATASET FOR BACHELIER FORMULA, CONTAINING IMPLIED VOL AND TM
    # VOL IS IN BASIS POINTS, 1BP = 0.01
    DATASET = [[1, 70.58598], [2, 86.61775], [3, 87.91504], [4, 88.40403], [5, 88.88835],
              [6, 89.16321], [7, 89.43583], [8, 89.45311], [9, 89.47049],
              [10, 89.48782], [15, 87.76504], [20, 84.99587],
              [25, 83.77717], [30, 84.75736]]


    
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

    Intervals = np.arange(0.0, MAT+dt, dt)

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
    interpolated_rates_XY = [linear_interpolation(interval, Plus_initial_X, Plus_initial_Y) for interval in Intervals]
    PD_list = [rate*dt for rate in interpolated_rates_XY]

    #This is the end of the interpolation chunk!
    ###############################################################################



    # Functions
    ###############################################################################

    # Theta(t) in the Hull-White model. We calibrate this using a formula
    # from the syllabus. We estimate the differential with the central
    # difference method
    def theta(t,a,vol,dt,maturity,f_grid):
        #Compute the required part
        i = int(t/dt)
        temp = a*f_grid[i] + ((vol**2)/(2*a))*(1-np.exp(-2*a*t))
        if 0 < t < maturity:
            return (f_grid[i+1] - f_grid[i-1])/(2*dt) + temp
        elif t == 0:
            return (f_grid[1] - f_grid[0])/dt + temp
        elif t == maturity:
            return (f_grid[-2] - f_grid[-1])/dt + temp
     

    # Function to simulate 1 path of Hull-White and obtain short rate path
    # Here note that the expected rate is the sum over all path
    def hull_white(r0,vol, maturity, a, dt,f_grid):
        num_steps = int(maturity/dt)
        short_rate = np.zeros(num_steps + 1)
        short_rate[0] = r0  # Initial interest rate
        sqrt_dt = np.sqrt(dt)  # Precompute square root of dt
        dW = np.random.normal(0, sqrt_dt,num_steps)
        for i in tqdm(range(1,num_steps+1),desc='Simulating Hull-White'):
            dR = (theta(i*dt,a,vol,dt,maturity,f_grid) - a*short_rate[i-1])*dt + vol*dW[i-1]
            short_rate[i] = short_rate[i-1] + dR
        return short_rate

    
        
    # Retrieve the value of P(s,t) based on r(s) from Hull-White  
    def P_faster(s, t, short_rate, p_s, p_t, fM, vol, a):
        b_st = (1/a)*(1-np.exp(-a*(s-t)))
        return ((p_t/p_s)*np.exp(b_st*fM - ((vol**2)/(4*a))*((b_st)**2)*(1-np.exp(-2*a*s))))*np.exp(-b_st*(short_rate[int(s/dt)]))
    
    # Determine the instantanous forward rate f(0,s) = - d/dt log(p(0,s))
    # via a numerical scheme of the derivative
    # this is to be used in calculation of A hence P
    def f(s):
        return f_grid[int(s/dt)]


    # Now we have functions to determine SWAP value/NPV/EPE

    # Function to compute net present value for any t
    # given an interest path
    def NPV(t, short_rate,a,maturity_grid, sig):
        T0 = 0.25
        if(t >= maturity_grid[-1]):
            return 0,0
        # First, simulate the short rate and compute the fixed rate
        fixed_grid = []
        p_t_T_grid = []
        p_T0_grid = []
        t_array = np.array(maturity_grid)
        p_t_array = np.array(p_grid)[np.floor(t_array/dt).astype(int)]

        # pre-compute with vectorization 
        #b_st = (1/a)*(1-np.exp(-a*(s-t)))
        fixed_b_st_array = (1/a) * (1-np.exp(a*t_array))
        discount_b_st_array = (1/a)*(1-np.exp(-a*(t-t_array)))
        fixed_exp_array = np.exp(-fixed_b_st_array*(short_rate[0]))
        discount_exp_array = np.exp(-discount_b_st_array*(short_rate[int(t/dt)]))
        
        disc_b = (1/a)*(1-np.exp(-a*(T0-t_array)))
        disc_exp = np.exp(-discount_b_st_array*(short_rate[int(T0/dt)]))
        
        # A = (p_t/p_s)*np.exp(b_st*fM - ((vol**2)/(4*a))*((b_st)**2)*(1-np.exp(-2*a*s)))
        fixed_A_array = ((p_t_array/p_grid[0])*np.exp(fixed_b_st_array*f(0)))
        discount_A_array = ((p_t_array/p_grid[int(t/dt)])*np.exp(discount_b_st_array*f(t) - ((sig**2)/(4*a))*((discount_b_st_array)**2)*(1-np.exp(-2*a*t))))
        disc_A = ((p_t_array/p_grid[int(T0/dt)])*np.exp(disc_b*f(T0) - ((sig**2)/(4*a))*((disc_b)**2)*(1-np.exp(-2*a*T0))))
        # P = A * exp(b_st * short_rate)
        fixed_grid = fixed_A_array * fixed_exp_array
        p_t_T_grid = discount_A_array * discount_exp_array
        p_T0_grid = disc_A * disc_exp
        
        annuity = Coupon_frequency*sum(p_T0_grid)
        # for T in tqdm(maturity_grid,desc='Computing NPV'):
        #     fixed_grid.append(P_faster(s=0, t=T, short_rate=short_rate, p_s = p_grid[0], p_t = p_grid[int(T/dt)], fM = f(0), vol = vol, a = a))
        #     p_t_T_grid.append(P_faster(s=t, t=T, short_rate = short_rate, p_s = p_grid[int(t/dt)], p_t = p_grid[int(T/dt)], fM = f(t), vol = vol, a = a))
        swap_rate = (fixed_grid[0] - fixed_grid[-1])/(Coupon_frequency*(sum(fixed_grid) - fixed_grid[0]))
        upcoming_maturities = [T for T in maturity_grid if T > t]
        
        if t <= maturity_grid[0]:
            temp = swap_rate*Coupon_frequency*(sum(p_t_T_grid) - p_t_T_grid[0])
            return max(notional*(p_t_T_grid[0] - p_t_T_grid[-1] - temp),0), annuity
        else:
            Next_coupon = maturity_grid.index(upcoming_maturities[0])
            J = upcoming_maturities[0]-Coupon_frequency
            first_p = P_faster(t, J, short_rate,p_grid[int(t/dt)], p_grid[int(J/dt)], f(t), sig, a)
            next_leg = p_t_T_grid[Next_coupon]*Coupon_frequency*notional*(-swap_rate + (1/Coupon_frequency)*(1-(p_t_T_grid[Next_coupon]/first_p))/(p_t_T_grid[Next_coupon]/first_p))  
            if len(upcoming_maturities) == 1:
                return max(next_leg,0), annuity
            elif(len(upcoming_maturities) == 0):
                return 0, annuity
            else: 
                temp = swap_rate*Coupon_frequency*(sum(p_t_T_grid[Next_coupon:]) - p_t_T_grid[Next_coupon])
                return max(notional*(p_t_T_grid[Next_coupon] - p_t_T_grid[-1] - temp) + next_leg,0), annuity

    
    # Here we compute the price of a swaption using the EPE profile
    def price_swaption(MCA, t, sig, maturity):
        EPE_grid = np.zeros(MCA)
        maturity_grid = [k*Coupon_frequency for k in range(1, 1+int(maturity/Coupon_frequency))]
        for n in range(MCA):
            T0 = 0.25
            short_rate = hull_white(r0, sig, maturity, a, dt, f_grid)
            NP, ann = NPV(T0, short_rate, a, maturity_grid, sig)
            EPE = max(NP, 0)
            EPE_grid[n] = EPE
        return np.mean(EPE_grid)
        
    def bachelier(DATA):
        T = DATA[0]
        maturities = [k*Coupon_frequency for k in range(2, 1+int(T/Coupon_frequency))]
        annuity = sum([p_grid[int(t/dt)] for t in maturities])*Coupon_frequency
        imp_vol = 0.01*DATA[1]
        return annuity*imp_vol*np.sqrt(0.25+T)/np.sqrt(2*math.pi)
        
    
    
    def calibrate_sigma():
        vol_end = 0.05
        vol_grid = np.linspace(0, vol_end, num=1+int(vol_end/0.001))
        error_grid = []
        error_mean = np.zeros(len(vol_grid))
        DATA_MAT = [1,2,3,4,5,6,7,8,9,10,15,20,25,30]
        i=0
        datapoints = [bachelier(DATA) for DATA in DATASET]
        for sig in vol_grid:
            error_vol = []
            error_vol = [price_swaption(monte_carlo_amount, 0.25, sig, maturity) for maturity in DATA_MAT]
            error = math.sqrt(sum([((error_vol[k] - datapoints[k])/datapoints[k])**2 for k in range(len(datapoints))]))
            
            error_mean[i] = error
            error_grid.append(error_vol)
            i+=1
        #for i in range(len(vol_grid)):
         #   plt.plot(DATA_MAT, error_grid[i], label='Vol = '+str(vol_grid[i]))    
        #plt.plot(DATA_MAT, datapoints, color='0')
        sigma_hat = vol_grid[np.argmin(error_mean)]
        min_error = np.min(error_mean)
        mins = [min_error for vol in vol_grid]
        plt.plot(vol_grid, error_mean, label='Error for different volatilities', color='0')
        plt.plot(vol_grid, mins, label='Minimal error for $\sigma_{cal}$ = '+str(round(sigma_hat,3)), color='r', linestyle = 'dashed')
        plt.xlabel('Volatility $\sigma$')
        plt.ylabel('Mean-squared error')
        plt.xlim(0, 0.05)
        plt.legend()
        print(sigma_hat)
        print(min_error)
        
    
    calibrate_sigma()
  
    
  
if __name__ == "__main__":
    main()
    # cProfile.run('main()', 'profile_results')
    
    # p = pstats.Stats('profile_results')
    # p.sort_stats('cumulative').print_stats(10)
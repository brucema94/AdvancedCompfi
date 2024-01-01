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
    N=monte_carlo_amount
    maturity = 10
    #MAT = 30
    vol=0.05
    vol_cal = 0.026
    r0 = 0.01
    a=0.01
    notional=100
    Coupon_frequency = 0.25
    theta0 = 0.001


    #DATASET = [[1, 99.91054], [2, 101.33254], [3, 101.43563], [4, 100.42322]]
    DATASET = [[4, 100.42322]]
    # These are possible values o]f T0, T1, ... ,Tm where Tm = maturity.
    # Note: THIS INCLUDES T0
    maturity_grid = [k*Coupon_frequency for k in range(1, 1+int(maturity/Coupon_frequency))]

    # We obtain historical data through interpolation

    ###############################################################################
    #Here we load in the data. The filepath is of course different per user

    # DATASET FOR BACHELIER FORMULA, CONTAINING IMPLIED VOL AND TM
    # VOL IS IN BASIS POINTS, 1BP = 0.01
    #DATASET = [[1, 70.58598], [2, 86.61775], [3, 87.91504], [4, 88.40403], [5, 88.88835],
             # [6, 89.16321], [7, 89.43583], [8, 89.45311], [9, 89.47049],
             # [10, 89.48782], [15, 87.76504], [20, 84.99587],
             # [25, 83.77717], [30, 84.75736]]


    
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
        for i in range(1,num_steps+1):
            dR = (theta(i*dt,a,vol,dt,maturity,f_grid) - a*short_rate[i-1])*dt + vol*dW[i-1]
            short_rate[i] = short_rate[i-1] + dR
        return short_rate

    # def hull_white_faster(N,r0,vol, maturity, a, dt,f_grid):
    #     num_steps = int(maturity/dt)
    #     short_rate = np.zeros(num_steps + 1)
    #     short_rate[0] = r0  # Initial interest rate
    #     dW = np.random.normal(0, np.sqrt(dt),(num_steps,N))
    #     t_array = dt * np.arange(1,num_steps+1)
    #     i_array = np.floor(t_array/dt).astype(int)
    #     f_grid_array = np.array(f_grid)[i_array]
    #     temp_array = a*f_grid_array + ((vol**2)/(2*a))*(1-np.exp(-2*a*t_array))
    #     mask_0_maturity = (t_array > 0) & (t_array < maturity)        
    #     theta_array_middle = (f_grid_array[1:] - f_grid_array[:-1])/dt + temp_array[mask_0_maturity]
    #     if t_array[0] == 0:
    #         theta_0 = (f_grid_array[1] - f_grid_array[0])/dt + temp_array[0]
    #     else:
    #         theta_0 = None     
    #     theta_maturity = (f_grid_array[-2] - f_grid_array[-1])/dt + temp_array[-1]
    #     theta_array = theta_array_middle.tolist() + [theta_maturity]
    #     if theta_0:
    #         theta_array = [theta_0] + theta_array
    #     dR_array = ((np.array(theta_array) - a*short_rate[0])*dt).reshape(-1,1) + vol*np.sqrt(dt)*dW
    #     dR_array = np.concatenate((np.zeros((1,N)),dR_array),axis=0)
    #     short_rate_array = np.cumsum(dR_array, axis=0) + r0
    #     return short_rate_array
        
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
        if(t >= maturity_grid[-1]):
            return 0
        # First, simulate the short rate and compute the fixed rate
        fixed_grid = []
        p_t_T_grid = []
        
        t_array = np.array(maturity_grid)
        p_t_array = np.array(p_grid)[np.floor(t_array/dt).astype(int)]

        # pre-compute with vectorization 
        #b_st = (1/a)*(1-np.exp(-a*(s-t)))
        fixed_b_st_array = (1/a) * (1-np.exp(a*t_array))
        discount_b_st_array = (1/a)*(1-np.exp(-a*(t-t_array)))
        fixed_exp_array = np.exp(-fixed_b_st_array*(short_rate[0]))
        discount_exp_array = np.exp(-discount_b_st_array*(short_rate[int(t/dt)]))
        
        # A = (p_t/p_s)*np.exp(b_st*fM - ((vol**2)/(4*a))*((b_st)**2)*(1-np.exp(-2*a*s)))
        fixed_A_array = ((p_t_array/p_grid[0])*np.exp(fixed_b_st_array*f(0)))
        discount_A_array = ((p_t_array/p_grid[int(t/dt)])*np.exp(discount_b_st_array*f(t) - ((sig**2)/(4*a))*((discount_b_st_array)**2)*(1-np.exp(-2*a*t))))
        
        # P = A * exp(b_st * short_rate)
        fixed_grid = fixed_A_array * fixed_exp_array
        p_t_T_grid = discount_A_array * discount_exp_array
        # for T in tqdm(maturity_grid,desc='Computing NPV'):
        #     fixed_grid.append(P_faster(s=0, t=T, short_rate=short_rate, p_s = p_grid[0], p_t = p_grid[int(T/dt)], fM = f(0), vol = vol, a = a))
        #     p_t_T_grid.append(P_faster(s=t, t=T, short_rate = short_rate, p_s = p_grid[int(t/dt)], p_t = p_grid[int(T/dt)], fM = f(t), vol = vol, a = a))
        swap_rate = (fixed_grid[0] - fixed_grid[-1])/(Coupon_frequency*(sum(fixed_grid) - fixed_grid[0]))
        upcoming_maturities = [T for T in maturity_grid if T > t]
        
        if t <= maturity_grid[0]:
            temp = swap_rate*Coupon_frequency*(sum(p_t_T_grid) - p_t_T_grid[0])
            return max(notional*(p_t_T_grid[0] - p_t_T_grid[-1] - temp),0)
        else:
            Next_coupon = maturity_grid.index(upcoming_maturities[0])
            J = upcoming_maturities[0]-Coupon_frequency
            first_p = P_faster(t, J, short_rate,p_grid[int(t/dt)], p_grid[int(J/dt)], f(t), sig, a)
            next_leg = p_t_T_grid[Next_coupon]*Coupon_frequency*notional*(-swap_rate + (1/Coupon_frequency)*(1-(p_t_T_grid[Next_coupon]/first_p))/(p_t_T_grid[Next_coupon]/first_p))  
            if len(upcoming_maturities) == 1:
                return max(next_leg,0)
            elif(len(upcoming_maturities) == 0):
                return 0
            else: 
                temp = swap_rate*Coupon_frequency*(sum(p_t_T_grid[Next_coupon:]) - p_t_T_grid[Next_coupon])
                return max(notional*(p_t_T_grid[Next_coupon] - p_t_T_grid[-1] - temp) + next_leg,0)

    def CVA(N,a,vol):
        T = maturity_grid[-1]-dt
        t_grid = np.linspace(0, T, num=int(1+(T/dt)))
        swap_grid = np.zeros((N, len(t_grid)))
        mean_grid = np.zeros(len(t_grid))
        CVA = 0
        # First compute N paths from 0 to T
        # where per path we use one interest rate path
        #short_rate_array = hull_white_faster(N,r0,vol, maturity, a, dt,f_grid)
        for i in tqdm(range(N),desc='Computing CVA'):
            short_rate = hull_white(r0,vol, maturity, a, dt,f_grid)
            for j in range(len(t_grid)):
                swap_grid[i,j] = NPV(t_grid[j], short_rate,a,maturity_grid)
                #swap_grid[i,j] = NPV(t_grid[j], short_rate_array[:,i],a,maturity_grid)
        # Next compute Monte Carlo average
        for k in tqdm(range(len(t_grid)),desc='compute Monte Carlo average'):
            mean_grid[k] = np.mean(swap_grid[:, k])
            CVA += mean_grid[k] * PD_list[k]
        
        return CVA
    
    
    def CVA_against_maturity(maturities, N, a, vol):
        CVA_grid = []
        for T in tqdm(maturities, desc='Computing maturities'):
            t_grid = np.linspace(0, T, num=int(1+(T/dt)))
            swap_grid = np.zeros((N, len(t_grid)))
            mean_grid = np.zeros(len(t_grid))
            CVA = 0
            coupons = [k*Coupon_frequency for k in range(1, 1+int(T/Coupon_frequency))]
            for i in range(N):
                short_rate = hull_white(r0,vol, T, a, dt,f_grid)
                for j in range(len(t_grid)):
                    swap_grid[i,j] = NPV(t_grid[j], short_rate,a,coupons, vol)
                    #swap_grid[i,j] = NPV(t_grid[j], short_rate_array[:,i],a,maturity_grid)
            # Next compute Monte Carlo average
            for k in range(len(t_grid)):
                mean_grid[k] = np.mean(swap_grid[:, k])
                CVA += mean_grid[k] * PD_list[k]
            CVA_grid.append(CVA)
            
        #Now we plot
        plt.plot(maturities, CVA_grid, label='CVA as function of maturity date T', color='0')
        plt.xlim(0.5, maturities[-1])
        plt.xlabel('Final maturity of the swap contract, $\Delta t$ = '+str(dt))
        plt.ylabel('CVA (% of notional)')
        plt.legend()
        plt.show()
        print(CVA_grid)
    
    
    
    def graph_EPE(N, a, vol):
        T = maturity_grid[-1]-dt
        values = [bachelier(DATA) for DATA in DATASET]
        times, results = zip(*values)
        t_grid = np.linspace(0, T, num=int(1+(T/dt)))
        swap_grid = np.zeros((N, len(t_grid)))
        mean_grid = np.zeros(len(t_grid))
        for i in tqdm(range(N),desc='Computing CVA'):
            short_rate = hull_white(r0,vol, maturity, a, dt,f_grid)
            for j in range(len(t_grid)):
                swap_grid[i,j] = NPV(t_grid[j], short_rate,a,maturity_grid, vol)
                #swap_grid[i,j] = NPV(t_grid[j], short_rate_array[:,i],a,maturity_grid)
        # Next compute Monte Carlo average
        for k in tqdm(range(len(t_grid)),desc='compute Monte Carlo average'):
            mean_grid[k] = np.mean(swap_grid[:, k])
        # Now we graph EPE
        plt.plot(t_grid, mean_grid, label='Estimation of the EPE', color='0')
        plt.xlabel('Time (t), $\Delta$t = '+str(dt))
        plt.ylabel('EPE (% of notional), N = '+str(N))
        plt.scatter([1], results, label='Bachelier Values', color='r', marker='o')
        plt.xlim(0, maturity)
        #value = mean_grid[int(1/dt)]
        #print(value)
        #print(results)
        #print(abs(value - results))
        plt.legend()
        plt.show()
        
        
    def graph_EPE2(N, a, vol1, vol2):
        T = maturity_grid[-1]-dt
        t_grid = np.linspace(0, T, num=int(1+(T/dt)))
        swap_grid = np.zeros((N, len(t_grid)))
        swap2_grid = np.zeros((N, len(t_grid)))
        mean_grid = np.zeros(len(t_grid))
        mean2_grid = np.zeros(len(t_grid))
        for i in tqdm(range(N),desc='Computing CVA'):
            short_rate = hull_white(r0,vol, maturity, a, dt,f_grid)
            for j in range(len(t_grid)):
                swap_grid[i,j] = NPV(t_grid[j], short_rate,a,maturity_grid, vol1)
                swap2_grid[i,j] = NPV(t_grid[j], short_rate,a,maturity_grid, vol2)
                #swap_grid[i,j] = NPV(t_grid[j], short_rate_array[:,i],a,maturity_grid)
        # Next compute Monte Carlo average
        for k in tqdm(range(len(t_grid)),desc='compute Monte Carlo average'):
            mean_grid[k] = np.mean(swap_grid[:, k])
            mean2_grid[k] = np.mean(swap2_grid[:, k])
        # Now we graph EPE
        plt.plot(t_grid, mean_grid, label='Estimation of the EPE using $\sigma$ = 0.05', color='0')
        plt.plot(t_grid, mean2_grid, label='Estimation of EPE using $\sigma$ = $\sigma_{cal}$', color='blue')
        plt.xlabel('Time (t), $\Delta$t = '+str(dt))
        plt.ylabel('EPE (% of notional), N = '+str(N))
        #plt.scatter([1], results, label='Bachelier Values', color='r', marker='o')
        plt.xlim(0, maturity)
        #value = mean_grid[int(1/dt)]
        #print(value)
        #print(results)
        #print(abs(value - results))
        plt.legend()
        plt.show()
    
    # Compute EPE for certain maturity T
    def compute_EPE(N, a, vol, T):
        t_grid = np.linspace(0, T, num=int(1+(T/dt)))
        swap_grid = np.zeros((N, len(t_grid)))
        mean_grid = np.zeros(len(t_grid))
        #maturity_grid2 = [k*Coupon_frequency for k in range(1, 1+int(T/Coupon_frequency))]
        for i in tqdm(range(N),desc='Computing CVA'):
            short_rate = hull_white(r0,vol, T, a, dt,f_grid)
            for j in range(len(t_grid)):
                swap_grid[i,j] = NPV(t_grid[j], short_rate,a,maturity_grid, vol)
                #swap_grid[i,j] = NPV(t_grid[j], short_rate_array[:,i],a,maturity_grid)
        # Next compute Monte Carlo average
        for k in tqdm(range(len(t_grid)),desc='compute Monte Carlo average'):
            mean_grid[k] = np.mean(swap_grid[:, k])
        return mean_grid[-1]
    
    def analyse_EPE(maturities, N, a, vol):
        EPE_grid = []
        #Here we compute the EPE outsight
        for T in maturities:
            EPE_grid.append(compute_EPE(N, a, vol, T))
        mean_grid = [sum(grid)/len(grid) for grid in EPE_grid]
        max_grid = [max(grid) for grid in EPE_grid]
        plt.plot(maturities, mean_grid, label= 'Average EPE', color='0')
        plt.plot(maturities, max_grid, label='Maximum EPE on [0,T]', color='b')
        
        plt.xlim(0.5, maturities[-1])
        plt.xlabel('Maturity T of the contract, $\Delta t$ = '+str(dt))
        plt.ylabel('Value of interest (% of notional), N = '+str(N))
        
        plt.legend()
        plt.show()
        
    def CVA_against_volatility(T, N, a, vol_grid):
        t_grid = np.linspace(0, T, num=int(1+(T/dt)))
        swap_grid = np.zeros((N, len(t_grid)))
        mean_grid = np.zeros(len(t_grid))
        CVA_grid = []
        coupons = [k*Coupon_frequency for k in range(1, 1+int(T/Coupon_frequency))]
        for sig in vol_grid:
            CVA = 0
            for i in tqdm(range(N),desc='Computing CVA'):
                short_rate = hull_white(r0,sig, T, a, dt,f_grid)
                for j in range(len(t_grid)):
                    swap_grid[i,j] = NPV(t_grid[j], short_rate,a,coupons, sig)
                    #swap_grid[i,j] = NPV(t_grid[j], short_rate_array[:,i],a,maturity_grid)
            # Next compute Monte Carlo average
            for k in tqdm(range(len(t_grid)),desc='compute Monte Carlo average'):
                mean_grid[k] = np.mean(swap_grid[:, k])
                CVA += mean_grid[k] * PD_list[k]
            CVA_grid.append(CVA)
        plt.plot(vol_grid, CVA_grid, label='CVA for given volatility $\sigma$', color='0')
        plt.title('CVA against volatility for T  = '+str(T))
        plt.xlabel('Volatility $\sigma$')
        plt.ylabel('CVA (% of notional)')
        plt.xlim(vol_grid[0], vol_grid[-1])
        
        plt.show()
            
    def price_swaption(MCA, t, sig, maturity):
        EPE_grid = np.zeros(MCA)
        swaption_grid = np.zeros(MCA)
        maturity_grid = [k*Coupon_frequency for k in range(1, 1+int(MAT/Coupon_frequency))]
        for n in range(MCA):
            T0 = 0.25
            short_rate = hull_white(r0, sig, maturity, a, dt, f_grid)
            fixed_grid = []
            p_t_T_grid = []
            p_T0_grid = []
            t_array = np.array(maturity_grid)
            p_t_array = np.array(p_grid)[np.floor(t_array/dt).astype(int)]
    
            fixed_b_st_array = (1/a) * (1-np.exp(a*t_array))
          
            discount_b_st_array = (1/a)*(1-np.exp(-a*(t-t_array)))
            disc_b = (1/a)*(1-np.exp(-a*(T0-t_array)))
            
            fixed_exp_array = np.exp(-fixed_b_st_array*(short_rate[0]))
            
            discount_exp_array = np.exp(-discount_b_st_array*(short_rate[int(t/dt)]))
            disc_exp = np.exp(-discount_b_st_array*(short_rate[int(T0/dt)]))
            
            fixed_A_array = ((p_t_array/p_grid[0])*np.exp(fixed_b_st_array*f(0)))
            
            discount_A_array = ((p_t_array/p_grid[int(t/dt)])*np.exp(discount_b_st_array*f(t) - ((sig**2)/(4*a))*((discount_b_st_array)**2)*(1-np.exp(-2*a*t))))
            disc_A = ((p_t_array/p_grid[int(T0/dt)])*np.exp(disc_b*f(T0) - ((sig**2)/(4*a))*((disc_b)**2)*(1-np.exp(-2*a*T0))))
            
            fixed_grid = fixed_A_array * fixed_exp_array
            
            swap_rate = (fixed_grid[0] - fixed_grid[-1])/(Coupon_frequency*(sum(fixed_grid) - fixed_grid[0]))
            
            p_t_T_grid = discount_A_array * discount_exp_array
            p_T0_grid = disc_A * disc_exp
            annuity = Coupon_frequency*sum(p_T0_grid[1:])
            ST0 = (1- p_T0_grid[-1])/(sum(p_T0_grid[1:]))
           
            EPE = max(NPV(0.25, short_rate, a, maturity_grid, sig),0)
            #swaption = notional*annuity*max(ST0 - swap_rate, 0)
            EPE_grid[n] = EPE
            #swaption_grid[n] = swaption
        return np.mean(EPE_grid)
        
    def bachelier(DATA):
        T = DATA[0]
        maturities = [k*Coupon_frequency for k in range(2, 1+int(T/Coupon_frequency))]
        annuity = sum([p_grid[int(t/dt)] for t in maturities])*Coupon_frequency
        imp_vol = 0.01*DATA[1]
        result = annuity*imp_vol*np.sqrt(5)/np.sqrt(2*np.pi)
        return T,result
        
    
    
    def calibrate_sigma():
        vol_grid = np.linspace(0, 0.2, num=100)
        error_grid = []
        DATA_MAT = [1,2,3,4,5,6,7,8,9,10,15,20,25,30]
        for sig in vol_grid:
            error_vol = []
            datapoints = [bachelier(DATA) for DATA in DATASET]
            
            error_vol = [price_swaption(monte_carlo_amount, 0.25, sig, maturity) for maturity in DATA_MAT]
            error = math.sqrt(sum([((error_vol[k] - datapoints[k])/datapoints[k])**2 for k in range(len(datapoints))]))
            error_grid.append(error)
        plt.plot(vol_grid, error_grid)
        plt.show()
        
    
    #calibrate_sigma()
    #print(price_swaption(monte_carlo_amount, 0.25, vol))
    graph_EPE(monte_carlo_amount, a, vol_cal)
    #graph_EPE2(monte_carlo_amount, a, 0.05, vol_cal)
    #MAT=10
    #vol_grid=np.linspace(0, 0.1, num=50)
    #maturities = [k*Coupon_frequency for k in range(2, 1+int(maturity/Coupon_frequency))]
    #analyse_EPE(maturities, monte_carlo_amount, a, vol)
    #CVA_against_maturity(maturities, monte_carlo_amount, a, vol_cal)
    #CVA_against_volatility(10, monte_carlo_amount, a, vol_grid)
if __name__ == "__main__":
    main()
    # cProfile.run('main()', 'profile_results')
    
    # p = pstats.Stats('profile_results')
    # p.sort_stats('cumulative').print_stats(10)
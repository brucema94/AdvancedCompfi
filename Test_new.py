import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from tqdm import tqdm
from scipy.stats import norm
import pandas as pd

def main():
    np.random.seed(123)
    dt = 0.001
    monte_carlo_amount = 100
    maturity=2
    vol=0.05
    r0 = 0.01
    a=0.01
    notional=100
    Coupon_frequency = 0.25

    # These are possible values of T0, T1, ... ,Tm where Tm = maturity.
    # Note: THIS INCLUDES T0
    maturity_grid = [k*Coupon_frequency for k in range(1, 1+int(maturity/Coupon_frequency))]

    # We obtain historical data through interpolation

    ###############################################################################
    #Here we load in the data. The filepath is of course different per user

    loaded_data_discount = np.load('C:/Users/bruce/Desktop/Compfi/std_tenor_eonia.npy', allow_pickle=True)
    loaded_data_forward = np.load('C:/Users/bruce/Desktop/Compfi/std_tenor_eur3m.npy', allow_pickle=True)


    # Define a function for linear interpolation
    def linear_interpolation(x, x_values, y_values):
        f = interpolate.interp1d(x_values, y_values, kind='linear',fill_value='extrapolate')
        return f(x)

    Intervals = np.arange(0.0, maturity+dt, dt)

    # taking a single day entry as our t0
    Test_discount = loaded_data_discount[:20]
    Test_forward = loaded_data_forward[:20]

    # split columns for interpolation 
    tenors = Test_discount[:, 1]
    interest_rates_discount = Test_discount[:, 2]
    interest_rates_forward = Test_forward[:, 2]

    # discount rate interpolation, final product contains all P(0,t)
    interpolated_rates_discount = [linear_interpolation(interval, tenors, interest_rates_discount) for interval in Intervals]
    p_grid = [float(np.exp(-r * t)) for r, t in zip(interpolated_rates_discount, Intervals)]

    # Compute forward rates, final product contains f(0,t)
    f_grid = [float(linear_interpolation(interval, tenors, interest_rates_forward)) for interval in Intervals]

    swaption_file_path = 'C:/Users/bruce/Desktop/Compfi/swaption.csv'
    df = pd.read_csv(swaption_file_path,sep=';')
    df['Swap maturity'] = df['Swap maturity'].str.extract('(\d+)')
    df['Swap maturity'] = df['Swap maturity'].astype(int)
    swap_maturity_list = df['Swap maturity'].tolist()
    Volatitly_list = df['Volatitly'].tolist()
    #This is the end of the interpolation chunk!
    ###############################################################################



    # Functions
    ###############################################################################

    # Theta(t) in the Hull-White model. We calibrate this using a formula
    # from the syllabus. We estimate the differential with the central
    # difference method
    def theta(t,a,vol,dt,maturity,f_grid):
        # Compute the required part
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
        dW = np.random.normal(0, np.sqrt(dt),num_steps)
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

    def NPV(t, short_rate,a,maturity_grid):
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
        discount_A_array = ((p_t_array/p_grid[int(t/dt)])*np.exp(discount_b_st_array*f(t) - ((vol**2)/(4*a))*((discount_b_st_array)**2)*(1-np.exp(-2*a*t))))
        
        # P = A * exp(b_st * short_rate)
        fixed_grid = fixed_A_array * fixed_exp_array
        p_t_T_grid = discount_A_array * discount_exp_array
        # for T in tqdm(maturity_grid,desc='Computing NPV'):
        #     fixed_grid.append(P_faster(s=0, t=T, short_rate=short_rate, p_s = p_grid[0], p_t = p_grid[int(T/dt)], fM = f(0), vol = vol, a = a))
        #     p_t_T_grid.append(P_faster(s=t, t=T, short_rate = short_rate, p_s = p_grid[int(t/dt)], p_t = p_grid[int(T/dt)], fM = f(t), vol = vol, a = a))
        swap_rate = (fixed_grid[0] - fixed_grid[-1])/(Coupon_frequency*sum(fixed_grid[1:]))
        upcoming_maturities = [T for T in maturity_grid if T > t]
        
        if t < maturity_grid[0]:
            temp = swap_rate*Coupon_frequency*(sum(p_t_T_grid[1:]))
            return max(notional*(p_t_T_grid[0] - p_t_T_grid[-1] - temp),0)
        else:
            Next_coupon = maturity_grid.index(upcoming_maturities[0])
            J = upcoming_maturities[0]-Coupon_frequency
            first_p = P_faster(t, J, short_rate,p_grid[int(t/dt)], p_grid[int(J/dt)], f(t), vol, a)
            next_leg = p_t_T_grid[Next_coupon]*Coupon_frequency*notional*(-swap_rate + (1/Coupon_frequency)*(1-(p_t_T_grid[Next_coupon]/first_p))/(p_t_T_grid[Next_coupon]/first_p))  
            if len(upcoming_maturities) == 1:
                return max(next_leg,0)
            else: 
                temp = swap_rate*Coupon_frequency*(sum(p_t_T_grid[Next_coupon:]) - p_t_T_grid[Next_coupon])
                return max(notional*(p_t_T_grid[0] - p_t_T_grid[-1] - temp) + next_leg,0)

    def Swaption_pricer(t,vol,short_rate):
        # First, simulate the short rate and compute the fixed rate
        T0 = Coupon_frequency
        
        t_array = np.array(maturity_grid)
        p_t_array = np.array(p_grid)[np.floor(t_array/dt).astype(int)]

        # pre-compute with vectorization 
        #b_st = (1/a)*(1-np.exp(-a*(s-t)))
        fixed_b_st_array = (1/a) * (1-np.exp(a*t_array))
        swaption_b_st_array = (1/a)*(1-np.exp(-a*(0.25-t_array[1:])))
        fixed_exp_array = np.exp(-fixed_b_st_array*(short_rate[0]))
        swaption_exp_array = np.exp(-swaption_b_st_array*(short_rate[int(0.25/dt)]))
        
        # A = (p_t/p_s)*np.exp(b_st*fM - ((vol**2)/(4*a))*((b_st)**2)*(1-np.exp(-2*a*s)))
        fixed_A_array = ((p_t_array/p_grid[0])*np.exp(fixed_b_st_array*f(0)))
        swaption_A_array = ((p_t_array[1:]/p_grid[int(0.25/dt)])*np.exp(swaption_b_st_array*f(0.25) - ((vol**2)/(4*a))*((swaption_b_st_array)**2)*(1-np.exp(-2*a*t))))
        
        # P = A * exp(b_st * short_rate)
        fixed_grid = fixed_A_array * fixed_exp_array
        swaption_grid = swaption_A_array * swaption_exp_array
        # for T in tqdm(maturity_grid,desc='Computing NPV'):
        #     fixed_grid.append(P_faster(s=0, t=T, short_rate=short_rate, p_s = p_grid[0], p_t = p_grid[int(T/dt)], fM = f(0), vol = vol, a = a))
        #     p_t_T_grid.append(P_faster(s=t, t=T, short_rate = short_rate, p_s = p_grid[int(t/dt)], p_t = p_grid[int(T/dt)], fM = f(t), vol = vol, a = a))
        StrikeRate = (fixed_grid[0] - fixed_grid[-1])/(Coupon_frequency*sum(fixed_grid[1:]))
        SwapRate = (1 - swaption_grid[-1])/(Coupon_frequency*sum(swaption_grid[1:])) 

        
        Swaption_price = notional * (sum(swaption_grid[1:])) * Coupon_frequency * max(SwapRate-StrikeRate,0)
        discoount_price = Swaption_price * p_grid[int(t/dt)]
        vol_bachilier = 70.58598
        d1 = (SwapRate-StrikeRate)/(vol_bachilier * np.sqrt(Coupon_frequency))
        #Swaption_price_bachilier = vol_bachilier * np.sqrt(Coupon_frequency/(2*np.pi))
        Swaption_price_bachilier = notional *(SwapRate-StrikeRate) * norm.cdf(d1) + vol_bachilier * np.sqrt(Coupon_frequency) * norm.pdf(d1)

        return discoount_price
    
    # P=Notional×(SwapRate−StrikeRate)×Φ(d1) 
    # d1 = (SwapRate - StrikeRate)/Vol * sqrt(Option Maturity)
    def Bachilier():
        return 

    def Objective_function(vol):
        1

    
    short_rate = hull_white(r0,vol, maturity, a, dt,f_grid)
    result = NPV(0.1, short_rate,a,maturity_grid)
    result2 = Swaption_pricer(0.1,vol,short_rate)
    print(result)
    print(result2)



if __name__ == "__main__":
    main()
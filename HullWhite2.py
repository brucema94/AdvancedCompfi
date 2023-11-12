import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate as integrate


# Parameters
dt = 0.001
N = 100
maturity=2
vol=0.1
r0 = 0
a=0.1
notional=100
t_begin = 0
t_end = 1

# These are test values for P(0,t)
p_grid = np.linspace(1, 0.6, num=2+int(maturity/dt))

# These are possible values of T0, T1, ... ,Tm
maturity_grid = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

# For computational purpose
timeGrid = np.linspace(0, maturity_grid[0], num = 1+int(maturity_grid[0]/dt))

# Functions
###############################################################################

# Theta(t) in the Hull-White model
def theta(t):
    return -0.5*abs(math.sin(t))

# As in the notes
def B(s,t):
    return (1/a)*(1-math.e**(-1*a*(s-t)))

def A(s,t, p_s, p_t, fM):
    return (p_t/p_s)*math.e**(B(s,t)*fM - ((vol**2)/(4*a))*((B(s,t))**2)*(1-math.e**(-2*a*s)))

# Function to simulate N paths of Hull-White and take an average
def hull_white(N, vol, maturity, a, dt):
    M = int(maturity/dt)
    r_grid = np.zeros((N, M+1))
    avg_grid = []
    norm = np.zeros((N, M))
    for j in range(N):
        norm[j, :] = np.random.normal(0, 1, size = M)
    for i in range(1,M+1):
        r_grid[:,i] = r_grid[:,i-1] + (theta(i*dt) - a*r_grid[:,i-1])*dt + vol*math.sqrt(dt)*norm[:,i-1]
    for i in range(M+1):
        avg_grid.append(np.mean(r_grid[:,i]))
    return r_grid, avg_grid


# Retrieve the value of P(s,t) based on r(s) from Hull-White
def P(s,t, short_rate, p_s, p_t, fM):
    return A(s,t, p_s, p_t, fM)*math.e**(-1*B(s,t)*(short_rate[int(s/dt)]))

# Determine the forward rate f(0,s) = - d/dt log(p(0,s))
# via a numerical scheme of the derivative
def f(s, p_grid):
    return -1*(math.log(p_grid[1+int(s/dt)]) - math.log(p_grid[int(s/dt)]))/dt

# Function to determine the value of a swap at time t, where we make a 
# distintion between t <= T0 and t > T0
def V(t, maturity_grid, notional):
    short_rate = hull_white(N, vol, maturity_grid[-1], a, dt)[1]
    fixed_grid = []
    delta = maturity_grid[-1] - maturity_grid[-2]
    # If t <= T0 we simply use the formula given in the notes
    if (t <= maturity_grid[0]):
        p_t_T_grid = []
        for T in maturity_grid:
            p_t_T_grid.append(P(t, T, short_rate, p_grid[int(t/dt)], p_grid[int(T/dt)], f(t, p_grid)))
            fixed_grid.append(P(0, T, short_rate, p_grid[int(0/dt)], p_grid[int(T/dt)], f(0, p_grid)))
        fixed = (fixed_grid[0] - fixed_grid[-1])/(delta*(sum(fixed_grid) - fixed_grid[0]))
        temp = fixed*delta*(sum(p_t_T_grid) - p_t_T_grid[0])
        return notional*(p_t_T_grid[0] - p_t_T_grid[-1] - temp), fixed, short_rate
    # Otherwise (t > T0) we make a small change to the formula
    else:
        for T in maturity_grid:
            fixed_grid.append(P(0, T, short_rate, p_grid[int(0/dt)], p_grid[int(T/dt)], f(0, p_grid)))
        fixed = (fixed_grid[0] - fixed_grid[-1])/(delta*(sum(fixed_grid) - fixed_grid[0]))
        return 0, fixed, short_rate
    
# Function to compute net present value
def NPV(t, maturity_grid, notional):
    v, K, short_rate = V(t, maturity_grid, notional) #Fixed rate
    PV_fixed = 0
    p_t_T_grid = []
    forward_grid = []
    delta = maturity_grid[1] - maturity_grid[0]
    for T in [T for T in maturity_grid if t <= T]:
        p_t_T_grid.append(P(t, T, short_rate, p_grid[int(t/dt)], p_grid[int(T/dt)], f(t, p_grid)))
    forward_grid.append((1/delta)*p_t_T_grid[0]*(1/p_t_T_grid[0] - 1))
    for i in range(1,len(p_t_T_grid)):  
        forward_grid.append((1/delta)*p_t_T_grid[i-1]*(p_t_T_grid[i-1]/p_t_T_grid[i] - 1))
    PV_fixed = delta*K*sum(p_t_T_grid)
    PV_float = delta*sum(forward_grid)
    return PV_float - PV_fixed

# We compute the EPE based on NPV
def EPE(t, maturity_grid, notional):
    return max(NPV(t, maturity_grid, notional), 0)
    
    
        
    

###############################################################################

#HW, avg_HW = hull_white(N, vol, maturity, 0.1, dt)

t_grid = np.linspace(t_begin, t_end, num = int(1+(t_end - t_begin)/dt))

EPE_grid = []
for T in t_grid:
    EPE_grid.append(NPV(T, maturity_grid, notional))
    
plt.plot(t_grid, EPE_grid, label='EPE of swap at time t', color='b')
    
plt.legend()
plt.show()


        
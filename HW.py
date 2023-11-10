# Script to simulate using Hull-White model
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate as integrate

# Introduce parameters

T = 10
dt = 0.001
M = int(T/dt)
timeGrid = np.linspace(0, T, num = int(1+T/dt))
p_grid = np.linspace(1, 0.6, num=int(1+T/dt))
N = 20
epsilon = 0
b = 0.2
r0 = 1
a=0.3
vol=0.2

# Hull-White is dX(t) = (theta(t) - aX(t))dt + sigma*dW(t)
# The solution is computable


# Here comes a large amount of functions

#########################################################################

# This is the time-dependent deterministic part
def theta(t):
    return b*t

def mean(lst):
    return sum(lst)/len(lst)

def B(s,t):
    return (1/a)*(1-math.e**(a*(s-t)))

def A(s,t):
    sum1 = (-(vol**2)*(B(s,t)**2)/(4*a))*(1-math.e**(-2*s))
    sum2 = (-1*(vol/a)**2)*B(s,t)*((1-math.e**(-a*s))**2)
    return math.e**(sum1 + sum2)

# Generate a Brownian motion from 0 to T with step size dt
def generateBrownianIncrements(T, dt):
    if (T/dt).is_integer(): #Just a double-check
        N = int(T/dt)
        temp = []
        temp.append(0)
        for i in range(N):
            temp.append(np.random.normal(0, math.sqrt(dt)))
        return temp
    else:
        return np.zeros(N+1)
    
def generateBrownian(increments):
    res = []
    for i in range(len(increments)):
        temp=0
        for j in range(0,i):
            temp += increments[j]
        res.append(temp)
    return res

# Function to sample N amount of Brownian motions
def sampleBrownianMotion(N, T, dt):
    res = []
    res2 = []
    for i in range(N):
        increments = generateBrownianIncrements(T, dt)
        temp = generateBrownian(increments)
        res2.append(increments)
        res.append(temp)
    return res, res2
    

# Compute the final values of an array x[i][j]
def computeFinalValue(sample, N):
    res = []
    for i in range(N):
        res.append(sample[i][-1])
    return res

# Compute the minimum and maximum of Brownian motions
def computeMinMax(sample):
    N = len(sample)
    dim1min = []
    dim1max = []
    for i in range(N):
        dim1min.append(min(sample[i]))
        dim1max.append(max(sample[i]))
    return min(dim1min), max(dim1max)
    
# Function to compute stochastic integral 
def stochasticIntegral(integrand, Brownian):
    # We need len(integrand)+1 = len(Brownian)
    sum = 0
    for i in range(len(integrand)):
        sum += integrand[i]*(Brownian[i+1] - Brownian[i])
    return sum

# Function to sample solutions to the Hull-White equation
# using predetermined Brownian motions    
def HullWhite(a, vol, increments, Brownian):
    solution_grid = []
    solution_grid.append(r0)
    theta_grid = []
    for i in range(len(timeGrid)):
        theta_grid.append(theta(timeGrid[i]))
    for i in range(1, len(Brownian)):
        sum = 0
        integrand = []
        for j in range(i):
            integrand.append(vol*math.e**(a*(timeGrid[j]-timeGrid[i])))
        sum += r0*math.e**(-a*timeGrid[i]) + stochasticIntegral(integrand, Brownian[:i+1])
        sum += integrate.quad(lambda x: theta(x)*math.e**(a*(x-timeGrid[i])), 0, timeGrid[i])[0]
        solution_grid.append(sum)
    return solution_grid
        


def CouponPrice(HullWhiteGrid):
    coupon_grid = []
    for i in range(N):
        temp = []
        for j in range(M+1):
            temp.append((p_grid[-1]/p_grid[j])*A(timeGrid[j], T)*math.e**(-1*B(timeGrid[j], T)*HullWhiteGrid[i][j]))
        coupon_grid.append(temp)
    return coupon_grid
    
#########################################################################
    
    
BrownianMotion, increments = sampleBrownianMotion(N, T, dt)
HullWhiteGrid = []
for i in range(N):
    HullWhiteGrid.append(HullWhite(a, vol, increments[i], BrownianMotion[i]))

#coupon_grid = CouponPrice(HullWhiteGrid)
#mean_grid = [p_grid[-1]]
#for i in range(M):
#    temp = [coupon_grid[j][i] for j in range(N)]
#    mean_grid.append(mean(temp))
mean_grid = []
for i in range(M+1):
    temp = [HullWhiteGrid[j][i] for j in range(N)]
    mean_grid.append(mean(temp))
                     
# Now plot
for i in range(N): 
 #   plt.plot(timeGrid, coupon_grid[i])
    plt.plot(timeGrid, HullWhiteGrid[i])
plt.plot(timeGrid, mean_grid, label='Empirical average', color='0')
    
#plt.plot(timeGrid, mean_grid, label='Empirical average' ,color='0')

y_min, y_max = computeMinMax(HullWhiteGrid)
#y_min, y_max = computeMinMax(coupon_grid) 

#final = computeFinalValue(BrownianMotion, N)

plt.xlabel('Time (t)')
#plt.ylabel('Simulated value of P(t, ' + str(T) +'), size = '+ str(N))
plt.ylabel('Simulated solution to Hull-White equation, size = '+ str(N))
plt.xlim(-1*epsilon, T + epsilon)
plt.ylim(y_min - epsilon, y_max + epsilon)
plt.legend()
plt.show

    
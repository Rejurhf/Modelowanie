# Mateusz Ziomek
# 26.11.2018

# Spreading
# dA(t)/dt = (ksp*V^(4/3))/A
# ksp is spreading constant ~150 per second
# A is spill area (m^2)
#   A0 is the initial area for stage 2 of spreading(m2)
#   A0 = pi(k2^4/k1^2)*((d*g*V0^5)/vw)
#   k2, k1 are empirical constants (1.14 and 1.45 respectively)
#   g is the gravitational acceleration 9.8
#   vw is the kinematic viscosity of water(m2/s), for 283K 1.308, 313K 3.03
#   Poil is oil density (kg/m3) 895
#   Psw density of sea water (kg/m3) 1025
# t0 is time to end stage one of spreading (s),
#   t0 = (k2/k1)^4 * (V0/(vw*g*d))
# V is initial spill volume (m^2)

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dy/dt
def model(a,t,ksp,V):
    dadt = (ksp * (V**(4/3)))/a
    return dadt

# initial condition
psw = 1025
poil = 832
d = (psw - poil)/psw
k2 = 1.14
k1 = 1.45
g = 9.8
V0 = 1  # 1m3
vw = 3.03
a0 = np.pi*(k2**4/k1**2)*((d*g*V0**5)/vw)
t0 = (k2/k1)**2 * (V0/(vw*g*d))
ksp = 150

print('----------------')
print(t0)

# time points
t = np.linspace(t0,60*60*24*2)

# solve ODE
a = odeint(model,a0,t,args=(ksp,V0,))

# plot results
plt.plot(t, a, 'r-', linewidth=2)
plt.xlabel('time')
plt.ylabel('Spill area')
plt.legend()
plt.show()

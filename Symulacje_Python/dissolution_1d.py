# Mateusz Ziomek
# 26.11.2018

# Dissolution
# dFd(t)/dt = Kdiss*A(S/(1000*Poil))
# Fd is the volume fraction of oil dissolved in sea water (%),
# A is spill area (m^2)
# Kdiss is the mass transfer coefficient of dissolution (m/s),
#   nie wiem ile to jest, dla uproszczenia Kdiss = kevp
# S =S0*exp(-12.0*Fe)
# Fe is the volume fraction of oil evaporated (%)
# S the solubility of oil at time t (g/m3)
# S0 is the initial solubility of oil in water (g/m3).
# Poil is oil density (kg/m3) 832

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dy/dt
def model(f,t,Kdiss,A,S0,Poil,fe,i):
    S = S0*np.exp(-12*fe[round(t/(60*60*24*2*1.07)*49)]) # potrzebna jest
    # wartość odparowanego oleju dla danego t, a fe to tablica 50 elementw,
    # i to jest jakiś tam moj nieudolny sposb na załątwienie tego na szybko.
    # TRZEBA TO POPRAWIC
    dfdt = Kdiss*A*(S/(1000*Poil))
    return dfdt

# Evaporation is needed to calculate dissolution
def modelevap(f,t,kevp,A,V,a,b,T0,Tg,Toil):
    dfdt = ((kevp * A)/V) * np.exp(a-((b*(T0 + (Tg*f)))/Toil))
    return dfdt


# time points
t = np.linspace(0,60*60*24*2)

# initial condition evaporation
fe0 = 0.0
Ws10 = 5.0
kevp = 0.0025 * (Ws10 ** 0.78)
V = 1.0
a = 6.3
b = 10.3
API = 38.4
T0 = 532.98 - 3.125 * API
Tg = 985.62 - 13.597 * API
Toil = 298

# initial condition
f0 = 0.0
Kdiss = kevp
S0 = 1.0
A = 10.0
Poil = 832
i=1

# solve ODE evaporation
fe = odeint(modelevap,fe0,t,args=(kevp,A,V,a,b,T0,Tg,Toil,))

# solve ODE
f = odeint(model,f0,t,args=(Kdiss,A,S0,Poil,fe,i,))

# plot results
plt.plot(t, f, 'r-', linewidth=2)
plt.xlabel('time')
plt.ylabel('Fd(t) (%)')
plt.legend()
plt.show()

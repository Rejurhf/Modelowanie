# Mateusz Ziomek
# 26.11.2018

# Emulsification
# dy(t)/dt = (k*(1+Ws^2)) * (1 - (Yw/YF))
# Y is percentage of volume fraction in oil (%);
# Ws10 is the wind speed at height of 10 m from sea surface (m/s)
# Yf is maximum water content or final water cut (~0.7 for heavy crude oils).
#   Statfjord crude 0.9
# kemul is the mass transfer coefficient of emulsification (2.0*10^(-6), m/s),
# kevp is mass transfer coefficient of evaporation(m/s);
#   (2.5*10^(-3)*Ws10^0.78)
# kexp_corr is the corrected mass transfer coefficient of evaporation
#   considering the effect of emulsion formation (m/s)
# kexp_corr = kevp(1-Y)

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dy/dt
def model(y,t,kemul,Ws10,Yf):
    dydt = kemul*(1 + Ws10**2)*(1 - y/Yf)
    return dydt

# initial condition
y0 = 0.0
kemul = 2.0 * (10 ** (-6))
Ws10 = 4.0
Yf = 0.9

# time points
t = np.linspace(0,60*60*24*2)

# solve ODE
y = odeint(model,y0,t,args=(kemul,Ws10,Yf,))

# plot results
plt.plot(t, y, 'r-', linewidth=2)
plt.xlabel('time')
plt.ylabel('Y(t) (%)')
plt.legend()
plt.show()

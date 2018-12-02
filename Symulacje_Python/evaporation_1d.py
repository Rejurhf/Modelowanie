# Mateusz Ziomek
# 26.11.2018

# Evaporation
# dFe(t)/dt = ((KL*A)/V0)exp(a-((b*(T0+Tg*Fe))/Toil))
#  Fe is the volume fraction of oil evaporated (%)
# A is spill area (m^2)
# V is initial spill volume (m^2)
# Fe is volume fraction of oil evaporated;
# kevp is mass transfer coefficient of evaporation(m/s);
#   (2.5*10^(-3)*Ws10^0.78)
# Ws10 is the wind speed at height of 10 m from sea surface (m/s)
# T0 is the initial boiling point temperature of the oil (K) (532.98-3.125API);
# Tg gradient of the oil distillation plot (K) (985.62-13.587API);
# API refer to the American Petroleum Institute gravity scale, for
#   Statfjord crude 38.4 (light oil), Kuwait crude 31.31,
#   Prudhoe Bay crude 25.74
# a, b are constants (~ 6.3 and 10.3).
# Toil oil temperature (K).

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dy/dt
def model(f,t,kevp,A,V,a,b,T0,Tg,Toil):
    dfdt = ((kevp * A)/V) * np.exp(a-((b*(T0 + (Tg*f)))/Toil))
    return dfdt

# initial condition
f0 = 0.0
Ws10 = 5.0
kevp = 0.0025 * (Ws10 ** 0.78)
A = 10.0
V = 1.0
a = 6.3
b = 10.3
API = 38.4
T0 = 532.98 - 3.125 * API
Tg = 985.62 - 13.597 * API
Toil = 298

# time points
t = np.linspace(0,60*60*24*7)

# solve ODE
f = odeint(model,f0,t,args=(kevp,A,V,a,b,T0,Tg,Toil))

# plot results
plt.plot(t, f, 'r-', linewidth=2)
plt.xlabel('time')
plt.ylabel('f(t)')
plt.legend()
plt.show()

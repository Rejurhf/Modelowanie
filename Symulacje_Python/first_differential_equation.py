# Mateusz Ziomek
# 26.11.2018

# dy(t)/dt = -ky(t)
#  k=0.3, y0 = 5

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dy/dt
def model(y,t,k):
    dydt = -k * y
    return dydt

# initial condition
y0 = 5

# time points
t = np.linspace(0,20)

# solve ODE
k = 0.1
y1 = odeint(model,y0,t,args=(k,))
k = 0.2
y2 = odeint(model,y0,t,args=(k,))
k = 0.5
y3 = odeint(model,y0,t,args=(k,))

# plot results
plt.plot(t, y1, 'r-', linewidth=2, label='k=0.1')
plt.plot(t, y2, 'b--', linewidth=2, label='k=0.2')
plt.plot(t, y3, 'g:', linewidth=2, label='k=0.5')
plt.xlabel('time')
plt.ylabel('y(t)')
plt.legend()
plt.show()

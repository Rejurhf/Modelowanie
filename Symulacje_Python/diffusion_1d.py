# Rejurhf
# 2.12.2018

import numpy
from matplotlib import pyplot as plt

# du/dt = v(d^2u)/(dx^2) 1d diffusion equation
# After discretizing
# (d^2u)/(dx^2) = (u[i+1]-2u[i]+u[i-1])/delx^2 + O(delx^2)

# Initial Conditions
nx = 41
dx = 2 / (nx - 1)
nt = 20    # the number of timesteps we want to calculate
nu = 0.3   # the value of viscosity
sigma = .2 # sigma is a parameter
dt = sigma * dx**2 /nu # dt is defined using sigm
print(dt)

u = numpy.ones(nx)      # a numpy array with nx elements all equal to 1.
#setting u = 2 between 0.5 and 1 as per our I.C.s
u[int(.5 / dx):int(1 / dx + 1)] = 2

# Calculation
un = numpy.ones(nx) # our placeholder array, un, to advance the solution in time
for n in range(nt):  # iterate through time
    un = u.copy() # copy the existing values of u into un
    for i in range(1, nx - 1):
        u[i] = un[i] + nu * dt / dx**2 * (un[i+1] - 2 * un[i] + un[i-1])

    plt.plot(numpy.linspace(0, 2, nx), u);
plt.show();

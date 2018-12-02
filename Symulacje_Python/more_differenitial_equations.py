# Mateusz Ziomek
# 26.11.2018

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# # -----------------------------------------------------------------------------
# # dy(t)/dt = -y(t) + 1
# # y0 = 0
# # function that returns dy/dt
# def model(y,t):
#     dydt = -y + 1
#     return dydt
#
# # initial condition
# y0 = 0
#
# # time points
# t = np.linspace(0,20)
#
# # solve ODE
# y = odeint(model,y0,t)
#
# # plot results
# plt.plot(t, y)
# plt.xlabel('time')
# plt.ylabel('y(t)')
# plt.legend()
# plt.show()

# # -----------------------------------------------------------------------------
# # 5*dy(t)/dt = -y(t) + u(t)
# # y0 = 1, u steps form 0 to 2 at t=10
# # function that returns dy/dt
# def model(y,t):
#     if t < 10.0:
#         u = 0
#     else:
#         u = 2
#     dydt = (-y + u)/5
#     return dydt
#
# # initial condition
# y0 = 1
#
# # time points
# t = np.linspace(0,40,1000)
#
# # solve ODE
# y = odeint(model,y0,t)
#
# # plot results
# plt.plot(t, y, 'r-', label='Output (y(t))')
# plt.plot([0,10,10,40], [0,0,2,2], 'b-', label='Input (u(t))')
# plt.xlabel('time')
# plt.ylabel('y(t)')
# plt.legend(loc='best')
# plt.show()

# # -----------------------------------------------------------------------------
# # dx(t)/dt = 3exp(-t)
# # dy(t)/dt = 3 - y(t)
# # x0 = 0, y0 = 0
# # function that returns dy/dt
# def model(z,t):
#     dxdt = 3.0 * np.exp(-t)
#     dydt = -z[1] + 3
#     return [dxdt, dydt]
#
# # initial condition
# z0 = [0,0]
#
# # time points
# t = np.linspace(0,5)
#
# # solve ODE
# z = odeint(model,z0,t)
#
# # plot results
# plt.plot(t, z[:,0], 'b-', label=r'$\frac{dx}{dt}=3 \; \exp(-t)$')
# plt.plot(t, z[:,1], 'r--', label=r'$\frac{dy}{dt}=-y+3$')
# plt.xlabel('time')
# plt.ylabel('response')
# plt.legend(loc='best')
# plt.show()

# # -----------------------------------------------------------------------------
# # 2 * dx(t)/dt = -x(t) + u(t)
# # 5 * dy(t)/dt = -y(t) + x(t)
# # x0 = 0, y0 = 0, u = 2S(t-5),
# # where S(t−5) is a step function that changes from zero to one at
# # t=5. When it is multiplied by two, it changes from zero to two at
# # that same time, t=5.
# # function that returns dy/dt
# def model(z,t,u):
#     x = z[0]
#     y = z[1]
#     dxdt = (-x + u)/2.0
#     dydt = (-y + x)/5.0
#     return [dxdt, dydt]
#
# # initial condition
# z0 = [0,0]
#
# # number of time points
# n = 401
#
# # time points
# t = np.linspace(0,40,n)
#
# # step input
# u = np.zeros(n)
# # change to 2.0 at time t = 5.0
# u[51:] = 2.0
#
# # store solutions
# x = np.empty_like(t)
# y = np.empty_like(t)
# # record initial conditions
# x[0] = z0[0]
# y[0] = z0[1]
#
# # solve ODE
# for i in range(1,n):
#     # span for next time steps
#     tspan = [t[i-1], t[i]]
#     # solve for next step
#     z = odeint(model,z0,t,args=(u[i],))
#     # store solutions for ploting
#     x[i] = z[1][0]
#     y[i] = z[1][1]
#     z0 = z[1]
#
# # plot results
# plt.plot(t, x, 'b-', label='x(t)')
# plt.plot(t, y, 'r--', label='y(t)')
# plt.plot([0,5,5,40], [0,0,2,2], 'g:', label='u(t)')
# plt.xlabel('time')
# plt.ylabel('values')
# plt.legend(loc='best')
# plt.show()

# -----------------------------------------------------------------------------
# du(t)/dt = -x(v*u(t))
# u0 = 0, x0 = 0, v = 2
# function that returns dy/dt
def model(u,t,v):
    dudt = -(v * u)
    return dudt

# initial condition
u0 = 1
v=2

# time points
t = np.linspace(0,40)

# solve ODE
u = odeint(model,u0,t,args=(v,))

# plot results
plt.plot(t, u, 'b-', label='Przemieszczenie')
plt.xlabel('time')
plt.ylabel('response')
plt.legend(loc='best')
plt.show()

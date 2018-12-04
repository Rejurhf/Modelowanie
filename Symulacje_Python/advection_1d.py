# Rejurhf
# 4.12.2018

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

class Advection1D:
    """Advection"""
    def __init__(self,
                 C,     # value
                 u,     # velocity
                 X,     # position
                 dx,    # x step
                 ):
        self.C = np.asarray(C, dtype='float')
        self.u = np.asarray(u, dtype='float')
        self.X = np.asarray(X, dtype='float')
        self.dx = dx
        self.time_elapsed = 0

    def solution(self):
        """Advection solution"""
        Cn = self.C
        Cn[0] = Cn[-1]
        Cnn = Cn
        for i in range(1, len(Cn)):
            # Advection term
            A = u[i] * (Cn[i] - Cn[i-1])/self.dx
            # Euler's Method
            Cnn[i] = Cn[i] + dt*(-A)

        self.C = Cnn
        x = self.X
        y = self.C
        return (x, y)

    def step(self, dt):
        """execute one time step of length dt and update state"""
        self.time_elapsed += dt

#------------------------------------------------------------
Lx = 10         # x len
dx = 0.5        # Every 0.2m
nx = int(Lx/dx) # number of steps
x = np.linspace(0, Lx, nx);
C = np.zeros(nx)
u = np.full(nx, 1)
C[1] = 2
C[1] = 1.9


# set up initial state and global variables
advect = Advection1D(C, u, x, dx)
dt = 1./30 # 30 fps

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False,
                     xlim=(0, Lx), ylim=(0, 1))
ax.grid()

line, = ax.plot([], [], '-o', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

def init():
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    energy_text.set_text('')
    return line, time_text, energy_text

def animate(i):
    """perform animation step"""
    global advect, dt
    advect.step(dt)

    line.set_data(*advect.solution())
    time_text.set_text('time = %.1f' % advect.time_elapsed)
    energy_text.set_text('Position X = f')
    return line, time_text, energy_text

# choose the interval based on dt and the time to animate one step
from time import time
t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=300,
                              interval=interval, blit=True, init_func=init)

plt.show()
#

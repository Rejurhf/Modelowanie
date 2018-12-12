# Rejurhf
# 4.12.2018

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

class Burgers2D:
    """Burgers"""
    def __init__(self,
                 C,     # value
                 u,     # velocity
                 K,     # Diffusion
                 v,
                 dx,
                 dy,
                 ):
        self.C = np.asarray(C, dtype='float')
        self.u = np.asarray(u, dtype='float')
        self.K = np.asarray(K, dtype='float')
        self.v = np.asarray(v, dtype='float')
        self.time_elapsed = 0

    def solution(self):
        """Advection solution"""
        Cn = self.C
        Cnn = Cn
        Cnn[:, 0] = Cnn[:, 1]
        Cnn[:, -1] = Cnn[:, -2]
        Cnn[0, :] = Cnn[1, :]
        Cnn[-1, :] = Cnn[-2, :]

        if self.time_elapsed < 2:
            Cnn[10][31] = Cnn[9][3] + 50
            Cnn[10][30] = Cnn[9][4] + 50
            Cnn[9][31] = Cnn[9][3] + 50
            Cnn[9][30] = Cnn[9][4] + 50

        for i in range(1, len(Cn)-1):
            for j in range(1, len(Cn[0])-1):
                # Advection term
                A = (u[i][j] + v[i][j]) * (Cn[i+1][j] - Cn[i-1][j])/(2*dx) + (u[i][j] + v[i][j]) * (Cn[i][j+1] - Cn[i][j-1])/(2*dy)
                # Diffusion term
                D = K[i][j] * (Cn[i+1][j] - 2*Cn[i][j] + Cn[i-1][j])/dx**2 + (K[i][j] * (Cn[i][j+1] - 2*Cn[i][j] + Cn[i][j-1])/dy**2)
                # Euler's Method
                Cnn[i][j] = Cn[i][j] + dt*(-A + D)
                if Cnn[i][j] < 0:
                    Cnn[i][j]=np.abs(Cnn[i][j]);

        self.C = Cnn
        return self.C

    def step(self, dt):
        """execute one time step of length dt and update state"""
        self.time_elapsed += dt

#------------------------------------------------------------
Lx = 7         # x len
Ly = 5         # x len
dx = dy = 0.1        # Every 0.2m
nx = int(Lx/dx)
ny = int(Ly/dy)     # number of steps
C = np.zeros((ny,nx))
u = np.zeros((ny,nx))
u[:, :] = 0.0
K = np.zeros((ny,nx))
K[:, :] = 0.001
v = np.zeros((ny,nx))
for i in range(ny):
    v[i, :] = np.sin(np.pi*i/ny)
print(v)

# set up initial state and global variables
burgers = Burgers2D(C, u, K, v, dx, dy)
dt = 1./30 # 30 fps

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False,
                     xlim=(0, nx-1), ylim=(0, ny-1))

line = ax.imshow(burgers.solution(), animated=True)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

def init():
    """initialize animation"""
    global burgers, dt
    line.set_array(burgers.solution())
    time_text.set_text('')
    energy_text.set_text('')
    return line, time_text, energy_text

def animate(*i):
    """perform animation step"""
    global burgers, dt
    burgers.step(dt)

    line.set_array(burgers.solution())
    time_text.set_text('time = %.1f' % burgers.time_elapsed)
    energy_text.set_text('Position X = f')
    return line, time_text, energy_text

# choose the interval based on dt and the time to animate one step
from time import time
t0 = time()
animate(0)
t1 = time()
interval = 500 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=300,
    interval=interval, blit=True, init_func=init)

plt.show()

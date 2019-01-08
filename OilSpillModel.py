# Rejurhf
# 8.01.2019

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation


class Layer:

    def __init__(self,
                 mass,          # array of oil mass
                 velocity_x,    # velocity in X direction (array)
                 diffusion,     # array of diffusion
                 velocity_y,    # velocity in Y direction (array)
                 dx,
                 dy,
                 dt):

        self.mass = np.asarray(mass, dtype='float')             # main array of sipll
        self.velocity_x = np.asarray(velocity_x, dtype='float') # velocity moving in y direction
        self.velocity_y = np.asarray(velocity_y, dtype='float') # velocity moving in x direction
        self.diffusion = np.asarray(diffusion, dtype='float')   # array of Diffusion
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.time_elapsed = 0


    def update(self):

        current_mass = self.mass    # array of mass in t
        next_mass = current_mass    # array of mass in t+1

        # managing edges of array edge pixel is equal to neighbor pixel
        next_mass[:, 0] = next_mass[:, 1]
        next_mass[:, -1] = next_mass[:, -2]
        next_mass[0, :] = next_mass[1, :]
        next_mass[-1, :] = next_mass[-2, :]

        # spilling oil
        if self.time_elapsed < 1:
            for m in range(15, 20):
                for n in range(5, 10):
                    next_mass[m,n] += 50

        # calculate oil spill
        for i in range(1, len(current_mass)-1):
            for j in range(1, len(current_mass[0])-1):
                # Advection term
                A = self.velocity_y[i][j] * (current_mass[i+1][j] - current_mass[i-1][j])/(2*self.dx) + \
                    self.velocity_x[i][j] * (current_mass[i][j+1] - current_mass[i][j-1])/(2*self.dy)
                # Diffusion term
                D = self.diffusion[i][j] * (current_mass[i+1][j] - 2*current_mass[i][j] + current_mass[i-1][j])/self.dx**2 + \
                    (self.diffusion[i][j] * (current_mass[i][j+1] - 2*current_mass[i][j] + current_mass[i][j-1])/self.dy**2)
                # Euler's Method
                next_mass[i][j] = current_mass[i][j] + self.dt*(-A + D)
                if next_mass[i][j] < 0:
                    next_mass[i][j]=np.abs(next_mass[i][j])

        self.mass = next_mass
        return self.mass


    def step(self):
        self.time_elapsed += self.dt


#------------------------------------------------------------
Lx = 12.8         # x len
Ly = 12.8         # x len
dx = dy = 0.05        # Every 0.2m
dt = 1./30 # 30 fps
nx = int(Lx/dx)
ny = int(Ly/dy)     # number of steps
m = np.zeros((ny, nx))
u = np.zeros((ny, nx))   # velocity moving in x direction advection
u[:, :] = 0.5
K = np.zeros((ny, nx))   # array of Diffusion
K[:, :] = 0.01
v = np.zeros((ny, nx))   # velocity moving in y direction advection
for i in range(ny):
    for j in range(nx):
        v[i, j] = (0.1 + 0.001*(i-Ly) + np.sin(np.pi*j/Lx)/4)

# set up initial state and global variables
layer1 = Layer(m, u, K, v, dx, dy, dt)

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False,
                     xlim=(0, nx-1), ylim=(0, ny-1))

line = ax.imshow(layer1.update(), animated=True)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)


def init():
    """initialize animation"""
    global layer1, dt
    line.set_array(layer1.update())
    time_text.set_text('')
    energy_text.set_text('')
    return line, time_text, energy_text


def animate(*i):
    """perform animation step"""
    global layer1, dt
    layer1.step()

    line.set_array(layer1.update())
    time_text.set_text('time = %.1f' % layer1.time_elapsed)
    energy_text.set_text('Position X = f')
    return line, time_text, energy_text


# choose the interval based on dt and the time to animate one step
from time import time
t0 = time()
animate(0)
t1 = time()
interval = 500 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=300, interval=interval, blit=True, init_func=init)

plt.show()

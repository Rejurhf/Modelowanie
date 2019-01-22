# Rejurhf
# 8.01.2019

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.integrate as integrate
import matplotlib.animation as animation
from filecontroller import getArrayFromJSON

land_array = []
evaporation_rate = 0.00002
temperature = 20
time_step = 10

spill_area_array = []
evaporation_array = []

class Layer:

    def __init__(self,
                 mass,          # array of oil mass
                 land,
                 velocity_x,    # velocity in X direction (array)
                 diffusion,     # array of diffusion
                 velocity_y,    # velocity in Y direction (array)
                 dx,
                 dy,
                 dt):
        # main array of sipll
        self.mass = np.asarray(mass, dtype='float')
        # array of land (value of -1 means water 0 means land and
        # positive value means mass of shoreline deposition)
        self.land = np.asarray(land, dtype='float')
        # velocity moving in y direction
        self.velocity_x = np.asarray(velocity_x, dtype='float')
        # velocity moving in x direction
        self.velocity_y = np.asarray(velocity_y, dtype='float')
        # array of Diffusion
        self.diffusion = diffusion
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.time_elapsed = 0

        self.shorelineConst = 0.15
        self.maximumShorelineDeposition = 12
        self.evapPrc = 0.0000001
        self.ifSpilledArray = m = np.zeros((len(self.mass), len(self.mass[0])))

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
            for m in range(90, 100):
                for n in range(70, 80):
                    next_mass[m,n] += 10

        # calculate oil spill
        for i in range(1, len(current_mass)-1):
            for j in range(1, len(current_mass[0])-1):

                if self.land[i][j] < 0:
                    # Advection term
                    A = self.velocity_y[i][j] * (current_mass[i+1][j] - \
                        current_mass[i-1][j])/(2*self.dx) + \
                        self.velocity_x[i][j] * (current_mass[i][j+1] - \
                        current_mass[i][j-1])/(2*self.dy)
                    # Diffusion term
                    D = self.diffusion * (current_mass[i+1][j] - \
                        2*current_mass[i][j] + current_mass[i-1][j])/self.dx**2 + \
                        (self.diffusion * (current_mass[i][j+1] - \
                        2*current_mass[i][j] + current_mass[i][j-1])/self.dy**2)

                    # Euler's Method
                    next_mass[i][j] = current_mass[i][j] + self.dt*(-A + D)

                    # calculate evaporated percent
                    if self.time_elapsed > 0.02:
                        evapPrcPrv = self.evapPrc
                        self.evapPrc = ((evaporation_rate + 0.045*(temperature-15))*np.log(self.time_elapsed*60))/2.5

                        totalMass = next_mass[i][j]/evapPrcPrv  # Calculate total mass without evaporation
                        if self.evapPrc <= 0:   # in case devision by 0
                            self.evapPrc = 0.0000001
                        next_mass[i][j] -= totalMass*(self.evapPrc-evapPrcPrv)    # substract

                    if next_mass[i][j] < 0:
                        next_mass[i][j] = np.abs(next_mass[i][j])
                    if next_mass[i][j] < 0.1:
                        next_mass[i][j] = 0
                    if next_mass[i][j] > 1:
                        self.ifSpilledArray[i][j] = 1


        # shoreline deposition
        for i in range(1, len(next_mass)-1):
            for j in range(1, len(next_mass[0])-1):
                if 0 <= self.land[i][j] <= self.maximumShorelineDeposition:
                    if self.land[i-1][j] < 0:
                        self.land[i][j] += self.shorelineConst * next_mass[i-1][j]
                        next_mass[i-1][j] -= self.shorelineConst * next_mass[i-1][j]
                    if self.land[i+1][j] < 0:
                        self.land[i][j] += self.shorelineConst * next_mass[i+1][j]
                        next_mass[i+1][j] -= self.shorelineConst * next_mass[i+1][j]
                    if self.land[i][j-1] < 0:
                        self.land[i][j] += self.shorelineConst * next_mass[i][j-1]
                        next_mass[i][j-1] -= self.shorelineConst * next_mass[i][j-1]
                    if self.land[i][j+1] < 0:
                        self.land[i][j] += self.shorelineConst * next_mass[i][j+1]
                        next_mass[i][j+1] -= self.shorelineConst * next_mass[i][j+1]

        self.mass = next_mass

        # do wyswietlania:
        tmp = np.zeros((len(current_mass), len(current_mass[0])))
        land_sum = 0.0
        for i in range(1, len(current_mass)-1):
            for j in range(1, len(current_mass[0])-1):
                if self.land[i][j] < 0:
                    tmp[i][j] = next_mass[i][j]
                else:
                    tmp[i][j] = self.land[i][j]
                    land_sum += self.land[i][j]
        land_array.append(land_sum)
        evaporation_array.append(self.evapPrc)
        spill_area_array.append((self.ifSpilledArray == 1).sum())

        return tmp

    def step(self, dt):
        self.time_elapsed += dt


#------------------------------------------------------------
Lx = 12.8         # x len
Ly = 12.8         # x len
dx = dy = 0.05        # Every 0.2m
dt = 1./30 # 30 fps
nx = int(Lx/dx)
ny = int(Ly/dy)     # number of steps
m = np.zeros((ny, nx))
# land test:
l = getArrayFromJSON("maps", "zatoka2")

K = 0.01    # Diffusion constant

u = getArrayFromJSON("leftright", "zatokatest")
v = getArrayFromJSON("updown", "zatokatest")
u *= 300
v *= 300
# u = np.zeros((ny, nx))   # velocity moving in x direction advection
# u[:, :] = 0.5
# v = np.zeros((ny, nx))   # velocity moving in y direction advection
# for i in range(ny):
#     for j in range(nx):
#         v[i, j] = (0.1 + 0.001*(i-Ly) + np.sin(np.pi*j/Lx)/4)

# set up initial state and global variables
layer1 = Layer(m, l, u, K, v, dx, dy, dt)

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()

ax = fig.add_subplot(111, autoscale_on=False,
                     xlim=(0, nx-1), ylim=(0, ny-1))

line = ax.imshow(layer1.update(), animated=True)
img = mpimg.imread('res/zatoka_256.png')  # get image of coast
img = np.flipud(img)    # had to be fliped other case it is up side down
image = ax.imshow(img)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
fig.colorbar(line, ax=ax)

def init():
    """initialize animation"""
    global layer1, dt, image
    image.set_array(img)
    line.set_array(layer1.update())
    time_text.set_text('')
    return line, image, time_text


def animate(*i):
    """perform animation step"""
    global layer1, dt, image, img
    layer1.step(dt)
    image.set_array(img)
    line.set_array(layer1.update())
    time_text.set_text('time = %.1f' % layer1.time_elapsed)
    return line, image, time_text


# choose the interval based on dt and the time to animate one step
from time import time
t0 = time()
animate(0)
t1 = time()
interval = 500 * dt - (t1 - t0)

ani = animation.FuncAnimation(
    fig, animate, frames=300, interval=interval, blit=True, init_func=init)

plt.show()

#plt.figure(1)
plt.plot(land_array)
plt.ylabel('masa ropy osadzona na brzegu')
plt.xlabel('krok czasowy')
plt.show()

# i = 1
# while i < len(evaporation_array):
#     evaporation_array[i] += evaporation_array[i-1]
#     i = i+1

#plt.figure(2)
plt.plot(evaporation_array)
plt.ylabel('Procent odparowanej ropy')
plt.xlabel('krok czasowy')
plt.show()

plt.plot(spill_area_array)
plt.ylabel('Powierzchnia objęta wyciekiem')
plt.xlabel('krok czasowy')
plt.show()

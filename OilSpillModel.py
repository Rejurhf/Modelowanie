# Rejurhf
# 8.01.2019

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.integrate as integrate
import matplotlib.animation as animation
from filecontroller import getArrayFromJSON
from menu import choserArray, choserTemperature

land_array = []
evaporation_rate = 0.00002
time_step = 10

spill_area_array = []
evaporation_array = []

array1 = array2 = np.zeros((256, 256, 800))

class Layer:

    def __init__(self,
                 mass,          # array of oil mass
                 land,
                 velocity_x,    # velocity in X direction (array)
                 diffusion,     # array of diffusion
                 velocity_y,    # velocity in Y direction (array)
                 dx,
                 dy,
                 dt,
                 vertical_dispersion,
                 layer_number,
                 temperature = 20):
        # main array of spill
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
        self.vertical_dispersion = vertical_dispersion
        self.layer_number = layer_number
        self.time_elapsed = 0
        self.temperature = temperature

        self.shorelineConst = 0.15
        self.maximumShorelineDeposition = 12
        self.evapPrc = 0.0000001
        self.totalMassEvap = 0
        self.ifSpilledArray = m = np.zeros((len(self.mass), len(self.mass[0])))

        self.iteration = 0

    def update(self):
        current_mass = self.mass    # array of mass in t
        next_mass = current_mass    # array of mass in t+1

        # managing edges of array edge pixel is equal to neighbor pixel
        next_mass[:, 0] = next_mass[:, 1]
        next_mass[:, -1] = next_mass[:, -2]
        next_mass[0, :] = next_mass[1, :]
        next_mass[-1, :] = next_mass[-2, :]

        # spilling oil
        if self.time_elapsed < 1 and self.layer_number == 1:
            for m in range(90, 100):
                for n in range(70, 80):
                    next_mass[m,n] += 10
                    self.totalMassEvap += 10

        if 0.5 < self.time_elapsed < 1.5 and self.layer_number == 2:
            for m in range(90, 100):
                for n in range(70, 80):
                    next_mass[m,n] += 0.8

        if 1 < self.time_elapsed < 2 and self.layer_number == 3:
            for m in range(90, 100):
                for n in range(70, 80):
                    next_mass[m,n] += 0.1

        # calculate oil spill
        for i in range(1, len(current_mass)-1):
            for j in range(1, len(current_mass[0])-1):

                if self.layer_number == 2 and self.iteration > 0:
                    current_mass[i][j] += self.vertical_dispersion * array1[i][j][self.iteration-1]
                    current_mass[i][j] -= self.vertical_dispersion * array2[i][j][self.iteration-1]

                if self.layer_number == 3 and self.iteration > 0:
                    current_mass[i][j] += self.vertical_dispersion * array2[i][j][self.iteration-1]

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
                    if self.time_elapsed > 0.02 and self.layer_number == 1:
                        evapPrcPrv = self.evapPrc
                        self.evapPrc = ((evaporation_rate + 0.045*(self.temperature-15))*np.log(self.time_elapsed*60))/14

                        totalMass = next_mass[i][j]/evapPrcPrv  # Calculate total mass without evaporation
                        if self.evapPrc <= 0:   # in case devision by 0
                            self.evapPrc = 0.0000001
                        evaporated = totalMass*(self.evapPrc-evapPrcPrv)
                        if evaporated > 0:
                            next_mass[i][j] -= evaporated   # substract
                            # self.totalMassEvap += evaporated

                    if next_mass[i][j] < 0:
                        next_mass[i][j] = np.abs(next_mass[i][j])
                    if next_mass[i][j] > 1:
                        self.ifSpilledArray[i][j] = 1


        # shoreline deposition
        if self.layer_number == 1:
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
        
        if self.layer_number == 1:
            land_array.append(land_sum)
            evaporation_array.append(self.totalMassEvap * self.evapPrc / 10)
            spill_area_array.append((self.ifSpilledArray == 1).sum())

        if self.layer_number == 1:
            for i in range(1, len(next_mass)-1):
                for j in range(1, len(next_mass[0])-1):
                    array1[i][j][self.iteration] = next_mass[i][j]
        elif self.layer_number == 2:
            for i in range(1, len(next_mass)-1):
                for j in range(1, len(next_mass[0])-1):
                    array2[i][j][self.iteration] = next_mass[i][j]

        self.iteration += 1

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
K = 0.01    # Diffusion constant
Rw = 0.01   # Vertical dispersion constant

# land test:
name = choserArray("maps", "Wybierz mapę (preferowana zatoka2):")
if name == "":
    l = np.zeros((ny, nx))
else:
    l = getArrayFromJSON("maps", name)

name = choserArray("leftright", "Wybierz prądy wodne (preferowana zatokatest):")
if name == "":
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
else:
    u = getArrayFromJSON("leftright", name)
    v = getArrayFromJSON("updown", name)

temperature = choserTemperature("Wybierz temperaturę:")

u *= 300
v *= 300
# u = np.zeros((ny, nx))   # velocity moving in x direction advection
# u[:, :] = 0.5
# v = np.zeros((ny, nx))   # velocity moving in y direction advection
# for i in range(ny):
#     for j in range(nx):
#         v[i, j] = (0.1 + 0.001*(i-Ly) + np.sin(np.pi*j/Lx)/4)

# set up initial state and global variables
m = np.zeros((ny, nx))
layer1 = Layer(m, l, u, K, v, dx, dy, dt, Rw, 1, temperature)
m = np.zeros((ny, nx))
layer2 = Layer(m, l, u, K, v, dx, dy, dt, Rw, 2)
m = np.zeros((ny, nx))
layer3 = Layer(m, l, u, K, v, dx, dy, dt, Rw, 3)

#------------------------------------------------------------
img = mpimg.imread('res/zatoka_256.png')  # get image of coast
img = np.flipud(img)    # had to be fliped other case it is up side down
# set up figure and animation
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, autoscale_on=False,xlim=(0, nx-1), ylim=(0, ny-1))
line1 = ax1.imshow(layer1.update(), animated=True)
image1 = ax1.imshow(img)
time_text1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
fig1.colorbar(line1, ax=ax1)

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, autoscale_on=False,xlim=(0, nx-1), ylim=(0, ny-1))
line2 = ax2.imshow(layer2.update(), animated=True)
image2 = ax2.imshow(img)
time_text2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes)
line2.set_clim(0,5)
fig2.colorbar(line2, ax=ax2)

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111, autoscale_on=False,xlim=(0, nx-1), ylim=(0, ny-1))
line3 = ax3.imshow(layer3.update(), animated=True)
image3 = ax3.imshow(img)
time_text3 = ax3.text(0.02, 0.95, '', transform=ax3.transAxes)
line3.set_clim(0,2.5)
fig3.colorbar(line3, ax=ax3)


def init1():
    """initialize animation"""
    global layer1, dt, image1
    image1.set_array(img)
    line1.set_array(layer1.update())
    time_text1.set_text('')
    return line1, image1, time_text1


def animate1(*i):
    """perform animation step"""
    global layer1, dt, image1, img
    layer1.step(dt)
    image1.set_array(img)
    line1.set_array(layer1.update())
    time_text1.set_text('time = %.1f' % layer1.time_elapsed)
    return line1, image1, time_text1


def init2():
    """initialize animation"""
    global layer2, dt, image2
    image2.set_array(img)
    line2.set_array(layer2.update())
    time_text2.set_text('')
    return line2, image2, time_text2


def animate2(*i):
    """perform animation step"""
    global layer2, dt, image2, img
    layer2.step(dt)
    image2.set_array(img)
    line2.set_array(layer2.update())
    time_text2.set_text('time = %.1f' % layer2.time_elapsed)
    return line2, image2, time_text2


def init3():
    """initialize animation"""
    global layer3, dt, image3
    image3.set_array(img)
    line3.set_array(layer3.update())
    time_text3.set_text('')
    return line3, image3, time_text3


def animate3(*i):
    """perform animation step"""
    global layer3, dt, image3, img
    layer3.step(dt)
    image3.set_array(img)
    line3.set_array(layer3.update())
    time_text3.set_text('time = %.1f' % layer3.time_elapsed)
    return line3, image3, time_text3


# choose the interval based on dt and the time to animate one step
from time import time
t0 = time()
animate1(0)
t1 = time()
interval = 500 * dt - (t1 - t0)

ani1 = animation.FuncAnimation(
    fig1, animate1, frames=300, interval=interval, blit=True, init_func=init1)

ani2 = animation.FuncAnimation(
    fig2, animate2, frames=300, interval=interval, blit=True, init_func=init2)

ani3 = animation.FuncAnimation(
    fig3, animate3, frames=300, interval=interval, blit=True, init_func=init3)

plt.show()

plt.figure(1)
plt.plot(land_array)
plt.ylabel('Masa ropy osadzona na brzegu [kg]')
plt.xlabel('czas w godzinach')
plt.show()

# i = 1
# while i < len(evaporation_array):
#     evaporation_array[i] += evaporation_array[i-1]
#     i = i+1

plt.figure(2)
plt.plot(evaporation_array)
plt.ylabel('Ilość odparowanej ropy [kg]')
plt.xlabel('czas w godzinach')
plt.show()

plt.plot(spill_area_array)
plt.ylabel('Powierzchnia objęta wyciekiem [km]')
plt.xlabel('czas w godzinach')
plt.show()

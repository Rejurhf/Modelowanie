# Rejurhf
# 8.01.2019

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.integrate as integrate
import matplotlib.animation as animation
from filecontroller import getArrayFromJSON

#land_array = []
#evaporation_rate = 0.00002
#temperature = 20
#time_step = 10

#evaporation_array = []

global array1, array2

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
                 layer_number):
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

        self.shorelineConst = 0.15
        self.maximumShorelineDeposition = 12

        self.iteration = 0

    def update(self):

        current_mass = self.mass    # array of mass in t
        next_mass = current_mass    # array of mass in t+1
        #evaporated_sum = 0

        # managing edges of array edge pixel is equal to neighbor pixel
        next_mass[:, 0] = next_mass[:, 1]
        next_mass[:, -1] = next_mass[:, -2]
        next_mass[0, :] = next_mass[1, :]
        next_mass[-1, :] = next_mass[-2, :]

        # spilling oil
        if self.time_elapsed < 1 and self.layer_number == 1:
            for m in range(90, 100):
                for n in range(70, 80):
                    next_mass[m,n] += 50

        # calculate oil spill
        for i in range(1, len(current_mass)-1):
            for j in range(1, len(current_mass[0])-1):

                # odejmowanie ropy, która spłynie do niższej warstwy i dodawanie tej, która spłynęła z wyższej
                if self.layer_number == 1 and self.iteration > 0:
                    current_mass[i][j] -= self.vertical_dispersion * array1[i][j][self.iteration-1]

                if self.layer_number == 2 and self.iteration > 0:
                    current_mass[i][j] += self.vertical_dispersion * array1[i][j][self.iteration-1]
                    current_mass[i][j] -= self.vertical_dispersion * array2[i][j][self.iteration-1]

                if self.layer_number == 3 and self.iteration > 1:
                    current_mass[i][j] += self.vertical_dispersion * array2[i][j][self.iteration-1]

                A = D = 0
                if self.land[i][j] < 0 and self.land[i+1][j] >= 0 and \
                        self.land[i-1][j] >= 0 and self.land[i][j+1] < 0 and \
                        self.land[i][j-1] < 0:
                    # Advection term
                    A = self.velocity_x[i][j] * (current_mass[i][j+1] - current_mass[i][j-1])/(2*self.dy)
                    # Diffusion term
                    D = self.diffusion * (current_mass[i][j+1] - 2*current_mass[i][j] + current_mass[i][j-1])/self.dy**2
                elif self.land[i][j] < 0 and self.land[i+1][j] < 0 and \
                        self.land[i-1][j] < 0 and self.land[i][j+1] >= 0 and \
                        self.land[i][j-1] >= 0:
                    # Advection term
                    A = self.velocity_y[i][j] * \
                        (current_mass[i+1][j] - current_mass[i-1][j])/(2*self.dx)
                    # Diffusion term
                    D = self.diffusion * (current_mass[i+1][j] - \
                        2*current_mass[i][j] + current_mass[i-1][j])/self.dx**2

                elif self.land[i][j] < 0:
                        # and self.land[i+1][j] < 0 and self.land[i-1][j] < 0 and
                        # self.land[i][j+1] < 0 and self.land[i][j-1] < 0:
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

                # # Euler's Method
                #evaporated = current_mass[i][j]*evaporation_rate*time_step*temperature
                next_mass[i][j] = current_mass[i][j] + self.dt*(-A + D)# - evaporated
                #evaporated_sum += evaporated
                if next_mass[i][j] < 0:
                    next_mass[i][j] = np.abs(next_mass[i][j])
                if next_mass[i][j] < 0.1:
                    next_mass[i][j] = 0


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

        if self.layer_number == 2 and self.iteration == 0:
            next_mass = np.zeros((256,256))

        if self.layer_number == 3 and (self.iteration == 0 or self.iteration == 1):
            next_mass = np.zeros((256, 256))

        self.mass = next_mass

        if self.layer_number == 1:  #zapisywanie w 3-wymiarowych tablicach(współ x, współ y, iteracja update()) ile było wtedy ropy,
                                    #żeby wiedzieć, ile ma "spłynąć" w dół
            for i in range(1, len(next_mass)-1):
                for j in range(1, len(next_mass[0])-1):
                    array1[i][j][self.iteration] = next_mass[i][j]
        elif self.layer_number == 2:
            for i in range(1, len(next_mass)-1):
                for j in range(1, len(next_mass[0])-1):
                    array2[i][j][self.iteration] = next_mass[i][j]

        self.iteration += 1

        # do wyswietlania:
        tmp = np.zeros((len(current_mass), len(current_mass[0])))
        #land_sum = 0.0
        for i in range(1, len(current_mass)-1):
            for j in range(1, len(current_mass[0])-1):
                if self.land[i][j] < 0:
                    tmp[i][j] = next_mass[i][j]
                else:
                    tmp[i][j] = self.land[i][j]
                    #land_sum += self.land[i][j]
        #land_array.append(land_sum)
        #evaporation_array.append(evaporated_sum)

        return tmp

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
# land test:
l = getArrayFromJSON("maps", "zatoka2")

K = 0.01    # Diffusion constant
Rw = 0.01   # Vertical dispersion constant

array1 = array2 = np.zeros((256, 256, 1000))

u = getArrayFromJSON("leftright", "zatokatest")
v = getArrayFromJSON("updown", "zatokatest")
u *= 100
v *= 100
# u = np.zeros((ny, nx))   # velocity moving in x direction advection
# u[:, :] = 0.5
# v = np.zeros((ny, nx))   # velocity moving in y direction advection
# for i in range(ny):
#     for j in range(nx):
#         v[i, j] = (0.1 + 0.001*(i-Ly) + np.sin(np.pi*j/Lx)/4)

# set up initial state and global variables
layer1 = Layer(m, l, u, K, v, dx, dy, dt, Rw, 1)
layer2 = Layer(m, l, u, K, v, dx, dy, dt, Rw, 2)
layer3 = Layer(m, l, u, K, v, dx, dy, dt, Rw, 3)

#------------------------------------------------------------
# set up figure and animation
fig1 = plt.figure(1)
fig2 = plt.figure(2)
fig3 = plt.figure(3)
ax1 = fig1.add_subplot(111, autoscale_on=False,xlim=(0, nx-1), ylim=(0, ny-1))
ax2 = fig2.add_subplot(111, autoscale_on=False,xlim=(0, nx-1), ylim=(0, ny-1))
ax3 = fig3.add_subplot(111, autoscale_on=False,xlim=(0, nx-1), ylim=(0, ny-1))

line1 = ax1.imshow(layer1.update(), animated=True)
line2 = ax2.imshow(layer2.update(), animated=True)
line3 = ax3.imshow(layer3.update(), animated=True)
img = mpimg.imread('res/zatoka_256.png')  # get image of coast
img = np.flipud(img)    # had to be fliped other case it is up side down
image1 = ax1.imshow(img)
image2 = ax2.imshow(img)
image3 = ax3.imshow(img)
time_text1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
time_text2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes)
time_text3 = ax3.text(0.02, 0.95, '', transform=ax3.transAxes)
fig1.colorbar(line1, ax=ax1)
fig2.colorbar(line2, ax=ax2)
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
    layer1.step()
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
    layer2.step()
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
    layer3.step()
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
plt.show()

ani2 = animation.FuncAnimation(
    fig2, animate2, frames=300, interval=interval, blit=True, init_func=init2)
plt.show()

ani3 = animation.FuncAnimation(
    fig3, animate3, frames=300, interval=interval, blit=True, init_func=init3)
plt.show()

#plt.figure(4)
#plt.plot(land_array)
#plt.ylabel('masa ropy osadzona na brzegu')
#plt.xlabel('krok czasowy')
#plt.show()

#i = 1
#while i < len(evaporation_array):
    #evaporation_array[i] += evaporation_array[i-1]
    #i = i+1

#plt.figure(5)
#plt.plot(evaporation_array)
#plt.ylabel('masa ropy odparowana')
#plt.xlabel('krok czasowy')
#plt.show()

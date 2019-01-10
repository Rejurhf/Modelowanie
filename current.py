# Rejurhf
# 10.01.2019

import numpy as np
from PIL import Image
from filecontroller import saveArrayOfTuplesToJSON

rowmax = colmax = 256
tovisitqueue = [] # que of pixels to visit

current = np.empty((rowmax,colmax),object)    # array with current
array = np.zeros((rowmax,colmax)) # init array with shema of current

try:
    im = Image.open("res/zatoka_current.png")   # read image
    gray = im.convert('L')  # conversion to gray scale
    array = np.asarray(gray)    # conversion to array
except IOError:
    print("cannot create image")

array = np.where(array == 155, 254, array)    # repleacing all 155(orange) with 254
array = np.where(array == 255, 0, array)
redlist = list(zip(*np.where(array == 91)))    # geting tuple of red dot
greenlist = list(zip(*np.where(array == 122)))   # geting tuple of green dot

#---------------------------------------------------------------------------
# manage red and grin pixel
for red, green in zip(redlist, greenlist):   # comment this line if got only one current
    print(red[0], red[1])
    print(green[0], green[1])
    print("----------------")
    if green[0]-red[0] != 0 and green[1]-red[1] != 0:   # check if green in the angle
        current[red[0],red[1]] = \
            (round(((green[0]-red[0])/(2**(1/2))),2), \
            round(((green[1]-red[1])/(2**(1/2))),2))
    else:
        current[red[0],red[1]] = (float(green[0]-red[0]), float(green[1]-red[1]))
    #---------------------------------------------------------------------------
    # manage green pixel
    # if pixel in the edge of array
    if green[0] == rowmax - 1 or green[1] == colmax - 1 or green[0] == 0 or green[1] == 0:
        current[green[0], green[1]] = current[red[0],red[1]]
        for r in range(-1,2):   # get informations about neightboars
            for c in range(-1,2):
                if green[0]+r >= 0 and green[1]+c >= 0 and green[0]+r < rowmax and \
                        green[1]+c < colmax and (r != 0 and c != 0):
                    tovisitqueue.append((green[0]+r,green[1]+c))
    else:
        rowtoadd = 0    # sum of neighbours positions
        coltoadd = 0
        for r in range(-1,2):   # get informations about neightboars
            for c in range(-1,2):
                if array[green[0]+r,green[1]+c] == 254:
                    rowtoadd += r
                    coltoadd += c
                    tovisitqueue.append((green[0]+r,green[1]+c))

        if rowtoadd == 0 and coltoadd == 0:
            current[green[0], green[1]] = current[red[0],red[1]]
        elif rowtoadd == 0 or coltoadd == 0:
            if rowtoadd == 0:
                current[green[0], green[1]] = (0., coltoadd/np.abs(coltoadd))
            else:
                current[green[0], green[1]] = (rowtoadd/np.abs(rowtoadd), 0.)
        else:
            current[green[0], green[1]] = \
                (round(((rowtoadd/(np.abs(rowtoadd)+np.abs(coltoadd)))/(2**(1/2))),2), \
                round(((coltoadd/(np.abs(rowtoadd)+np.abs(coltoadd)))/(2**(1/2))),2))

print(tovisitqueue)
#--------------------------------------------------------------------------
# manage pixels == 254
while len(tovisitqueue) > 0:
    (rtmp, ctmp) = tovisitqueue.pop(0)  # position of curr pixel
    rbackup, cbackup = -1, -1   # here will be saved already visited pixel
    array[rtmp, ctmp] = 1

    if rtmp == rowmax - 1 or ctmp == colmax - 1 or rtmp == 0 or ctmp == 0:
        for r in range(-1,2):   # get informations about neightboars
            for c in range(-1,2):
                if rtmp+r >= 0 and ctmp+c >= 0 and rtmp+r < rowmax and \
                        ctmp+c < colmax:
                    if current[rtmp+r, ctmp+c] is not None:
                        current[rtmp, ctmp] = current[rtmp+r, ctmp+c]

                    if (rtmp+r, ctmp+c) not in tovisitqueue and \
                            array[rtmp+r,ctmp+c] == 254:
                        tovisitqueue.append((rtmp+r,ctmp+c))
        if current[rtmp, ctmp] is None:
            current[rtmp, ctmp] = (0.,0.)
    else:
        rowtoadd = 0    # sum of neighbours positions
        coltoadd = 0
        for r in range(-1,2):   # get informations about neightboars
            for c in range(-1,2):
                if array[rtmp+r,ctmp+c] == 254:
                    rowtoadd += r
                    coltoadd += c
                    if (rtmp+r, ctmp+c) not in tovisitqueue:
                        tovisitqueue.append((rtmp+r,ctmp+c))
                # if curr pixel then skip
                if array[rtmp+r,ctmp+c] == 1 and (r != 0 or c != 0):
                    rbackup, cbackup = rtmp+r, ctmp+c

        if rowtoadd == 0 and coltoadd == 0:
            if rbackup == -1:
                current[rtmp, ctmp] = (0.,0.)
            else:
                current[rtmp, ctmp] = current[rbackup,cbackup]
        elif rowtoadd == 0 or coltoadd == 0:
            if rowtoadd == 0:
                current[rtmp, ctmp] = (0., round(coltoadd/np.abs(coltoadd),2))
            else:
                current[rtmp, ctmp] = (round(rowtoadd/np.abs(rowtoadd),2), 0.)
        else:
            current[rtmp, ctmp] = \
                round((((rowtoadd/(np.abs(rowtoadd)+np.abs(coltoadd)))/(2**(1/2)))),2), \
                round(((coltoadd/(np.abs(rowtoadd)+np.abs(coltoadd)))/(2**(1/2))),2)

#--------------------------------------------------------------------------
# spred influence of current
for i in range (0, 10):
    for r in range(0, rowmax):
        for c in range(0, colmax):
            sumtuple = (0.,0.)  # tuple with sum speed
            counter = 0   # count of added tuples

            if current[r,c] is None or current[r,c] == (0.,0.):
                if r == rowmax - 1 or c == colmax - 1 or r == 0 or c == 0:
                    for ri in range(-1,2):   # get informations about neightboars
                        for ci in range(-1,2):
                            if r+ri >= 0 and c+ci >= 0 and r+ri < rowmax \
                                    and c+ci < colmax and current[r+ri,c+ci] != (0.,0.) \
                                    and current[r+ri,c+ci] is not None:
                                # adding tuples and increse counter
                                sumtuple = \
                                    tuple(map(lambda x, y: x + y, sumtuple, current[r+ri,c+ci]))
                                counter += 1
                else:
                    for ri in range(-1,2):   # get informations about neightboars
                        for ci in range(-1,2):
                            if current[r+ri,c+ci] != (0.,0.) \
                                    and current[r+ri,c+ci] is not None and (current[r+ri,c+ci][0] > 1/(i+5) or current[r+ri,c+ci][0] > 1/(i+5)):
                                # adding tuples and increse counter
                                sumtuple = \
                                    tuple(map(lambda x, y: x + y, sumtuple, current[r+ri,c+ci]))
                                counter += 1


            if counter != 0:
                sumtuple = tuple([round(z/(counter*1.01),2) for z in sumtuple])
                if not (np.abs(sumtuple[0]) < 0.01 and np.abs(sumtuple[1]) < 0.01):
                    current[r,c] = sumtuple

for r in range(0,rowmax):
    for c in range(colmax):
        if current[r,c] is None:
            current[r,c] = (0.,0.)

current = np.flipud(current)

saveArrayOfTuplesToJSON("updown", "leftright", "zatoka", "zatoka", current)

# print(current)

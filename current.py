# Rejurhf
# 10.01.2019

import numpy as np
from PIL import Image
from filecontroller import saveArrayOfTuplesToJSON

try:
    im = Image.open("res/zatoka_current.png")   # read image
except IOError:
    print("cannot create image")

gray = im.convert('L')  # conversion to gray scale
array = np.asarray(gray)    # conversion to array
array = np.flipud(array)
array = np.where(array == 155, 254, array)    # repleacing all 155(orange) with 254
array = np.where(array == 255, 0, array)
redlist = list(zip(*np.where(array == 91)))    # geting tuple of red dot
greenlist = list(zip(*np.where(array == 122)))   # geting tuple of green dot

rowmax = colmax = len(array)
current = np.empty((rowmax,colmax),object)    # array with current
tovisitqueue = [] # que of pixels to visit


#---------------------------------------------------------------------------
# manage red and grin pixel
for red, green in zip(redlist, greenlist):   # comment this line if got only one current
    # print(red[0], red[1])
    # print(green[0], green[1])
    # print("----------------")
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
toVisitCurr = redlist[:]    # get visiting starting position
visitedCurr = []            # list with visited points
# spred influence of current
for i in range(0, 10):
    # find places to visit
    neighbors = [(0,-1), (0,1), (-1, 0), (1,0)]
    tmpToVisit = []     # tmp list of positions to visit in next iteration

    while len(toVisitCurr):
        currField = toVisitCurr.pop(0)  # pop next position
        visitedCurr.append(currField)   # ad current point to visited
        for neighbor in neighbors:
            row = currField[0]+neighbor[0]
            col = currField[1]+neighbor[1]
            # check if out of array
            if row < 0 or col < 0 or row >= rowmax or col >= rowmax or \
                    (row, col) in visitedCurr:
                continue
            # if not visited, visit and update
            elif current[row, col] is None or current[row, col] == (0.,0.):
                sumtuple = (0.,0.)  # tuple with sum speed
                counter = 0   # count of added tuples

                for ri in range(-1,2):   # get informations about neightboars
                    for ci in range(-1,2):
                        if row+ri >= 0 and col+ci >= 0 and row+ri < rowmax \
                                and col+ci < colmax and current[row+ri,col+ci] != (0.,0.) \
                                and current[row+ri,col+ci] is not None:
                            # adding tuples and increse counter
                            sumtuple = \
                                tuple(map(lambda x, y: x + y, sumtuple, current[row+ri,col+ci]))
                            counter += 1

                # if neighbors have values ad avg to current point
                if counter != 0:
                    sumtuple = tuple([round(z/(counter*1.1),2) for z in sumtuple])
                    if not (np.abs(sumtuple[0]) < 0.01 and np.abs(sumtuple[1]) < 0.01):
                        current[row,col] = sumtuple
                        if (row,col) not in visitedCurr and (row,col) not in toVisitCurr and \
                                (row,col) not in tmpToVisit:
                            tmpToVisit.append((row, col))   # add to visit list in next iteration
            # if point has a value and was not visited add to visit list
            elif (row,col) not in visitedCurr and (row,col) not in toVisitCurr and \
                    (row,col) not in tmpToVisit:
                toVisitCurr.append((row, col))

    toVisitCurr = tmpToVisit    # save visit list in next iteration

for r in range(0,rowmax):
    for c in range(colmax):
        if current[r,c] is None:
            current[r,c] = (0.,0.)

saveArrayOfTuplesToJSON("updown", "leftright", "zatokatest", "zatokatest", current)

# print(current)

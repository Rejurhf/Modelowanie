# Rejurhf
# 8.01.2019

from PIL import Image
from filecontroller import saveArrayToJSON
import numpy as np

size = 256, 256
try:
    im = Image.open("zatoka.png") # read image
    im = im.resize(size, Image.ANTIALIAS)   # resize image to size
    gray = im.convert('L')  # conversion to gray scale
    bw = gray.point(lambda x: 0 if x<209 else 255, '1')  # binarization 0 and 255
    bw.save("test_bw.png") # save it

    bw = bw.rotate(90)
    array = np.asarray(bw)
    tmp = np.zeros(size)
    for i in range(len(array)):
        for j in range(len(array[i])):
                if array[j,i]:
                    tmp[254-i,254-j] = 0
                else:
                    tmp[254-i,254-j] = -1

    saveArrayToJSON("maps.json", "zatoka", tmp) # save to json
except IOError:
    print("cannot create image")

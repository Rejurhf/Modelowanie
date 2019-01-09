# Rejurhf
# 8.01.2019

from PIL import Image
from filecontroller import saveArrayToJSON
import numpy as np

size = 256, 256
try:
    im = Image.open("zatokabw.png") # read image
    im = im.resize(size, Image.ANTIALIAS)   # resize image to size
    gray = im.convert('L')  # conversion to gray scale
    bw = gray.point(lambda x: 0 if x<209 else 255, '1')  # binarization 0 and 255
    bw.save("test_bw.png") # save it

    bw = np.flipud(bw)
    array = np.asarray(bw)
    tmp = np.zeros(size)
    for i in range(len(array)):
        for j in range(len(array[i])):
                if array[i,j]:
                    tmp[i,j] = -1
                else:
                    tmp[i,j] = 0

    saveArrayToJSON("maps.json", "zatoka2", tmp) # save to json
except IOError:
    print("cannot create image")

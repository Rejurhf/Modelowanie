# Rejurhf
# 8.01.2019

from PIL import Image
from filecontroller import saveArrayToJSON

size = 256, 256
try:
    im = Image.open("test.jpg") #read image
    im = im.resize(size, Image.ANTIALIAS)
    gray = im.convert('L')  #conversion to gray scale
    bw = gray.point(lambda x: 0 if x<128 else 255, '1')  #binarization
    bw.save("test_bw.png") #save it

    # print(list(bw.getdata())
    bw_bin = gray.point(lambda x: 0 if x<128 else 1, '1')
    saveArrayToJSON("maps.json", "europe", list(bw_bin.getdata()))
except IOError:
    print("cannot create image")

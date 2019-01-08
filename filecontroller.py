# Rejurhf
# 7.01.2019

import json
import numpy as np
import os

def saveArrayToJSON(filename, name, array):
    json_str = ""
    if filename[-5:] != ".json":
        filename += ".json"

    if os.path.exists("res/" + filename):
        fr = open("res/" + filename, 'r')
        json_str = fr.read()
        fr.close()

    fw = open("res/" + filename, 'w')
    if not json_str:
        tmp_dict = {}
    else:
        tmp_dict = json.loads(json_str)
    tmp_dict[name] = array.tolist()
    fw.write(json.dumps(tmp_dict))

    fw.close()

def printJSON(filename):
    if filename[-5:] != ".json":
        filename += ".json"

    if os.path.exists("res/" + filename):
        f = open("res/" + filename, "r")
        json_str = f.read()
        print(json.loads(json_str))
        f.close()

def getArrayFromJSON(filename, arrayname):
    if filename[-5:] != ".json":
        filename += ".json"

    if os.path.exists("res/" + filename):
        f = open("res/" + filename, "r")
        json_str = f.read()
        f.close()

        tmp_dict = json.loads(json_str)
        return tmp_dict[arrayname]


def deleteFile(filename):
    if os.path.exists("res/" + filename):
        os.remove("res/" + filename)
    else:
        print("File does not exist")

# ------------------------------------------------------------
deleteFile("test.json")
K = np.zeros((5,5))
K1 = np.ones((5,5))

saveArrayToJSON("test.json", "test", K)
saveArrayToJSON("test", "test1", K1)

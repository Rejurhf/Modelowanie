# Rejurhf
# 7.01.2019

import json
import numpy as np
import os

def saveArrayToJSON(filename, name, array):
    ''' Convert list or array to json and save it to a file
        filename - name of output file, will be createt if it is not existing;
        name - key in the dict; array - list or numpay array
     '''
    json_str = ""
    if filename[-5:] != ".json":    # check if filename ends with .json
        filename += ".json"

    if os.path.exists("res/" + filename):   # if file exists get content from it
        fr = open("res/" + filename, 'r')
        json_str = fr.read()
        fr.close()

    fw = open("res/" + filename, 'w')   # open file to write data
    # if file is not empty save data to tmp_dict else create empty dict
    if not json_str:
        tmp_dict = {}
    else:
        tmp_dict = json.loads(json_str)

    if isinstance(array,(list,)):   # if array is not list convert it to list
        tmp_dict[name] = array  # create dict from name and array
    else:
        tmp_dict[name] = array.tolist()

    fw.write(json.dumps(tmp_dict))  # save dict to file
    fw.close()

def saveArrayOfTuplesToJSON(filename1, filename2, name1, name2, array):
    ''' Convert list or array to json and save it to a file
        filename - name of output file, will be createt if it is not existing;
        name - key in the dict; array - list or numpay array which wil be splited
    '''
    list1 = []
    list2 = []
    for r in range(0, len(array)):
        tmplist1 = []
        tmplist2 = []
        for c in range(0, len(array[0])):
            x, y = array[r,c]
            tmplist1.append(x)
            tmplist2.append(y)
        list1.append(tmplist1)
        list2.append(tmplist2)

    saveArrayToJSON(filename1, name1, list1)
    saveArrayToJSON(filename2, name2, list2)

def printJSON(filename):
    ''' Print data from json file. filename - name of output file'''
    if filename[-5:] != ".json":    # check if filename ends with .json
        filename += ".json"

    if os.path.exists("res/" + filename):   # if file exists get content from it
        f = open("res/" + filename, "r")
        json_str = f.read() # read data
        print(json.loads(json_str)) # print data
        f.close()

def getArrayFromJSON(filename, arrayname):
    ''' Get array from file
        filename - file with arrays;
        name - key in the dict, specyfiing which array to get
     '''
    if filename[-5:] != ".json":
        filename += ".json"

    if os.path.exists("res/" + filename):
        f = open("res/" + filename, "r")
        json_str = f.read()
        f.close()

        tmp_dict = json.loads(json_str)
        # if array with kay == arrayname exists return it
        if arrayname in tmp_dict:
            return tmp_dict[arrayname]  # return array
    return [[]] # else return empty list

def deleteFile(filename):
    if os.path.exists("res/" + filename):
        os.remove("res/" + filename)
    else:
        print("File does not exist")

def getArrayNamesFromJSON(filename):
    ''' Get array names from file
        filename - file with arrays;
     '''
    if filename[-5:] != ".json":
        filename += ".json"

    if os.path.exists("res/" + filename):
        f = open("res/" + filename, "r")
        json_str = f.read()
        f.close()

        tmp_dict = json.loads(json_str)
        # if array with kay == arrayname exists return it
        keys = list(tmp_dict.keys())
        keys.sort()
        return keys
    return [] # else return empty list
# ------------------------------------------------------------
# deleteFile("test.json")
# K = np.zeros((5,5))
# K1 = np.ones((5,5))
#
# saveArrayToJSON("test.json", "test", K)
# saveArrayToJSON("test", "test1", K1)
# print(getArrayFromJSON("test", "test1"))
# print(getArrayFromJSON("test", "test2"))

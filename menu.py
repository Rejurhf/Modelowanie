# Rejurhf
# 8.01.2019

from filecontroller import getArrayNamesFromJSON, getArrayFromJSON

def choserArray(fileName, informationStr):
    dictionary = getArrayNamesFromJSON(fileName)
    print(informationStr)
    for i in range(0, len(dictionary)):
        print("{} - {}".format(i, dictionary[i]))
    print("{} - Brak".format(len(dictionary)))

    chosen = -1
    while chosen < 0 or chosen > len(dictionary):
        instr = input(">: ")
        if instr.isdigit():
            chosen = int(instr)

    chosenArrayName = ""
    if chosen != len(dictionary):
        chosenArrayName = dictionary[chosen]
    return chosenArrayName

def choserTemperature(informationStr):
    print(informationStr)

    chosen = -1
    while chosen < 0 or chosen > 50:
        instr = input(">: ")
        if instr.isdigit():
            chosen = int(instr)

    return chosen

# maps = getArrayNamesFromJSON("maps")
# chosenMap = choserMenu(maps, "Wybierz mapę: ")
# print(chosenMap)
#
# currents = getArrayNamesFromJSON("updown")
# chosenCurrent = choserMenu(currents, "Wybierz prądy: ")
# print(chosenCurrent)

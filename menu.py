# Rejurhf
# 8.01.2019

from filecontroller import getArrayNamesFromJSON, getArrayFromJSON

def choser(dictionary, informationStr):
    print(informationStr)
    for i in range(0, len(dictionary)):
        print("{} - {}".format(i, dictionary[i]))
    print("{} - Brak".format(len(dictionary)))

    chosen = -1
    while chosen < 0 or chosen > len(maps):
        instr = input(">: ")
        if instr.isdigit():
            chosen = int(instr)

    chosenArrayName = ""
    if chosen != len(dictionary):
        chosenArrayName = dictionary[chosen]
    return chosenArrayName


maps = getArrayNamesFromJSON("maps")
chosenMap = choser(maps, "Wybierz mapę: ")
print(chosenMap)

currents = getArrayNamesFromJSON("updown")
chosenCurrent = choser(currents, "Wybierz prądy: ")
print(chosenCurrent)

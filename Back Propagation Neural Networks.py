import numpy as np
def openfile(road):
    data = np.loadtxt(open(road), delimiter=",", skiprows=1)
    return data
 

import matplotlib.pyplot as plt
import numpy as np

with open('ImagePlaneValues.txt') as file:
    array2d = [[float(digit) for digit in line.split()] for line in file]

Map = plt.imshow(array2d, interpolation='gaussian', cmap='viridis')

plt.colorbar(Map)
plt.title("ImagePlane")
plt.show()

import numpy as np
from matplotlib import pyplot as plot

image_data = np.load('images.npy')

image_data_list = []

for i in range(0, len(image_data), 28):
    image_data_list.append(image_data[i:28])

plot.imshow(image_data_list[0], cmap='gray')
plot.show()
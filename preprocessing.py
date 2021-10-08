import random

import numpy
import numpy as np
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plot

image_data = np.load('images.npy')
label_data = np.load('labels.npy')

flattened_image_data = numpy.reshape(image_data, (6500, -1))

# Split into strata for proper sampling
stratified_labels = [[], [], [], [], [], [], [], [], [], [], []]
stratified_images = [[], [], [], [], [], [], [], [], [], [], []]

# Create training, test, validation sets
training_images = []
test_images = []
validation_images = []
training_labels = []
test_labels = []
validation_labels = []


def split_into_strata():
    i = 0
    while i < 6500:
        class_num = label_data[i]
        stratified_labels[class_num].append(label_data[i])
        stratified_images[class_num].append(flattened_image_data[i])
        i += 1


def split_data_into_training_testing_validation():
    for j in range(10):
        current_label_strata = stratified_labels[j]
        current_image_strata = stratified_images[j]
        current_length = len(current_label_strata)

        for i in range(current_length):
            n = random.randint(0, 100)
            current_categorical_label = to_categorical(current_label_strata[i], 10)
            current_image = current_image_strata[i]
            if n < 60:  # 60% go to training data
                training_images.append(current_image)
                training_labels.append(current_categorical_label)
            elif n < 75:  # 15% go to validation data
                validation_images.append(current_image)
                validation_labels.append(current_categorical_label)
            else:  # 25% go to test data
                test_images.append(current_image)
                test_labels.append(current_categorical_label)


split_into_strata()
split_data_into_training_testing_validation()

print("")
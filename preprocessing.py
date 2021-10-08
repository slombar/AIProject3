import random

import numpy
import numpy as np
from keras.utils.np_utils import to_categorical
import preprocessing
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
        # stratified_labels[class_num] = np.append(stratified_labels[class_num], label_data[i])
        stratified_labels[class_num].append(label_data[i])
        # stratified_images[class_num] = np.append(stratified_images[class_num], flattened_image_data[i])
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
                # np.append(training_images, current_image, axis=0)
                # np.append(training_labels, current_categorical_label, axis=0)
                training_images.append(current_image)
                training_labels.append(current_categorical_label)
            elif n < 75:  # 15% go to validation data
                # np.append(validation_images, current_image, axis=0)
                # np.append(validation_labels, current_categorical_label, axis=0)
                validation_images.append(current_image)
                validation_labels.append(current_categorical_label)
            else:  # 25% go to test data
                # np.append(test_images, current_image, axis=0)
                # np.append(test_labels, current_categorical_label, axis=0)
                test_images.append(current_image)
                test_labels.append(current_categorical_label)


split_into_strata()
split_data_into_training_testing_validation()

print("Num training images: " + str(len(training_images)))
print("Num training labels: " + str(len(training_labels)))


np_training_images = np.asarray(training_images)
np_training_labels = np.asarray(training_labels)

np_validation_images = np.asarray(validation_images)
np_validation_labels = np.asarray(validation_labels)

np_test_images = np.asarray(test_images)
np_test_labels = np.asarray(test_labels)

# print("Shape of training images" + str(training_images.shape))
# print("Shape of training labels" + str(training_labels.shape))
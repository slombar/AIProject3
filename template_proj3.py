from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
import numpy as np
import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

NUM_LAYERS = 2
NUM_NODES_PER_HIDDEN_LAYER = 50
BATCH_SIZE = 50
EPOCHS = 50
ACTIVATION_FUNCTION_1 = 'relu'
ACTIVATION_FUNCTION_2 = 'sigmoid'
WEIGHT_INIT_FUNCTION = 'truncated_normal'

# Model Template

# Beginning of model, don't change:
model = Sequential()  # declare model
model.add(Dense(10, input_shape=(28 * 28,), kernel_initializer='he_normal'))  # first layer
model.add(Activation('relu'))
#####

# Fill in Model Here
model.add(Dense(NUM_NODES_PER_HIDDEN_LAYER, kernel_initializer=WEIGHT_INIT_FUNCTION, activation=ACTIVATION_FUNCTION_1))
if NUM_LAYERS == 2:
    model.add(Dense(NUM_NODES_PER_HIDDEN_LAYER, kernel_initializer=WEIGHT_INIT_FUNCTION,
                    activation=ACTIVATION_FUNCTION_2))

# End of model, don't change
model.add(Dense(10, kernel_initializer='he_normal'))  # last layer
model.add(Activation('softmax'))

# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model

history = model.fit(x=preprocessing.np_training_images, y=preprocessing.np_training_labels,
                    validation_data=(preprocessing.np_validation_images, preprocessing.np_validation_labels),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE)

# Report Results
print(history.history)

score = model.evaluate(preprocessing.np_test_images, preprocessing.np_test_labels, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

predictions = model.predict(x=preprocessing.np_test_images)
conf_matrix = tf.math.confusion_matrix(labels=preprocessing.np_test_labels.argmax(axis=1),
                                       predictions=predictions.argmax(axis=1))

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epoch_array = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,45,46,47,48,49,50]

#plot
plt.plot(epoch_array, accuracy, val_accuracy)
plt.ylabel('Accuracy & Validation Accuracy')
plt.xlabel('Epochs')
plt.show()

print(keras.backend.get_value(conf_matrix))

# Find overall accuracy of model
total_sum = 0
correct_sum = 0
for i in range(10):
    for j in range(10):
        current_num = conf_matrix[i][j]
        if i == j:
            correct_sum += current_num
        total_sum += current_num

total_accuracy = correct_sum / total_sum

# Find precision of each class:
precision_list = []
for row in range(10):
    current_precision_total = 0
    correct_precision_num = 0
    for col in range(10):
        current_num = conf_matrix[col][row]
        current_precision_total += current_num
        if row == col:
            correct_precision_num = current_num
    current_precision = correct_precision_num / current_precision_total
    precision_list.append(keras.backend.get_value(current_precision))

# Find precision of each class:
recall_list = []
for row in range(10):
    current_recall_total = 0
    correct_recall_num = 0
    for col in range(10):
        current_num = conf_matrix[row][col]
        current_recall_total += current_num
        if row == col:
            correct_recall_num = current_num
    current_recall = correct_recall_num / current_recall_total
    recall_list.append(keras.backend.get_value(current_recall))

print("Calculated Accuracy of Model: " + str(keras.backend.get_value(total_accuracy)))
print()
print("Calculated Precisions of Model: ")
print(precision_list)
print()
print("Calculated Recall of Model: ")
print(recall_list)


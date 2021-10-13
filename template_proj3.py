from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
import tensorflow as tf
import preprocessing
import matplotlib.pyplot as plt
import numpy as np

# Experimental variables. The current values are what we used for our best model.
NUM_LAYERS = 2
NUM_NODES_PER_HIDDEN_LAYER = 50
BATCH_SIZE = 50
EPOCHS = 50
ACTIVATION_FUNCTION_1 = 'relu'
ACTIVATION_FUNCTION_2 = 'sigmoid'
WEIGHT_INIT_FUNCTION = 'truncated_normal'

# Input layer, from the project 3 template
model = Sequential()  # declare model
model.add(Dense(10, input_shape=(28 * 28,), kernel_initializer='he_normal'))  # first layer
model.add(Activation('relu'))
#####

# Our model:
model.add(Dense(NUM_NODES_PER_HIDDEN_LAYER, kernel_initializer=WEIGHT_INIT_FUNCTION, activation=ACTIVATION_FUNCTION_1))
if NUM_LAYERS == 2:
    model.add(Dense(NUM_NODES_PER_HIDDEN_LAYER, kernel_initializer=WEIGHT_INIT_FUNCTION,
                    activation=ACTIVATION_FUNCTION_2))

# Output layer, from the project 3 template
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
print("Model History:")
print(history.history)

# Get the overall loss and accuracy of our model.
score = model.evaluate(preprocessing.np_test_images, preprocessing.np_test_labels, verbose=0)
print("Model Evaluation:")
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}' + "\n")

# Create the model's prediction
predictions = model.predict(x=preprocessing.np_test_images)

# Create confusion matrix with test data
conf_matrix = tf.math.confusion_matrix(labels=preprocessing.np_test_labels.argmax(axis=1),
                                       predictions=predictions.argmax(axis=1))

# variables for the matrix -> image conversion
newpredicts = predictions.argmax(axis=1)
newtestlabels = preprocessing.np_test_labels.argmax(axis=1)
storedMatricies = []

# loop until we find 3 images that are incorrect
for i in range(1000):
    # if the value for the prediction is not equal to the value of the test label, then we add this incorrect matrix
    if newpredicts[i] != newtestlabels[i]:
        storedMatricies.append(preprocessing.np_test_images[i])
    # if we have found three images, then break
    if len(storedMatricies) == 3:
        break

# matrices to images
for i in range(3):
    plt.imshow(np.reshape(storedMatricies[i], (28, 28)), cmap="gray")
    plt.show()

# get the accuracy from the model's history to create chart
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# array used to display plot values accurately
epoch_array = range(50)

# Plot the test and validation accuracy over the epochs
plt.plot(epoch_array, accuracy, val_accuracy)
plt.ylabel('Accuracy & Validation Accuracy')
plt.xlabel('Epochs')
plt.show()

print("Confusion Matrix:")
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

print("Calculated Accuracy of Model: " + str(keras.backend.get_value(total_accuracy)) + "\n")
print("Calculated Precisions of Model: ")
print(precision_list)
print()
print("Calculated Recall of Model: ")
print(recall_list)

model.save("best_trained_model")

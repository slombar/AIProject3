from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
import numpy as np
import preprocessing

NUM_LAYERS = 2
NUM_NODES_PER_HIDDEN_LAYER = 50
BATCH_SIZE = 50
EPOCHS = 50
ACTIVATION_FUNCTION_1='relu'
ACTIVATION_FUNCTION_2='sigmoid'
WEIGHT_INIT_FUNCTION='truncated_normal'

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
conf_matrix = tf.math.confusion_matrix(labels=preprocessing.np_test_labels.argmax(axis=1), predictions=predictions.argmax(axis=1))

print(conf_matrix)
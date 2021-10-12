from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import preprocessing

# Model Template

# Beginning of model, don't change:
model = Sequential()  # declare model
model.add(Dense(10, input_shape=(28 * 28,), kernel_initializer='he_normal'))  # first layer
model.add(Activation('relu'))
#####

# Fill in Model Here

# Activation functions to try: selu, tanh, relu
"""
Notes: we should just use a simple network to start with (2 hidden layers) and see if it can get any amount of accuracy in
the final model. Once we have our model working, we can do more experimentation.

All of the possible input arguments we can use for a dense layer:
model.add(Dense(units=10, activation="selu", use_bias=False, kernel_initializer='he_normal', bias_initializer='',
                kernel_regularizer='', bias_regularizer='', activity_regularizer='', kernel_constraint='',
               bias_constraint=''))
"""

# model.add(Dense(32, kernel_initializer='lecun_normal', activation='selu'))
# model.add(Dense(16, kernel_initializer='lecun_normal', activation='selu'))


model.add(Dense(512, kernel_initializer='random_normal', activation='relu'))
# model.add(Dense(100, kernel_initializer='random_normal', activation='selu'))
model.add(Dense(100, kernel_initializer='random_normal', activation='sigmoid'))

# End of model, don't change
model.add(Dense(10, kernel_initializer='he_normal'))  # last layer
model.add(Activation('softmax'))

# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
"""
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=10,
                    batch_size=512)
"""

history = model.fit(x=preprocessing.np_training_images, y=preprocessing.np_training_labels,
                    validation_data=(preprocessing.np_validation_images, preprocessing.np_validation_labels),
                    epochs=50,
                    batch_size=50)

# Report Results

print(history.history)

score = model.evaluate(preprocessing.np_test_images, preprocessing.np_test_labels, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

print(model.predict(x=preprocessing.np_test_images))

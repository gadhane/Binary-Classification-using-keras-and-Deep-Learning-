from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Activation, Dropout
from keras.layers import Dense
from keras import callbacks
import os.path

from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

# Step 1 - Convolution
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
model.add(Activation("relu"))
# Step 2 - Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# Adding a second convolutional layer
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='softmax'))
model.add(Activation("relu"))
model.add(Dropout(0.5))
# Compiling the CNN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Part 2 - Fitting the CNN to the images
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(
    rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')
"""
Tensorboard log
"""
log_dir = 'tf_log'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
cbks = [tb_cb]

# Part 3 - If the CNN weight model already exists make predictions
if os.path.isfile("weights.h5") & os.path.isfile("model.h5"):
    print("CNN Weight and models already exists, make the predictions")
# Part 3 - Else Load the data and store the CNN weight model
else:
    model.fit_generator(training_set,
                        steps_per_epoch=3000,
                        epochs=5,
                        validation_data=test_set,
                        callbacks=cbks,
                        validation_steps=1000)

    model.save("model.h5")
    model.save_weights('weights.h5', overwrite=True)
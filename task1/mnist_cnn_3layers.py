'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
import os
import time

# Options

TRAIN = 1
TEST = 1
SAVE_MODEL = 1
SAVE_WEIGHTS = 1
LOAD_MODEL = 0
LOAD_WEIGHTS = 0 

# Paths to set
model_name = "mnist_cnn_3conv"
model_path = "models_trained/" +model_name+"/"
weights_path = "models_trained/"+model_name+"/weights/"

# Create directories for the models
if not os.path.exists(model_path):
    os.makedirs(model_path)
    os.makedirs(weights_path)

# Network Parameters

batch_size = 128
nb_classes = 10
nb_epoch = 12

# Input image dimensions
img_rows, img_cols = 28, 28
# Number of convolutional filters to use CL1,CL2
nb_filters1 = 32
# Number of convolutional filters to use CL3
nb_filters2 = 16
# Size of pooling area for max pooling
nb_pool = 2
# Convolution kernel size
nb_conv = 3

# Data Loading and Preprocessing

# The data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Data Augmentation


# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


# Model Definition
model = Sequential()

model.add(Convolution2D(nb_filters1, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters1, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters2, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

# Save model architecture

if(SAVE_MODEL):
	json_string = model.to_json()
	f = open(model_path+model_name+".json", 'w')
	f.write(json_string)
	f.close()

# Load model (The epoch has to be set)

if(LOAD_MODEL):
	model = model_from_json(open(model_path+model_name+".json").read())

if(LOAD_WEIGHTS):
	model.load_weights(weights_path+model_name+"weights_epoch1.h5")
	print("Loaded model from disk: "+weights_path+model_name+"weights_epoch1.h5")


model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


# Train the network: Iterate nb_epochs, at the end of each epoch store the model weights, the loss function & the accuracy values

if (TRAIN):
	start_time = time.time()
	f = open(model_path+model_name+"_scores_training.txt", 'w')
	for epoch in range(1,nb_epoch+1):
		print ("Number of epoch: " +str(epoch)+"/"+str(nb_epoch))
		scores = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,
	          verbose=1, validation_data=(X_test, Y_test))
		f.write(str(scores.history)+"\n")
		if(SAVE_WEIGHTS):
			model.save_weights(weights_path+model_name+"weights_epoch"+str(epoch)+".h5")
			print("Saved model to disk in: "+weights_path+model_name+"weights_epoch"+str(epoch)+".h5")
	f.close()

	# Compute time elapsed and save it
	time_elapsed = time.time() - start_time
	print("Time Elapsed: "+str(time_elapsed))
	f = open(model_path+model_name+"_time_elapsed.txt", 'w')
	f.write(str(time_elapsed))
	f.close()

# Test 

if(TEST):
	score = model.evaluate(X_test, Y_test, verbose=0)
	f = open(model_path+model_name+"scores_test.txt", 'w')
	f.write(str(score))
	f.close()
	print('Test score:', score[0])
	print('Test accuracy:', score[1])


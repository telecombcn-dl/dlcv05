'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pylab as pl
import matplotlib.cm as cm
np.random.seed(1337)  # for reproducibility
import theano
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras import backend as K
import numpy.ma as ma
import os
import time


# Options

TRAIN = 0
TEST = 0
SAVE_MODEL = 0
SAVE_WEIGHTS = 0
LOAD_MODEL = 0
LOAD_WEIGHTS = 1 



# Paths to set
model_name = "mnist_cnn_3conv_v2"
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


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)


def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic

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


# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


if(LOAD_MODEL):
	model = model_from_json(open(model_path+model_name+".json").read())

else:
	# Model Definition
	model = Sequential()

	model.add(Convolution2D(nb_filters1, nb_conv, nb_conv,
	                        border_mode='valid',
	                        input_shape=(1, img_rows, img_cols)))
	convout1 = Activation('relu')
	model.add(convout1)
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


# Save model architecture

if(SAVE_MODEL):
	json_string = model.to_json()
	f = open(model_path+model_name+".json", 'w')
	f.write(json_string)
	f.close()

# Load model (The epoch has to be set)

if(LOAD_WEIGHTS):
	model.load_weights(weights_path+model_name+"_weights_epoch1.h5")
	print("Loaded model from disk: "+weights_path+model_name+"_weights_epoch1.h5")

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


# Train the network: Iterate nb_epochs, at the end of each epoch store the model weights, the loss function & the accuracy values
time_elapsed = 0

if (TRAIN):
	f_train = open(model_path+model_name+"_scores_training.txt", 'w')
	f_test = open(model_path+model_name+"_scores_test.txt", 'w')
	f_scores = open(model_path+model_name+"_scores.txt", 'w')
	for epoch in range(1,nb_epoch+1):
		t0 = time.time()
		print ("Number of epoch: " +str(epoch)+"/"+str(nb_epoch))
		scores = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,
	          verbose=1, validation_data=(X_test, Y_test))
		time_elapsed = time_elapsed + time.time() - t0
		print ("Time Elapsed: " +str(time_elapsed))
		if(SAVE_WEIGHTS):
			model.save_weights(weights_path+model_name+"_weights_epoch"+str(epoch)+".h5",overwrite=True)
			print("Saved model to disk in: "+weights_path+model_name+"_weights_epoch"+str(epoch)+".h5")

		score_train = model.evaluate(X_train, Y_train, verbose=0)
		print('Train Loss:', score_train[0])
		print('Train Accuracy:', score_train[1])
		f_train.write(str(score_train)+"\n")
		score_test = model.evaluate(X_test, Y_test, verbose=0)
		print('Test Loss:', score_test[0])
		print('Test Accuracy:', score_test[1])
		f_test.write(str(score_test)+"\n")
		f_scores.write(str(score_train[0])+","+str(score_train[1])+","+str(score_test[0])+","+str(score_test[1])+"\n")

	f_train.close()
	f_test.close()
	f_scores.close()

	# Compute time elapsed and save it
	print("Time Training: "+str(time_elapsed))
	f = open(model_path+model_name+"_time_elapsed.txt", 'w')
	f.write(str(time_elapsed))
	f.close()

# Test 

if(TEST):
	score = model.evaluate(X_test, Y_test, verbose=0)
	f = open(model_path+model_name+"scores_test1.txt", 'w')
	f.write(str(score))
	f.close()
	print('Test score:', score[0])
	print('Test accuracy:', score[1])


W = model.layers[0].W.get_value(borrow=True)
W = np.squeeze(W)
print("W shape : ", W.shape)

pl.figure(figsize=(15, 15))
pl.title('conv1 weights')
nice_imshow(pl.gca(), make_mosaic(W, 6, 6), cmap=cm.binary)
pl.show()


# K.learning_phase() is a flag that indicates if the network is in training or
# predict phase. It allow layer (e.g. Dropout) to only be applied during training
inputs = [K.learning_phase()] + model.inputs

_convout1_f = K.function(inputs, [convout1.output])

def convout1_f(X):
    # The [0] is to disable the training phase flag
    return _convout1_f([0] + [X])

i = 508
X = X_test[i:i+1]
# Visualize convolution result (after activation)
C1 = convout1_f(X)
C1 = np.squeeze(C1)
print("C1 shape : ", C1.shape)

pl.figure(figsize=(15, 15))
pl.suptitle('convout1')
nice_imshow(pl.gca(), make_mosaic(C1, 6, 6), cmap=cm.binary)
pl.show()


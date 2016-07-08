'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
import numpy as np
import load_terrassa as lt
import h5py
import os
import time

# Options

TRAIN = 1
TEST = 1
SAVE_MODEL = 0
SAVE_WEIGHTS = 1
LOAD_MODEL = 0
LOAD_WEIGHTS = 1 
AUGMENTATION = 0

time_elapsed = 0

# Paths to set
model_name_to_load = "cifar10_cnn_v1_batch"
weights_path_to_load = "models_trained/"+model_name_to_load+"/"+"/weights/"
model_name = "cifar10_cnn_v1_terrassa"
model_path = "models_trained/" +model_name+"/"
weights_path = "models_trained/"+model_name+"/weights/"


# Create directories for the models
if not os.path.exists(model_path):
	os.makedirs(model_path)
	os.makedirs(weights_path)

# Network Parameters
batch_size = 32
nb_classes = 13
nb_epoch = 20


# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#X_train, Y_train , X_val , Y_val = lt.load_data(32)

#np.save ("x_train_32",X_train)
#np.save ("y_train_32",Y_train)
#np.save ("x_val_32",X_val)
#np.save ("y_val_32",Y_val)

X_train = np.load("x_train.npy")
Y_train = np.load("y_train.npy")
X_val = np.load("x_val.npy")
Y_val = np.load("y_val.npy")

print('y_train shape:', Y_train.shape)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'test samples')

#X_train = np.load("x_train.npy")
#Y_train = np.load("y_train.npy")
#X_val = np.load("x_val.npy")
#Y_val = np.load("y_val.npy")


print (str(Y_train))

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_val = np_utils.to_categorical(Y_val, nb_classes)

print (str(Y_train))
print (str(Y_train.shape))

if(LOAD_MODEL):
	model = model_from_json(open(model_path+model_name+".json").read())

else:
	model = Sequential()

	model.add(Convolution2D(32, 3, 3, border_mode='same',
							input_shape=(img_channels, img_rows, img_cols)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(64, 3, 3, border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(BatchNormalization())
	model.add(Activation('softmax'))


if(LOAD_WEIGHTS):
	model.load_weights("models_trained/cifar10_cnn_v1_batch/weigths/cifar10_cnn_v1_batch_weights_epoch20.h5")
	print("Loaded model from disk: "+weights_path_to_load + model_name_to_load+"_weights_epoch20.h5")

model.layers.pop()
model.layers.pop()
model.layers.pop()
model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []
model.add(Dense(13))
model.add(BatchNormalization())
model.add(Activation('softmax'))


model.summary()

# Train the model using SGD + momentum 
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
			  optimizer=sgd,
			  metrics=['accuracy'])

# Save model architecture

if(SAVE_MODEL):
	json_string = model.to_json()
	f = open(model_path+model_name+".json", 'w')
	f.write(json_string)
	f.close()


X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_val /= 255



if (TRAIN):
	print ("Training the model")
	f_scores = open(model_path+model_name+"_scores.txt", 'w')

	if(AUGMENTATION):
		print('Using real-time data augmentation')

		# this will do preprocessing and realtime data augmentation
		datagen = ImageDataGenerator(
			featurewise_center=False,  # set input mean to 0 over the dataset
			samplewise_center=False,  # set each sample mean to 0
			featurewise_std_normalization=False,  # divide inputs by std of the dataset
			samplewise_std_normalization=False,  # divide each input by its std
			zca_whitening=False,  # apply ZCA whitening
			rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
			width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
			height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
			horizontal_flip=True,  # randomly flip images
			vertical_flip=False)  # randomly flip images

		# compute quantities required for featurewise normalization
		# (std, mean, and principal components if ZCA whitening is applied)
		datagen.fit(X_train)

		for epoch in range(1,nb_epoch+1):
			t0 = time.time()
			print ("Number of epoch: " +str(epoch)+"/"+str(nb_epoch))
			scores = model.fit_generator(datagen.flow(X_train, Y_train,
							batch_size=batch_size),
							samples_per_epoch=X_train.shape[0],
							nb_epoch=1,
							validation_data=(X_test, Y_test))
			time_elapsed = time_elapsed + time.time() - t0
			print ("Time Elapsed: " +str(time_elapsed))
			
			if(SAVE_WEIGHTS):
				model.save_weights(weights_path+model_name+"_weights_epoch"+str(epoch)+".h5")
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
		
	else:

		print('Not using data augmentation')
		
		for epoch in range(1,nb_epoch+1):
			t0 = time.time()
			print ("Number of epoch: " +str(epoch)+"/"+str(nb_epoch))
			scores = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,
				  verbose=1, validation_data=(X_val, Y_val))
			time_elapsed = time_elapsed + time.time() - t0
			print ("Time Elapsed: " +str(time_elapsed))

			if(SAVE_WEIGHTS):
				model.save_weights(weights_path+model_name+"_weights_epoch"+str(epoch)+".h5")
				print("Saved model to disk in: "+weights_path+model_name+"_weights_epoch"+str(epoch)+".h5")
			
			score_train = model.evaluate(X_train, Y_train, verbose=0)
			print('Train Loss:', score_train[0])
			print('Train Accuracy:', score_train[1])
			
			score_test = model.evaluate(X_val, Y_val, verbose=0)
			print('Test Loss:', score_test[0])
			print('Test Accuracy:', score_test[1])
			
			f_scores.write(str(score_train[0])+","+str(score_train[1])+","+str(score_test[0])+","+str(score_test[1])+"\n")
		
	f_scores.close()

	# Save time elapsed
	f = open(model_path+model_name+"_time_elapsed.txt", 'w')
	f.write(str(time_elapsed))
	f.close()

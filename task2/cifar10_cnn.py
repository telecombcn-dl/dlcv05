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
import os
import time

# Options

TRAIN = 0
TEST = 0
SAVE_MODEL = 0
SAVE_WEIGHTS = 0
LOAD_MODEL = 0
LOAD_WEIGHTS = 0 
AUGMENTATION = 0

time_elapsed = 0

# Paths to set
model_name = "cifar10_cnn_v1_AUG"
model_path = "models_trained/" +model_name+"/"
weights_path = "models_trained/"+model_name+"/weights/"

# Create directories for the models
if not os.path.exists(model_path):
	os.makedirs(model_path)
	os.makedirs(weights_path)

# Network Parameters
batch_size = 32
nb_classes = 10
nb_epoch = 20


# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('y_train shape:', y_train.shape)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print (str(y_train))

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print (str(Y_train))
print (str(Y_train.shape))

if(LOAD_MODEL):
	model = model_from_json(open(model_path+model_name+".json").read())

else:
	model = Sequential()

	model.add(Convolution2D(32, 3, 3, border_mode='same',
							input_shape=(img_channels, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(64, 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
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
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255



# Load model (The epoch has to be set)

if(LOAD_WEIGHTS):
	model.load_weights(weights_path+model_name+"_weights_epoch1.h5")
	print("Loaded model from disk: "+weights_path+model_name+"_weights_epoch1.h5")

if (TRAIN):
	print ("Training the model")
	f_train = open(model_path+model_name+"_scores_training.txt", 'w')
	f_test = open(model_path+model_name+"_scores_test.txt", 'w')
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
				  verbose=1, validation_data=(X_test, Y_test))
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
		
	f_train.close()
	f_test.close()
	f_scores.close()

	# Save time elapsed
	f = open(model_path+model_name+"_time_elapsed.txt", 'w')
	f.write(str(time_elapsed))
	f.close()

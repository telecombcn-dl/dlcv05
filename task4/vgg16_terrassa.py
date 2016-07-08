from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
import time
import numpy as np
import load_terrassa as lt
import os
import h5py


SAVE_WEIGHTS = 0

# Paths to set
model_name = "vgg_terrassa_ft"
model_path = "models_trained/" +model_name+"/"
weights_path = "models_trained/"+model_name+"/weights/"

# Create directories for the models
if not os.path.exists(model_path):
	os.makedirs(model_path)
	os.makedirs(weights_path)

nb_classes = 13
nb_epoch = 5
batch_size = 16

#X_train, Y_train , X_val , Y_val = lt.load_data()

#np.save ("x_train",X_train)
#np.save ("y_train",Y_train)
#np.save ("x_val",X_val)
#np.save ("y_val",Y_val)

X_train = np.load("x_train.npy")
Y_train = np.load("y_train.npy")
X_val = np.load("x_val.npy")
Y_val = np.load("y_val.npy")

Y_train = Y_train.reshape(450,1)
Y_val = Y_val.reshape(180,1)

print('y_train shape:', Y_train.shape)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'test samples')

Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_val = np_utils.to_categorical(Y_val, nb_classes)

print("Ytrain")
print str(Y_train)
print (str(Y_train.shape))
print("test")
print str(Y_val)
print (str(Y_val.shape))

X_train = X_train.astype('float32')
X_test = X_val.astype('float32')
X_train /= 255
X_val /= 255

time_elapsed = 0

def VGG_16_Terrassa(weights_path=""):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
	model.add(Convolution2D(64, 3, 3, activation='relu',trainable = False))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu',trainable = False))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu',trainable = False))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu',trainable = False))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu',trainable = False))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu',trainable = False))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu',trainable = False))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu',trainable = False))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu',trainable = False))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu',trainable = False))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu',trainable = False))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu',trainable = False))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu',trainable = False))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu',trainable = False))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu',trainable = False))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='softmax'))

	if weights_path:
		print ("Weights Loaded")
		model.load_weights(weights_path)

	model.layers.pop()
	model.layers.pop()
	model.outputs = [model.layers[-1].output]
	model.layers[-1].outbound_nodes = []
	model.add(Dense(13, activation='softmax'))

	return model

model = VGG_16_Terrassa('vgg16_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


f_scores = open(model_path+model_name+"_scores.txt", 'w')


for epoch in range(1,nb_epoch+1):
	t0 = time.time()
	print ("Number of epoch: " +str(epoch)+"/"+str(nb_epoch))
	model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1, verbose=1, validation_data=(X_val, Y_val))
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




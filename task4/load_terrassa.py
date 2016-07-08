from scipy.misc import imread, imresize, imsave
import numpy as np

def preprocess_class(str):
	if (str == "mnactec"):
		return 0
	elif (str =="mercat_independencia"):
		return 1
	elif (str =="ajuntament"):
		return 2
	elif (str =="societat_general"):
		return 3
	elif (str =="estacio_nord"):
		return 4
	elif (str =="dona_treballadora"):
		return 5
	elif (str =="escola_enginyeria"):
		return 6
	elif (str =="catedral"):
		return 7
	elif (str =="teatre_principal"):
		return 8
	elif (str =="farmacia_albinyana"):
		return 9
	elif (str =="masia_freixa"):
		return 10
	elif (str =="castell_cartoixa"):
		return 11
	elif (str =="desconegut"):
		return 12
	else:
		#print ("Error")
		return -1


def preprocess_image(image_path,img_size):
	img = imresize(imread(image_path), (img_size, img_size))
	return img

def load_data(img_size):
	image_train_list_path = "TerrassaBuildings900/train/train.txt"
	image_train_labels_path = "TerrassaBuildings900/train/annotation.txt"
	image_train_path = "TerrassaBuildings900/train/images/"
	image_val_list_path = "TerrassaBuildings900/val/val.txt"
	image_val_labels_path = "TerrassaBuildings900/val/annotation.txt"
	image_val_path = "TerrassaBuildings900/val/images/"
	

	nb_train_samples = 450

	nb_val_samples = 180

	X_train = np.zeros((nb_train_samples, 3, img_size, img_size), dtype="uint8")
	y_train = np.zeros((nb_train_samples,), dtype="uint8")
	X_val = np.zeros((nb_val_samples, 3, img_size, img_size), dtype="uint8")
	y_val = np.zeros((nb_val_samples,), dtype="uint8")

	i = 0
	for line in open(image_train_list_path):  
		line = line.rstrip('\n')
	 	print line
	 	X_train[i,:,:,:] = np.transpose(np.array(preprocess_image(image_train_path+line,img_size)))
	 	i += 1

	i = 0
	for line in open(image_val_list_path):  
	 	line = line.rstrip('\n')
	 	print line
	 	X_val[i,:,:,:] = np.transpose(np.array(preprocess_image(image_val_path+line,img_size)))
	 	i += 1

	i = 0
	for line in open(image_train_labels_path):
		category = line.rstrip('\n')
		print category
		num = preprocess_class(category)
		if(num!=-1):
			y_train[i] = np.array([num])
			i += 1

	print str(y_train)

	i = 0
	for line in open(image_val_labels_path):
		category = line.rstrip('\n')
		print category
		num = preprocess_class(category)
		if(num!=-1):
			y_val[i] = np.array([num])
			i += 1
	
	return X_train, y_train , X_val ,y_val



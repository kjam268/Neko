from utils import *
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy.ndimage.measurements import center_of_mass


def create_dataset(path = "../Data/InputData/Testing Set/", nrrd_MRI_file= "lgemri.nrrd", nrrd_label_file = 'laendo.nrrd', img_size = 576, nb_data = 5, savedir = 'Results/Session_Dump', test = False):

	create_folder(savedir)

	data_names = os.listdir(path)

	if nb_data > len(data_names):
		nb_data = len(data_names)
		print ('Number of testing files =', len(data_names),'\n\n')

	image_array = np.zeros([nb_data*88, img_size, img_size])
	centroid_array = np.zeros([nb_data*88,2])
	x = 0

	for number in tqdm(range(nb_data), ncols = 100):

		image = load_nrrd(os.path.join(path, data_names[number], nrrd_MRI_file))
		label = load_nrrd(os.path.join(path, data_names[number], nrrd_label_file))//255

		###Size normalization for all dataset (image > 576*576 resized)
		if image.shape[1] > 576:
			image = cropping(img_size, image)
			label = cropping(img_size, label)

		if test:	
			# Creating the folder for the testing data
			folder_name = savedir+"OutputData/"+str(number)+' - '+data_names[number]
			create_folder (folder_name+'/Prediction')

			# Initialization of arrays for the testing data
			image_array = np.zeros([image.shape[0], image.shape[1], image.shape[2]])
			centroid_array = np.zeros([image.shape[0], 2])
			x = 0

		for slice in range(label.shape[0]):

			if np.max(label[slice]) != 0:
				centroid = center_of_mass(label[slice])
			else:
				centroid = np.array([label[slice].shape[1]//2, label[slice].shape[1]//2])

			image_array[x] = image[slice]
			centroid_array[x] = centroid

			x += 1

		if test:
			np.save(folder_name+"/test_image.npy",image_array)
			np.save(folder_name+"/test_centroid.npy",centroid_array)

		
	if not test:
		# Normalization (data - mean)/std
		image_array, mean, std = image_norm(image_array)
		## Between [-1,1]
		# centroid_array = 2*((centroid_array-np.min(centroid_array))/(np.max(centroid_array)-np.min(centroid_array))) - 1
		## Between [0,1]
		#centroid_array = (centroid_array-np.min(centroid_array))/(np.max(centroid_array)-np.min(centroid_array))
		centroid_array = centroid_array/img_size
		# One hot array change
		image_array = np.reshape(image_array, newshape=[-1, image_array.shape[1], image_array.shape[2], 1])
		#image_array, centroid_array = data_reshape(image_array, centroid_array)

		# dataset = {"images":image_array, 'centroids':centroid_array}
		# return dataset

		return	image_array, centroid_array, mean, std
	else:
		return



def image_norm(dataset):

	# calculate mean and standard deviation
	mean = np.mean(dataset)
	std = np.std(dataset)

	# Sample wise normalisation
	dataset = (dataset - mean)/std

	return dataset, mean, std


def data_augmentation(images, labels, data_aug, probability = 0.5):

	print ('\n\n- Performing data augmentation...\n\n')

	images_aug_array = np.zeros([images.shape[0],images.shape[1],images.shape[2]])
	labels_aug_array = np.zeros([labels.shape[0],labels.shape[1],labels.shape[2]])

	for slice in range(images.shape[0]):

		image_aug, label_aug = img_aug(images[slice], labels[slice], data_aug, probability)

		images_aug_array[slice] = image_aug
		labels_aug_array[slice] = label_aug

	return images_aug_array, labels_aug_array


def data_reshape(images, labels, img_size):

	# One-hot encoding
	temp = np.empty(shape=[labels.shape[0], img_size, img_size, 2])
	temp[:,:,:,0] = 1-labels
	temp[:,:,:,1] = labels

	images = np.reshape(images, newshape=[-1, img_size, img_size, 1])
	labels = np.reshape(temp, newshape=[-1, img_size, img_size, 2])

	return images, labels


def show_keypoints(image, key_point):
	plt.imshow(image, interpolation='nearest', cmap='bone')
	plt.scatter(key_point[1],key_point[0], s=15, marker='.', c='lightgreen')

	## LINES TO ADD TO SHOW THE PICTURE
	# fig = plt.figure()
	# ax = plt.subplot(1,3,1)
	# ax.set_title('List')
	# show_keypoints(test_image_list[45], test_centroid_list[45])
	# plt.show()


# def create_training_dataset(training_path = "../Data/InputData/Training Set/",   nrrd_MRI_file= "lgemri.nrrd", nrrd_label_file = 'laendo.nrrd', img_size = 576, nb_train = 1, data_aug = False, savedir = 'Results/Session_Dump'):

# 	print ('\n- Creating training dataset\n\n')

# 	create_folder(savedir)

# 	training_files = os.listdir(training_path)

# 	if nb_train > len(training_files):
# 		nb_train = len(training_files)
# 		print ('Number of training file =', len(training_files),'\n\n')


# 	train_image_array = np.zeros([nb_train*88, 576, 576])
# 	train_centroid_array = np.zeros([nb_train*88, 2])
# 	x = 0

# 	for file in training_files[0:nb_train]:

# 		train_image = load_nrrd(os.path.join(training_path, file, nrrd_MRI_file))
# 		train_label = load_nrrd(os.path.join(training_path, file, nrrd_label_file))//255.0

# 		###Size normalization for all dataset (image > 576*576 resized)
# 		if train_image.shape[1] > 576:
# 			train_image = cropping(576, train_image)
# 			train_label = cropping(576, train_label)

# 		for slice in range(train_label.shape[0]):

# 			if np.max(train_label[slice]) != 0:
# 				centroid = center_of_mass(train_label[slice])
# 			else:
# 				centroid = (train_label[slice].shape[0]//2, train_label[slice].shape[1]//2)

# 			train_image_array[x] = train_image[slice]
# 			train_centroid_array[x] = centroid

# 			x += 1

# 	training_set = {"images":train_image_array, 'centroids':train_centroid_array}

# 	return training_set



# def create_testing_dataset(testing_path = "../Data/InputData/Testing Set/",   nrrd_MRI_file= "lgemri.nrrd", nrrd_label_file = 'laendo.nrrd', img_size = 576, nb_test = 5, data_aug = False, savedir = 'Results/Session_Dump'):

# 	print ('\n- Creating testing dataset\n\n')

# 	create_folder(savedir)

# 	testing_files = os.listdir(testing_path)

# 	if nb_test > len(testing_files):
# 		nb_test = len(testing_files)
# 		print ('Number of testing file =', len(testing_files),'\n\n')


# 	test_image_array = np.zeros([nb_test*88,576,576])
# 	test_centroid_array = np.zeros([nb_test*88,2])
# 	x = 0

# 	for file in testing_files[0:nb_test]:

# 		test_image = load_nrrd(os.path.join(testing_path, file, nrrd_MRI_file))
# 		test_label = load_nrrd(os.path.join(testing_path, file, nrrd_label_file))//255

# 		###Size normalization for all dataset (image > 576*576 resized)
# 		if test_image.shape[1] > 576:
# 			test_image = cropping(576, test_image)
# 			test_label = cropping(576, test_label)

# 		for slice in range(test_label.shape[0]):

# 			if np.max(test_label[slice]) != 0:
# 				centroid = center_of_mass(test_label[slice])
# 			else:
# 				centroid = (test_label[slice].shape[0]//2, test_label[slice].shape[1]//2)

# 			test_image_array[x] = test_image[slice]
# 			test_centroid_array[x] = centroid

# 			x += 1


# 	testing_set = {"images":test_image_array, 'centroids':test_centroid_array}

# 	return testing_set

if __name__ == '__main__':

	start = time.time()

	#create_training_dataset()
	#create_testing_dataset()
	create_dataset()
	print("Time ",time.time()-start) 

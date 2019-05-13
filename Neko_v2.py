import os
from utils import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

if os.uname()[1] != "hpc2.bioeng.auckland.ac.nz":
	os.environ["CUDA_VISIBLE_DEVICES"]="0"
else:
	os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time
from models import *
from data_process_v2 import *
from utils import *

from PIL import Image

def main():

	start_time = time.time()

	#################
	# Variable preparation
	#################

	# Variable set paths
	training_path = "../Data/InputData/Training Set/"
	testing_path = "../Data/InputData/Testing Set/"
	nrrd_MRI_file = "lgemri.nrrd"
	nrrd_LABEL_file = "laendo.nrrd"

	comments = "Neko - 1"

	# Variable set values
	session = "1"
	epoch = 50
	nb_train = 80
	nb_test = 20
	if training_path == "../Data/InputData/_Training Set/":
		nb_train = 100
		nb_test = 54
	img_size = 576
	batch_size = 10
	kernel_size = 3
	lr = 0.001
	keep_rate = 0.8
	loss_function = "dice_loss" #"categorical_crossentropy"
	weights = False #"Neko_centroid"# baseline_Zhao/Zhao_5x5" #"baseline/Vnet_baseline_weights"


	data_aug = False #["aug_rotation(image, label, (-20,20), probability = 1)","aug_scale(image, label, probability = 1)","aug_flipLR(image, label, probability = 1)"]

	''' Data augmentation available to put in the list of data augmentation
	"aug_rotation(image, label, (-20,20), probability = 1)" 
	"aug_scale(image, label, probability = 1)"
	"aug_perspective(image, label, probability = 1)"
	"aug_add(image, label, probability = 1)"
	"aug_contrastNorm(image, label, probability = 1)"
	"aug_gammaContrast(image, label, probability = 1)"
	"aug_elastic_T(image, labe, image.shape[0]*3, image.shape[0]*0.05)"
	"aug_unsharp(image, label, probability = 1)"
	'''

	# Directories creation
	savedir = "Results/Session_"+session+"/"
	mod_dir = savedir+'model/'
	log_dir = savedir+"logs/"

	if not os.path.lexists(savedir):
		os.makedirs(savedir)
		os.makedirs(mod_dir)
		os.makedirs(log_dir)



	###########################################
	# Testing data processing and model loading
	###########################################


	print(NEKO())

	log(nrrd_LABEL_file,
		img_size,
		session,
		epoch,
		nb_train,
		nb_test,
		kernel_size,
		lr,
		loss_function, 
		batch_size, 
		weights,
		data_aug,
		comments,
		savedir)


	### Loading model

	print ('\nLoading model...\n\n')
	# Building the model and loading the weights
	model = Znet((None, img_size, img_size, 1), 8, kernel_size, keep_rate, lr, log_dir)
	if weights:
		model.load("Weights/"+weights)


	################################
	# Datasets creation and Pre-processing
	################################

	#Creation of training dataset without data augmentation
	train_image, train_centroid, train_mean, train_SD = create_dataset(training_path, nrrd_MRI_file, nrrd_LABEL_file, img_size, nb_train, savedir, test = False)
	
	#Creation of the testing dataset
	create_dataset(testing_path, nrrd_MRI_file, nrrd_LABEL_file, img_size, nb_test, savedir, test = True)

	best_epoch = 0

	centroid_mse = (0,0)
	best_mse = (0,0)

	################################
	# Training and Data augmentation
	################################


	for e in range(epoch):

		tic = time.time()

		print ("\n\n\n --- Session", session, "- Epoch",str(e+1)+"/"+str(epoch)+'\n')

		Epoch_details = open(savedir+'Epoch_details.txt','a')
		Epoch_details.write("-"*75+" Epoch "+str(e+1)+"\n\n")
		Epoch_details.close()


		#Data augmentation and training
		if data_aug:
			
			train_image_aug, train_label_aug = data_augmentation(train_image, train_centroid, data_aug, probability = 0.5)
			model.fit(train_image_aug, train_label_aug, n_epoch=1, show_metric=True, batch_size=batch_size, shuffle=True)

		else :
			#train_image_list, train_label_list = data_reshape(train_image_list, train_label_list, img_size)
			model.fit(train_image, train_centroid, n_epoch=1, show_metric=True, batch_size=batch_size, shuffle=True)


		###Prediction and evaluation
		centroid_mse = pred(model, train_mean, train_SD, img_size, savedir)

		# ###Save the best results
		if centroid_mse[0] >= best_mse[0] and centroid_mse[1] >= best_mse[1]:
			best_mse = centroid_mse
			best_epoch = e+1

		###Saving the best model after epoch 20
		if e > 20 and centroid_mse >= best_mse:
			model.save(mod_dir+"model.tfl")	

		###RAW scores 
		Raw_scores = open(savedir+'Raw_scores.txt','a')
		Raw_scores.write(str(centroid_mse)+'\n')
		Raw_scores.close()

		toc = time.time()
		print ('Remaining time:', time.strftime("%H:%M:%S", time.gmtime((toc-tic)*(epoch-(e+1)))), '\n')

	quit()
	###Final details
	elapsed_time = time.time() - start_time

	Session_log = open(savedir+'Details.txt','a')
	Session_log.write("\n"+"-"*30+'\n')
	Session_log.write("Best Epoch: "+str(best_epoch)+"\n")
	Session_log.write("\nCentroid mse: ("+str(centroid_mse[0])+','+str(centroid_mse[1])+')')
	Session_log.write('\n\nTotal time : ' + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + '\n')
	Session_log.close()

	total = open('Results/Resume.txt','a')
	total.write(savedir+": "+str(best_mse)+'\n')
	total.close()


	plot_learn(savedir)

	print('\n\n')
	print ('*'*15,'Best score','*'*15)
	print ('*')
	print ('* \tDice score:', np.round(best_mse, 4))
	print ('* \tCentroid MSE:', np.round(centroid_mse, 4))
	print ('*')
	print ('*'*40,'\n')
	print ('\n--- Total time :', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),'\n\n')

if __name__ == '__main__':

	main()
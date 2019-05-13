import os, glob, re, random, time, datetime
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
import cv2

from scipy import ndimage
import matplotlib.pyplot as plt

def atoi(text):
	return int(text) if text.isdigit() else text


def natural_keys(text):
	''' alist.sort(key=natural_keys) sorts in human order
	http://nedbatchelder.com/blog/200712/human_sorting.html
	(See Toothy's implementation in the comments)'''
	return [ atoi(c) for c in re.split('(\d+)', text) ]


def glab(path):
	'''Glob with natural sorting'''
	return sorted(glob.glob(path),key=natural_keys)


def create_folder(full_path_filename):
	# this function creates a folder if its not already existed
	if not os.path.exists(full_path_filename):
		os.makedirs(full_path_filename)

	return


def load_nrrd(full_path_filename):
	'''this function loads .nrrd files into a 3D matrix and outputs it
	the input is the specified file path to the .nrrd file'''
	data = sitk.ReadImage( full_path_filename )
	data = sitk.Cast( sitk.RescaleIntensity(data), sitk.sitkUInt8 )
	data = sitk.GetArrayFromImage(data)

	return data



def log(img_type, 
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
	savedir):

	'''Writing the log file containing important information
	to track the changes and parameters'''

	file = open(savedir+'Details.txt','w')

	date = datetime.datetime.now()

	file.write(RTBF())

	file.write("Date : "+date.strftime("%d/%m/%Y, %H:%M:%S")+'\n')
	file.write("Machine:"+os.uname()[1]+'\n')
	file.write("-"*30+'\n')
	file.write("Saving directory : "+savedir+'\n')
	file.write("Images size : "+str(img_size)+'x'+str(img_size)+'\n')
	file.write("Images processed : "+str(img_type)+'\n')
	file.write("Number of Epoch = "+str(epoch)+'\n')
	file.write("Number of Training data = "+str(nb_train)+'\n')
	file.write("Number of Testing data = "+str(nb_test)+'\n')
	file.write("Filters size = "+str(kernel_size)+'\n')
	file.write("Learning rate = "+str(lr)+'\n')
	file.write("Loss function = "+str(loss_function)+'\n')
	file.write("Batch size = "+str(batch_size)+'\n')
	file.write("Weights loaded = "+str(weights)+'\n')
	file.write("Data augmentation = "+str(data_aug)+'\n\n')
	file.write("Comments :"+str(comments)+"\n")

	file.close()

	#Displaying the different components of the run
	print ('*'*15,"Session",session,'*'*15,"\n")
	print (" - Date :", date.strftime("%d/%m/%Y, %H:%M:%S"))
	print (" - Saving directory :", savedir)
	print (" - Images size :", str(img_size), 'x', str(img_size))
	print (" - Number of Epoch :", str(epoch))
	print (" - Number of Training data :", str(nb_train))
	print (" - Number of Testing data :", str(nb_test))
	print (" - Filters size :", str(kernel_size))
	print (" - Learning rate :", str(lr))
	print (" - Loss function :", str(loss_function))
	print (" - Batch size :", str(batch_size))
	print (" - Weights loaded :", str(weights))
	print (" - Data augmentation :", str(data_aug), '\n')
	print (" - Comments :", str(comments),"\n")
	print ("*"*41,"\n")


def cropping(img_size, data):

	# Base on the image size cropped image according to the desired size

	if len(data.shape) == 4:
		midpoint = data.shape[2]//2	
	else:
		midpoint = data.shape[1]//2	

	start, end = midpoint - int(img_size/2), midpoint + int(img_size/2)

	if len(data.shape) == 3:
		data = data[0:data.shape[0], start:end, start:end]
	elif len(data.shape) == 2:
		data = data[start:end, start:end]
	elif len(data.shape) == 4:
		data = data[0:data.shape[0], 0:data.shape[1], start:end, start:end]
	else:
		print ("\nERROR: Bad cropping shape")
		quit()

	return data


def pred(CNN_model, mu = 0.2, sd = 0.1, img_size = 576, savedir = 'Dump'):

	test_folders = glab(savedir+"OutputData/*")

	details = open(savedir+"Epoch_details.txt","a")
	
	print ("\nPredicting ...\n\n")

	predict_dict = {}
	groundt_dict = {}

	
	temp_mse = []
	mse_average = (0,0)
	k = 1 #Value for the class evaluated

	for folder in test_folders:
		#groundT = []
		test_image = np.load(folder+'/test_image.npy')
		GT_centroid = np.load(folder+'/test_centroid.npy')
		#GT_centroid = 2*((GT_centroid-np.min(GT_centroid))/(np.max(GT_centroid)-np.min(GT_centroid))) - 1
		#GT_centroid_norm = (GT_centroid-np.min(GT_centroid))/(np.max(GT_centroid)-np.min(GT_centroid))
		GT_centroid_norm = GT_centroid/img_size

		prediction = np.zeros(shape=[test_image.shape[0], img_size, img_size])
		temp_Input = np.zeros(shape=[test_image.shape[0], img_size, img_size])
		temp_Output = np.zeros(shape=[test_image.shape[0], 2])
		centroid_mse = np.zeros([88, 2])
		for number, slice in enumerate(test_image):

			temp_Input[number,:,:] = (slice - mu)/sd
			temp_Output[number,:] = CNN_model.predict([temp_Input[number,:,:,None]])

			centroid_mse[number,0] = (temp_Output[number][0] - GT_centroid_norm[number][0])**2
			centroid_mse[number,1] = (temp_Output[number][1] - GT_centroid_norm[number][1])**2

			mse_average = (mse_average[0]+centroid_mse[number,0], mse_average[1]+centroid_mse[number,1])

		print ('GT',GT_centroid_norm[number,0]*576,GT_centroid_norm[number,1]*576)
		#print ('GT',GT_centroid_norm[number][0]*576,GT_centroid_norm[number][1]*576)
		print ('Pred',temp_Output[number,0]*576, temp_Output[number,1]*576)
		#print ('Pred',centroid_mse[number][0]*576, centroid_mse[number][1]*576)
		#prediction = np.argmax(temp_Output, 1)
		# print ('\n', 'GT:',GT_centroid_norm[number],'PRED',temp_Output[number])
		# print ('\n',GT_centroid_norm[number][0], temp_Output[number][0])
		# print (GT_centroid_norm[number][1], temp_Output[number][1])

	mse_average = ((mse_average[0]/(88*len(test_folders))), (mse_average[1]/(88*len(test_folders))))

	print (mse_average)

	return mse_average

		#dice_average, jacquard_average, dice = scoring(prediction, centroid, k)

	# 	details.write(os.path.basename(folder)+" Dice score: "+str(dice)+"\n")

	# 	# Prediction save
	# 	predict_dict[folder] = np.array(prediction)
	# 	groundt_dict[folder] = np.array(groundT)	


	# print ('\n\nDice score : ', dice_average)
	# print ('\nJacquard score : ', jacquard_average)
	# details.write("\nOverall Dice Average = "+str(dice_average))
	# details.write("\nOverall Jacquard Average = "+str(jacquard_average)+"\n\n")
	# details.close()


	# if dice_average > best_score:

	# 	centroid_mse = centroiD(predict_dict, groundt_dict, savedir)
	# 	print ('\nCentroid MSE',centroid_mse)

	# 	print ('\n\nSaving prediction...\n\n')
	# 	for folder in predict_dict:
	# 		np.save(folder+'/prediction.npy', predict_dict[folder])
	# 		for number, slice in enumerate(predict_dict[folder]):

	# 			#Saving the predicted images
	# 			cv2.imwrite(folder+"/Prediction/Slice%03d.tiff"%(number), 255 * slice)

	# return dice_average, jacquard_average, centroid_mse


def centroiD(predict_dict, groundt_dict, savedir):

	centroid_file = open(savedir+'Centroid.txt','w')
	centroid_file.write('CENTROID\n')
	centroid_file.write('*'*30+'\n')

	# 1) centroid_dict[foler_name] = {slice, (centroid pred), (centroid truth)}
	# 2) centroid_list (Slice, (centroid[0] pred, centroid[1] pred), (centroid[0] truth, centroid[1] truth))
	# 3) mse: Numpy array for Mean Squarred Error calculation 
	#    mse shape = (number of values in the dictionnary, number of values in the dictionnary)

	centroid_dict = {}
	centroid_list = np.zeros([88,2,2])
	mse = np.zeros([sum(map(len, predict_dict.values())), sum(map(len, predict_dict.values()))])
	x = 0
	for folder in predict_dict:

		centroid_file.write('Folder : '+str(folder)+'\n')

		for number, slice in enumerate(predict_dict[folder]):

			#Calculating centroid
			if np.max(groundt_dict[folder][number]) != 0:
				centroid_pred = ndimage.measurements.center_of_mass(predict_dict[folder][number])
				centroid_grud = ndimage.measurements.center_of_mass(groundt_dict[folder][number])

			else:
				centroid_pred = centroid_grud = (slice.shape[0]//2, slice.shape[1]//2)

			centroid_list[number,0] = centroid_pred
			centroid_list[number,1] = centroid_grud
			centroid_file.write('Slice number:'+str(number)+' ground ='+str(centroid_grud)+' pred ='+str(centroid_pred)+'\n')

		centroid_dict[folder] = centroid_list

	for folder in (centroid_dict.keys()):
		for slice in range(len(centroid_dict[folder])):
			mse[x,0] = (centroid_dict[folder][slice][0][0]-centroid_dict[folder][slice][1][0])**2
			mse[x,1] = (centroid_dict[folder][slice][0][1]-centroid_dict[folder][slice][1][1])**2
			x += 1
		try:
			centroid_file.write('\nMSE folder = '+str(folder)+' ('+str(np.int(np.mean(mse[:,0])))+','+str(np.int(np.mean(mse[:,1])))+')\n')
		except ValueError:
			pass

	centroid_mse = (np.mean(mse[:,0]), np.mean(mse[:,1]))
	np.save(savedir+'centroid.npy', centroid_dict)
	centroid_file.close()

	return centroid_mse


def scoring(prediction, ground_truth, k):
	# Scoring : Dice Score (f1 score) & Jacquard Index (intersection over union)

	dice_scores, jacquard_scores = [], []

	dice = np.sum(prediction[ground_truth==k]==k)*2.0 / (np.sum(prediction[prediction==k]==k) + np.sum(ground_truth[ground_truth==k]==k))
	IoU = np.sum(prediction[ground_truth==k]==k) / (np.sum(prediction[prediction==k]==k) + np.sum(ground_truth[ground_truth==k]==k) - (np.sum(prediction[ground_truth==k]==k)))

	dice_scores.append(dice)
	jacquard_scores.append(IoU)

	dice_average = np.mean(np.array(dice_scores))
	jacquard_average = np.mean(np.array(jacquard_scores))	

	return dice_average, jacquard_average, dice


def file_len(fname):
    return sum(1 for line in open(fname))


def plot_learn(savedir):

	file_path = savedir+"Raw_scores.txt"

	num_lines = file_len(file_path)

	#Collecting the scores
	with open(file_path, "r") as file:
	
		scores = []
		epochs = np.arange(1, num_lines+1)
		for number, line in enumerate(file):
			scores.append(float(line.rstrip()))

	##Finding max position
	ymax = max(scores)
	xmax = epochs[scores.index(ymax)]

	##Creating the figure
	fig = plt.figure()
	ax = fig.add_subplot(111)

	##Generating the graph
	line, = ax.plot(epochs, scores, linewidth=0.8)

	##Setting the limit
	ax.set_ylim(min(scores), 0.96)

	##Marking max position
	#With X
	ax.plot(xmax, ymax, "x", ms=10, markerfacecolor="None",
         markeredgecolor='red', markeredgewidth=0.8)
	#With the Dice score value
	ymax = round(ymax,4)
	ax.annotate(ymax, xy=(xmax, ymax), xytext=(xmax-5.5, ymax+0.005))

	##Title and axis label
	ax.set_xlabel('Epochs')
	ax.set_ylabel('Dice score')
	ax.set_title('Training dice score evolution -'+savedir)

	##Saving the images (PDF for final figures)
	plt.savefig(savedir+'Training.png', bbox_inches='tight')
	#plt.savefig(savedir+')Training.pdf', bbox_inches='tight')
	#plt.show()

def NEKO():
	return ('''
  ███╗   ██╗███████╗██╗  ██╗ ██████╗ 
  ████╗  ██║██╔════╝██║ ██╔╝██╔═══██╗
  ██╔██╗ ██║█████╗  █████╔╝ ██║   ██║
  ██║╚██╗██║██╔══╝  ██╔═██╗ ██║   ██║
  ██║ ╚████║███████╗██║  ██╗╚██████╔╝
  ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ 
''')

def RTBF():
	return ('''

  _____       _        _ _     
 |  __ \     | |      (_) |    
 | |  | | ___| |_ __ _ _| |___ 
 | |  | |/ _ \ __/ _` | | / __|
 | |__| |  __/ || (_| | | \__ \

 |_____/ \___|\__\__,_|_|_|___/
                               
                               
''')



if __name__ == '__main__':

	predict_dict = {}

	ar = np.zeros([15,2,2])
	L = ['folder1','folder2','folder3']

	for l in L:	
		for i in range(0,15):
			cent = (i+3, i+1)
			grud = (i+5, i+10)
			ar[i,0] = cent
			ar[i,1] = grud

		predict_dict[l] = ar

	all_ctr = np.ones([45,45])
	all_ctr0 = []
	all_ctr1 = []
	x = 0
	for l in L:
		for w in range(len(predict_dict[l])):

			all_ctr0.append((predict_dict[l][w][0][0]-predict_dict[l][w][1][0])**2)
			all_ctr1.append((predict_dict[l][w][0][1]-predict_dict[l][w][1][1])**2)

			all_ctr[x,0] = (predict_dict[l][w][0][0]-predict_dict[l][w][1][0])**2
			all_ctr[x,1] = (predict_dict[l][w][0][1]-predict_dict[l][w][1][1])**2
			x += 1

	print (L)
	print (x)
	print (np.mean(all_ctr[:,0]), np.mean(all_ctr[:,1]))
	ar0 = np.mean(np.array(all_ctr0))
	ar1 = np.mean(np.array(all_ctr1))

	print (ar0, ar1)
	print (len(predict_dict['folder1']))
	print (sum(map(len, predict_dict.values())))

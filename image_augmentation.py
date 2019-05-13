from imgaug import augmenters as iaa
import numpy as np
import os, random
from PIL import Image
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
#from skimage.filters import unsharp_mask

def img_aug(image, label, data_aug, probability = 0.5):

	'''Image augmentation list, each is applied individually with the same probability
	partial list of augmentation available:

	function_list = ["aug_rotation(image, label, (-20,20), probability = 1)",
	 "aug_scale(image, label, probability = 1)",
	 "aug_perspective(image, label, probability = 1)",
	 "aug_add(image, label, probability = 1)",
	 "aug_contrastNorm(image, label, probability = 1)",
	 "aug_gammaContrast(image, label, probability = 1)"]
	 "aug_elastic_T(image, labe, image.shape[0]*3, image.shape[0]*0.05)"'''


	#function_list = ["aug_rotation(image, label, (-25,25), probability = 1)", "aug_scale(image, label, probability = 1)", "aug_add(image, label, probability = 1)"]
	#function_list = ["aug_elastoc(image, label, image.shape[0]*3, image.shape[0]*0.05)"]#, "aug_flipUD(image, label, probability = 1)"]#, "aug_add(image, label, probability = 1)"]

	image_aug, label_aug = aug_add(image, label, 1)

	if random.random() < probability:
		image_aug, label_aug = eval(random.choice(data_aug))

		return image_aug, label_aug

	else:
		return image, label


###############
# MANIPULATION
###############


def aug_rotation(image, label, rotation = (-10, 10), probability = 0.5):

	if random.random() < probability:
		seq = iaa.Sequential([iaa.Affine(rotate=(rotation))])
		aug_parameters = seq.to_deterministic()

		image_aug = aug_parameters.augment_image(image)
		label_aug = aug_parameters.augment_image(label)

		return image_aug, label_aug

	else:
		return image, label


def aug_flipLR(image, label, probability = 0.5):

	seq = iaa.Sequential([iaa.Fliplr(probability)])
	aug_parameters = seq.to_deterministic()

	image_aug = aug_parameters.augment_image(image)
	label_aug = aug_parameters.augment_image(label)

	return image_aug, label_aug


def aug_flipUD(image, label, probability = 0.5):

	seq = iaa.Sequential([iaa.Flipud(probability)])
	aug_parameters = seq.to_deterministic()

	image_aug = aug_parameters.augment_image(image)
	label_aug = aug_parameters.augment_image(label)

	return image_aug, label_aug


def aug_scale(image, label, probability = 0.5):

	if random.random() < probability:
		seq = iaa.Sequential([iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})])
		aug_parameters = seq.to_deterministic()

		image_aug = aug_parameters.augment_image(image)
		label_aug = aug_parameters.augment_image(label)

		return image_aug, label_aug

	else:
		return image, label


def aug_translate(image, label, probability = 0.5):

	if random.random() < probability:
		seq = iaa.Sequential([iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)})])
		aug_parameters = seq.to_deterministic()

		image_aug = aug_parameters.augment_image(image)
		label_aug = aug_parameters.augment_image(label)

		return image_aug, label_aug

	else:
		return image, label


###############
# DEFORMATION
###############

def aug_shear(image, label, probability = 0.5):

	if random.random() < probability:
		seq = iaa.Sequential([iaa.Affine(shear=(-8, 8))])
		aug_parameters = seq.to_deterministic()

		image_aug = aug_parameters.augment_image(image)
		label_aug = aug_parameters.augment_image(label)

		return image_aug, label_aug

	else:
		return image, label


def aug_elastoc(image, label, probability = 0.5):

	if random.random() < probability:
		seq = iaa.Sequential([iaa.ElasticTransformation(alpha=(2, 5), sigma=0.05)])
		aug_parameters = seq.to_deterministic()

		image_aug = aug_parameters.augment_image(image)
		label_aug = aug_parameters.augment_image(label)

		return image_aug, label_aug

	else:
		return image, label


def aug_warp(image, label, probability = 0.5):

	if random.random() < probability:
		seq = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01, 0.03))])
		aug_parameters = seq.to_deterministic()

		image_aug = aug_parameters.augment_image(image)
		label_aug = aug_parameters.augment_image(label)

		return image_aug, label_aug

	else:
		return image, label


def aug_perspective(image, label, probability = 0.5):

	if random.random() < probability:
		seq = iaa.Sequential([iaa.PerspectiveTransform(scale=(0.05, 0.100))])
		aug_parameters = seq.to_deterministic()

		image_aug = aug_parameters.augment_image(image)
		label_aug = aug_parameters.augment_image(label)

		return image_aug, label_aug

	else:
		return image, label


###############
# SMOOTHING
###############

def aug_GaussianBlur(image, label, probability = 0.5):

	if random.random() < probability:
		seq = iaa.Sequential([iaa.GaussianBlur(sigma=(0, 2))])
		aug_parameters = seq.to_deterministic()

		image_aug = aug_parameters.augment_image(image)

		return image_aug, label

	else:
		return image, label


def aug_AverageBlur(image, label, probability = 0.5):

	if random.random() < probability:
		seq = iaa.Sequential([iaa.AverageBlur(k=(2, 5))])
		aug_parameters = seq.to_deterministic()

		image_aug = aug_parameters.augment_image(image)

		return image_aug, label

	else:
		return image, label


def aug_MedianBlur(image, label, probability = 0.5):

	if random.random() < probability:
		seq = iaa.Sequential([iaa.MedianBlur(k=(3, 7))])
		aug_parameters = seq.to_deterministic()

		image_aug = aug_parameters.augment_image(image)

		return image_aug, label

	else:
		return image, label


def aug_MotionBlur(image, label, probability = 0.5):

	if random.random() < probability:
		seq = iaa.Sequential([iaa.MotionBlur(k=(3,6), angle=(-90, 90))])
		aug_parameters = seq.to_deterministic()

		image_aug = aug_parameters.augment_image(image)

		return image_aug, label

	else:
		return image, label


######################
# CONTRAST/BRIGHTNESS
######################

def aug_add(image, label, probability = 0.5):

	if random.random() < probability:
		seq = iaa.Sequential([iaa.Add((-40, 40))])
		aug_parameters = seq.to_deterministic()

		image_aug = aug_parameters.augment_image(image)

		return image_aug, label

	else:
		return image, label



def aug_multiply(image, label, probability = 0.5):

	if random.random() < probability:
		seq = iaa.Sequential([iaa.Multiply((0.5, 1.5))])
		aug_parameters = seq.to_deterministic()

		image_aug = aug_parameters.augment_image(image)

		return image_aug, label

	else:
		return image, label


def aug_contrastNorm(image, label, probability = 0.5):

	if random.random() < probability:
		seq = iaa.Sequential([iaa.ContrastNormalization((0.5, 1.5))])
		aug_parameters = seq.to_deterministic()

		image_aug = aug_parameters.augment_image(image)

		return image_aug, label

	else:
		return image, label


def aug_gammaContrast(image, label, probability = 0.5):

	if random.random() < probability:
		seq = iaa.Sequential([iaa.GammaContrast((0.3, 1.7))])
		aug_parameters = seq.to_deterministic()

		image = image.clip(min=0)
		image_aug = aug_parameters.augment_image(image)

		return image_aug, label

	else:
		return image, label



def aug_CLAHE(image, label, probability = 0.5):

	if random.random() < probability:
		seq = iaa.Sequential([iaa.CLAHE(clip_limit=1, tile_grid_size_px=(2, 2))])
		aug_parameters = seq.to_deterministic()

		image_aug = aug_parameters.augment_image(image)

		return image_aug, label

	else:
		return image, label



def aug_unsharp(image, label, probability = 0.5):
# linear sharpening of the image : http://scikit-image.org/docs/dev/auto_examples/filters/plot_unsharp_mask.html
	if random.random() < probability:
		image_aug = unsharp_mask(image, radius=5, amount=2)

		return image_aug, label
	else:
		return image, label


def aug_elastic_T(image, label, alpha, sigma):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       based on https://gist.github.com/fmder/e28813c1e8721830ff9c
    """

    shape = image.shape
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape), map_coordinates(label, indices, order=1).reshape(shape)


if __name__ == '__main__':


	print ('Goo')

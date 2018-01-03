import sys
import os
import argparse
import numpy as np

from datetime import datetime

from keras.models import Model
from keras.layers import Input, Dense, Activation, concatenate, UpSampling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.losses import mean_squared_error, mean_absolute_error
from keras.optimizers import RMSprop
from keras.initializers import RandomNormal
from keras.callbacks import ModelCheckpoint

from utils import subsampling, load_image_data, multi_gpu_model

# Data loading
ANALYZE_DATA_EXTENSION_IMG = ".img"

# Training set construction
NUM_SAMPLE_SLICES = 35

# Neural Network Parameters
RMS_WEIGHT_DECAY = .9
LEARNING_RATE = .001
FNET_ERROR_MSE = "mse"
FNET_ERROR_MAE = "mae"

# Logging
CHECKPOINT_FILE_PATH_FORMAT = "fnet-{epoch:02d}.hdf5"

class FNet:

	def __init__(self, error):
		self.architecture_exists = False
		self.error = error

	def train(self, y_folded, y_original, batch_size, num_epochs):
		"""
		Trains the specialized U-net for the MRI reconstruction task

		Parameters
		------------
		y_folded : [np.ndarray]
			A set of folded images obtained by subsampling k-space data
		y_original : [np.ndarray]
			The ground truth set of images, preprocessed by applying the inverse
			f_{cor} function and removing undersampled k-space data
		batch_size : int
			The training batch size
		num_epochs : int
			The number of training epochs
		"""

		if not self.architecture_exists:
			self._create_architecture()

		checkpoint_callback = ModelCheckpoint(CHECKPOINT_FILE_PATH_FORMAT, monitor='val_loss', period=1)

		self.model.fit(y_folded, 
					   y_original, 
					   batch_size=batch_size, 
					   nb_epoch=num_epochs, 
					   shuffle=True, 
					   validation_split=.2,
					   verbose=1,
					   callbacks=[checkpoint_callback])

	def _parse_error(self):
		if self.error == FNET_ERROR_MSE:
			return mean_squared_error
		elif self.error == FNET_ERROR_MAE:
			return mean_absolute_error
		else:
			raise("Attempted train with an invalid loss function!")


	def _get_initializer_seed(self):
		epoch = datetime.utcfromtimestamp(0)
		curr_time = datetime.now()
		millis_since_epoch = (curr_time - epoch).total_seconds() * 1000
		return int(millis_since_epoch)

	def _create_architecture(self, num_gpus):
		inputs = Input(shape=(256,256,1))

		weights_initializer = RandomNormal(mean=0.0, stddev=.01, seed=self._get_initializer_seed())

		# Using the padding=`same` option is equivalent to zero padding
		conv2d_1 = Conv2D(
			filters=64, kernel_size=(3,3), strides=(1,1), padding='same', 
			activation='relu', kernel_initializer=weights_initializer)(inputs)

		conv2d_2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', 
			activation='relu', kernel_initializer=weights_initializer)(conv2d_1)

		maxpool_1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv2d_2)

		conv2d_3 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', 
			activation='relu', kernel_initializer=weights_initializer)(maxpool_1)

		conv2d_4 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', 
			activation='relu', kernel_initializer=weights_initializer)(conv2d_3)

		maxpool_2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv2d_4)

		conv2d_5 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', 
			activation='relu', kernel_initializer=weights_initializer)(maxpool_2)

		conv2d_6 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', 
			activation='relu', kernel_initializer=weights_initializer)(conv2d_5)

		unpool_1 = concatenate([UpSampling2D(size=(2,2))(conv2d_6), conv2d_4], axis=3)

		conv2d_7 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', 
			activation='relu', kernel_initializer=weights_initializer)(unpool_1)

		conv2d_8 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', 
			activation='relu', kernel_initializer=weights_initializer)(conv2d_7)

		unpool_2 = concatenate([UpSampling2D(size=(2,2))(conv2d_8), conv2d_2], axis=3)

		conv2d_9 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', 
			activation='relu', kernel_initializer=weights_initializer)(unpool_2)
		conv2d_10 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', 
			activation='relu', kernel_initializer=weights_initializer)(conv2d_9)

		# Conv2d_10 is 256 x 256 x 64. We now need to reduce the number of output
		# channels via a convolution with `n` filters, where `n` is the original
		# number of channels. We therefore choose `n` = 1.

		outputs = Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), padding='same', 
			activation=None, kernel_initializer=weights_initializer)(conv2d_10)

		optimizer = RMSprop(lr=LEARNING_RATE, rho=RMS_WEIGHT_DECAY, epsilon=1e-08, decay=0)

		self.model = Model(inputs=[inputs], outputs=[outputs])
		if num_gpus >= 2:
			self.model = multi_gpu_model(self.model, gpus=num_gpus)	

		self.model.compile(optimizer=optimizer, loss=self._parse_error(), metrics=[mean_squared_error])

		self.architecture_exists = True

def get_img_file_paths(dir_path):
	img_paths = []
	dir_walk = os.walk(dir_path)
	for walk_item in dir_walk:
		dir_name, _, file_subpaths = walk_item
		relevant_subpaths = [path for path in file_subpaths if ANALYZE_DATA_EXTENSION_IMG in path]
		relevant_paths = [os.path.join(dir_name, path) for path in relevant_subpaths]
		img_paths += relevant_paths

	return img_paths

def load_and_subsample(raw_img_path, substep, low_freq_percent):
	original_img = load_image_data(analyze_image_path=raw_img_path)
	subsampled_img = subsample(analyze_image_data=original_img, substep=substep, low_freq_percent=low_freq_percent)

	original_img = np.moveaxis(original_img, -1, 0)
	original_img = np.expand_dims(original_img, -1)
	subsampled_img = np.moveaxis(np.expand_dims(subsampled_img, 3), -2, 0)

	if len(original_img) > NUM_SAMPLE_SLICES:
		relevant_idx_low = (img_len - NUM_SAMPLE_SLICES) // 2
		relevant_idx_high = relevant_idx_low + NUM_SAMPLE_SLICES

		subsampled_img = subsampled_img[relevant_idxs]
		original_img = original_img[relevant_idxs]

	return subsampled_img, original_img

def load_and_subsample_images(disk_path, num_imgs, substep, low_freq_percent):
	"""
	Parameters
	------------
	disk_path : str
		A path to a disk (directory) of MRI images in Analyze 7.5 format
	num_imgs : int
		The number of images to load
	substep : int
		The substep to use when each image
	low_freq_percent : float
		The percentage of low frequency data to retain when subsampling training images

	Returns
	------------
	A tuple of training data and ground truth images, each represented
	as numpy float arrays of dimension N x 256 x 256 x 1.
	"""
	file_paths = get_img_file_paths(disk_path)

	num_output_imgs = 0

	x_train = None
	y_train = None

	for i in range(len(file_paths)):
		raw_img_path = file_paths[i]

		subsampled_img, original_img = load_and_subsample(raw_img_path=raw_img_path, 
														  substep=substep, 
														  low_freq_percent=low_freq_percent)

		if i == 0:
			x_train = subsampled_img
			y_train = original_img
		else:
			x_train = np.vstack([x_train, subsampled_img])
			y_train = np.vstack([y_train, original_img])

		num_output_imgs += 1
		if num_output_imgs >= num_imgs:
			break

	return x_train, y_train

def main():
    parser = argparse.ArgumentParser(description='Train FNet on MRI image data')
    parser.add_argument('-d', '--disk_path', type=str, help="The path to a disk (directory) containing Analyze-formatted MRI images")
    parser.add_argument('-t', '--training_size', type=int, default=1400, help="The size of the training dataset")
    parser.add_argument('-e', '--training_error', type=str, help="The type of error to use for training the reconstruction network (either 'mse' or 'mae')")
    parser.add_argument('-f', '--lf_percent', type=float, default=.04, help="The percentage of low frequency data to retain when subsampling training images")
    parser.add_argument('-s', '--substep', type=int, default=4, help="The substep to use when subsampling training images")
    parser.add_argument('-n', '--num_epochs', type=int, default=2000, help='The number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='The training batch size. This will be sharded across all available GPUs')
    parser.add_argument('-g', '--num_gpus', type=int, default=0, help='The number of GPUs on which to train the model')

    args = parser.parse_args()

    x_train, y_train = load_and_subsample_images(disk_path=args.disk_path, 
    											 num_imgs=args.training_size,
    											 substep=args.substep,
    											 low_freq_percent=args.lf_percent)

    if len(x_train) > args.training_size:
    	# Select the most relevant slices from each image
    	# until the aggregate number of slices is equivalent to the
    	# specified training dataset size
    	training_idxs = range(args.training_size)
    	x_train = x_train[training_idxs]
    	y_train = y_train[training_idxs]

    net = FNet(args.training_error)
    net.train(x_train, y_train)

if __name__ == "__main__":
	main()


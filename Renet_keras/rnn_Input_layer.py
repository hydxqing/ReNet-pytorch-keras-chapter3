'''
ReNet-Keras
Code written by: Xiaoqing Liu
If you use significant portions of this code or the ideas from our paper, please cite it :)

'''
import numpy as np
import sys
#from torch.autograd import gradcheck
import time
import math
import keras
from keras.layers import *
from keras.models import *
import keras.backend
from keras import initializers, layers

from keras import backend as K
import tensorflow as tf

class rnn_input_layer(layers.Layer):

	def __init__(self, dim=3,receptive_filter_size=4, batch_size=1):

		super(rnn_input_layer, self).__init__()
                
                self.batch_size = batch_size
		self.receptive_filter_size = receptive_filter_size
		self.input_size1 = receptive_filter_size * receptive_filter_size * dim
                self.dim = dim


        def get_image_patches(self, X, receptive_filter_size):
		"""
		creates image patches based on the dimension of a receptive filter
		"""
		image_patches = []

		_, X_height, X_width, X_channel= X.get_shape()
               # print X.get_shape()##(?, 32, 32, 3)

		for i in range(0, X_height, receptive_filter_size):
			for j in range(0, X_width, receptive_filter_size):
				X_patch = X[:,i: i + receptive_filter_size, j : j + receptive_filter_size, :]
				image_patches.append(X_patch)

		image_patches_height = (X_height // receptive_filter_size)
		image_patches_width = (X_width // receptive_filter_size)


		image_patches = K.stack(image_patches)
               # print image_patches.get_shape()  ###(64, ?, 4, 4, 3)
                image_patches = keras.backend.permute_dimensions(image_patches, (1, 0, 2, 3, 4))
                #print image_patches.get_shape() ###(?, 64, 4, 4, 3)

		image_patches =tf.reshape(image_patches,[int(self.batch_size),int(image_patches_height),int(image_patches_height), int(receptive_filter_size) * int(receptive_filter_size) * int(X_channel)])
                #print image_patches.shape

		return image_patches#(1, 8, 8, 48)

        def get_vertical_rnn_inputs(self, image_patches, forward):
		"""
		creates vertical rnn inputs in dimensions 
		(num_patches, batch_size, rnn_input_feature_dim)
		num_patches: image_patches_height * image_patches_width
		"""
		vertical_rnn_inputs = []

		_, image_patches_height, image_patches_width, feature_dim = image_patches.shape

		#image_transpose = Lambda(lambda x: K.tf.transpose(image_patches, perm=[0, 2, 1, 3]))(image_patches)
		#_, image_transpose_height, image_transpose_width, feature_dim = image_transpose.shape

		if forward:
			for i in range(image_patches_height):
				for j in range(image_patches_width):
					vertical_rnn_inputs.append(image_patches[:, j, i, :])

		else:
			for i in range(image_patches_height-1, -1, -1):
				for j in range(image_patches_width-1, -1, -1):
					vertical_rnn_inputs.append(image_patches[:, j, i, :])

		vertical_rnn_inputs = K.stack(vertical_rnn_inputs)

               # print rnn_inputs.shape#(64, 1, 48)

		vertical_rnn_inputs=keras.backend.permute_dimensions(vertical_rnn_inputs, (1, 0, 2))
                #vertical_rnn_inputs = keras.backend.squeeze(vertical_rnn_inputs,axis=0)
               # print rnn_inputs.shape#(1, 64, 48)
		return vertical_rnn_inputs

	def get_horizontal_rnn_inputs(self, image_patches, forward):
		"""
		creates vertical rnn inputs in dimensions 
		(num_patches, batch_size, rnn_input_feature_dim)
		num_patches: image_patches_height * image_patches_width
		"""
		horizontal_rnn_inputs = []
		_, image_patches_height, image_patches_width, feature_dim = image_patches.shape

		if forward:
			for i in range(image_patches_height):
				for j in range(image_patches_width):
					horizontal_rnn_inputs.append(image_patches[:, i, j, :])
		else:
			for i in range(image_patches_height-1, -1, -1):
				for j in range(image_patches_width -1, -1, -1):
					horizontal_rnn_inputs.append(image_patches[:, i, j, :])
		
		horizontal_rnn_inputs = K.stack(horizontal_rnn_inputs)
		horizontal_rnn_inputs = keras.backend.permute_dimensions(horizontal_rnn_inputs, (1, 0, 2))

		return horizontal_rnn_inputs

	def call(self, X):
        	# divide input input image to image patches
		image_patches = self.get_image_patches(X, self.receptive_filter_size)

                #print image_patches.get_shape()
		_, image_patches_height, image_patches_width, feature_dim = image_patches.get_shape()

		# process vertical rnn inputs
		vertical_rnn_inputs_fw = self.get_vertical_rnn_inputs(image_patches, forward=True)
                #print vertical_rnn_inputs_fw.shape
		vertical_rnn_inputs_rev = self.get_vertical_rnn_inputs(image_patches, forward=False)
                #print vertical_rnn_inputs_rev.shape	
		horizontal_rnn_inputs_fw = self.get_horizontal_rnn_inputs(image_patches, forward=True)
		horizontal_rnn_inputs_rev = self.get_horizontal_rnn_inputs(image_patches, forward=False)

		return [vertical_rnn_inputs_fw,vertical_rnn_inputs_rev,horizontal_rnn_inputs_fw,horizontal_rnn_inputs_rev]
                #return vertical_rnn_inputs_fw

        def compute_output_shape(self, input_shape):
		image_patches_height = input_shape[1]/self.receptive_filter_size
		image_patches_width = input_shape[2]/self.receptive_filter_size
		return [(input_shape[0],image_patches_height*image_patches_width,self.input_size1),(input_shape[0],image_patches_height*image_patches_width,self.input_size1),(input_shape[0],image_patches_height*image_patches_width,self.input_size1),(input_shape[0],image_patches_height*image_patches_width,self.input_size1)]


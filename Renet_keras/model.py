'''
ReNet-Keras
Code written by: Xiaoqing Liu
If you use significant portions of this code or the ideas from our paper, please cite it :)

'''
#coding:utf-8
from keras.models import *
from keras.layers import *
#from Input_layer import vertical_layer
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Lambda, K, Reshape, Bidirectional, merge, CuDNNLSTM
from renet_layer import renet_module
import keras

class build_model(keras.Model):

	def __init__(self, nClasses ,  input_height=32, input_width=32): 
		super(build_model,self).__init__()

		self.renet_module = renet_module(X_height=input_height, X_width=input_width, dim=3,receptive_filter_size=4, batch_size=1, hidden_size=320)
		self.conv = Conv2D(1, kernel_size=(1, 1))
		self.upsample = convolutional.UpSampling2D(size=(4, 4), data_format=None)

	def call(self, inputs):  

		renet = self.renet_module(inputs)
        	print renet.shape
		conv = self.conv(renet)
        	#print conv.shape  #(?, 8, 8, 1)
        	upsample = self.upsample(conv)

		return upsample

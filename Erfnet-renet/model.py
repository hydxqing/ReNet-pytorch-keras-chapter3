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
from keras.layers import Conv2D, Lambda, K, Reshape, Bidirectional, merge, CuDNNLSTM, ReLU,Concatenate
from renet_layer import renet_module
import keras
from keras.layers.core import Activation, Flatten, Reshape, Permute

def Conv_BN(inputs,filters, kernel_size, strides=1, dilation_rate=1):
       	x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate,padding='same')(inputs)
        x = BatchNormalization(epsilon=1e-3, momentum=0.993)(x)
	x = ReLU()(x)

        return x


def Transpose_Conv_BN(inputs,filters,kernel_size, strides=1,last=False):

        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,padding='same')(inputs)
        if not last:
            	x = BatchNormalization(epsilon=1e-3, momentum=0.993)(x)
            	x = ReLU()(x)

        return x


def ERFNetModule(inputs, filters, dilation_rate=1, dropout=0.3):

        x = Conv_BN(inputs,filters, kernel_size=[3, 1], dilation_rate=1)

        x =  Conv_BN(x, filters, kernel_size=[1, 3], dilation_rate=1)

        x = Conv_BN(x, filters, kernel_size=[3, 1], dilation_rate=dilation_rate)

        x = Conv_BN(x, filters, kernel_size=[1, 3], dilation_rate=dilation_rate)

        x = Dropout(rate=dropout)(x)
	x = concatenate([x, inputs], axis=-1)

        return x 


def ERFNetDownsample(inputs, filters, hidden_size,renet=False):

        x = Conv_BN(inputs, filters, kernel_size=3, dilation_rate=1, strides=2)
	if renet == True:
        	x2 = renet_module(inputs, hidden_size=hidden_size)
	else:
		x2 = MaxPooling2D(pool_size=(2, 2),strides=2)(inputs)
        x = concatenate([x, x2], axis=-1)
	#x = merge([x, x2], mode='concat') 
        return x



def ERFNetUpsample(inputs,filters, last=False ):

        x = Transpose_Conv_BN(inputs,filters, kernel_size=3, strides=2, last=last)
		#print x.shape
        return x



def ERFNet(input_shape,num_classes=2):

	inputs = Input(shape=input_shape)
	#print inputs.shape
        x = ERFNetDownsample(inputs, filters=16-3, hidden_size=20,renet=False)
       	#print x.shape##(?, 120, 120, 16)

        x = ERFNetDownsample(x, filters=64-16, hidden_size=20,renet=False)
       	#print x.shape##(?, 60, 60, 64)

        x = ERFNetModule(x, 64, dilation_rate=1)
	#print x.shape##(?, 60, 60, 128)
        x = ERFNetModule(x, 64, dilation_rate=1)
	#print x.shape##(?, 60, 60, 192)
	#x = ERFNetDownsample(x, filters=64-16, hidden_size=20,renet=True)
	#print x.shape
        x = ERFNetModule(x, 64, dilation_rate=1)
	#print x.shape##(?, 60, 60, 256)
        x = ERFNetModule(x, 64, dilation_rate=1)
	#print x.shape##(?, 60, 60, 320)
        x = ERFNetModule(x, 64, dilation_rate=1)
	#print x.shape##(?, 60, 60, 384)
        x = ERFNetDownsample(x, filters=128-64,hidden_size=20,renet=False)
	#print x.shape##(?, 30, 30, 448)
        x =  ERFNetModule(x, 128, dilation_rate=2)
        x =  ERFNetModule(x, 128, dilation_rate=4)
        x =  ERFNetModule(x, 128, dilation_rate=8)
        x =  ERFNetModule(x, 128, dilation_rate=16)
        x =  ERFNetModule(x, 128, dilation_rate=2)
        x =  ERFNetModule(x, 128, dilation_rate=4)
        x =  ERFNetModule(x, 128, dilation_rate=8)
        x =  ERFNetModule(x, 128, dilation_rate=16)

        x = ERFNetUpsample(x, 64)
	#print x.shape

        x = ERFNetModule(x, 64, dilation_rate=1)
        x = ERFNetModule(x, 64, dilation_rate=1)

        x = ERFNetUpsample(x, 16)
	#print x.shape
	
        x =  ERFNetModule(x, 16, dilation_rate=1)
        x =  ERFNetModule(x, 16, dilation_rate=1)

        x = ERFNetUpsample(x,num_classes, last=True)
	#print x.shape

	conv = Conv2D(1,1,activation = 'sigmoid')(x)

	#print conv.shape
	model = Model(inputs = inputs, outputs = conv)
        return model

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



def myRenet(nClasses ,  input_height=32, input_width=32 ):
        x = Input(shape=(input_height,input_width,3))
        #print x.shape
	rnn_inputs_fw,rnn_inputs_rev = vertical_layer(dim=3)(x)

   	renet1 = Bidirectional(CuDNNLSTM(320, return_sequences=True))(rnn_inputs_fw)
        #print renet1.shape ###(1, 64, 640)
	renet2 = Bidirectional(CuDNNLSTM(320, return_sequences=True))(rnn_inputs_rev)
        #print renet2.shape
    	renet = concatenate([renet1, renet2], axis=2)
        #print renet.shape  #####(1, 64, 1280)
#vertical_reverse_hidden, vertical_reverse_cell = LSTM(vertical_rnn_inputs_rev, self.hidden)
    	renet = Reshape((8,8,-1))(renet)
       # print renet.shape  ###(?, 8, 8, ?)
	conv = Conv2D(1, kernel_size=(1, 1))(renet)
        #print conv.shape  #(?, 8, 8, 1)
        upsample = convolutional.UpSampling2D(size=(4, 4), data_format=None)(conv)
        #print upsample.shape
        #out = Lambda(lambda x: K.tf.transpose(upsample, perm=[0, 3, 1, 2 ]))(upsample)
        #print out.shape
        train_model = Model(inputs=x, outputs=upsample)###(?, 32, 32, 1)

        return train_model


def Renet(nClasses ,  input_height=32, input_width=32 ):
        x = Input(shape=(3,input_height,input_width))
        print x.shape
    	layer = Conv2D(512, kernel_size=(1, 1),data_format='channels_first')(x)  # (?, 512, 7, 7)
        print layer.shape
    	layer_transpose = Lambda(lambda x: K.tf.transpose(layer, perm=[0, 2, 1, 3]))(layer)
        print layer_transpose.shape

    	layer = Reshape((-1, 512))(layer)
        print layer.shape
    	layer_transpose = Reshape((-1, 512))(layer_transpose)  # (?, ?, 512)
        print layer_transpose.shape
   	renet1 = Bidirectional(CuDNNLSTM(512, return_sequences=True))(layer)  # (?, ?, 1024)
        print renet1.shape
   	renet2 = Bidirectional(CuDNNLSTM(512, return_sequences=True))(layer_transpose)  # (?, ?, 1024)
    	renet = merge([renet1, renet2], mode='concat')  # (?, ?, 2048)
	print renet.shape
    	renet = Reshape((-1,7,7))(renet)  # (?, ?, 7, 7)
	print renet.shape
    	renet = Conv2D(512, kernel_size=(1, 1),data_format='channels_first')(renet)
	print renet.shape

        train_model = Model(inputs=x, outputs=outputs)

        return train_model


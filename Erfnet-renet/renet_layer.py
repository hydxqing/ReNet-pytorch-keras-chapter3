'''
ReNet-Keras
Code written by: Xiaoqing Liu
If you use significant portions of this code or the ideas from our paper, please cite it :)

'''
from keras import initializers, layers
from keras import backend as K
from rnn_Input_layer import rnn_input_layer
from keras.layers import Concatenate, Reshape, Bidirectional, CuDNNLSTM, LSTM
import keras
#from keras.engine.topology import _to_list

def renet_module(X,hidden_size,receptive_filter_size=2, batch_size=1):


	_, X_height, X_width, X_channel= X.get_shape()
	print X_height
	vertical_rnn_inputs_fw,vertical_rnn_inputs_rev,horizontal_rnn_inputs_fw,horizontal_rnn_inputs_rev = rnn_input_layer(receptive_filter_size)(X)
	renet1 = CuDNNLSTM(hidden_size, return_sequences=True)(vertical_rnn_inputs_fw)
	print renet1.shape
	renet2 = CuDNNLSTM(hidden_size, return_sequences=True)(vertical_rnn_inputs_rev)
	print renet2.shape
	renet3 = CuDNNLSTM(hidden_size, return_sequences=True)(horizontal_rnn_inputs_fw)
	renet4 = CuDNNLSTM(hidden_size, return_sequences=True)(horizontal_rnn_inputs_rev)

	renet_concat =  Concatenate(axis=2)([renet1, renet2, renet3, renet4])
	print renet_concat.shape
	renet = Reshape((int(X_height)/receptive_filter_size, int(X_width)/receptive_filter_size, -1))(renet_concat)
	print renet.get_shape()

	return renet



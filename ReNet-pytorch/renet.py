#coding:utf-8
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import sys
from torch.autograd import gradcheck
import time
import math
import argparse

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage

from dataset import train,test
from transform import Relabel, ToLabel, Colorize

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=3, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
#args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cuda = False
if args.cuda:
    torch.cuda.manual_seed(args.seed)



receptive_filter_size = 4
hidden_size = 320
image_size_w = 32
image_size_h = 32


input_transform = Compose([
    Resize((32,32)),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
target_transform = Compose([
    Resize((32,32)),

    ToLabel(),

])

#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                        download=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
#                                          shuffle=True, num_workers=2)
trainloader = DataLoader(train(input_transform, target_transform),num_workers=1, batch_size=1, shuffle=True)
testloader = DataLoader(train(input_transform, target_transform),num_workers=1, batch_size=1, shuffle=True)
#testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                       download=True, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
#                                         shuffle=False, num_workers=2)




# renet with one layer
class ReNet(nn.Module):
	def __init__(self, receptive_filter_size, hidden_size, batch_size, image_patches_height, image_patches_width):

		super(ReNet, self).__init__()

		self.batch_size = batch_size
		self.receptive_filter_size = receptive_filter_size
		self.input_size1 = receptive_filter_size * receptive_filter_size * 3
		self.input_size2 = hidden_size * 2
		self.hidden_size = hidden_size

		# vertical rnns
		self.rnn1 = nn.LSTM(self.input_size1, self.hidden_size, dropout = 0.2)
		self.rnn2 = nn.LSTM(self.input_size1, self.hidden_size, dropout = 0.2)

		# horizontal rnns
		self.rnn3 = nn.LSTM(self.input_size2, self.hidden_size, dropout = 0.2)
		self.rnn4 = nn.LSTM(self.input_size2, self.hidden_size, dropout = 0.2)

		self.initHidden()

		#feature_map_dim = int(image_patches_height*image_patches_height*hidden_size*2)
		self.conv1 = nn.Conv2d(hidden_size*2, 2, 3,padding=1)#[1,640,8,8]->[1,1,8,8]
		self.UpsamplingBilinear2d=nn.UpsamplingBilinear2d(size=(32,32), scale_factor=None)
		#self.dense = nn.Linear(feature_map_dim, 4096)
		#self.fc = nn.Linear(4096, 10)

		self.log_softmax = nn.LogSoftmax()

	def initHidden(self):
		self.hidden = (Variable(torch.zeros(1, self.batch_size, self.hidden_size)), Variable(torch.zeros(1, self.batch_size, self.hidden_size)))


	def get_image_patches(self, X, receptive_filter_size):
		"""
		creates image patches based on the dimension of a receptive filter
		"""
		image_patches = []

		_, X_channel, X_height, X_width= X.size()


		for i in range(0, X_height, receptive_filter_size):
			for j in range(0, X_width, receptive_filter_size):
				X_patch = X[:, :, i: i + receptive_filter_size, j : j + receptive_filter_size]
				image_patches.append(X_patch)

		image_patches_height = (X_height // receptive_filter_size)
		image_patches_width = (X_width // receptive_filter_size)


		image_patches = torch.stack(image_patches)
		image_patches = image_patches.permute(1, 0, 2, 3, 4)

		image_patches = image_patches.contiguous().view(-1, image_patches_height, image_patches_height, receptive_filter_size * receptive_filter_size * X_channel)

		return image_patches



	def get_vertical_rnn_inputs(self, image_patches, forward):
		"""
		creates vertical rnn inputs in dimensions 
		(num_patches, batch_size, rnn_input_feature_dim)
		num_patches: image_patches_height * image_patches_width
		"""
		vertical_rnn_inputs = []
		_, image_patches_height, image_patches_width, feature_dim = image_patches.size()

		if forward:
			for i in range(image_patches_height):
				for j in range(image_patches_width):
					vertical_rnn_inputs.append(image_patches[:, j, i, :])

		else:
			for i in range(image_patches_height-1, -1, -1):
				for j in range(image_patches_width-1, -1, -1):
					vertical_rnn_inputs.append(image_patches[:, j, i, :])

		vertical_rnn_inputs = torch.stack(vertical_rnn_inputs)


		return vertical_rnn_inputs



	def get_horizontal_rnn_inputs(self, vertical_feature_map, image_patches_height, image_patches_width, forward):
		"""
		creates vertical rnn inputs in dimensions 
		(num_patches, batch_size, rnn_input_feature_dim)
		num_patches: image_patches_height * image_patches_width
		"""
		horizontal_rnn_inputs = []

		if forward:
			for i in range(image_patches_height):
				for j in range(image_patches_width):
					horizontal_rnn_inputs.append(vertical_feature_map[:, i, j, :])
		else:
			for i in range(image_patches_height-1, -1, -1):
				for j in range(image_patches_width -1, -1, -1):
					horizontal_rnn_inputs.append(vertical_feature_map[:, i, j, :])
		
		horizontal_rnn_inputs = torch.stack(horizontal_rnn_inputs)

		return horizontal_rnn_inputs


	def forward(self, X):

		"""ReNet """

		# divide input input image to image patches
		image_patches = self.get_image_patches(X, self.receptive_filter_size)
		_, image_patches_height, image_patches_width, feature_dim = image_patches.size()

		# process vertical rnn inputs
		vertical_rnn_inputs_fw = self.get_vertical_rnn_inputs(image_patches, forward=True)
		vertical_rnn_inputs_rev = self.get_vertical_rnn_inputs(image_patches, forward=False)

		# extract vertical hidden states
		vertical_forward_hidden, vertical_forward_cell = self.rnn1(vertical_rnn_inputs_fw, self.hidden)
		vertical_reverse_hidden, vertical_reverse_cell = self.rnn2(vertical_rnn_inputs_rev, self.hidden)

		# create vertical feature map
		vertical_feature_map = torch.cat((vertical_forward_hidden, vertical_reverse_hidden), 2)
		vertical_feature_map =  vertical_feature_map.permute(1, 0, 2)

		# reshape vertical feature map to (batch size, image_patches_height, image_patches_width, hidden_size * 2)
		vertical_feature_map = vertical_feature_map.contiguous().view(-1, image_patches_width, image_patches_height, self.hidden_size * 2)
		vertical_feature_map.permute(0, 2, 1, 3)

		# process horizontal rnn inputs
		horizontal_rnn_inputs_fw = self.get_horizontal_rnn_inputs(vertical_feature_map, image_patches_height, image_patches_width, forward=True)
		horizontal_rnn_inputs_rev = self.get_horizontal_rnn_inputs(vertical_feature_map, image_patches_height, image_patches_width, forward=False)

		# extract horizontal hidden states
		horizontal_forward_hidden, horizontal_forward_cell = self.rnn3(horizontal_rnn_inputs_fw, self.hidden)
		horizontal_reverse_hidden, horizontal_reverse_cell = self.rnn4(horizontal_rnn_inputs_rev, self.hidden)

		# create horiztonal feature map[64,1,320]
		horizontal_feature_map = torch.cat((horizontal_forward_hidden, horizontal_reverse_hidden), 2)
		horizontal_feature_map =  horizontal_feature_map.permute(1, 0, 2)

		# flatten[1,64,640]
		output = horizontal_feature_map.contiguous().view(-1, image_patches_height , image_patches_width , self.hidden_size * 2)
		output=output.permute(0,3,1,2)#[1,640,8,8]
		conv1=self.conv1(output)
		Upsampling=self.UpsamplingBilinear2d(conv1)
		# dense layer
		#output = F.relu(self.dense(output))
		 
		# fully connected layer
		#logits = self.fc(output)

		# log softmax
		logits = self.log_softmax(Upsampling)

		return logits


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since):
    now = time.time()
    s = now - since
    s = '%s' % (asMinutes(s))
    return s



if __name__ == "__main__":

	renet = ReNet(receptive_filter_size, hidden_size, args.batch_size, image_size_w/receptive_filter_size, image_size_h/receptive_filter_size)
	if args.cuda:
		renet.cuda()

	#criterion = nn.CrossEntropyLoss().cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(renet.parameters(), lr=args.lr, momentum = args.momentum)

	for epoch in range(args.epochs):  

		running_loss = 0.0
		start = time.time()

		for i, data in enumerate(trainloader, 0):
			# get the inputs
			inputs, labels = data

			if args.cuda:
				inputs, labels = inputs.cuda(), labels.cuda()

			# wrap them in Variable
			inputs, labels = Variable(inputs), Variable(labels)

			# # zero the parameter gradients
			optimizer.zero_grad()

			# # forward + backward + optimize
			renet.initHidden()
			renet.train()
			logits = renet(inputs)
			loss = criterion(logits, labels[:,0])
			loss.backward()
			optimizer.step()

			# get statistics
			running_loss += loss.data[0]
			_, predicted = torch.max(logits.data, 1)
			total = labels.size(0)
			correct = (predicted == labels.data).sum()
			train_accur = correct / total * 100


			eval_every = 50

			# print necessary info
			if i % eval_every == eval_every-1:    # print every 50 mini-batches
				print('[%d, %5d] loss: %.3f time: %s' % (epoch + 1, i + 1, running_loss/(i + 1), timeSince(start)))
				print("train accuracy", train_accur)



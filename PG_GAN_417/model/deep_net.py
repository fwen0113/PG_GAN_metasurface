import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from metalayers import * 



class GBlock(nn.Module):
	"""docstring for GBlock"""
	def __init__(self, in_channel, out_channel, kernel_size, stride, bias):
		super(GBlock, self).__init__()


		self.CONV_T = nn.Sequential(
			nn.BatchNorm2d(in_channel),
			nn.ReLU(),
			Conv2d_meta(in_channel, in_channel, 1, stride=1),
			nn.BatchNorm2d(in_channel),
			nn.ReLU(),
			ConvTranspose2d_meta(in_channel, out_channel, kernel_size, stride=stride, bias=bias),
			Conv2d_meta(out_channel, out_channel, 3, stride=1),
			nn.BatchNorm2d(out_channel),
			nn.ReLU(),
			Conv2d_meta(out_channel, out_channel, 3, stride=1),
			nn.BatchNorm2d(out_channel),
			nn.ReLU(),
			Conv2d_meta(out_channel, out_channel, 1, stride=1)
			)

		self.SKIP = nn.Sequential(
			nn.Upsample(scale_factor=stride),
			Conv2d_meta(in_channel, out_channel, 1, stride=1)
			)
		


	def forward(self, imgs):
		#print(self.SKIP(imgs).size())
		#print(self.CONV_T(imgs).size())
		return self.CONV_T(imgs) + self.SKIP(imgs)
		


class ResGenerator(nn.Module):
	def __init__(self, params):
		super().__init__()

		self.noise_dim = params.noise_dims
		self.label_dim = params.label_dims

		self.gkernel = gkern(params.gkernlen, params.gkernsig)

		self.FC = nn.Sequential(
			nn.Linear((self.noise_dim + self.label_dim), 256),
			nn.ReLU(),
            nn.Linear(256, 4*16*64, bias=False),
		)

		self.BlOCK = nn.Sequential(
			GBlock(64, 64, 5, stride=2, bias=False),
			GBlock(64, 32, 5, stride=2, bias=False),
			GBlock(32, 16, 5, stride=2, bias=False),
			GBlock(16, 1, 5, stride=1, bias=True)
			)
		


	def forward(self, labels, noise):
		net = torch.cat([labels, noise], -1)
		net = self.FC(net)
		net = net.view(-1, 64, 4, 16)
		net = self.BlOCK(net)
		net = conv2d_meta(net, self.gkernel)
		return torch.tanh(net)


class DBlock(nn.Module):
	"""docstring for GBlock"""
	def __init__(self, in_channel, out_channel, kernel_size, stride):
		super(DBlock, self).__init__()

		self.CONV = nn.Sequential(
			nn.LeakyReLU(0.2),
			Conv2d_meta(in_channel, out_channel, 1, stride=1),
			nn.LeakyReLU(0.2),
			Conv2d_meta(out_channel, out_channel, kernel_size, stride=1),
			nn.LeakyReLU(0.2),
			AvgPool2d_meta(5, stride=stride),
			Conv2d_meta(out_channel, out_channel, 1, stride=1)
			)

		self.SKIP = AvgPool2d_meta(5, stride=2)
		self.CONV1 = Conv2d_meta(in_channel, out_channel - in_channel, 1, stride=1)


	def forward(self,inputs):

		skip_inputs = self.SKIP(inputs)
		return torch.cat([skip_inputs, self.CONV1(skip_inputs)], dim=1) + self.CONV(inputs)




class ResDiscriminator(nn.Module):
	"""docstring for Discriminator"""
	def __init__(self, params):
		super().__init__()

		self.BLOCK = nn.Sequential(
			DBlock(1, 8, 3, stride=2),
			DBlock(8, 16, 3, stride=2),
			DBlock(16, 32, 3, stride=2)
			)

		self.FC = nn.Sequential(
			nn.LeakyReLU(0.2),
			nn.Linear(32*4*16 + params.label_dims, 512),
			nn.LayerNorm(512),
			nn.LeakyReLU(0.2),
			nn.Linear(512, 64),
			nn.LayerNorm(64),
			nn.LeakyReLU(0.2),
			nn.Linear(64, 1)
			)


	def forward(self, img, labels):
		net = img + torch.randn(img.size()).type(Tensor)*0.1
		net = self.BLOCK(net)
		net = net.view(net.size(0), -1)
		net = torch.cat([net, labels], -1)
		net = self.FC(net)
		return net




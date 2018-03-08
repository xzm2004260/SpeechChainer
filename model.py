import numpy as np
import chainer, uuid, os
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class SpeechModel(chainer.Chain):
	def __init__(self, features, words_size, ndim=128, n_block=3):
		w0 = chainer.initializers.HeNormal()
		w1 = chainer.initializers.HeNormal()
		w2 = chainer.initializers.HeNormal()
		super(SpeechModel, self).__init__(
			conv1d0 = L.ConvolutionND(ndim=1, in_channels=features, out_channels=ndim, ksize=1, initialW=w0, initial_bias=None),
			batchnormalization0 = L.BatchNormalization(size=ndim),
			residual0 = ResidualBlock(in_channels=ndim, out_channels=ndim),
			conv1d1 = L.ConvolutionND(ndim=1, in_channels=ndim, out_channels=ndim, ksize=1, initialW=w1, initial_bias=None),
			batchnormalization1 = L.BatchNormalization(size=ndim),
			conv1d2 = L.ConvolutionND(ndim=1, in_channels=ndim, out_channels=words_size, ksize=1, initialW=w2, initial_bias=getZerosb(words_size)),
			batchnormalization2 = L.BatchNormalization(size=words_size))

	def __call__(self, x):
		# 数据预处理，交换维度，使得满足x: [batch,in_channels,in_width]
		x = np.array([x_.transpose() for x_ in x], dtype='float32')
		out = F.tanh(self.batchnormalization0(self.conv1d0(x)))
		out = self.residual0(out)	
		out = F.tanh(self.batchnormalization1(self.conv1d1(out)))
		out = self.conv1d2(out)
		# out = self.batchnormalization2(self.conv1d2(out))
		return out

class ResidualBlock(chainer.ChainList):
	def __init__(self, in_channels, out_channels, n_block=3):
		super(ResidualBlock, self).__init__()
		for _ in range(n_block):
			for r in [1,2,4,8,16]:
				self.add_link(Aconv1dLayer(out_channels, out_channels, 7, r))

	def __call__(self, x):
		skip = 0
		out = x
		for f in self.children():
			out, s = f(out)
			skip += s
		return skip

class Aconv1dLayer(chainer.Chain):
	def __init__(self, in_channels, out_channels, size, rate):
		w0 = chainer.initializers.HeNormal()
		w1 = chainer.initializers.HeNormal()
		w2 = chainer.initializers.HeNormal()
		super(Aconv1dLayer, self).__init__()
		with self.init_scope():
			self.aconv1d0 = L.DilatedConvolution2D(in_channels=in_channels, out_channels=out_channels,ksize=(1,size), dilate=rate, pad=(0,((size-1)*(rate-1)+size-1)//2), initialW=w0, initial_bias=None)
			self.aconv1d1 = L.DilatedConvolution2D(in_channels=in_channels, out_channels=out_channels,ksize=(1,size), dilate=rate, pad=(0,((size-1)*(rate-1)+size-1)//2), initialW=w1, initial_bias=None)
			self.conv1d0 = L.ConvolutionND(ndim=1, in_channels=in_channels, out_channels=out_channels, ksize=1, initialW=w2, initial_bias=None)
			self.batchnormalization0 = L.BatchNormalization(size=out_channels)
			self.batchnormalization1 = L.BatchNormalization(size=out_channels)

	def __call__(self, x):
		conv_filter = self.aconv1d0(F.expand_dims(x, axis=-2))
		conv_filter = F.squeeze(conv_filter, axis=-2)
		conv_filter = self.batchnormalization0(conv_filter)
		conv_filter = F.tanh(conv_filter)
		conv_gate = self.aconv1d1(F.expand_dims(x, axis=-2))
		conv_gate = F.squeeze(conv_gate, axis=-2)
		conv_gate = self.batchnormalization1(conv_gate)
		conv_gate = F.sigmoid(conv_gate)
		out = F.tanh(self.conv1d0(conv_filter*conv_gate))
		return out+x, out


def getZerosb(dim):
	if dim >= 0:
		return np.array([0.1]*dim, dtype='float32')
	return 0
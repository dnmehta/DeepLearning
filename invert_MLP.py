from __future__ import print_function

__docformat__ = 'restructedtext en'


from MLP import HiddenLayer,MLP,LogisticRegression


import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import cv

import numpy

import theano
import theano.tensor as T
from theano import pp
import copy_reg
import types

def reduce_method(m):
	return (getattr, (m.__self__, m.__func__.__name__))

copy_reg.pickle(types.MethodType, reduce_method)


class inverse_HiddenLayer(object):     #different class created just for better understanding that W,b are fixed

	def __init__(self,n_in,n_out,W,b,input):

		self.W = W

		self.b = b


		self.input = input

		self.output = T.tanh(T.dot(self.input,self.W) + self.b )

		self.params = []



class inverse_LR(object):

	def __init__(self,n_in,n_out,W,b,input):



		self.W = W
		self.b = b

		self.input = input

		self.p_y_given_x = T.nnet.softmax(T.dot(self.input,self.W)+self.b)

		self.y_pred = T.argmax(self.p_y_given_x,axis=1)

		self.params = []

	def negative_log_likelihood(self, y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
				'y should have the same shape as self.y_pred',
				('y', y.type, 'y_pred', self.y_pred.type)
			)

		if y.dtype.startswith('int'):

			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()

class inverse_MLP(object):

	def __init__(self):


		#rasterizing the image
		name='gray_image.png'
		im=cv.LoadImageM(name)
		q=numpy.asarray(im,dtype=theano.config.floatX) 
		ans=numpy.reshape(q,(1,-1))


		img = theano.shared(
					value=ans,
					name='img',
					borrow=True
					)

		self.img = img


		classifierrr = pickle.load(open('mlp_best_model.pkl'))

		self.inverse_hl1 = inverse_HiddenLayer( n_in = 250*250*3,n_out =20,
							W=classifierrr.hiddenlayer_1.W.get_value(),
							b=classifierrr.hiddenlayer_1.b.get_value(),
							input = self.img )

		self.inverse_hl2 = inverse_HiddenLayer( n_in = 20, n_out = 5,
							W=classifierrr.hiddenlayer_2.W.get_value(),
							b=classifierrr.hiddenlayer_2.b.get_value(),
							input = self.inverse_hl1.output
							 )

		self.inverse_lr = inverse_LR( n_in = 5, n_out = 2,
							W=classifierrr.logRegLayer.W.get_value(),
							b=classifierrr.logRegLayer.b.get_value(),
							input = self.inverse_hl2.output )

		self.L1= (
			abs(self.inverse_hl1.W).sum() +
			abs(self.inverse_hl2.W).sum() +
			abs(self.inverse_lr.W).sum()
				)

		self.L2_sq = (
			(self.inverse_hl1.W ** 2).sum() +
			(self.inverse_hl2.W ** 2).sum() +
			(self.inverse_lr.W ** 2).sum()
			)

		self.negative_log_likelihood = (self.inverse_lr.negative_log_likelihood)

		self.y_pred = self.inverse_lr.y_pred

		self.p_y_given_x = self.inverse_lr.p_y_given_x

		self.errors= self.inverse_lr.errors

	# def negative_log_likelihood(self,y):
		
	# 	return -T.mean(T.log(T.nnet.softmax(T.dot(T.tanh(T.dot(T.tanh(T.dot(self.img,self.inverse_hl1.W) + self.inverse_hl1.b ),self.inverse_hl2.W) + self.inverse_hl2.b ),self.inverse_lr.W)+self.inverse_lr.b))[T.arange(y.shape[0]), y])




def inverse_sgd():

	classifier=inverse_MLP()
	learning_rate = 1
	n_epochs = 1000
	patience = 500   #ignore for now, used for early stopping


	print('building')

	L1_reg = 0.0
	L2_sq = 0.0001

	y=T.ivector('y')

	cost = ( classifier.negative_log_likelihood(y) ) #other terms ignored as they dont depend on classifier.img


	g_image_1 = T.grad(cost=cost,wrt=classifier.img) #SOME ERROR HERE IN GRAD FUNCTION


	updates = [( classifier.img, classifier.img - learning_rate * g_image_1 )]

	train_model = theano.function(     
		inputs=[],
		outputs=g_image_1,             #output is kept as gradient just for debugging purpose
		updates=updates,
		givens={
		y:numpy.ones((1,),dtype=numpy.int32)
		}

	)

	print('training')

	validation_frequency = 10
								
	best_validation_probability = 1
	final_image = 0
	start_time = timeit.default_timer()

	print(classifier.img.get_value())

	done_looping = False
	epoch = 0
	while (epoch < n_epochs):
		epoch = epoch + 1

		minibatch_avg_cost = train_model()
		print(minibatch_avg_cost)
		
		if epoch % validation_frequency == 0 :
			print("HEYY")
			
			print(classifier.p_y_given_x.eval())

			if classifier.p_y_given_x[0][0].eval() < best_validation_probability:

				print("HELLO")

				best_validation_probability=classifier.p_y_given_x[0][0].eval()

				final_image = classifier.img.get_value()


	final_dream = numpy.reshape(final_image,(250,250,3))
	
	print (final_dream)

	end_time = timeit.default_timer()
	
if __name__ == "__main__":

	inverse_sgd()
	







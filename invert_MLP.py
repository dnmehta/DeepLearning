from __future__ import print_function

__docformat__ = 'restructedtext en'


from MLP import HiddenLayer,MLP,LogisticRegression

from theano import gradient
import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import cv
import numpy

from scipy.misc import imsave

import theano
import theano.tensor as T
from theano import pp
import copy_reg
import types

def reduce_method(m):
	return (getattr, (m.__self__, m.__func__.__name__))

copy_reg.pickle(types.MethodType, reduce_method)

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = numpy.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = numpy.clip(x, 0, 255).astype('uint8')
    x=numpy.reshape(x,(250,250,3))
    return x


# class inverse_HiddenLayer(object):     #different class created just for better understanding that W,b are fixed

# 	def __init__(self,n_in,n_out,W,b,input):

# 		self.W = W
# 		self.b = b



# 		self.input = input

# 		self.output = T.tanh(T.dot(self.input,self.W) + self.b )

# 		self.params = []



# class inverse_LR(object):

# 	def __init__(self,n_in,n_out,W,b,input):

# 		self.W=W
# 		self.b=b


# 		self.input = input

# 		self.p_y_given_x = T.nnet.softmax(T.dot(self.input,self.W)+self.b)

# 		self.y_pred = T.argmax(self.p_y_given_x,axis=1)

# 		self.params = []

# 	def negative_log_likelihood(self, y):
# 		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


# 	def errors(self, y):
# 		if y.ndim != self.y_pred.ndim:
# 			raise TypeError(
# 				'y should have the same shape as self.y_pred',
# 				('y', y.type, 'y_pred', self.y_pred.type)
# 			)

# 		if y.dtype.startswith('int'):

# 			return T.mean(T.neq(self.y_pred, y))
# 		else:
# 			raise NotImplementedError()

# class inverse_MLP(object):

# 	def __init__(self,name):


# 		#rasterizing the image
# 		im=cv.LoadImageM(name)
# 		q=numpy.asarray(im,dtype=theano.config.floatX) 
# 		ans=numpy.reshape(q,(1,-1))


# 		img = theano.shared(
# 					value=ans,
# 					name='img',
# 					borrow=True
# 					)

# 		self.img = img


# 		classifierrr = pickle.load(open('mlp_best_model.pkl'))

# 		self.inverse_hl1 = inverse_HiddenLayer( n_in = 250*250*3,n_out =20,
# 							W=classifierrr.hiddenlayer_1.W,
# 							b=classifierrr.hiddenlayer_1.b,
# 							input = self.img )

# 		self.inverse_hl2 = inverse_HiddenLayer( n_in = 20, n_out = 5,
# 							W=classifierrr.hiddenlayer_2.W,
# 							b=classifierrr.hiddenlayer_2.b,
# 							input = self.inverse_hl1.output
# 							 )

# 		self.inverse_lr = inverse_LR( n_in = 5, n_out = 2,
# 							W=classifierrr.logRegLayer.W,
# 							b=classifierrr.logRegLayer.b,
# 							input = self.inverse_hl2.output )

# 		self.L1= (
# 			abs(self.inverse_hl1.W).sum() +
# 			abs(self.inverse_hl2.W).sum() +
# 			abs(self.inverse_lr.W).sum()
# 				)

# 		self.L2_sq = (
# 			(self.inverse_hl1.W ** 2).sum() +
# 			(self.inverse_hl2.W ** 2).sum() +
# 			(self.inverse_lr.W ** 2).sum()
# 			)

# 		self.negative_log_likelihood = (self.inverse_lr.negative_log_likelihood)

# 		self.y_pred = self.inverse_lr.y_pred

# 		self.p_y_given_x = self.inverse_lr.p_y_given_x

# 		self.errors= self.inverse_lr.errors

# 	# def negative_log_likelihood(self,y):
		
# 	# 	return -T.mean(T.log(T.nnet.softmax(T.dot(T.tanh(T.dot(T.tanh(T.dot(self.img,self.inverse_hl1.W) + self.inverse_hl1.b ),self.inverse_hl2.W) + self.inverse_hl2.b ),self.inverse_lr.W)+self.inverse_lr.b))[T.arange(y.shape[0]), y])



def inverse_sgd():


	classifierrr=pickle.load(open('mlp_best_model.pkl'))  #this is the model computing using MLP code
															#weights are stored in this classifierrr

	learning_rate = 100
	n_epochs = 20
	patience = 500   #ignore for now, used for early stopping

	print('building')

	L1_reg = 0.0
	L2_sq = 0.0001
	im=cv.LoadImageM('init_dream1.png')   #any image of size 250*250*3 (after converting to linear)
	q=numpy.asarray(im,dtype=theano.config.floatX) 
	ans=numpy.reshape(q,(1,-1))     
	#ans = numpy.random.randint(1,100,250*250*3)

	#imsave('init_dream1.png',numpy.reshape(ans,(250,250,3)))
	img = theano.shared(
				value=numpy.asarray(ans,dtype=theano.config.floatX),
				name='ans',
				borrow=True
				)
	
	print(img.get_value())
	# 	inputs=[],
	# 	outputs=g_image_1,             #output is kept as gradient just for debugging purpose
	# 	updates=updates,
	# 	givens={
	# 		c:numpy.asmatrix(classifierrr.hiddenlayer_1.W.get_value()),
	# 		d:numpy.asmatrix(classifierrr.hiddenlayer_1.b.get_value()),
	# 		e:numpy.asmatrix(classifierrr.hiddenlayer_2.W.get_value()),
	# 		f:numpy.asmatrix(classifierrr.hiddenlayer_2.b.get_value()),
	# 		g:numpy.asmatrix(classifierrr.logRegLayer.W.get_value()),
	# 		h:numpy.asmatrix(classifierrr.logRegLayer.b.get_value())
	# 	}
	# )

#The are the weights of the layers...
	c=T.matrix('c')
	d=T.matrix('d')
	e=T.matrix('e')
	f=T.matrix('f')
	g=T.matrix('g')
	h=T.matrix('h')



#Calculating cost
	cost_for_grad=T.matrix('cost')
	cost_for_grad = -T.log(T.nnet.softmax(T.dot((T.dot((T.dot(img,c) + d),e)+f),g)+h))[0][0]#all values compute correctly

	p_y_given_x = T.nnet.softmax(T.dot(T.tanh(T.dot(T.tanh(T.dot(img,c) + d),e)+f),g)+h)

	g2=T.grad(cost=cost_for_grad,wrt=img) #computing gradiest. Works for all variables like c,d,f,etc other than img

	g2 /= T.sqrt(T.mean(T.square(g2)) + 1e-5)

	updates = [(img,img+0.01*g2)]

	train_image = theano.function(inputs=[],outputs=[p_y_given_x],givens={    #function to get value of gradient
			d:numpy.asmatrix(classifierrr.hiddenlayer_1.b.get_value()),
			e:numpy.asmatrix(classifierrr.hiddenlayer_2.W.get_value()),
			f:numpy.asmatrix(classifierrr.hiddenlayer_2.b.get_value()),
			g:numpy.asmatrix(classifierrr.logRegLayer.W.get_value()),
			h:numpy.asmatrix(classifierrr.logRegLayer.b.get_value()),
			c:numpy.asmatrix(classifierrr.hiddenlayer_1.W.get_value())
			},
			updates=updates,
			on_unused_input='ignore'
			)
	print('training')

	# validation_frequency = 10
								
	# best_validation_probability = 1
	# final_image = 0
	# start_time = timeit.default_timer()

	done_looping = False
	epoch = 0
	while (epoch < n_epochs):
		epoch = epoch + 1
		print(epoch)
		minibatch_avg_cost = train_image()
		print(minibatch_avg_cost)


	y = deprocess_image(img.get_value())

	imsave('dream1.png',y)
		



	# final_dream = numpy.reshape(final_image,(250,250,3))
	
	# print (final_dream)

	# end_time = timeit.default_timer()
	
if __name__ == "__main__":

	inverse_sgd()
	







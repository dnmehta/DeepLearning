from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

# to overcome pickling limitations of instance method objects
import copy_reg
import types

def reduce_method(m):
    return (getattr, (m.__self__, m.__func__.__name__))

copy_reg.pickle(types.MethodType, reduce_method)


class HiddenLayer(object):


    def __init__(self,rng,input,n_in,n_out,W=None,b=None,activation=T.tanh):
        
        self.input=input
    
        if W is None:

            W_values=numpy.asarray(rng.uniform(low=-numpy.sqrt(6./n_in+n_out),
                                    high=numpy.sqrt(6./n_in+n_out),size=(n_in,n_out))
                                                ,dtype=theano.config.floatX)
            W=theano.shared(value=W_values,name='W',borrow=True)
        
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        

        self.W=W
        self.b=b
        
        self.output = T.tanh( T.dot(input, self.W) + self.b )
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):


    def __init__(self,rng,input,n_in,n_hidden_1,n_hidden_2,n_out):

        self.hiddenlayer_1=HiddenLayer(rng=rng,input=input,n_in=n_in,
                            n_out=n_hidden_1,activation=T.tanh)

        self.hiddenlayer_2=HiddenLayer(rng=rng,input=self.hiddenlayer_1.output,
                            n_in=n_hidden_1,n_out=n_hidden_2,
                            activation=T.tanh)

        self.logRegLayer=LogisticRegression(
                    input=self.hiddenlayer_2.output,
                    n_in=n_hidden_2,
                    n_out=n_out
                )

        self.L1= (
            abs(self.hiddenlayer_1.W).sum() +
            abs(self.hiddenlayer_2.W).sum() +
            abs(self.logRegLayer.W).sum()
                )

        self.L2_sq = (
            (self.hiddenlayer_1.W ** 2).sum() +
            (self.hiddenlayer_2.W ** 2).sum() +
            (self.logRegLayer.W ** 2).sum()
            )
        self.negative_log_likelihood = (
            self.logRegLayer.negative_log_likelihood
            )
        self.y_pred = self.logRegLayer.y_pred

        self.errors= self.logRegLayer.errors

        self.params=self.hiddenlayer_1.params+self.hiddenlayer_2.params+self.logRegLayer.params 

        self.input=input


class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
        
        # initialize with 0 the weights W (n_in,n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
    
        self.params = [self.W, self.b]

        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):

        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset):

    #############
    # LOAD DATA #
    #############
    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def mlp_mnist(learning_rate=0.01, n_epochs=20,
                           dataset='final_data.pkl.gz',
                           batch_size=20,
                           n_hidden_1=20,n_hidden_2=5, L1_reg=0.0,L2_sq=0.0001,
                           ):

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    print (n_train_batches)
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 250*250

    rng = numpy.random.RandomState(1234)

    classifier = MLP(input=x,rng=rng, n_in=250*250*3,n_hidden_1=n_hidden_1,n_hidden_2=n_hidden_2,n_out=2)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = ( classifier.negative_log_likelihood(y) + L1_reg*classifier.L1 + L2_sq*classifier.L2_sq )

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates=[ (params,params - learning_rate*T.grad(cost,params)) for params in classifier.params ]


    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 1  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    with open('mlp_best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)
                        

                # save the best model

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)



    print(classifier.logRegLayer.W.get_value())
    
    print(classifier.hiddenlayer_1.W.get_value())
    print(classifier.hiddenlayer_1.b.get_value()) 

    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.logRegLayer.p_y_given_x)

    # We can test it on some examples from test test `  
    dataset='final_data.pkl.gz'
    datasets = load_data(dataset)
    train_set_x,train_set_y=datasets[0];
    train_set_x=train_set_x.get_value();
    predicted_values = predict_model(train_set_x[0:5])

    print("Predicted values for the first 5 examples in test set:")
    print(predicted_values)








def predict():

    # load the saved model
    # classifier = pickle.load(open('mlp_best_model.pkl'))

    # compile a predictor function
    # predict_model = theano.function(
    #     inputs=[classifier.input],
    #     outputs=classifier.y_pred)

    # # We can test it on some examples from test test `  
    # dataset='final_data.pkl.gz'
    # datasets = load_data(dataset)
    # train_set_x,train_set_y=datasets[0];
    # train_set_x=train_set_x.get_value();
    # predicted_values = predict_model(train_set_x[0:5])

    # print("Predicted values for the first 5 examples in test set:")
    # print(predicted_values)

    print(classifier.logRegLayer.W.get_value())


if __name__ == '__main__':
    mlp_mnist()
   # predict()





import pickle
import gzip
import os
    
#import numpy as np
##import theano as th
##from theano import tensor as T
#from numpy import random as rng
#import scipy

def load_data(dataset):
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib.request
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print ('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print ('... loading data')

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(f,encoding='bytes')
    f.close()
    '''
    train_set, valid_set, test_set format: tuple(input, target)
    input is an numpy.ndarray of 2 dimensions (a matrix)
    witch row's correspond to an example. target is a
    numpy.ndarray of 1 dimensions (vector)) that have the same length as
    the number of rows in the input. It should give the target
    target to the example with the same index in the input.
    '''
    return (train_set, valid_set, test_set)
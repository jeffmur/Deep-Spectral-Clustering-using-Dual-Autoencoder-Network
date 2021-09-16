import numpy as np
from keras.datasets import mnist
# from tensorflow.keras.datasets import mnist

def get_data(params, data=None):
    print("INFO: Fetching data")
    ret = {}

    # get data if not provided
    if data is None:
        x_train, x_test, y_train, y_test = load_data(params)
    else:
        print("WARNING: Using data provided in arguments. Must be tuple or array of format (x_train, x_test, y_train, y_test)")
        x_train, x_test, y_train, y_test = data

    ret['spectral'] = {}


    x_val=x_test
    y_val=y_test


    ret['spectral']['train_and_test'] = (x_train, y_train, x_val, y_val, x_test, y_test)


    return ret

def load_data(params):
    '''
    Convenience function: reads from disk, downloads, or generates the data specified in params
    '''
    if params['dset'] == 'mnist':
        x_train, x_test, y_train, y_test = get_mnist(params['use_all_data'])
    else:
        raise ValueError('Dataset provided ({}) is invalid!'.format(params['dset']))

    return x_train, x_test, y_train, y_test


def get_mnist(full_dataset):
    '''
    Returns the train and test splits of the MNIST digits dataset,
    where x_train and x_test are shaped into the tensorflow image data
    shape and normalized to fit in the range [0, 1]
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if not full_dataset:
        # Just the top 10% of the dataset
        # First 600 elements of 6000
        x_train = x_train[:600]
        y_train = y_train[:600]

    x_train = np.expand_dims(x_train, -1) / 255
    x_test = np.expand_dims(x_test, -1) / 255
    return x_train, x_test, y_train, y_test
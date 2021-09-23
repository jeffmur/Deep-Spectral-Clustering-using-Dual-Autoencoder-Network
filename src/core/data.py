import numpy as np
from keras.datasets import mnist
from extra_keras_datasets import usps


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
        x_train, x_test, y_train, y_test = get_set(mnist, params)
    elif params['dset'] == 'usps':
        x_train, x_test, y_train, y_test = get_set(usps, params)
    else:
        raise ValueError('Dataset provided ({}) is invalid!'.format(params['dset']))

    return x_train, x_test, y_train, y_test

def get_set(object, params):
    '''
    Returns the train and test splits of the MNIST digits dataset,
    where x_train and x_test are shaped into the tensorflow image data
    shape and normalized to fit in the range [0, 1]
    '''
    (x_train, y_train), (x_test, y_test) = object.load_data()
    shape = (0, params['img_dim'], params['img_dim'])

    # check for true or false entry
    if params['full_dataset'] == False: 
        if params['sample_size'] == 10:
            x_train, x_test, y_train, y_test = linear_partition(object, 0.1, shape)
        elif params['sample_size'] == 90:
            x_train, x_test, y_train, y_test = remain_90(object, shape)
        else: 
            raise "Expected 10 or 90 percent for param['sample_size']" 

    x_train = np.expand_dims(x_train, -1) / 255
    x_test = np.expand_dims(x_test, -1) / 255

    ## Resulting Shape
    print(f"X Training: {np.shape(x_train)}")
    print(f"X Testing : {np.shape(x_test)}")

    ## What should Y shape be??
    print(f"Y Training: {np.shape(y_train)}")
    print(f"Y Testing : {np.shape(y_test)}")
    return x_train, x_test, y_train, y_test

def remain_90(obj, shape):
    print("Using remaining 90%")
    mnist_labels = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    
    (x_train, y_train), (x_test, y_test) = obj.load_data()
    
    rx_train = rx_test = np.empty(shape)

    for label in mnist_labels:
        # Only keep 10% of train data
        pre_train = x_train[y_train == label]
        # SETS, ROWS, Columns
        label_train = pre_train[int(0.1 * len(pre_train)):]
        # Only keep 10%
        pre_test = x_test[y_test == label]
        label_test = pre_test[int(0.1 * len(pre_test)):]

        # concat result
        rx_train = np.concatenate((rx_train, label_train))
        rx_test = np.concatenate((rx_test, label_test))
    
    ry_train = y_train[:len(rx_train)]
    ry_test = y_test[:len(rx_test)]

    return rx_train, rx_test, ry_train, ry_test 

def linear_partition(obj, percent_step, shape):
    print(f"NOTE: Left Skew with {percent_step * 100}% linear partition")
    '''
    Input:      step == 10%
    Mnist label [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    Output: 100% of 9, 90% of 8 ... 10% of 0
    '''

    mnist_labels = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    
    (x_train, y_train), (x_test, y_test) = obj.load_data()
    
    rx_train = rx_test = np.empty(shape)

    partition = 1

    for label in mnist_labels:
        print(f"On: {label} with skew of {partition * 100}%")
        # Only keep 10% of train data
        pre_train = x_train[y_train == label]
        # SETS, ROWS, Columns
        # print("NOTE")
        # print(np.shape(pre_train))
        # print(len(pre_train[0]))
        label_train = pre_train[:int(0.1 * len(pre_train))]
        # Then partition current label (iterative by 0.1 default)
        # print(np.shape(label_train))
        skew_train = label_train[:int(partition*len(label_train))]
        # print(np.shape(skew_train))
        # Only keep 10%
        pre_test = x_test[y_test == label]
        label_test = pre_test[:int(0.1 * len(pre_test))]
        # Then partition test data
        skew_test = label_test[:int(partition*len(label_test))]
        print(f"Skew Train : {len(skew_train)} / {len(label_train)} ")
        print(f"Skew Test : {len(skew_test)} / {len(label_test)} ")
        partition -= percent_step

        # concat result
        rx_train = np.concatenate((rx_train, skew_train))
        rx_test = np.concatenate((rx_test, skew_test))
    
    ry_train = y_train[:len(rx_train)]
    ry_test = y_test[:len(rx_test)]

    return rx_train, rx_test, ry_train, ry_test 
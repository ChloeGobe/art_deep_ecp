from __future__ import print_function
from natsort import natsorted
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.models import load_model
from keras import optimizers
from read_activations import get_activations, display_activations

img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)
train_data_dir = "../../data/wikipaintings_train"
validation_data_dir = "../../data/wikipaintings_val"
nb_train_samples = 66549
nb_validation_samples = 7383
batch_size = 64
num_classes = 25
epochs = 50


def get_data():

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    #x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')
    #x_train /= 255
    #x_test /= 255
    print('x_train shape:', x_train.shape)
    #print(x_train.shape[0], 'train samples')
    #print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test



if __name__ == '__main__':

    checkpoints = ('checkpoints/resnet_chloe.h5')

    if len(checkpoints) > 0:

        checkpoints = natsorted(checkpoints)
        assert len(checkpoints) != 0, 'No checkpoints found.'
        checkpoint_file = checkpoints[-1]
        print('Loading [{}]'.format(checkpoint_file))
        model = load_model(checkpoint_file)

        model.compile(optimizer=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print(model.summary())

        x_train, y_train, x_test, y_test = get_data()

        # checking that the accuracy is the same as before 99% at the first epoch.
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1, batch_size=128)
        print('')
        assert test_acc > 0.98

        a = get_activations(model, x_test[0:1], print_shape_only=True)  # with just one sample.
        display_activations(a)

        #get_activations(model, x_test[0:200], print_shape_only=True)  # with 200 samples.

        import numpy as np
        import matplotlib.pyplot as plt
        plt.imshow(np.squeeze(x_test[0:1]), interpolation='None', cmap='gray')
    
    else:
        print("ERROR")
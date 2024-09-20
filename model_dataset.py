import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import *
import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.datasets import fashion_mnist



def normalize(X_train,X_test):
    """
    function for normalizing the fashion dataset.
    """
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test

class Model_Dataset:
    """
    This class was used to include the load of the datasets in the init function and 
    the model in separate functions per case.
    """
    def __init__(self, fashion=True):
        """
        In this init function, it is controlled the load of all the datasets.
        """
        if fashion == True:

            img_rows = 28
            img_cols = 28
        else:
            img_rows = 32
            img_cols = 32

        #Load fashion
        if fashion==True:
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

            train_X = x_train.astype('float32')
            test_X = x_test.astype('float32')
            
            train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
            test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))
            train_X, test_X = normalize(train_X, test_X)
            train_X, val_X, train_Y, val_Y = train_test_split(train_X, y_train, test_size=0.2, random_state=123)



            train_Y_one_hot = to_categorical(train_Y)
            val_Y_one_hot = to_categorical(val_Y)
            test_Y_one_hot = to_categorical(y_test)
            self.train_X, self.train_Y_one_hot = train_X, train_Y_one_hot
            self.val_X, self.val_Y_one_hot = val_X, val_Y_one_hot
            self.test_X, self.test_Y_one_hot   = test_X, test_Y_one_hot

            self._input_shape = (28, 28, 1)

            

        

    def get_train(self):
        return (self.train_X, self.train_Y_one_hot)
    def get_val(self):
        return (self.val_X, self.val_Y_one_hot)
    def get_test(self):
        return (self.test_X, self.test_Y_one_hot)
    
    

    @property
    def input_shape(self):
        return self._input_shape


    def CNNModel(self, hypers, out_len):
        """
        Model for the fashion dataset.
        :param hypers are the hyperparameters of the model.
        :param out_len is the length of the output of the model.
        """
        print("hypers: ", hypers)
        model = Sequential()
        model.add(Conv2D(int(hypers[0]), (int(hypers[3]), int(hypers[3])),
         activation='relu', kernel_initializer='he_uniform', padding='same',
          input_shape=self._input_shape))
        model.add(Conv2D(int(hypers[1]), (int(hypers[4]), int(hypers[4])), 
            activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((int(hypers[5]), int(hypers[5]))))
        model.add(Flatten())
        model.add(Dropout(hypers[7]))
        model.add(Dense(int(hypers[2]), activation='relu',
         kernel_initializer='he_uniform'))
        model.add(Dropout(hypers[8]))
        if out_len>1:
            model.add(Dense(out_len, activation='softmax'))
        else:
            model.add(Dense(out_len))

        # compile model
        if out_len>1:
            optimizers= {0: SGD, 1: Adam, 2: RMSprop}
            model.compile(optimizer=optimizers[int(hypers[6])](),
            loss='categorical_crossentropy', metrics=['acc'])
        else:
            optimizers= {0: Adam, 1: RMSprop}

            model.compile(optimizer=optimizers[int(hypers[6])](),
            loss='mse')
        return model

    
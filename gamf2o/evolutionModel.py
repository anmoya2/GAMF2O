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

class EvolutionModelInterface:
    def build_model(self):
        """
        Function for building a model
        """
        raise NotImplementedError


    def train(self, X_train, y_train, X_val, y_val, train_config):
        """
        Function for training a model
        :param X_train is the training dataset (data)
        :param y_train includes the labels for the training dataset
        :param X_val is the validation dataset (data)
        :param y_val includes the labels for the validation dataset
        :param train_config is the configuration for training the model. It is a dictionary that should include: epochs, batch_size, callbacks (without checkpoint), checkpoint_path and verbose. 
        """
        raise NotImplementedError

    def evaluate(self, X_test, y_test):
        """
        Function for evaluating
        :param X_test is the test dataset (data)
        :param y_test includes the labels for the test dataset
        """
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError


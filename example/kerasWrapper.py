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

from gamf2o import EvolutionModelInterface

class KerasModelWrapper(EvolutionModelInterface):
    def __init__(self, model_fn, model_args=None):
        self.model_fn = model_fn
        self.model_args = model_args
        self.model = None

    def build_model(self, model_args=None):
        K.clear_session()

        if model_args is not None:
            self.model_args = model_args
        self.model = self.model_fn(self.model_args)

    def train(self, X_train, y_train, X_val, y_val, train_config):
        epochs = train_config["epochs"]
        batch_size = train_config["batch_size"]
        callbacks = train_config["callbacks"]
        checkpoint_path = train_config["checkpoint_path"]
        verbose=train_config["verbose"]

        # Add model checkpoint callback
        checkpoint_cb = ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            
        )
        callbacks.append(checkpoint_cb)

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=verbose
        )
        self.load(checkpoint_path)
        return history

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=0)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)


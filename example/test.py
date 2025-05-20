from kerasWrapper import KerasModelWrapper
import logging
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
from gamf2o import Genetic_HB
import numpy as np
import os
def f_enf(x, init):
  return x**(1.5)+init


def DenseWine(hypers):
    
   model = Sequential()
   model.add(Dense(int(hypers[0]), activation = 'relu'))
   model.add(Dropout(hypers[3]))
   model.add(Dense(int(hypers[1]), activation = 'relu'))
   model.add(Dropout(hypers[4]))
   model.add(Dense(int(hypers[2]), activation = 'relu'))
   model.add(Dense(1))

   optimizers= {0: Adam, 1: RMSprop}
   model.compile(optimizer=optimizers[int(hypers[5])](), loss='mse')
   return model

def main():
   cut_point_options = [3, 5]

   t_at = ["int", "int", "int", "float", "float", "cat"]
   ranges= [(32, 512), (32, 512), (32,512), (0,0.3), (0,0.3),
     (0, 1)]

   n_at = 6
   max_ind = 20
   n_gen = 100000
   n_cut = 3
   cut_el = 3
   delta = 3

   options_mut = []
   options_mut.append({"type_m": 1, "mut_percent": 0.05})
   options_mut.append({"type_m": 1, "mut_percent": 0.15})

   options_mut.append({"type_m": 2, "mut_percent": 0.1})
   options_mut.append({"type_m": 2, "mut_percent": 0.2})


   train_X = np.load('./datasets/x_train_wine.npy')
   train_Y_one_hot = np.load('./datasets/y_train_wine.npy')
   val_X = np.load('./datasets/x_val_wine.npy')
   val_Y_one_hot = np.load('./datasets/y_val_wine.npy')
   test_X = np.load('./datasets/x_test_wine.npy')
   test_Y_one_hot = np.load('./datasets/y_test_wine.npy')

   batch_size = 32
   max_epochs = 25
   patience = 10

   e_s = tf.keras.callbacks.EarlyStopping(patience=patience, monitor = "val_loss")

   callbacks = [e_s]

   train_config = {"epochs": None, "batch_size":batch_size, "callbacks": callbacks,
        "checkpoint_path": None, "verbose": 0}

   n_exec = 5

   seed = n_exec


   N_EXEC = "output/exec_wine"+str(n_exec)
   logging.basicConfig(level=logging.DEBUG)

   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


   os.environ['KMP_WARNINGS'] = 'off'


   kw_wine = KerasModelWrapper(DenseWine) 
   epsilon = 0.0005

   g_hb = Genetic_HB(t_at, ranges, n_at, model_wrapper=kw_wine, train_X=train_X, epsilon=epsilon, out_path=N_EXEC, train_config = train_config,
    train_Y_one_hot=train_Y_one_hot, val_X=val_X, val_Y_one_hot=val_Y_one_hot, test_X=test_X, test_Y_one_hot=test_Y_one_hot,
      n_ind = max_ind, generations = n_gen, n_cut = n_cut
    , cut_el = cut_el, cut_point_options= cut_point_options, 
    w_init = 0.6, type_m = 2, f_enf = f_enf, options_mut = options_mut, seed = seed, delta=delta)

    


   


    

    

   g_hb.run_evol(batch_size, max_epochs)


main()

'''
 Copyright 2020 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

'''
Utility functions for tf.data pipeline
'''

'''
Author: Mark Harvey, Xilinx Inc
'''
import os
import numpy as np

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist


def loss_func(y_true, y_predict, encoder_mu, encoder_log_variance):
    '''
    Loss function: Kulback-Leibler
    '''
    reconstruction_loss_factor = 1000
    reconstruction_loss = K.mean(K.square(y_true-y_predict), axis=[1, 2, 3])
    vae_r_loss = reconstruction_loss_factor * reconstruction_loss

    kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance), axis=1)

    return vae_r_loss + kl_loss



def mnist_download():
    '''
    Download, normalize ,reshape MNIST dataset
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data() 
    x_train = (x_train/255.0).astype(np.float32)
    x_test = (x_test/255.0).astype(np.float32)
    x_train = x_train.reshape(x_train.shape[0],28,28,1)
    x_test = x_test.reshape(x_test.shape[0],28,28,1)
    return x_train, x_test




def input_fn(input_data,batchsize):
    '''
    Dataset creation and augmentation for training
    '''
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    dataset = dataset.shuffle(buffer_size=1000,seed=42)
    dataset = dataset.batch(batchsize, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset






import os
import numpy as np

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Conv2DTranspose
from tensorflow.keras.layers import Flatten, Dense, Input, Reshape, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


tf.compat.v1.disable_eager_execution()

# https://keras.io/examples/generative/vae/
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_space_dim=2
input_shape=(28,28,1)

'''encoder'''
encoder_input = Input(shape=input_shape)
net = Conv2D(filters=1, kernel_size=(3, 3), padding="same", strides=1)(encoder_input)
net = BatchNormalization()(net)
net = Activation('relu')(net)
net = Conv2D(filters=32, kernel_size=(3,3), padding="same", strides=1)(net)
net = BatchNormalization()(net)
net = Activation('relu')(net)
net = Conv2D(filters=64, kernel_size=(3,3), padding="same", strides=2)(net)
net = BatchNormalization()(net)
net = Activation('relu')(net)
net = Conv2D(filters=64, kernel_size=(3,3), padding="same", strides=2)(net)
net = BatchNormalization()(net)
net = Activation('relu')(net)
net = Conv2D(filters=64, kernel_size=(3,3), padding="same", strides=1)(net)
net = BatchNormalization()(net)
net = Activation('relu')(net)
shape_before_flatten = K.int_shape(net)[1:]
net = Flatten()(net)
encoder_mu = Dense(units=latent_space_dim)(net)
encoder_log_variance = Dense(units=latent_space_dim)(net)
encoder_z = Sampling()([encoder_mu, encoder_log_variance])
encoder=Model(inputs=encoder_input, outputs=[encoder_mu,encoder_log_variance,encoder_z])


''' decoder '''
decoder_input = Input(shape=latent_space_dim)
net = Dense(units=np.prod(shape_before_flatten))(decoder_input)
net = Reshape(target_shape=shape_before_flatten)(net)
net = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", strides=1)(net)
net = BatchNormalization()(net)
net = Activation('relu')(net)
net = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", strides=2)(net)
net = BatchNormalization()(net)
net = Activation('relu')(net)
net = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", strides=2)(net)
net = BatchNormalization()(net)
net = Activation('relu')(net)
net = Conv2DTranspose(filters=1, kernel_size=(3, 3), padding="same", strides=1)(net)
decoder_output = Activation('sigmoid')(net)
decoder = Model(inputs=decoder_input, outputs=decoder_output)


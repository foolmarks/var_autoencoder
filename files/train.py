

import os
import shutil
import cv2


# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint,LearningRateScheduler
from tensorflow.keras.layers import Input, Lambda


from vae import encoder, decoder
from utils import input_fn, loss_func, mnist_download


tf.compat.v1.disable_eager_execution()

learning_rate=0.001
batchsize = 100


def step_decay(epoch):
    """
    Learning rate scheduler used by callback
    Reduces learning rate depending on number of epochs
    """
    lr = learning_rate
    if epoch > 50:
        lr /= 100
    elif epoch > 20:
        lr /= 10
    return lr


'''
Variational encoder model
'''
image_dim = 28
image_chan = 1

input_layer = Input(shape=(image_dim,image_dim,image_chan))
encoder_mu, encoder_log_variance, encoder_z = encoder(input_layer)

dec_out = decoder(encoder_z)
model = Model(inputs=input_layer, outputs=dec_out)
model.summary()


'''
Prepare MNIST dataset
'''
x_train, x_test = mnist_download()
mnist_train = input_fn((x_train,x_train), batchsize)
mnist_test = input_fn((x_test,x_test), batchsize)


'''
Call backs
'''
tb_call = TensorBoard(log_dir='tb_logs')
chkpt_call = ModelCheckpoint(filepath=os.path.join('float_model','f_model.h5'), 
                             monitor='val_loss',
                             verbose=1,
                             save_weights_only=False,
                             save_best_only=True)
lr_scheduler_call = LearningRateScheduler(schedule=step_decay,
                                          verbose=1)
callbacks_list = [tb_call, chkpt_call, lr_scheduler_call]


'''
Compile
'''
model.compile(optimizer=Adam(lr=learning_rate),
              loss=lambda y_true,y_predict: loss_func(y_true,y_predict,encoder_mu,encoder_log_variance))

'''
Training
'''

# make folder for saving trained model checkpoint
os.makedirs('float_model', exist_ok = True)

# remake Tensorboard logs folder
shutil.rmtree('tb_logs', ignore_errors=True)
os.makedirs('tb_logs')
    
train_history = model.fit(mnist_train,
                          epochs=30,
                          steps_per_epoch=60000//batchsize,
                          validation_data=mnist_test,
                          validation_steps=10000//batchsize,
                          callbacks=callbacks_list,
                          verbose=1)

'''
make predictions
'''
predictions = model.predict(mnist_test,
                            steps=10000//batchsize,
                            verbose=1)

for i in range(10):
    p = predictions[i] * 255.0
    cv2.imwrite('pred_'+str(i)+'.png', p)

'''
Save architecture (no weights) to a JSON file
'''
with open(os.path.join('float_model','f_model.json'), 'w') as f:
    f.write(model.to_json())



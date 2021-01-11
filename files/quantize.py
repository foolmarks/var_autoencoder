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
Quantize the floating-point model
'''

'''
Author: Mark Harvey
'''


import argparse
import os
import shutil
import sys

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.optimizers import Adam

from utils import input_fn, mnist_download, loss_func
from vae import Sampling

DIVIDER = '-----------------------------------------'




def quant_model(float_model,quant_model,batchsize,evaluate):
    '''
    Quantize the floating-point model
    Save to HDF5 file
    '''

    # make folder for saving quantized model
    head_tail = os.path.split(quant_model) 
    os.makedirs(head_tail[0], exist_ok = True)

    # set learning phase for no training
#    tf.keras.backend.set_learning_phase(0)

    # load trained floating-point model    
    float_model = load_model(float_model, compile=False, custom_objects={'Sampling': Sampling} )

    # make dataset and image processing pipeline
    _, x_test = mnist_download()
    quant_dataset = input_fn((x_test,x_test), batchsize)

    # run quantization
    quantizer = vitis_quantize.VitisQuantizer(float_model)
    quantized_model = quantizer.quantize_model(calib_dataset=quant_dataset)

    # saved quantized model
    quantized_model.save(quant_model)
    print('Saved quantized model to',quant_model)


    return



def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--float_model',  type=str, default='float_model/f_model.h5', help='Full path of floating-point model. Default is float_model/k_model.h5')
    ap.add_argument('-q', '--quant_model',  type=str, default='quant_model/q_model.h5', help='Full path of quantized model. Default is quant_model/q_model.h5')
    ap.add_argument('-b', '--batchsize',    type=int, default=100,                      help='Batchsize for quantization. Default is 100')
    ap.add_argument('-e', '--evaluate',     action='store_true', help='Evaluate floating-point model if set. Default is no evaluation.')
    args = ap.parse_args()  

    print('\n------------------------------------')
    print('TensorFlow version : ',tf.__version__)
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --float_model  : ', args.float_model)
    print (' --quant_model  : ', args.quant_model)
    print (' --batchsize    : ', args.batchsize)
    print (' --evaluate     : ', args.evaluate)
    print('------------------------------------\n')


    quant_model(args.float_model, args.quant_model, args.batchsize, args.evaluate)


if __name__ ==  "__main__":
    main()

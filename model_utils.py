#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 10:46:03 2018

@author: jakob
"""
from keras.callbacks import Callback, TensorBoard, LambdaCallback
from vis_utils import to_tensor, load_2d
from keras import models
import math
import numpy as np
import imageio
import matplotlib.pylab as plt
import keras.backend.tensorflow_backend as Ki

class CustomCallback(Callback):
    
    class activationHistory(Callback):
            def __init__(self, shape, img_path='OASIS/Test/predict/anormal.png'):
                self.batch_activations_model = []
                img = load_2d(img_path, (shape))
                self.img_tensor = to_tensor(img) 
            def on_batch_end(self, batch, logs={}):
                with Ki.tf.device('/cpu:0'):
                    lays=len(self.model.layers)
                    #getting layer outputs, init new model
                    layer_outputs = [layer.output for layer in self.model.layers[:lays]]
                    activation_model = models.Model(inputs=self.model.input, outputs=layer_outputs)
                #with Ki.tf.device('/cpu:2'):    
                    activations = activation_model.predict(self.img_tensor) 
                    self.batch_activations_model.append(activations)
                    #self.batch_activations_model.append(self.model.get_weights())
                return
            def get_stack(self):
                return self.batch_activations_model
            
    def tensorCall():
        return TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, 
                                 write_images=True)
    def weightsCall():
    
        def get_weights(epoch,logs):
            print("end of epoch: " + str(epoch)) #for instance
    
        return LambdaCallback(on_epoch_end=get_weights)
        
def make_gif(stack, layer_to_vis=0):
        
    
    print('start model_acti_gif')
    
    writer = imageio.get_writer('normlayer.gif', mode='I', loop=1)
    shape = stack[0][layer_to_vis].shape[-2]
    features = stack[0][layer_to_vis].shape[-1]
    m = math.ceil(math.sqrt(features))
    grid = np.zeros((shape*m,shape*m))
    #grid = np.full((shape*m,shape*m),255)
    l = len(stack)
    
    for i in range(len(stack)):
        
        activations = stack[i]
        activations = activations[layer_to_vis]
        
        f=0 
        for c in range(m):
            for r in range(m):
                x=c*shape
                y=r*shape
                if f < features:
                    acti = proc(activations[0, : ,: ,f])
                    grid[x:x+shape, y:y+shape] = acti
                print('\r[%i/%i] %i - %i - %i' % (i,l,f,r,c), end='')
                f += 1
    
        grid = grid.astype('uint8')
        writer.append_data(grid)

    writer.close()
        
def proc(activations):
    
    channel_image = activations
    channel_image -= channel_image.mean()
    channel_image /= channel_image.std()
    channel_image *= 64
    channel_image += 128
    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
    return channel_image

def plot_history(history):
    
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
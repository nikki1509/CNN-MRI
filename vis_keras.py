# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 21:19:43 2018
Contains Model_explorer class. This class provides an easy investigation tool for a keras sequential models.
@author: jakpo_000
"""


#from copy import deepcopy
#from keras.models import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt
import vis_core as vc
import vis_utils as vu
from debug import DEBUG
    

class Model_explorer():
    """
    Init with model or path to .h5 file 
    """
    def __init__(self, arg):

        debug = DEBUG()
        debug.pause()

        if isinstance(arg, str):
            self.model = load_model(arg)
            debug.time('Model load from path')
        elif arg.__class__.__name__ is 'Sequential':
            self.model = arg
            debug.time('Sequential model')
        else:
            print('input not supported! Init with Sequential or path to *.h5 Sequential')
            return    
        
        self.input_shape = self.model.input_shape
        self.summary() = self.model.summary()
        
    #def _get_weights_bias(model):
    def set_test_image(self, img_path):
        if len(self.input_shape) is 4:
            
            t_size = self.input_shape[1], self.input_shape[2]
            self.img = vu.load_2d(img_path, t_size)
            self.tensor = vu.to_tensor(self.img)
        
        elif len(self.input_shape) is 5:
             
            t_size = self.input_shape[1], self.input_shape[2], self.input_shape[3]
            self.img = vu.load_2d(img_path, t_size)
            self.tensor = vu.to_tensor(self.img)
            
        
    def filters(self):
        """
        shows the first conv layer kernels
        """
        vc.filters(self.model)
        
    def activations(self):
        a = vc.activations(self.model, self.tensor)
        vu.plot_tensor(a[2])
        
        
    def cam(self):
        vc.cam(self.model, self.tensor)
        
    def grad_ascent(self):
        #ga = vc.gradient_ascent(self.model) 
        stack = []
        for i in range(72):
            #stack.append(n_max(model, filter_index=i))
            stack.append(vc.gradient_ascent(self.model,filter_index=i))
       
        vu.plot_stack(stack)
    
  

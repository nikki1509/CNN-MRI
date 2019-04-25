# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 21:19:43 2018
Contains Model_explorer class. This class provides an easy investigation tool
for a keras sequential models.
@author: jakpo_000
"""


from copy import deepcopy
# from keras.models import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from . import vis_core as vc
from . import vis_utils as vu
# from debug import DEBUG
import os



class Model_explorer():
    """
    With this class a keras sequential 2D/3D grayscale model can be
    investigated. The class supports different visualization techniques and
    methods. From classical methods like filter kernel, activation maps to
    gradient ascent and grad_cam.

    """
    def __init__(self, arg):
        """
        Init with sequential 2D/3D grayscale model or path to .h5 file

        """
        # debug = DEBUG()
        # debug.pause()

        if isinstance(arg, str):
            self.model = load_model(arg)
            self.path_name = os.path.basename(arg)
            # debug.time('Model load from path')
        elif arg.__class__.__name__ is 'Sequential':
            self.model = arg
            self.path_name = 'not from path'
            # debug.time('Sequential model')
        else:
            print('input not supported! Init with Sequential or path to *.h5 Sequential')
            return

        # Mirror model attributes
        self.name = self.model.name
        self.input_shape = self.model.input_shape
        self.input_image_dim = vu.model_helper.model_indim(self.model)
        self.input_image_str = vu.model_helper.model_input(self.model)
        if self.input_image_dim is 2:
            self.t_size = self.input_shape[1], self.input_shape[2]
        elif self.input_image_dim is 3:
            self.t_size = self.input_shape[1], self.input_shape[2], self.input_shape[3]

        self.active_object = False
        self.object = None
        self.object_name = None
        self.summary = lambda: self.model.summary()
        Model_explorer.info(self)

    # def _get_weights_bias(model):
    def info(self):
        '''
        Print model information
        '''
        print('Name: %s' % self.name)
        print('Path_name: %s' % self.path_name)
        print('Input is %s with shape %s' % (self.input_image_str, self.t_size))


    def set_image_from_path(self, img_path, name=None):
        '''
        Plotting specific slice of a 4D tensor in one figure
        # Arguments
            img_path:  Path to image
            name:      Image name for saving options
        # Returns
            -
        '''
        self.object_name = name
        if self.object_name is None:
            self.object_name = 'unnamed'
        self.path_str = img_path
        self.object = vu.io.ext_load(img_path, target_size=self.t_size)
        self.active_object = True

    def set_image_from_array(self, array, name=None):
        ## check size
        self.object_name = name
        if self.object_name is None:
            self.object_name = 'unnamed'

        self.object = array
        self.active_object = True

    def filters(self, plot=True):
        '''
        Shows the first conv layer kernels
        # Arguments
            plot:  Path to image
        # Returns
            weights:   All weights
        '''
        if self.active_object is False:
            print('Error! No test object found, set first')
            return
        weights = vc.filters(self.model)
        if plot:
                if self.input_image_dim is 3:
                    vu.plot.plot_5dtensor(weights)
                elif self.input_image_dim is 2:
                    vu.plot.plot_tensor(weights)


        return weights
        #vu.plot.plot_tensor(weights, weights=True, cmap='gray')


    def activations(self, layer=0, plot=True):
        if self.active_object is False:
            print('Error! No test object found, set first')
            return

        else:
            activation = vc.activations(self.model, self.object)
            if plot:
                if self.input_image_dim is 3:
                    vu.plot.plot_5dtensor(activation[layer], cmap='seismic') # YIOrBr
                elif self.input_image_dim is 2:
                    vu.plot.plot_tensor(activation[layer])

            return activation

    def occ_info(self):
        return vu.model_helper.possible_kernel_stride(self.t_size, plot=True)

    def occlusion(self, kernel=None, stride=None, colour=0.5, arg=0, plot=True):
        if (kernel or stride) is None:
            combinations = vu.model_helper.possible_kernel_stride(self.t_size)
            le = len(combinations)
            le = int(le/2)
            kernel = combinations[le][1]
            stride = combinations[le][2]
            #print('Kernel %i and stride %i were chosen automatically!' % (kernel, stride))
        heatmap = vc.occlusion(self.model, self.object, stride, kernel, arg=arg,
                               k_value=colour)
        if plot:
                if self.input_image_dim is 3:
                    vu.plot.plot_3d(heatmap)
                elif self.input_image_dim is 2:
                    vu.plot.plot_tensor(heatmap)
        return heatmap

    def grad_cam(self, class_arg=None, values='pos', save_imposed=False,
                 destination_path='/', plot=True, layer=-1):

        if self.active_object is None:
            print('Error! No test object found, set first')
            return

        heatmap = vc.grad_cam(self.model, self.object, class_arg, out=values,
                              layer=layer)

        if save_imposed:
            base = os.path.basename(self.path_str)
            base = os.path.splitext(base)[0]
            name_str = 'Grad-CAM-Heatmap-'+ self.object_name
            vu.plot.superimpose(heatmap, self.path_str, destination_path,
                                name=name_str)
            print(1)
        if plot:
            if self.input_image_dim is 3:
                vu.plot.plot_3d(heatmap)
            elif self.input_image_dim is 2:
                vu.plot.plot_tensor(heatmap)
        return heatmap

    def grad_ascent(self, input_image=True, filter_index=0, layer=-1, plot=True):
        # ga = vc.gradient_ascent(self.model)
        last_conv = lambda x: vu.model_helper.count_same(x, 'conv')[-2][layer]
        name = last_conv(self.model)
            # stack.append(n_max(model, filter_index=i))
        print(name)
        input_image = None
        if input_image:
            input_image = self.object
        maximized=vc.gradient_ascent(self.model, img=input_image,
                                     filter_index=filter_index,
                                     layer_name=name)
        #vu.plot.plot_stack(stack)
        if plot:
            if self.input_image_dim is 3:
                middle = int((maximized.shape[-1]) / 2)
                plt.imshow(maximized[:,:,middle])
                #vu.plot.plot_3d(maximized)
#            elif self.input_image_dim is 2:
#                vu.plot.plot_tensor(maximized)
        return maximized

    def gradient(self):
        grad = vc.gradient(self.model, self.object)
        grad = np.squeeze(grad, axis=-1)
        return grad

    def predict(self):
        pred = self.model.predict(self.object)
        return pred





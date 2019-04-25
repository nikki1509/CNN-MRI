#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 12:37:00 2018

@author: jakob dexl
"""
# import matplotlib.pyplot as plt
from keras import models
from keras import backend as K
import numpy as np
from copy import deepcopy

from . import vis_utils as vu


def activations(model, img_tensor):
    '''
    Returns every activationmap for given model and input tensor
    # Arguments
        model:    	 Keras model
        img_tensor:  Image as tensor
    # Returns
        activations: Activationmaps
    '''
    # getting layer outputs, init new model
    layer_outputs = [layer.output for layer in model.layers[:]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)

    return activations


def filters(model, layer=None, layer_class='conv'):
    '''
    Returns neuron weights from given model and specified layer/layertype
    # Arguments
        model:    	 Keras model
        layer:  	 Layer by default first one
        layer_class: Specify layer type e.g 'conv' (convlutional layers) or
                     'core' (dense layers)
    # Returns
        weights: Weights
    '''
    # find first 'conv' by default
    # otherwise name prefered layer
    if layer is None:
        first_conv = lambda x: vu.model_helper.count_same(x, layer_class)[1][0]
        first_conv_count = first_conv(model)
    else:
        first_conv = lambda x: vu.model_helper.count_same(x, layer_class)[1][layer+1]
        first_conv_count = first_conv(model)
    # get weights
    weights = model.layers[first_conv_count].get_weights()[0]
    # plot if 2d
    # vu.plot.plot_tensor(weights, weights=True,cmap = 'gray')
    model.get_weights()

    return weights

def bias(model):
    return model.layers.get_weights()[1]

def grad_cam(model, img_tensor, class_arg=None, class_names=None, out='pos', layer=-1):
    '''
    Method proposed by Selvaraju RR, Cogswell M, Das A et al. Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization (2017/03/21), 2017, http://arxiv.org/pdf/1610.02391
    Implementation proposed by jacob gildenblat https://github.com/jacobgil/keras-grad-cam, MIT lic.
    This function is adopted from: (Francois Chollet Deep Learning with Python,
    pages 172ff, 2018, MIT lic.)
    Execute grad_cam for given model and input tensor. Special parameter are
    provided
    # Arguments
        model:    	 Keras model
        img_tensor:  Image as tensor
        class_arg:
        class_names:
        out:
    # Returns
        Heatmap :    Heatmap for given parameter
    '''
    # Decide 2D/3D and predict output
    tensor_len = len(model.input_shape)
    preds = model.predict(img_tensor)

    # Decide calculation in regard to which class, default highest prediction
    if class_arg is None:
        number = np.argmax(preds)
    else:
        number = class_arg

    # Get output tensor for specified class, find last conv layer name
    brain_output = model.output[:, number]
    last_conv = lambda x: vu.model_helper.count_same(x, 'conv') [-2] [layer]
    last_conv_name = last_conv(model)
    last_conv_layer = model.get_layer(last_conv_name)
    print(last_conv_name)
    # Get gradient tensor (per channel), mean(global average pooling) to get
    # gradient for each map
    grads = K.gradients(brain_output, last_conv_layer.output)[0]

    grads = grads / (np.max(grads) + K.epsilon())


    if tensor_len is 4:
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    if tensor_len is 5:
        pooled_grads = K.mean(grads, axis=(0, 1, 2, 3))

    # Provide access to grads and activationmap, get feature len
    iterate = K.function([model.input],
                         [pooled_grads, last_conv_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate([img_tensor])
    # return pooled_grads_value, conv_layer_output_value
    pre_out = deepcopy(conv_layer_output_value)

    features = last_conv_layer.output.shape[-1].value

    # Weighing every activationmap with its gradient, mean all maps
    if tensor_len is 4:
        for i in range(features):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    if tensor_len is 5:
        for i in range(features):
            conv_layer_output_value[:, :, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)

    # Test cases to force a division by zero
    test = np.mean(np.abs(heatmap))

    eps = 1e-25
    if class_names is None:
        print('arg: %i, pred:%.e,' % (number, preds[0][number]), end='')
    else:
        print('arg_high: %8s, pred:%37s,' % (class_names[number], preds),
              end='')
    if test > 0:
        if out is 'pos':
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)+eps
            print('pos, mean:%.e' % np.mean(heatmap))
        if out is 'neg':
            heatmap = np.minimum(heatmap, 0)
            #heatmap /= np.min(heatmap)-eps
            print('neg, mean: %.e' % np.mean(heatmap))

    if test == 0:
        print('Heatmap is empty; sum pooled_grads: %i; sum layer out: %i' %
              (np.sum(pooled_grads_value), np.sum(pre_out)))

    heatmap = vu.preprocess.np_normalize(heatmap)

    return heatmap


def gradient(model, img, output_index=0):
    '''
    Excerpt from https://github.com/experiencor/deep-viz-keras Apache License, Version 2.0
    Method proposed Springenberg JT, Dosovitskiy A, Brox T et al. Striving for Simplicity: The All Convolutional Net
    Captures the gradient from the optimizer
    # Arguments
        model:    	 	  Keras model
        img:  		  	  Image as starting point for loss calculation
        output_index:	  Specify layer output
    # Returns
        grad:    		  Gradient information
        '''
    input_tensors = [model.input, K.learning_phase()]
    gradients = model.optimizer.get_gradients(model.output[0][output_index],
                                              model.input)
    computed_gradients = K.function(inputs=input_tensors, outputs=gradients)
    grad = computed_gradients([img, 0])[0][0]

    return grad


def gradient_ascent(model, img=None, filter_index=0, layer_name=None,
                    iterations=1000, cancel_iterator=40):
    '''
    Method proposed by Dumitru Erhan, Yoshua Bengio, Aaron Courville, and Pascal Vincent. Visualizing Higher-Layer
    Features of a Deep Network. Technical Report 1341 2009
    This function is adopted from: (Francois Chollet Deep Learning with Python,
    pages 167ff, 2018, MIT lic.) 
    Executes a gradient ascent
    # Arguments
        model:    	 	  Keras model
        img:  		  	  Image as starting point for loss calculation
        filter_index:	  Specify filter index
        layer_name:		  Specify layer name
        cancel_iteratior: Stop condition for loss calculation. If this count is
                          reached whithout changes in loss image a break occurs
    # Returns
        img:    		  Calculated image
        '''
    model_input_size = vu.model_helper.model_indim(model)
    if model_input_size is 2:
        img_h = model.input_shape[1]
        img_w = model.input_shape[2]

    elif model_input_size is 3:
        img_h = model.input_shape[1]
        img_w = model.input_shape[2]
        img_d = model.input_shape[3]

    else:
        print('Error! Input is not supported')
        return


    if img is not None:
        size_image = len(img.shape)
        input_tensor = model_input_size+2
        if (input_tensor) is not size_image:
            print('Error! Wrong input image size')
            print('\n tensor length should be %s!' % input_tensor)
            return
    channel = 1
    step_size = 1

    if layer_name is None:
        last_conv = lambda x: vu.model_helper.count_same(x, 'conv')[-2][-1]
        layer_name = last_conv(model)

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    input_img = model.input

    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    if vu.model_helper.model_indim(model) is 2:
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

    if vu.model_helper.model_indim(model) is 3:
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, :, filter_index])
    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    # grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    grads = vu.preprocess.normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # we start from a gray image with some noise
    print(model_input_size)
    if img is not None:
        input_img_data = img
    else:
        if model_input_size is 2:

            if K.image_data_format() == 'channels_first':
                input_img_data = np.random.random((1, channel, img_h, img_w))
            else:
                input_img_data = np.random.random((1, img_h, img_w, channel))

                #if model_input_size is 3:
        elif model_input_size is 3:
            if K.image_data_format() == 'channels_first':
                input_img_data = np.random.random((1, channel, img_h, img_w,
                                                   img_d))
            else:
                input_img_data = np.random.random((1, img_h, img_w, img_d,
                                                   channel))
        input_img_data = (input_img_data - 0.5) * 20 + 128

    # run gradient ascent for 20 steps
    for i in range(iterations):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step_size

        print('\r\rCurrent loss value:%.3f , filter: %d, iteration: %d '
              % (loss_value, filter_index, i+1), end='')
        if loss_value <= 0. and i > cancel_iterator:
            # some filters get stuck to 0, we can skip them
            print('break')
            break
    print('\n')
    img = input_img_data[0]
    img = vu.preprocess.deprocess_image(img)
    img = np.squeeze(img)

    return img


def occlusion(model, img_tensor, stride, kernel, arg=0, k_value=0.5):
    '''
    As proposed by Zeiler MD, Fergus R. Visualizing and Understanding Convolutional Networks (2013/11/28), 2013,
    http://arxiv.org/pdf/1311.2901
    Occlusion experiment which systematically occludes a part of a given input
    image and feeds it into the given model. Prediction is measured and copied
    into the Heatmap
    # Arguments
        model:    	 Keras model
        img_tensor:  Image as tensor
        stride:		 Strdide have to be smaller smaller than kernel+1
        kernel:		 Size of kernel have to be smaller than image size
        k_value:	 Int value for colour of patch e.g. 0 is black
    # Returns
        Heatmap :    Heatmap for given parameter
    '''
    if not vu.model_helper.model_tensor_test(model, img_tensor):
        print('Failed: Model/tensor shape not same')
        return

    if vu.model_helper.model_indim(model) is 2:
        if not isinstance(kernel, tuple):
            kernel = (kernel, kernel)
        img_shape = img_tensor[0, :, :, 0].shape

        if not vu.model_helper.kernel_stride_test(kernel, stride, img_shape):
            print('Failed! Kernel, stride, image size error')
            print('Possibe combinations are:')
            vu.model_helper.possible_kernel_stride(img_shape, kernel)
            return

        # calculate the number of testimages produced
        n_x = int(((img_shape[0]-kernel[0])/stride)+1)
        n_y = int(((img_shape[1]-kernel[1])/stride)+1)
        n_z = 1
        num_occ = int(n_x*n_y)
        reshape_vec = (n_x, n_y)

    if vu.model_helper.model_indim(model) is 3:
        if not isinstance(kernel, tuple):
            kernel = (kernel, kernel, kernel)
        img_shape = img_tensor[0, :, :, :, 0].shape

        if not vu.model_helper.kernel_stride_test(kernel, stride, img_shape):
            print('Failed! Kernel, stride, image size error')
            print('Possibe combinations are:')
            vu.model_helper.possible_kernel_stride(img_shape, kernel)
            return

        # calculate the number of test images beeing produced
        n_x = int(((img_shape[0]-kernel[0])/stride)+1)
        n_y = int(((img_shape[1]-kernel[1])/stride)+1)
        n_z = int(((img_shape[2]-kernel[2])/stride)+1)
        num_occ = int(n_x*n_y*n_z)
        reshape_vec = (n_x, n_y, n_z)

    # initialize heatmap
    heatvec = np.zeros(num_occ)
    count = 0

    # pred_zero = deepcopy(img_tensor[0])
    pred = model.predict(img_tensor)
    standard_prediction = pred[0, arg]

    print('Kernel:     %s' % str(kernel))
    print('Stride:     %s' % (stride))
    print('Filter:     %s' % (num_occ))
    print('Prediction: %s' % (standard_prediction))
    for z in range(n_z):
        for y in range(n_y):
            for x in range(n_x):
                manipul_img = deepcopy(np.squeeze(img_tensor, axis=(0, -1)))
                if n_z == 1:
                    manipul_img[x:(x+kernel[0]), y:(y+kernel[1])] = k_value
                else:
                    manipul_img[x:(x+kernel[0]), y:(y+kernel[1]),
                                z:(z+kernel[2])] = k_value

                t = vu.io.to_tensor(manipul_img)
                pred = model.predict(t)
                heatvec[count] = pred[0, arg]
                count += 1
                prog = (count/num_occ)*100

                if (count % 1) == 0:

                    print('\r[%4d/%i] %.2f %%, Prediction: %f' % (count,
                          num_occ, prog, pred[0, arg]), end='')

    heatmap = np.reshape(heatvec, reshape_vec)
    heatmap = heatmap - standard_prediction
#    title = 'heatmap'
#    # title = os.path.basename('heatmap')
#    if plot is True:
#        plt.imshow(heatmap)
#        plt.colorbar()
#        plt.title('%s\nstandard_prediction=%s\nsize=%s kernel=%s stride=%s'
#                  % (title, standard_prediction, size, kernel, stride))
#        plt.xticks([])
#        plt.yticks([])
#        plt.show

    return heatmap

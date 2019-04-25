# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:20:30 2018
Utilities for keras Sequential models
@author: jakpo_000
"""
from copy import deepcopy


def make_tuple(size, dim=2):
    for i in range(dim-1):
        size = (size + size)
    return size


def possible_kernel_stride(size, dim=2, plot=False):
    if not isinstance(size, tuple):
        size = make_tuple(size, dim)

    erg = []
    possible = []
    for c in size:
        for k in range(c-1, 1, -1):
            for s in range(1, k):
                if ((c - k) % s) == 0:
                    erg.append((k, s))

        if not possible:
            possible = deepcopy(erg)
            erg = []
        else:
            s = set(erg)
            common = (s & set(possible))
            erg = []
            possible = list(common)

    final = []
    for i in possible:

        res = 1
        for dim in size:
            res *= ((dim-i[0])/i[1])+1
        tmp = list(i)
        tmp.insert(0, res)
        tmp = tuple(tmp)
        final.append(tmp)

    final.sort(reverse=True)
    if plot:
        for element in final:
            print('Resolution: %10i, Kernel: %3i, Stride: %2i' % element)
    final.insert(0, ('Resolution', 'Kernel', 'Stride'))
    return final


def kernel_stride_test(kernel, stride, image):
        if len(kernel) is 2:
            if kernel[0] < stride or kernel[1] < stride:
                print('stride bigger than kernel!')
                return

            im_h = image[0]
            im_w = image[1]
            k_h = kernel[0]
            k_w = kernel[1]

            tmp_h = im_h - k_h
            tmp_w = im_w - k_w

            boolean = False

            if (tmp_h % stride) != 0:
                print('Stride error in dim 1')
            if (tmp_w % stride) != 0:
                print('Stride error in dim 2')
            else:
                boolean = True

            return boolean

        if len(kernel) is 3:
            im_h = image[0]
            im_w = image[1]
            im_d = image[2]
            k_h = kernel[0]
            k_w = kernel[1]
            k_d = kernel[2]

            tmp_h = im_h - k_h
            tmp_w = im_w - k_w
            tmp_d = im_d - k_d

            boolean = False

            if (tmp_h % stride) != 0:
                print('Stride error in dim 1')
            if (tmp_w % stride) != 0:
                print('Stride error in dim 2')
            if (tmp_d % stride) != 0:
                print('Stride error in dim 3')
            else:
                boolean = True

            return boolean


def model_tensor_test(model, tensor):
    input_type = model_indim(model)
    if (len(tensor.shape)-2) is not input_type:
        return False
    else:
        return True

def model_indim(model):
    shape = model.input_shape
    shape_l = len(shape)
    if shape_l is 3:
        dim = 1
    if shape_l is 4:
        dim = 2
    if shape_l is 5:
        dim = 3
    return dim


def model_input(model):
    shape = model.input_shape
    shape_l = len(shape)
    string = ''
    if shape_l is 3:
        string = '1d_no_image'
    if shape_l is 4:
        string = '2d'
    if shape_l is 5:
        string = '3d'
    if shape_l is 6:
        string = '4d_vid'
    if shape[-1] is 1:
        string += '_grayscale'
    if shape[-1] is 3:
        string += '_rgb'
    return string


def count_same(model, layer_name='conv'):

    conv_classes = {
        'Conv1D',
        'Conv2D',
        'Conv3D',
        'Conv2DTranspose',
    }

    core_classes = {
        'Dense',
        'Flatten',
    }

    if layer_name is 'conv':
        classes = conv_classes
    elif layer_name is 'core':
        classes = core_classes
    elif isinstance(layer_name, set):
        print('test')
        classes = layer_name

    name = []
    layer_type = []
    pos = []
    pos_count = 0
    count = 0
    for layer in model.layers:
        if layer.__class__.__name__ in classes:
            count += 1
            pos.append(pos_count)
            name.append(layer.name)
            layer_type.append(layer.__class__.__name__)
        pos_count += 1

    return count, pos, name, layer_type

def num_conv_filter(model, idx):
    conv = lambda x: count_same(x, 'conv')[1][idx]
    place = conv(model)
    num = model.layers[place].output_shape[-1]
    return num

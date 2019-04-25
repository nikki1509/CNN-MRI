# -*- coding: utf-8 -*-
"""
Created on Wed May 30 16:15:56 2018

@author: jakpo_000
"""
import numpy as np
from keras import backend as K


def deprocess_image(x):
    '''
    Utility function to convert a tensor into a valid image
    # Arguments
        x:    Tensor
    # Returns
        x:    Deprocessed image
    '''
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    '''
    Utility function to normalize a tensor by its L2 norm
    # Arguments
        x:    Tensor
    # Returns
        Normalized tensor
    '''
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def np_normalize(img):
    '''
    # Arguments
        img:    Image
    # Returns
        img:    Normalized image
    '''
    thres = 1e-07
    tmp = np.sqrt(np.mean(np.square(img)))
    if tmp == 0:
        tmp += thres
    return img / tmp


def np_clip(img):
    '''
    # Arguments
        img:    Image
    # Returns
        img:    Standardized image (between 0 and 1)
    '''
    thres = 1e-07
    ma = np.max(img)
    mi = np.min(img)
    img = (img-mi)/(ma-mi+thres)
    return img


# draft functions
def np_clip_2(img):
    thres = 1e-07
    ma = np.max(img)/2
    ra = ma-np.min(img)

    img = ((img-(ma))/(ra+thres))
    return img

# def standardize(x):
#        """Apply the normalization configuration to a batch of inputs.
#
#        # Arguments
#            x: batch of inputs to be normalized.
#
#        # Returns
#            The inputs, normalized.
#        """
#        if self.rescale:
#            x *= self.rescale
#        if self.samplewise_center:
#            x -= np.mean(x, keepdims=True)
#        if self.samplewise_std_normalization:
#            x /= (np.std(x, keepdims=True) + K.epsilon())
#
#        if self.featurewise_center:
#            if self.mean is not None:
#                x -= self.mean
#            else:
#                warnings.warn('This ImageDataGenerator specifies '
#                              '`featurewise_center`, but it hasn\'t '
#                              'been fit on any training data. Fit it '
#                              'first by calling `.fit(numpy_data)`.')
#        if self.featurewise_std_normalization:
#            if self.std is not None:
#                x /= (self.std + K.epsilon())
#            else:
#                warnings.warn('This ImageDataGenerator specifies '
#                              '`featurewise_std_normalization`, but it hasn\'t '
#                              'been fit on any training data. Fit it '
#                              'first by calling `.fit(numpy_data)`.')
#        if self.zca_whitening:
#            if self.principal_components is not None:
#                flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
#                whitex = np.dot(flatx, self.principal_components)
#                x = np.reshape(whitex, x.shape)
#            else:
#                warnings.warn('This ImageDataGenerator specifies '
#                              '`zca_whitening`, but it hasn\'t '
#                              'been fit on any training data. Fit it '
#                              'first by calling `.fit(numpy_data)`.')
#        return x

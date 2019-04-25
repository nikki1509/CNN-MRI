# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:31:53 2018

@author: jakpo_000
"""
from copy import deepcopy
from math import sqrt
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .import io
from .import preprocess

try:
    import cv2
except ImportError:
    print('opencv not found! Functions cv_load, superimpose are not available')
    cv2 = None


def plot_3dstack(volume, axis=1, slices=20, cmap='viridis'):
    '''
    Plotting slices of a 3D volume in one figure
    # Arguments
        volume:    3D volume
        axis:      Slice axis
        cmap:      Colourmap
    # Returns
        -
    '''
#    volume = preprocess.np_clip(volume)
#    volume = np.uint8(255*volume)
    #shape = volume[0].shape
    le = len(volume)
    x = 4
    y = (le//x)+1
    plt.figure(figsize=(11, y+(y)))


    limits = (np.min(volume), np.max(volume))


    for i in range(len(volume)):

        c = slices
        if axis == 0:
            image = volume[i][c, :, :]

        if axis == 1:
            image = volume[i][:, c, :]

        if axis == 2:
            image = volume[i][:, :, c]

        plt.subplot(y, x, i+1)
        plt.imshow(image, cmap, clim=limits)
        plt.title('%s von %s' % (i+1, le))
        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

    #cax = plt.axes([0.85, 0.25, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.show()

def heatmap(stack, model, plot=True):
    '''
    Utility function to convert a tensor into a valid image
    # Arguments
        x:    Tensor
    # Returns
        x:    Deprocessed image
    '''
    pic_list = deepcopy(stack)
    length = len(pic_list)
    size = int(sqrt(length))

#    if (size % 2) is not :
#        return
    heatvec = np.zeros(length)

    for i in range(length):
        pic = deepcopy(pic_list[i])
        pic = np.expand_dims(pic, axis=0)
        pic = np.expand_dims(pic, axis=4)
        x = model.predict(pic)
        heatvec[i] = x[0, 0]

    heatmap = np.reshape(heatvec, (size, size))

    if plot is True:
        plt.imshow(heatmap)
        plt.colorbar()
        plt.show
    return heatmap


def plot_filter(filters, savepath):
    '''
    Plotting 3D volume filters and save them
    # Arguments
        volume:    3D volume
        savepath:  Destination
    # Returns
        -
    '''
    limits = (np.min(filters), np.max(filters))
    for i in range(filters.shape[-1]):
        name = 'Filter kernel %i' %i
        plot_3d(filters[:,:,:,0,i], axis=2, title=name, limits=limits, width=4,
                fil=True, save=True, spath=savepath+str(i))


def plot_3d(volume, axis=1, cmap='seismic', title=None, limits=None, width=5,
            fil=False, save=False, spath=''):
    '''
    Plotting slices of a 3D volume in one figure
    # Arguments
        volume:    3D volume
        axis:      Slice axis
        cmap:      Colourmap
    # Returns
        -
    '''
#    volume = preprocess.np_clip(volume)
#    volume = np.uint8(255*volume)
    shape = volume.shape
    le = shape[axis]
    x = width
    y = (le//x)+1
    plt.figure(figsize=(11, y+(y)))

    if limits is None:
        limits = (np.min(volume), np.max(volume))

        if (np.min(volume)) >= 0:
            volume = preprocess.np_clip(volume)
            limits = (0, 1)
            cmap = 'hot'
            #print('red')

        if (np.min(volume)) < 0:
            ma = np.maximum(np.max(volume), np.abs(np.min(volume)))
            limits = (-ma, ma)
            print('seis')
            #cmap = 'seismic'


    for c in range(le):
        if axis == 0:
            image = volume[c, :, :]

        if axis == 1:
            image = volume[:, c, :]

        if axis == 2:
            image = volume[:, :, c]

        plt.subplot(y, x, c+1)
        plt.imshow(image, cmap, clim=limits)
        plt.title('%s / %s' % (c+1, le))
        #plt.subplots_adjust( top=4.0)
        plt.xticks([])
        plt.yticks([])


    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    param = 0.85
    if fil :
        param = 0.65
    cax = plt.axes([param, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    if title is not None:
        plt.suptitle(title, fontsize=16, x=(param-0.25), y=1.2)
        #plt.subplots_adjust( top=1.5)
    if save:
        plt.savefig(spath)
    plt.show()


def plot_stack(liste, cmap='viridis'):
    '''
    Plotting a list of 2D images
    # Arguments
        volume:    List of 2D images
        cmap:      Colourmap
    # Returns
        -
    '''
    length = len(liste)
    x = 5
    y = (length//x)+1
    plt.figure(figsize=(11, y+(y)))
    liste = normalize(liste)
    for c in range(length):
        plt.subplot(y, x, c+1)
        plt.imshow(liste[c], cmap, clim=(np.min(liste), np.max(liste)))
        plt.title('%s von %s' % (c+1, length))
        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.show()


def plot_5dtensor(tensor, axis=2, slices=None, cmap='gray', save=False,
                  path=''):
    '''
    Plotting specific slice of a 5D tensor in one figure
    # Arguments
        tensor:    5D tensor
        axis:      Slice axis
        slices:    Specific slice
        cmap:      Colourmap
        save:      Abillity to save figure
        path:      Save path
    # Returns
        -
    '''
    le = tensor.shape[-1]

    x = 4
    y = round((le//x))
    plt.figure(figsize=(11, y+(y)))

    limits = (np.min(tensor), np.max(tensor))

    if np.min(tensor) >= 0:
        tensor = preprocess.np_clip(tensor)
        limits = (0, 1)
        cmap = 'hot'
        #print('red')

    if np.min(tensor) < 0:
        ma = np.maximum(np.max(tensor), np.abs(np.min(tensor)))
        limits = (-ma, ma)
        #print('seis')
        cmap = 'seismic'

    if slices is None:
        shape = tensor.shape[-2]
        slices = int(shape/2)

    for c in range(le):
        if axis == 0:
            image = tensor[0, slices, :, :, c]

        if axis == 1:
            image = tensor[0, :, slices, :, c]

        if axis == 2:
            image = tensor[0, :, :, slices, c]

        plt.subplot(y, x, c+1)
        plt.imshow(image, cmap=cmap, clim=limits)
        plt.title('%s von %s' % (c+1, le))
        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
#
#    divider = make_axes_locatable(plt.Axes)
#    cax = divider.append_axes("right", size="10%", pad=0.05)
    plt.colorbar(cax=cax)
    if save:
        plt.savefig(path)
    plt.show()


def plot_tensor(tensor, weights=False, cmap='gray'):
    '''
    Plotting specific slice of a 4D tensor in one figure
    # Arguments
        tensor:    5D tensor
        weights:   Abillity to swap tensor axis for plotting weights
        cmap:      Colourmap
    # Returns
        -
    '''
    length = tensor.shape[-1]

    if weights:
        tens = tensor[:, :, 0, :]
    else:
        tens = tensor[0, :, :, :]

    x = 5
    y = (length//x)+1

    plt.figure(figsize=(11, y+(y)))
    for c in range(length):
        plt.subplot(y, x, c+1)
        plt.imshow(tens[:, :, c], cmap)
        plt.title('%s von %s' % (c+1, length))
        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.show()


def superimpose(heatmap, img_path, dest_path, name='Heatmap', axis=2, target_size=(128,128,128)):
    '''
    Function for interpolating an image or volume to another and weighing the
    first with the factor 0.4
    # Arguments
        heatmap:     2D or 3D heatmap
        img_path:    Path to original image/ volume
        dest_path:   Path for saving superimposed
        name:        Save name
        axis:        For 3D volumes specify save axis
        Target_size: Target interpolation size
    # Returns
        -
    '''
    img = io.ext_load(img_path, target_size)

    shape = img.shape[1:-1]


    if len(shape) == 2:
        heatmap = cv2.resize(heatmap, shape)
        heatmap = np.uint8(255*heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed = heatmap * 0.4 + img
        cv2.imwrite(dest_path + '%s.jpg' % name, superimposed)

    # plt.imshow(superimposed)
    else:
        heatmap = io.refit(heatmap, shape)

        #heatmap = np.uint8(255*heatmap)
        for i in range(shape[axis]):
            if axis == 0:
                image = heatmap[i,:,:]
                o_image = img[0,i,:,:,:]
            if axis == 1:
                image = heatmap[:,i,:]
                o_image = img[0,:,i,:,:]
            if axis == 2:
                image = heatmap[:,:,i]
                o_image = img[0,:,:,i,:]

            image = preprocess.np_clip(image)
            o_image = preprocess.np_clip(o_image)
            image = np.uint8(255*image)
            o_image = np.uint8(255*o_image)
            image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

            cube = np.dstack((o_image,o_image,o_image))
            superimposed = (image * 0.4) + cube

            cv2.imwrite(dest_path + '%s_%i.jpg' % (name, i), superimposed)


def g_inter(gradient, heatmap):
    shape = gradient.shape
    heatmap = io.refit(heatmap, shape)
    result = np.multiply(gradient, heatmap)
    return result


'''
Draft
3d plotting area
https://terbium.io/2017/12/matplotlib-3d/
https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
https://matplotlib.org/gallery/images_contours_and_fields/layer_images.html#sphx-glr-gallery-images-contours-and-fields-layer-images-py
https://matplotlib.org/gallery/animation/animation_demo.html#sphx-glr-gallery-animation-animation-demo-py
https://matplotlib.org/gallery/event_handling/image_slices_viewer.html#sphx-glr-gallery-event-handling-image-slices-viewer-py
https://matplotlib.org/gallery/specialty_plots/mri_demo.html#sphx-glr-gallery-specialty-plots-mri-demo-py
https://matplotlib.org/gallery/mplot3d/voxels_numpy_logo.html#sphx-glr-gallery-mplot3d-voxels-numpy-logo-py
https://matplotlib.org/gallery/mplot3d/bars3d.html#sphx-glr-gallery-mplot3d-bars3d-py
'''


def normalize(arr):
    arr_min = np.min(arr)
    tmp1 = (arr-arr_min)
    tmp2 = (np.max(arr)-arr_min)
    if tmp2 == 0:
        return arr
    else:
        erg = tmp1 / tmp2
        return erg

def show_histogram(values):
    n, bins, patches = plt.hist(values.reshape(-1), 50, normed=1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for c, p in zip(normalize(bin_centers), patches):
        plt.setp(p, 'facecolor', plt.cm.viridis(c))

    plt.show()

    # show_histogram(arr)


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def scroller():
    fig, ax = plt.subplots(1, 1)

    X = np.random.rand(20, 20, 40)
    tracker = IndexTracker(ax, X)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


def frame(data):
    fig, ax = plt.subplots()

    for i in range(len(data)):
        ax.cla()
        ax.imshow(data[i])
        ax.set_title("frame {}".format(i))
        # Note that using time.sleep does *not* work here!
        plt.pause(0.1)


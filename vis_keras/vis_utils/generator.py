#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 08:52:07 2018
The generator class is a fork from the official keras library. The loading functions
were modified to support 3D medical formats.
@author: jakob
"""
import os
import threading
from time import time
import numpy as np
from copy import deepcopy
from abc import abstractmethod
from . import io as  vio
from . import preprocess
from multiprocessing import Process, Pool
import multiprocessing

class Sequence(object):
    """Base object for fitting to a sequence of data, such as a dataset.

    Every `Sequence` must implements the `__getitem__` and the `__len__` methods.
    If you want to modify your dataset between epochs you may implement `on_epoch_end`.
    The method `__getitem__` should return a complete batch.

    # Notes

    `Sequence` are a safer way to do multiprocessing. This structure guarantees that the network will only train once
     on each sample per epoch which is not the case with generators.

    # Examples

    ```python
        from skimage.io import imread
        from skimage.transform import resize
        import numpy as np

        # Here, `x_set` is list of path to the images
        # and `y_set` are the associated classes.

        class CIFAR10Sequence(Sequence):

            def __init__(self, x_set, y_set, batch_size):
                self.x, self.y = x_set, y_set
                self.batch_size = batch_size

            def __len__(self):
                return np.ceil(len(self.x) / float(self.batch_size))

            def __getitem__(self, idx):
                batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

                return np.array([
                    resize(imread(file_name), (200, 200))
                       for file_name in batch_x]), np.array(batch_y)
    ```
    """

    @abstractmethod
    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.
        """
        raise NotImplementedError

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass

    def __iter__(self):
        """Create an infinite generator that iterate over the Sequence."""
        while True:
            for item in (self[i] for i in range(len(self))):
                yield item
    def test(self):
        pass

class Iterator(Sequence):
    """Abstract base class for image data iterators.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)
        self.seed = seed
        self.index_array = None

    def reset(self):
        self.batch_index = 0

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def _shuffle(self):
        arr = deepcopy(self.index_array)
       #  print('shuffle')
        self.index_array = np.random.permutation(arr)
        del(arr)

    def set_index_array(self, arr):
        self.index_array = arr

    def get_index_array(self):
        return self.index_array

    def on_epoch_end(self):
        self._shuffle()

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                if self.index_array is None:
                    # print('bug')
                    index_array = np.arange(n)
                    if shuffle:
                        index_array = np.random.permutation(n)
                else:
                    self._shuffle()
                    index_array = self.index_array

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            # print('bug2')
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batch(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class generator(Iterator):
    """Iterator capable of reading images from a directory on disk.
    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of sudirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, directory, batch_size=3, target_size=(20, 20), shuffle=True,
                 classes=True, seed=None, class_mode='binary'):

        self.directory = directory
        #self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.class_mode = class_mode

        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')

        # first, count the number of samples and classes
        self.samples = 0

        if os.path.isdir(self.directory):
            if len(self.target_size) is 2:
                self.formats = vio._FORMATS2D
                print('Init 2D generator')
            if len(self.target_size) is 3:
                self.formats = vio._FORMATS3D
                print('Init 3D generator')
    #        else:
    #            print('Unsupported target_size')

            self.path_str = []
            self.folder_list = []
            # dirslist = []
            path = vio.convert_path(self.directory)

            for root, dirs, files in os.walk(path):
                for file in files:
                    if vio.check_ext(file, self.formats):
                        tmp = os.path.join(root, file)
                        tmp = vio.convert_path(tmp)
                        self.path_str.append(tmp)
                        self.folder_list.append(vio.get_folder(tmp))
                # dirslist.append(dirs)

            self.dif_classes, self.classes = vio.to_binary(self.folder_list)
            self.num_classes = len(self.dif_classes)
            self.samples = len(self.path_str)
            print('Found %d images belonging to %d classes.' % (self.samples,
                                                                self.num_classes))

            super(generator, self).__init__(self.samples, batch_size, shuffle, seed)

    def names(self):
        return self.path_str

    def classnames(self):
        print(self.dif_classes)
        return self.dif_classes

    def len(self):
        return self.samples

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        return self._get_batch(index_array)


    def _get_batch(self, index_array):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((len(index_array),) + self.target_size, dtype=np.float32)
        batch_x = np.expand_dims(batch_x, axis=-1)
        # build batch of image data
        for i, j in enumerate(index_array):
            img = vio.ext_load(self.path_str[j], self.target_size)
            batch_x[i] = deepcopy(img)

        self.classes = np.asarray(self.classes, dtype=np.float32)
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_classes),dtype=np.float32)
            for i, label in enumerate(self.classes[index_array]):

                batch_y[i, int(label)] = 1.
        else:
            return batch_x

        return batch_x, batch_y


class yielder(Iterator):
    """Iterator capable of reading images from a directory on disk.
    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of sudirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, directory, batch_size=3, target_size=(20, 20), shuffle=True,
                 classes=True, seed=None, class_mode='binary', datapath=None,
                 save=False):

        # Load specific
        self.directory = directory
        self.target_size = tuple(target_size)
        self.samples = 0
        self.path_str = []
        self.folder_list = []
        self.data = None

        self.class_mode = class_mode
        self.classes = classes
        self.dif_classes = None
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')

        # For Iterator
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # Validation specific
        self.status = 'Yield all'
        self.test_fold = []
        self.train_fold = []
        self.folds = None
        self.num_folds = 0
        self.holdout_fold = []

        # Poss implementation for random data augmentation
        # self.image_data_generator = image_data_generator

        # first, count the number of samples and classes
        if os.path.isdir(self.directory):
            if len(self.target_size) is 2:
                self.formats = vio._FORMATS2D
                print('Init 2D yielder')
            if len(self.target_size) is 3:
                self.formats = vio._FORMATS3D
                print('Init 3D yielder')
    #        else:
    #            print('Unsupported target_size')


            # dirslist = []
            path = vio.convert_path(self.directory)


            for root, dirs, files in os.walk(path):
                for file in files:
                    if vio.check_ext(file, self.formats):

                        tmp = os.path.join(root, file)
                        tmp = vio.convert_path(tmp)
                        self.path_str.append(tmp)
                        self.folder_list.append(vio.get_folder(tmp))
                # dirslist.append(dirs)

            self.dif_classes, self.classes = vio.to_binary(self.folder_list)
            self.num_classes = len(self.dif_classes)
            self.samples = len(self.path_str)
            print('Found %d images belonging to %d classes.' % (self.samples,
                                                                self.num_classes))

            super(yielder, self).__init__(self.samples, batch_size, shuffle, seed)
            self._set_index_array()
            print('Init Loading all')
            self.data = np.zeros(((self.samples,)+ target_size + (1,)), np.float32)
#
            if datapath is not None:
                self.data = np.load(datapath)
            else:
                self.load()
                if save:
                    res = os.path.split(directory)
                    if res[1] is '':
                        res = os.path.split(res[0])
                    nam = res[1] + '.npy'
                    np.save(res[0], nam)

            print(self.dif_classes,self.classes)
#            kt = int(self.samples / 4)
#            print(kt)


#            start = time()

            #pool = multiprocessing.Pool(processes=4)
#            su = 0
#            aa = range(0,kt-1)
#            bb = range(kt,(2*kt)-1)
#            cc = range(2*kt,(3*kt)-1)
#            dd = range(3*kt,(4*kt)-1)
#            self.runInParallel(self.load(aa, su),self.load(bb, su),self.load(cc, su),self.load(dd, su))
#            pool.map(self.load, [aa,bb,cc,dd])
#
#            pool.close()
#            pool.join()
#            print(time()-start)
#            p = Process(target=self.load,)
#            p.start()
#            p.join()

#    def runInParallel(*fns):
#        proc = []
#        for fn in fns:
#            p = Process(target=fn)
#            p.start()
#            proc.append(p)
#        for p in proc:
#            p.join()

    def load(self):
        for i in range(self.samples):
            img = vio.ext_load(self.path_str[i], self.target_size)
            self.data[i] = deepcopy(img)
            del(img)
            prog = i/self.samples
            print('\r[%i/%i] %.2f %%' % (i+1, self.samples, prog*100),
                  end='')
    def save(self, save_path):
        print('Start saveing')

        np.save(save_path, self.data)
        print('Finished saveing')

    def names(self):
        return self.path_str

    def y_reset(self):
        n = self.samples
        super(yielder, self).__init__( n, self.batch_size, self.shuffle, self.seed)
        self._set_index_array()
        self.train_fold = []
        self.test_fold = []
        self.status = 'all'

    def k_fold(self, k):
        self.y_reset()
        self.status = 'k_fold'
        self.num_folds = k
        if k is not 0:
            def partition(lst, n):
                div = len(lst) / float(n)
                return [lst[int(round(div*i)): int(round(div*(i+1)))] for i in range(n)]

            self.folds = partition(self.index_array, k)

            for i in range(k-1):
                self.train_fold = np.append(self.train_fold, self.folds[i]).astype(int)
                print(self.train_fold)

            self.test_fold = self.folds[-1]
            self.active_test_fold = k

            super(yielder, self).__init__(len(self.train_fold), self.batch_size, self.shuffle, self.seed)
            self.set_index_array(self.train_fold.astype(int, copy=False))
            self._shuffle()
            print('Yielder changed to %i fold yielder. \nActive testfold: %i'
                  % (k, k))
            print('Fold size: ~%i' % len(self.test_fold))

    def set_fold(self, num):
        if num not in range(self.num_folds):
            print('Not in fold rage. Max testfold is %i - 1' % self.num_folds)
            return
        if self.status is 'holdout':
            print('Holdout yielder cant set k folds')

        if self.status is 'all':
            print('Apply obj.k_fold(x) first')

        self.train_fold = []
        self.test_fold = []
        self.active_test_fold = num
        for i in range(self.num_folds):
            if i is num:
                self.test_fold = self.folds[i]

            else:
                self.train_fold = np.append(self.train_fold, self.folds[i]).astype(int)

        self.active_test_fold = num
        super(yielder, self).__init__(len(self.train_fold), self.batch_size, self.shuffle, self.seed)
        self.set_index_array(self.train_fold.astype(int, copy=False))
        num2 = num + 1
        print('Active testfold: %i/%i' % (num2,self.num_folds))

    def activate_test(self):
        if self.num_folds is 0:
            print('no fold set')
            return
        super(yielder, self).__init__(len(self.test_fold), self.batch_size, self.shuffle, self.seed)
        self.set_index_array(self.test_fold)
        self._shuffle()

    def train_gen(self):
        self.activate_train()
        pa = partial((self.get_index_array()), self.data, self.batch_size,
                     self.shuffle, self.seed, self.target_size, self.classes,
                     self.class_mode, self.num_classes)
        return pa

    def test_gen(self):
        self.activate_test()
        pa = partial((self.get_index_array()), self.data, self.batch_size,
                     self.shuffle, self.seed, self.target_size, self.classes,
                     self.class_mode, self.num_classes)
        return pa


    def get_test_prop(self):
        tmp = []
        part = []
        if self.num_folds is 0:
            print('no folds set')
            return

        for i in self.test_fold:
            tmp.append(self.classes[i])

        for i in range(self.num_classes):
            count = tmp.count(i)
            part.append(count)
        if self.num_classes <= 2:
            print('%i %s zu %i %s' % (part[0], self.dif_classes[0], part[1],
                                      self.dif_classes[1]))
        else:
            print(self.dif_classes)
            print(part)

    def activate_train(self):
        if self.num_folds is 0:
            print('no fold set')
            return
        #self.n = len(self.train_fold)
        super(yielder, self).__init__( len(self.train_fold), self.batch_size, self.shuffle, self.seed)
        self.set_index_array(self.train_fold)
        self._shuffle()

    def reset_batch_size(self, batch_size):
        old = self.batch_size
        self.batch_size = batch_size
        super(yielder, self).__init__(self.samples, batch_size, self.shuffle,
                                      self.seed)
        self._set_index_array()
        print('Reseted batch size from %i to %i' % (old, batch_size))

    def holdout(self, partial):
        self.y_reset()
        self.status = 'holdout'
        self.num_folds = 2
        div = int(round(float(self.samples)*partial))
        idx_arr = self.get_index_array()
        tmp = np.split(idx_arr, [div])
        self.test_fold = tmp[0]
        self.train_fold = tmp[1]
        super(yielder, self).__init__(len(tmp[1]), self.batch_size, self.shuffle, self.seed)
        self.set_index_array(self.train_fold)
        self._shuffle()

    def classnames(self):
        print(self.dif_classes)
        return self.dif_classes

    def len(self):

        return self.samples

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        return self._get_batch(index_array)

    def _get_batch(self, index_array):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((len(index_array),) + self.target_size,
                           dtype=np.float32)
        batch_x = np.expand_dims(batch_x, axis=-1)
        # build batch of image data
        for i, j in enumerate(index_array):
            batch_x[i] = self.data[j]


        self.classes = np.asarray(self.classes, dtype=np.float32)
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_classes),
                               dtype=np.float32)
            for i, label in enumerate(self.classes[index_array]):

                batch_y[i, int(label)] = 1.
        else:
            return batch_x

        return batch_x, batch_y

class partial(Iterator):

    def __init__(self, index_array, data, batch_size ,shuffle, seed,
                 target_size, classes, class_mode, num_classes ):

        n = len(index_array)
        self.samples = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.target_size = target_size
        self.classes = classes
        self.class_mode = class_mode
        self.num_classes = num_classes
        self.data = data
        super(partial, self).__init__(n,batch_size,shuffle,seed)
        self.set_index_array(index_array)
        self._shuffle()

    def len(self):

        return self.samples

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        return self._get_batch(index_array)

    def _get_batch(self, index_array):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((len(index_array),) + self.target_size,
                           dtype=np.float32)
        batch_x = np.expand_dims(batch_x, axis=-1)
        # build batch of image data
        for i, j in enumerate(index_array):
            batch_x[i] = self.data[j]
        self.classes = np.asarray(self.classes, dtype=np.float32)
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_classes),
                               dtype=np.float32)
            for i, label in enumerate(self.classes[index_array]):

                batch_y[i, int(label)] = 1.
        else:
            return batch_x

        return batch_x, batch_y

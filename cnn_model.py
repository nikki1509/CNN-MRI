#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 15:00:51 2018
Rudimentary, not optimized keras CNN Model for classifying MRI Brain images into
normal and abnormal
@author: jakob
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import keras.backend.tensorflow_backend as K
import model_utils as mu

def main():
    """
    main function
    """
    with K.tf.device('/gpu:0'):

        shape = 126#128
        kernel = 9#9
        filters = 9
        batch = 16#64
        epoch = 4

        model = Sequential()
        model.add(Conv2D(filters, (kernel, kernel), input_shape=(shape, shape, 1),
                         padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.02))
        model.add(Conv2D(4*filters, (kernel, kernel), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(8*filters, (kernel, kernel), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.02))
        model.add(Flatten())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        train_datagen = ImageDataGenerator(rescale=1./255)
    #                                       ,shear_range = 0.2,
    #                                       zoom_range = 0.2,
    #                                       horizontal_flip = True)

        test_datagen = ImageDataGenerator(rescale=1./255)

        training_set = train_datagen.flow_from_directory('OASIS/Test/train',
                                                         target_size=(shape, shape),
                                                         color_mode='grayscale',
                                                         batch_size=batch,
                                                         class_mode='binary')

        test_set = test_datagen.flow_from_directory('OASIS/Test/test',
                                                    target_size=(shape, shape),
                                                    color_mode='grayscale',
                                                    batch_size=batch,
                                                    class_mode='binary')

        tb_callback = mu.CustomCallback.tensorCall()
        acti_callback = mu.CustomCallback.activationHistory(model.input_shape[-2])
        history = model.fit_generator(training_set,
                                      steps_per_epoch=193,
                                      epochs=epoch,
                                      validation_data=test_set,
                                      validation_steps=40,
                                      callbacks=[tb_callback, acti_callback])

        mu.plot_history(history)

        stack = acti_callback.get_stack()
        mu.make_gif(stack, layer_to_vis=1)

main()

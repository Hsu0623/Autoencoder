# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:07:44 2020
    It' s a uncompleted code. I havn' t join the bottleneck layer that
is a vector between decoder and encoder.  20/10/9.
@author: User
"""
from __future__ import print_function
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, UpSampling2D, Dropout, Reshape
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt


class AUTOENCODER(object):
    """ Generative Adversarial Network class """
    def __init__(self, width=28, height=28, channels=1):

        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (self.width, self.height, self.channels)

        self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)

        self.E = self.__encoder()
        self.E.compile(loss='mse', optimizer=self.optimizer)

        self.D = self.__decoder()
        self.D.compile(loss='mse', optimizer=self.optimizer, metrics=['accuracy'])

        self.autoencoder = self.__autoencoder()
        self.autoencoder.compile(loss='mse', optimizer=self.optimizer)


    def __encoder(self):
        
        model = Sequential()
        model.add(Conv2D(filters=32,
                         kernel_size=(5,5),
                         padding='same',
                         input_shape=self.shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))       
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Conv2D(filters=64,
                     kernel_size=(5,5),
                     padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))       
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Conv2D(filters=128,
                     kernel_size=(5,5),
                     padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))   
        return model
        #model.add(MaxPooling2D(pool_size=(2,2)))
        """
        model.add(Flatten())
        model.add(Dense(10, activation='relu'))
        """
    
     
    def __decoder(self):
        
        model = Sequential()
        '''
        model.add(Reshape)
        '''
        model.add(Conv2D(filters=128,
                     kernel_size=(5,5),
                     padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))       
        model.add(UpSampling2D(size=(2,2)))
        
        model.add(Conv2D(filters=64,
                     kernel_size=(5,5),
                     padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))       
        model.add(UpSampling2D(size=(2,2)))
        
        model.add(Conv2D(filters=32,
                         kernel_size=(5,5),
                         padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))       
        #model.add(UpSampling2D(size=(2,2)))
        
        model.add(Conv2D(filters=1,
                         kernel_size=(5,5),
                         padding='same'))
        
        return model

    
    def __autoencoder(self):

        model = Sequential()
        
        model.add(self.E)
        model.add(self.D)
        
        return model
        
  
    def train(self, X_train, epochs=30, batch = 64, save_interval = 100):

        self.autoencoder.fit(X_train, X_train, epochs = epochs, batch_size=batch)      


    def plot_images(self, X, save2file=False, samples=16):
        # Plot images 
       n = 10
       decoded_imgs = self.autoencoder.predict(X[3990:4000])
       plt.figure(figsize=(20, 4))
       for i in range(n):  
           # display original
           ax = plt.subplot(2, n, i+1)
           image = X[3990+i]
           plt.imshow(image.reshape(28, 28, 1)) #gray image
           ax.axis('off')
           ax.get_xaxis().set_visible(False)
           ax.get_yaxis().set_visible(False)

           # display reconstruction
           ax = plt.subplot(2, n, n + i + 1)
           plt.imshow(decoded_imgs[i].reshape(28, 28, 1))#gray image
           ax.get_xaxis().set_visible(False)
           ax.get_yaxis().set_visible(False)
           plt.show()







if __name__ == '__main__':
    #read data
    (X_train,Y_train),(X_test,Y_test)= mnist.load_data()
    
    '''
    train_filter = np.where((Y_train == 0))
    test_filter = np.where((Y_test == 7))
    
    X_train, Y_train = X_train[train_filter], Y_train[train_filter]
    X_test, Y_test = X_test[test_filter], Y_test[test_filter]
    '''
    
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],X_train.shape[2],1)).astype('float32') / 255.0
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],X_test.shape[2],1)).astype('float32') / 255.0
    
    autoencoder = AUTOENCODER()
    autoencoder.train(X_test)
    autoencoder.plot_images(X_test)
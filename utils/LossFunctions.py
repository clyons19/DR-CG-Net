# Loss Functions
# * Created by: Carter Lyons
# * Last Edited: **7/15/2022**

# Contains:
# * Loss Functions for Image estimation Neural Networks
 
# Package Requirements (in addition to those required in Initialize.py):
# * Tensorflow 2.5.0, numpy

import tensorflow as tf
import numpy as np

# Classes of Loss Functions

class MSE_loss: # Mean Squared Error
    def __init__(self, power = 1, normalize = False):
        self.power = power    # power applied to each norm (e.g. power = 1 gives mean square error, power = 1/2 gives root MSE)
        self.normalize = normalize # normalizing each error (True or False)
        
    def call(self, targets, estimates):
        norm = tf.cast(tf.linalg.norm(targets, axis = 1)**(2*self.normalize), float)
        error = tf.keras.losses.mean_squared_error(targets, estimates)
        return tf.math.reduce_mean((error/norm)**self.power)
    
class MAE_loss: # Mean Absolute Error
    def __init__(self, power = 1, normalize = False):
        self.power = power    # power applied to each norm (e.g. power = 1 gives mean absolute error, power = 1/2 gives root MAE)
        self.normalize = normalize # normalizing each error by l2 norm of each targets (True or False)

    def call(self, targets, estimates):
        norm = tf.cast(tf.linalg.norm(targets, axis = 1, ord = 1)**self.normalize, float)
        error = tf.keras.losses.mean_absolute_error(targets, estimates)
        return tf.math.reduce_mean((error/norm)**self.power)

class SSIM_loss: # Structural Similarity Index
    def __init__(self, Phi, img_size, scale):
        self.Phi = tf.constant(Phi, dtype = float)
        self.size, self.s = img_size, scale
        
    def call(self, targets, estimates):
        num_samples, size = tf.shape(targets)
        I_est = tf.reshape(estimates@tf.transpose(self.Phi), (num_samples, self.size, self.size, 1))
        I_act = tf.reshape(targets@tf.transpose(self.Phi), (num_samples, self.size, self.size, 1))
        ssim = tf.cast(tf.image.ssim(I_est, I_act, max_val = self.s) , float)
        return tf.math.reduce_mean(1-ssim)
    
class MS_SSIM_loss: # Multiscale Structural Similarity Index
# Note (img_size/2^4 >= filter_size) that or change number of power_factors
# Documentation for MS-SSIM: https://www.tensorflow.org/api_docs/python/tf/image/ssim_multiscale    
    def __init__(self, Phi, img_size, scale):
        self.Phi = tf.constant(Phi, dtype = float)
        self.size, self.s = img_size, scale
        
    def call(self, targets, estimates):
        num_samples, size = tf.shape(targets)
        I_est = tf.reshape(estimates@tf.transpose(self.Phi), (num_samples, self.size, self.size, 1))
        I_act = tf.reshape(targets@tf.transpose(self.Phi), (num_samples, self.size, self.size, 1))
        ssim = tf.cast(tf.image.ssim_multiscale(I_est, I_act, max_val = self.s, filter_size = np.floor(self.size/2**4)) , float)
        return tf.math.reduce_mean(1-ssim)

class weighted_SSIM_loss: # Weighted loss between SSIM an a second loss
    def __init__(self, Phi, img_size, scale, loss, loss_params, weight):
        self.SSIM_loss = SSIM_loss(Phi, img_size, scale)
        self.w = weight
        if loss == 'MSE':
            self.loss = MSE_loss(*loss_params)
        elif loss == 'MAE':
            self.loss = MAE_loss(*loss_params)
            
    def call(self, targets, estimates):
        return self.w*self.SSIM_loss.call(targets, estimates)+(1-self.w)*self.loss.call(targets, estimates)
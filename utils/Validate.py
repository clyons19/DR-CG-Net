# Validate.ipynb 
# * Created by: Carter Lyons
# * Last Edited: **9/20/2023**
# 
# Contains:
# * Class for calculating quality metrics (SSIM, PSNR, MSE, MAE) between input target and estimated wavelet coeficients
# 
# Package Requirements:
# * Tensorflow 2.5.0, numpy

import tensorflow as tf
import numpy as np
import time

class Ssim_PsnrMetrics: # Structural Similarity Index and Peak Signal to Noise Ratio
    def __init__(self, Phi, img_size, scale):
        self.Phi = tf.constant(Phi, dtype = float) if Phi is not None else Phi
        self.size, self.s = img_size, scale
        
    def call(self, targets, estimates):
        num_samples, size = tf.shape(targets)
        if self.Phi is None:
            I_est = tf.reshape(estimates, (num_samples, self.size, self.size, 1))
            I_act = tf.reshape(targets, (num_samples, self.size, self.size, 1))
        else:
            I_est = tf.reshape(estimates@tf.transpose(self.Phi), (num_samples, self.size, self.size, 1))
            I_act = tf.reshape(targets@tf.transpose(self.Phi), (num_samples, self.size, self.size, 1))
        ssim = np.array(tf.image.ssim(I_est, I_act, max_val = self.s))
        psnr = np.array(tf.image.psnr(I_est, I_act, max_val = self.s))
        return [ssim, psnr]
    
class NormMetrics: # Mean Square Error and Mean Absolute Error
    def __init__(self, power = 1, normalize = False):
        self.power = power
        self.normalize = normalize
      
    def call(self, targets, estimates):
        norm = tf.cast(tf.linalg.norm(targets, axis = 1)**(2*self.normalize), float)
        error = tf.cast(tf.keras.metrics.mean_squared_error(targets, estimates), float)
        mse = np.array((error/norm)**self.power)
        norm = tf.cast(tf.linalg.norm(targets, axis = 1, ord = 1)**self.normalize, float)
        error = tf.cast(tf.keras.metrics.mean_absolute_error(targets, estimates), float)
        mae = np.array((error/norm)**self.power)
        return [mse, mae]

class AverageMetrics: # Compute Metrics for supplied targets and estimates
    def __init__(self, Phi, img_size, scale, power = 1, normalize = False):
        self.spMetric = Ssim_PsnrMetrics(Phi, img_size, scale)
        self.normMetric = NormMetrics(power, normalize)
        self.data = {'SSIM':[], 'PSNR':[], 'MSE':[], 'MAE':[]}
            
    def call(self, targets, estimates):
        new = self.spMetric.call(targets, estimates)
        self.data['SSIM'] = self.data['SSIM'] + list(new[0])
        self.data['PSNR'] = self.data['PSNR'] + list(new[1])
        new = self.normMetric.call(targets, estimates)
        self.data['MSE'] = self.data['MSE'] + list(new[0])
        self.data['MAE'] = self.data['MAE'] + list(new[1])
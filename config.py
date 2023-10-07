import os
import sys
import platform
from datetime import datetime

import tensorflow as tf

from utils import LossFunctions
# ----------------------------------------------------------------------------------------------------------------------
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".
# taken from: https://github.com/tkarras/progressive_growing_of_gans/blob/master/config.py#L8
# ----------------------------------------------------------------------------------------------------------------------


class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]


# ----------------------------------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------------------------------

##### General Configuration ######
general_config = EasyDict()             # generic configurations/options for output, checkpoints, etc.
general_config.max_increases       = 6       # int: maximum number of epoch checks without a decrease in test loss before terminating training
general_config.quiet               = False   # bool: de-/activates printing of training loss results
general_config.best_model_test     = True    # bool: de-/activates testing on best trained model
general_config.TrainDataBasePath   = os.path.join('Train Datasets')   # os.PathLike:  base path to train datasets
general_config.TestDataBasePath   = os.path.join('Test Datasets')   # os.PathLike:  base path to train datasets
# general_config.SaveBasePath        = os.path.join('')     # os.PathLike:  base path to save results at
general_config.SaveBasePath   = r'C:\Users\LyonsC\Desktop\Testing'
##### ##################### ######

##### Model Configuration ######
model_config = EasyDict()               # configuration of model building
model_config.method           = 'ISTA DR-CG-Net'   # string: model to use in training. Options: ['ISTA DR-CG-Net', 'PGD DR-CG-Net', 'ISTA DR-CG-Net GDu', 'PGD DR-CG-Net GDu']
model_config.K                = 3                  # int:  number of unrolled iterations
model_config.J                = 4                  # int:  number of scale variable updates per iteration
model_config.GDsteps          = 100                 # int:  number of gradient descent steps used to estimate the Tikhonov solution for u.
model_config.D                = 8                  # int:  number of CNN subnetwork layers in each scale variable update
model_config.lamb_init        = {60: 0.1, 40: 0.1, 30: 0.1, 20: 0.1} # dict of floats: initial value of u covariance diagonal. Dict key values correspond to SNR of measurements.
# model_config.lamb_init        = {60: 10, 40: 10, 30: 10, 20: 10} # Gaussian Measurements
model_config.cov_style        = 'const' # str: u covariance matrix structure. Options: 'const', 'diag', 'tridiag'
model_config.scale_cov        = (False, 1.0)  # tuple (bool, float): de-/activates scaling the u covariance matrix, initial scaling constant
model_config.u0_style         = 'Tik'  # str: How to calculate the initial u estimate. Options: 'Tik', 'Ones'
model_config.normalize_A_in_z0 = True  # bool: de-/activates normalizing the measurement matrix in the initial z estimate
model_config.b                = 10.0   # float: maximum entry value in initial z estimate (10.0 Radon msmnts, 0.5 Gaussian msmnts)
model_config.num_filters      = 32     # int:  number of filter channels in each convolution
model_config.kernel_size      = 3      # int:  size of each convolutional kernel
model_config.step_size        = 1.0    # float:  step size in gradient descent step for z estimation
model_config.normalize_grad   = True   # bool: de-/activates normalizing the gradient for each z gradient descent step such that the gradient lies in the Eucledian ball of radius B
model_config.B                = 1.0    # float: radius of Euclidean ball to normalize the gradient to if model_config.normalize_grad = True   
model_config.epsilon          = 1e-4   # float: stabilization Parameter for u covariance calculations
model_config.use_bias         = False  # bool: de-/activates bias in convolutional subnetworks
model_config.project_z_est    = True   # bool: de-/activates projecting z0 onto [0,b]^n and projecting every z estimate onto [0, infty)^n.
model_config.use_refinement   = True   # bool: de-/activates use of final refienment block
model_config.shared_u_weights = True   # bool: de-/activates weight sharing in u update layers
##### ################### ######

##### Training Configuration ######
train_config = EasyDict()               # configuration of train parameters
train_config.batch_size               = 20          # int: training batch size
train_config.test_batch_size          = 50          # int: testing batch size
train_config.loss                     = LossFunctions.MAE_loss()      # class:  base loss function to train network with
train_config.epoch_check              = 25          # int: number of epochs to run model on test data
train_config.use_sparsity             = False       # bool: de-/activates use of a sparsity transformation (i.e. whether network will output image or sparsity coefficients)
train_config.sparsity_type            = 'dct'       # string: options ['biowavelet', 'dct']. Type of sparsity transformation to be used if use_sparsity = True
train_config.rearrange_inputs         = False       # bool:  de-/activates transposing measurements prior to vectorizing (True for iRadonMAP as then FBP layers actually correspond to the FBP)
train_config.random_seed              = 19          # int:  random seed for numpy and tensorflow
##### ###################### ######

##### Dataset Configuration ######
data_config = EasyDict()                # configuration of data set pipeline
data_config.true_data_type = 'coefs'          # str: Options 'imgs', 'coefs'. Whether the train and test data sets contain the actual images or wavelet coefficients.
data_config.num_validation_points = 200       #  int: number of data points in the validation dataset
data_config.num_test_points = 200             #  int: number of data points in the test dataset
data_config.scale = 1                         #  float:  value which the network input data is scaled (multiplied) by
##### ##################### ######

##### Optimizer Configuration ######
optimizer_config = EasyDict()           # configuration of adam optimizers for both generator and discriminator
optimizer_config.starting_lr          = 1e-4        # float: starting learning rate
optimizer_config.use_lr_scheduler     = False       # bool: de-/activates using a learning rate scheduler
optimizer_config.ending_lr            = 1e-4        # float: end learning rate (when using a learning rate scheduler)
optimizer_config.lr_schedule          = tf.keras.optimizers.schedules.PolynomialDecay       # tf.keras.optimizers.schedules:  method for scheduling the learning rate
optimizer_config.optimizer            = tf.keras.optimizers.Adam        # tf.keras.optimizers:  optimization method
##### ####################### ######

conf = EasyDict()                       # configuration summary
conf.general                            = general_config
conf.model                              = model_config
conf.train                              = train_config
conf.data                               = data_config
conf.optimizer                          = optimizer_config

# ----------------------------------------------------------------------------------------------------------------------
# These configurations are automatically set
# ----------------------------------------------------------------------------------------------------------------------

if model_config.method == 'ISTA DR-CG-Net' or model_config.method == 'ISTA DR-CG-Net GDu':
    general_config.SaveName = 'ISTADRCGNet'
    general_config.FolderStart = 'ISTA'
    model_config.CNN_option = 'ISTAConv'
elif model_config.method == 'PGD DR-CG-Net' or model_config.method == 'PGD DR-CG-Net GDu':
    general_config.SaveName = 'PGDDRCGNet'
    general_config.FolderStart = 'PGD'
    model_config.CNN_option = 'GradConv'

if 'GDu' in model_config.method:
    folder_label = ', GDu'
else:
    folder_label = ''
if model_config.use_refinement:
    folder_label += ', denoise'
else:
    folder_label += ''
if not train_config.use_sparsity:
    train_config.sparsity_type = 'No'
if model_config.scale_cov[0]:
    general_config.SaveFolder = '{}, {} sparisty, K = {} J = {} D = {}, {} scaled cov, {} u0'.format(general_config.FolderStart, train_config.sparsity_type, model_config.K, model_config.J, model_config.D, model_config.cov_style, model_config.u0_style) + folder_label
else:  
    general_config.SaveFolder = '{}, {} sparisty, K = {} J = {} D = {}, {} cov, {} u0'.format(general_config.FolderStart, train_config.sparsity_type, model_config.K, model_config.J, model_config.D, model_config.cov_style, model_config.u0_style) + folder_label
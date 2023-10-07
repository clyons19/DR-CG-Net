# We edit the implementation of DR-CG-Net for the 128x128 image reconstruction experiements to avoid memory overflow errors. (Specifically, a memory error occurs when attempting to construct the covariance matrix Pu which is of size 128^2 x 128^2. I am unsure how much more memory will be required to perform the linear solve but I assume that will be a lot as well.) Actually the memory error was resulting from trying to construct tf.linalg.diag(z) we could replace this by a sparse tensor representation
## The changes: Change the Tikhonov implementation into a series of unrolled GD steps on u. Added backtracking linesearch implementation.

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.activations as Act
import numpy as np
import time # used here to checking computation time

from config import conf
seed = conf.train.random_seed

class InitialLayer(L.Layer):
    def __init__(self, A_norm, normalize = False, name = None): # Inputs: Measurement matrix A (tensor with dtype 'float32'). Whether to normalize the measurement matrix (boolean). Name of network (string), note name is required for saving network variables. 
        super(InitialLayer, self).__init__(name = name)
        self.A_norm = A_norm**normalize
        
    def call(self, sparse_A, inputs):
        return tf.sparse.sparse_dense_matmul(inputs, sparse_A)/self.A_norm

class Backtrack(L.Layer):
    def __init__(self, alpha, beta, cov_style = 'const', name = None):
        super(Backtrack, self).__init__(name = name)
        self.alpha = tf.constant(alpha, dtype = 'float32')
        self.beta = tf.constant(beta, dtype = 'float32')
        
        if cov_style == 'const' or cov_style == 'diag':
            self.quadratic_form = self.quadratic_form_const
            self.quadratic_form_grad = self.quadratic_form_grad_const
        elif cov_style == 'tridiag':
            self.quadratic_form = self.quadratic_form_tridiag
            self.quadratic_form_grad = self.quadratic_form_grad_tridiag
            
    def quadratic_form_const(self, Pu, u):
        return tf.reduce_sum(u*Pu*u, axis = -1, keepdims = True)
    
    def quadratic_form_tridiag(self, Pu, u):
        return tf.reduce_sum(u*tf.transpose(tf.linalg.tridiagonal_matmul(Pu, tf.transpose(u), diagonals_format = 'sequence')), axis = -1, keepdims = True)
    
    def quadratic_form_grad_const(self, Pu, u):
        return Pu*u
    
    def quadratic_form_grad_tridiag(self, Pu, u):
        return tf.transpose(tf.linalg.tridiagonal_matmul(Pu, tf.transpose(u), diagonals_format = 'sequence'))
    
    def f(self, sparse_At, Pu, z, u, y):
        return 0.5*tf.linalg.norm(y - tf.sparse.sparse_dense_matmul(z*u, sparse_At), ord = 2, axis = -1, keepdims = True)**2 + 0.5*self.quadratic_form(Pu, u)
    
    def grad_f(self, sparse_At, Pu, z, u, y):
        return z*tf.sparse.sparse_dense_matmul(tf.sparse.sparse_dense_matmul(z*u, sparse_At) - y, tf.sparse.transpose(sparse_At)) + self.quadratic_form_grad(Pu, u)
    
    def call(self, sparse_At, Pu, z, u, y):
        r = tf.zeros_like(z)[:,0:1]
        grad = self.grad_f(sparse_At, Pu, z, u, y)
        grad_norm = tf.linalg.norm(grad, ord = 2, axis = -1, keepdims = True)**2
        orig_loss = self.f(sparse_At, Pu, z, u, y)
        calc_eta = lambda r: self.beta**r
        check = lambda eta: tf.less(orig_loss-self.alpha*eta*grad_norm, self.f(sparse_At, Pu, z, u-eta*grad, y))
        cond = lambda r: tf.experimental.numpy.any(check(calc_eta(r)))
        body = lambda r: (tf.add(r, tf.cast(check(calc_eta(r)), dtype = 'float32')), )
        return calc_eta(tf.while_loop(cond, body, [r])[0])        
    
class CalcTikhonov(L.Layer):
    def __init__(self, steps, cov_style = 'const', scale = (False, 1.0), name = None):
        super(CalcTikhonov, self).__init__(name = name)
        self.steps = steps
#         self.lamb = self.add_weight(name = '{}_lamb'.format(name), shape = (), initializer =  tf.keras.initializers.Constant(scale[1]), trainable = scale[0]) # Previously we were not including this parameter so loading in the weights from an older training will throw and error. Instead use following code
        if scale[0]:
            self.lamb = self.add_weight(name = '{}_Pu_scale'.format(name), shape = (), initializer =  tf.keras.initializers.Constant(scale[1]), trainable = scale[0])
        else:
            self.lamb = scale[1]
            
        if cov_style == 'const' or cov_style == 'diag':
            self.quadratic_form_grad = self.quadratic_form_grad_const
        elif cov_style == 'tridiag':
            self.quadratic_form_grad = self.quadratic_form_grad_tridiag
            
        self.backtrack = Backtrack(0.5, 0.5, cov_style, name = 'backtrack')
            
    def quadratic_form_grad_const(self, Pu, u):
        return Pu*u
    
    def quadratic_form_grad_tridiag(self, Pu, u):
        return tf.transpose(tf.linalg.tridiagonal_matmul(Pu, tf.transpose(u), diagonals_format = 'sequence'))
            
    def call(self, sparse_A, Pu, z, u, y):
#### Basic Gradient Descent ####        
#         start = time.time()
#         print('Starting u Calc')
#         for GDstep in range(self.steps):
#             u = self.lamb*u - etas*(z*tf.sparse.sparse_dense_matmul(tf.sparse.sparse_dense_matmul(z*u, tf.sparse.transpose(sparse_A)) - y, sparse_A) + self.quadratic_form_grad(Pu, u))
#         print('Finished u Calc {:5f}'.format(time.time() - start))
#         return u
#### ###################### ####  

#### Gradient Descent with Nesterov Momentum ####
        start = time.time()
#         print('Starting u Calc')
        w = u
        v1 = u
        sparse_At = tf.sparse.transpose(sparse_A)
        for GDstep in range(self.steps):
#             print(GDstep)
            etas = tf.nest.map_structure(tf.stop_gradient, self.backtrack(sparse_At, Pu, z, w, y))
#             print(etas)
            v0 = v1
#             v1 = self.lamb*w - etas*(z*tf.sparse.sparse_dense_matmul(tf.sparse.sparse_dense_matmul(z*w, tf.sparse.transpose(sparse_A)) - y, sparse_A) + self.quadratic_form_grad(Pu, w))
            v1 = self.lamb*w - etas*(z*tf.sparse.sparse_dense_matmul(tf.sparse.sparse_dense_matmul(z*w, sparse_At) - y, sparse_A) + self.quadratic_form_grad(Pu, w))
            beta = 1.0 - 3.0/(6.0 + GDstep)
            w = v1 + beta*(v1 - v0)
#         print('Finished u Calc {:5f}'.format(time.time() - start))
        return w
#### ####################################### ####

#### Gradient Descent with Nesterov Momentum (different momentum ratio calc) ####
#         start = time.time()
#         print('Starting u Calc')
#         w = u
#         v1 = u
#         for GDstep in range(self.steps):
#             v0 = v1
#             v1 = self.lamb*w - etas*(z*tf.sparse.sparse_dense_matmul(tf.sparse.sparse_dense_matmul(z*w, tf.sparse.transpose(sparse_A)) - y, sparse_A) + self.quadratic_form_grad(Pu, w))
#             w = v1 + (GDstep/(3.0+GDstep))*(v1 - v0)
#         print('Finished u Calc {:5f}'.format(time.time() - start))
#         return w
#### ####################################################################### ####

#### Gradient Descent with Nesterov Momentum (different momentum ratio calc) ####
#         start = time.time()
#         print('Starting u Calc')
#         w = u
#         v1 = u
#         beta0 = 0.0
#         beta1 = 1.0
#         for GDstep in range(self.steps):
#             v0 = v1
#             v1 = self.lamb*w - etas*(z*tf.sparse.sparse_dense_matmul(tf.sparse.sparse_dense_matmul(z*w, tf.sparse.transpose(sparse_A)) - y, sparse_A) + self.quadratic_form_grad(Pu, w))
#             gamma = (1-beta0)/beta1
#             w = v1 + gamma*(v0 - v1)
#             beta = (1.0+tf.math.sqrt(1.0+4.0*beta1**2))/2
#             beta0 = beta1
#             beta1 = beta
#         print('Finished u Calc {:5f}'.format(time.time() - start))
#         return w
#### ####################################################################### ####

class ConstCov(L.Layer):
    def __init__(self, n, lamb_init, eps = 1e-2, name = None):
        super(ConstCov, self).__init__(name = name)
        self.n = n
        self.lamb = self.add_weight(name = '{}_lamb'.format(self.name), shape = (), initializer =  lamb_init, trainable = True)      
        self.eps = eps
        
    def call(self):
        return tf.math.reciprocal(tf.math.maximum(self.lamb, self.eps))
#         return tf.math.maximum(self.lamb, self.eps)
    
class DiagCov(L.Layer):
    def __init__(self, n, diag_init, eps = 1e-2, name = None):
        super(DiagCov, self).__init__(name = name)
        self.diag = self.add_weight(name = '{}_diag'.format(self.name), shape = (n,), initializer =  diag_init, trainable = True)      
        self.eps = eps
        
    def call(self):
        return tf.math.reciprocal(tf.math.maximum(self.diag, self.eps))
#         return tf.math.maximum(self.diag, self.eps)

class TriDiagCovSRDD(L.Layer):
    def __init__(self, n, diag_init, off_diag_init, eps = 1e-2, name = None):
        super(TriDiagCovSRDD, self).__init__(name = name)
        self.n = n
        self.diag = self.add_weight(name = '{}_diag'.format(self.name), shape = (n,), initializer =  diag_init, trainable = True)      
        self.offdiag = self.add_weight(name = '{}_offdiag'.format(self.name), shape = (n-1,), initializer = off_diag_init, trainable = True)      
        self.eps = eps
    
    def call(self):
        b = tf.math.abs(self.offdiag)
        adjusted_diagonal = tf.math.maximum(self.diag,tf.concat([[0.0],b],axis = -1)+tf.concat([b,[0.0]],axis = -1)+self.eps)
        return [tf.concat([self.offdiag, [0.0]], axis = -1), adjusted_diagonal, tf.concat([[0.0],self.offdiag], axis = -1)]
    
class TikhonovLayer(L.Layer):
    def __init__(self, steps, m, n, A_norm, cov_params, cov_style = 'const', eps = 1e-2, scale = (False, 1.0), name = None):
        super(TikhonovLayer, self).__init__(name = name)
        self.style = cov_style
        self.eps = self.add_weight(name = '{}_eps'.format(self.name), shape = (), initializer =  tf.keras.initializers.Constant(eps), trainable = False)
        if cov_style == 'const':
            lamb_init = cov_params[0]
            self.Cov = ConstCov(n, lamb_init, self.eps, name = '{}_cov'.format(name))
        if cov_style == 'diag':
            diag_init = cov_params[0]
            self.Cov = DiagCov(n, diag_init, self.eps, name = '{}_cov'.format(name))   
        if cov_style == 'tridiag':
            diag_init, off_diag_init, pd_req = cov_params
            if pd_req == 'SRDD':
                self.Cov = TriDiagCovSRDD(n, diag_init, off_diag_init, self.eps, name = '{}_cov'.format(name))
            
        self.Tik = CalcTikhonov(steps, cov_style = cov_style, scale = scale, name = 'CalcTik')
        self.A_norm = A_norm
        
    def call(self, sparse_A, z, u, y):
        return self.Tik(sparse_A,
                        self.Cov.call(),
                        z,
                        u,
                        y)
    
class mReLU(tf.keras.layers.Layer):
    def __init__(self, MinVal, MaxVal, name = None):
        super(mReLU, self).__init__(name = name)
        self.min = self.add_weight(name = '{}_min'.format(self.name), shape = (), initializer = tf.keras.initializers.Constant(MinVal), trainable = False)
        self.max = self.add_weight(name = '{}_max'.format(self.name), shape = (), initializer = tf.keras.initializers.Constant(MaxVal), trainable = False)
        
    def call(self, inputs):
        return self.min+tf.nn.relu(inputs-self.min)-tf.nn.relu(inputs-self.max)
    
class DRCGNet:
    def __init__(self, K, J, GDsteps, sparse_A, A_norm, img_size, method, cov_params, cov_style, scale_cov = (False, 1.0), u0_style = 'Tik', normalize_init = True, b = 10.0, D = 8, filters = 32, kernel_size = 3, eta = 1.0, normalize_grad = True, B = 1.0, eps = 1e-2, bias = False, project = False, denoise = False, shared_u_weights = True):
        self.K = K
        self.J = J
        self.sparse_A = sparse_A
        self.m, n = sparse_A.dense_shape
        self.Initial = InitialLayer(A_norm, normalize = normalize_init, name = 'z0')
        self.project = project
        if project:
            self.P = mReLU(0, b, name = 'P0')  # Radon measurements
        self.Tikhonov = TikhonovLayer(GDsteps, self.m, n, A_norm, cov_params, cov_style, eps, scale_cov, name = 'Tik')
        self.u0_style = u0_style
        self.shared_u_weights = shared_u_weights
        self.Z_updates = list()
        if shared_u_weights:
            
            if method == 'ISTAConv':
    #            Params = (img_size, num_layers, channels, filters, kernel_size, UseBias, eta, normalize, B)
                Params = (img_size, D, 1, filters, kernel_size, bias, eta, normalize_grad, B)
                from blocks import ISTAConvGDu
                for k in range(K):
                    self.Z_updates.append(ISTAConvGDu.ISTAConv(J, self.m, n, Params, project = project, name = 'ISTAConv_{}'.format(k)))
            elif method == 'GradConv':
    #            Params = (img_size, num_layers, channels, filters, kernel_size, UseBias, eta, normalize, B)
                Params = (img_size, D, 1, filters, kernel_size, bias, eta, normalize_grad, B)
                from blocks import GradConvGDu
                for k in range(K):
                    self.Z_updates.append(GradConvGDu.GradConv(J, self.m, n, Params, project = project, name = 'GradConv_{}'.format(k)))  
        else:
            self.U_updates = list()
            
            if method == 'ISTAConv':
    #            Params = (img_size, num_layers, channels, filters, kernel_size, UseBias, eta, normalize, B)
                Params = (img_size, D, 1, filters, kernel_size, bias, eta, normalize_grad, B)
                from blocks import ISTAConvGDu
                for k in range(K):
                    self.Z_updates.append(ISTAConvGDu.ISTAConv(J, self.m, n, Params, project = project, name = 'ISTAConv_{}'.format(k)))
                    self.U_updates.append(TikhonovLayer(GDsteps, self.m, n, A_norm, cov_params, cov_style, eps, scale_cov, name = 'Tik_{}'.format(k)))
            elif method == 'GradConv':
    #            Params = (img_size, num_layers, channels, filters, kernel_size, UseBias, eta, normalize, B)
                Params = (img_size, D, 1, filters, kernel_size, bias, eta, normalize_grad, B)
                from blocks import GradConvGDu
                for k in range(K):
                    self.Z_updates.append(GradConvGDu.GradConv(J, self.m, n, Params, project = project, name = 'GradConv_{}'.format(k))) 
                    self.U_updates.append(TikhonovLayer(GDsteps, self.m, n, A_norm, cov_params, cov_style, eps, scale_cov, name = 'Tik_{}'.format(k)))
                                          
                
#         self.method = method
        self.denoise = denoise
        if denoise:
            if method == 'ISTAConv':
                self.D = ISTAConvGDu.ISTAConv(1, self.m, n, Params, project = False, name = 'ISTAConv_final')
            if method == 'GradConv':
                self.D = GradConvGDu.GradConv(1, self.m, n, Params, project = False, name = 'GradConv_final')
                
    def call(self):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        inputs = tf.keras.Input(shape = (self.m, ))
        z = self.Initial(self.sparse_A, inputs)
        if self.project:
            z = self.P(z)
        if self.u0_style == 'Tik':
            u = self.Tikhonov(self.sparse_A, z, tf.zeros_like(z), inputs)
        elif self.u0_style == 'Ones':
            u = tf.ones_like(z)
        for k in range(self.K):
            z = self.Z_updates[k](self.sparse_A, z, u, inputs)
            if self.shared_u_weights:
                u = self.Tikhonov(self.sparse_A, z, u, inputs)
            else:
                u = self.U_updates[k](self.sparse_A, z, u, inputs)
        outputs = tf.multiply(z, u)
        if self.denoise:
            outputs = self.D(self.sparse_A, outputs, tf.ones_like(outputs), inputs)
        model = tf.keras.Model(inputs, outputs, name = 'DRCGNet')
        return model

# class Grad:
#     def __init__(self):
#         return

#     def call(self, model, loss, inputs, targets):
#         with tf.GradientTape() as tape:
#             start = time.time()
#             estimates = model(inputs, training=True)
#             loss_value = loss.call(targets, estimates)
#         return loss_value, tape.gradient(loss_value, model.trainable_variables)    
                  
class Grad:
    def __init__(self):
        return

    def call(self, model, loss, inputs, targets):
        with tf.GradientTape() as tape:
            start = time.time()
            estimates = model(inputs, training=True)
            print('Forward Pass Time {:.5f}s'.format(time.time() - start))
            loss_value = loss.call(targets, estimates)
            start = time.time()
            grads = tape.gradient(loss_value, model.trainable_variables) 
            print('Backward Pass Time {:.5f}s'.format(time.time() - start))      
        return loss_value, grads
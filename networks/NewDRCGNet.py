import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.activations as Act
import numpy as np
import time # used here to checking computation time

from config import conf
seed = conf.train.random_seed

class InitialLayer(L.Layer):
    def __init__(self, normalize = False, name = None): # Inputs: Measurement matrix A (tensor with dtype 'float32'). Whether to normalize the measurement matrix (boolean). Name of network (string), note name is required for saving network variables. 
        super(InitialLayer, self).__init__(name = name)
        self.normalize = normalize
        
    def call(self, A, inputs):
        return inputs@A/tf.linalg.norm(A, ord = 2)**self.normalize

class CalcTikhonov(L.Layer):
    def __init__(self, m, scale = (False, 1.0), name = None):
        super(CalcTikhonov, self).__init__(name = name)
        self.m = m
#         self.lamb = self.add_weight(name = '{}_lamb'.format(name), shape = (), initializer =  tf.keras.initializers.Constant(scale[1]), trainable = scale[0]) # Previously we were not including this parameter so loading in the weights from an older training will throw and error. Instead use following code
        if scale[0]:
            self.lamb = self.add_weight(name = '{}_lamb'.format(name), shape = (), initializer =  tf.keras.initializers.Constant(scale[1]), trainable = scale[0])
        else:
            self.lamb = scale[1]

    def call(self, A, Pu, z, y):
        Azt = tf.linalg.diag(z)@tf.transpose(A)
        B = Pu@Azt
        H = tf.linalg.cholesky(tf.transpose(Azt, perm = [0, 2, 1])@B+self.lamb*tf.eye(self.m, dtype = float))
        u = tf.linalg.cholesky_solve(H, tf.expand_dims(y, -1))
        return tf.reshape(B@u, tf.shape(z))

class ConstCov(L.Layer):
    def __init__(self, n, lamb_init, eps = 1e-2, name = None):
        super(ConstCov, self).__init__(name = name)
        self.n = n
        self.lamb = self.add_weight(name = '{}_lamb'.format(self.name), shape = (), initializer =  lamb_init, trainable = True)      
        self.eps = eps
        
    def call(self):
        return tf.math.maximum(self.lamb, self.eps)*tf.eye(self.n, dtype = float)
    
class DiagCov(L.Layer):
    def __init__(self, n, diag_init, eps = 1e-2, name = None):
        super(DiagCov, self).__init__(name = name)
        self.diag = self.add_weight(name = '{}_diag'.format(self.name), shape = (n,), initializer =  diag_init, trainable = True)      
        self.eps = eps
        
    def call(self):
        return tf.linalg.diag(tf.math.maximum(self.diag, self.eps))

class TriDiagCovSRDD(L.Layer):
    def __init__(self, n, diag_init, off_diag_init, eps = 1e-2, name = None):
        super(TriDiagCovSRDD, self).__init__(name = name)
        self.n = n
        self.diag = self.add_weight(name = '{}_diag'.format(self.name), shape = (n,), initializer =  diag_init, trainable = True)      
        self.offdiag = self.add_weight(name = '{}_offdiag'.format(self.name), shape = (n-1,), initializer = off_diag_init, trainable = True)      
        self.eps = eps
    
    def call(self):
        b = tf.math.abs(self.offdiag)
        return tf.linalg.diag(tf.math.maximum(self.diag,tf.concat([[0.0],b],axis = -1)+tf.concat([b,[0.0]],axis = -1)+self.eps))+tf.linalg.diag(self.offdiag,k = -1)+tf.linalg.diag(self.offdiag,k = 1)

class TriDiagCovEIG(L.Layer):
    def __init__(self, n, diag_init, off_diag_init, eps = 1e-2, name = None):
        super(TriDiagCovEIG, self).__init__(name = name)
        self.n = n
        self.diag = self.add_weight(name = '{}_diag'.format(self.name), shape = (n,), initializer =  diag_init, trainable = True)      
        self.offdiag = self.add_weight(name = '{}_offdiag'.format(self.name), shape = (n-1,), initializer = off_diag_init, trainable = True)      
        self.eps = eps
    
    def call(self):
        D, Q = tf.linalg.eigh(tf.linalg.diag(self.diag)+tf.linalg.diag(self.offdiag,k = -1)+tf.linalg.diag(self.offdiag,k = 1))
        return Q@tf.linalg.diag(tf.math.maximum(D, self.eps))@tf.transpose(Q)

class FullCov(L.Layer):
    def __init__(self, n, L_init, eps = 1e-2, name = None):
        super(FullCov, self).__init__(name = name)
        self.n = n
        self.L = self.add_weight(name = '{}_L'.format(self.name), shape = (n*(n+1)//2,), initializer =  L_init, trainable = True)      
        self.eps = eps
    
    def call(self):
        Ltri = tf.linalg.band_part(tf.reshape(tf.concat([self.L, self.L[self.n:][::-1]], 0), (self.n,self.n)), 0, -1)
        return Ltri@tf.transpose(Ltri)+self.eps*tf.eye(self.n, dtype = float)
    
class FullCovEIG(L.Layer):
    def __init__(self, n, L_init, eps = 1e-2, name = None):
        super(FullCovEIG, self).__init__(name = name)
        self.n = n
        self.L = self.add_weight(name = '{}_L'.format(self.name), shape = (n*(n+1)//2,), initializer =  L_init, trainable = True)      
        self.eps = eps
    
    def call(self):
        Ltri = tf.linalg.band_part(tf.reshape(tf.concat([self.L, self.L[self.n:][::-1]], 0), (self.n,self.n)), 0, -1)
        D, Q = tf.linalg.eigh((Ltri+tf.transpose(Ltri))/2)
        return Q@tf.linalg.diag(tf.math.maximum(D, self.eps))@tf.transpose(Q)

class TikhonovLayer(L.Layer):
    def __init__(self, m, n, cov_params, cov_style = 'const', eps = 1e-2, scale = (False, 1.0), name = None):
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
            elif pd_req == 'EIG':
                self.Cov = TriDiagCovEIG(n, diag_init, off_diag_init, self.eps, name = '{}_cov'.format(name))
        if cov_style == 'full':
            L_init, pd_req = cov_params
            if pd_req == 'DEFAULT':
                self.Cov = FullCov(n, L_init, self.eps, name = '{}_cov'.format(name))
            elif pd_req == 'EIG':
                self.Cov = FullCovEIG(n, L_init, self.eps, name = '{}_cov'.format(name))                        
        self.Tik = CalcTikhonov(m, scale = scale, name = 'CalcTik')
        
    def call(self, A, z, y):
        Pu = self.Cov.call()
        return self.Tik(A, Pu, z, y)
    
class mReLU(tf.keras.layers.Layer):
    def __init__(self, MinVal, MaxVal, name = None):
        super(mReLU, self).__init__(name = name)
        self.min = self.add_weight(name = '{}_min'.format(self.name), shape = (), initializer = tf.keras.initializers.Constant(MinVal), trainable = False)
        self.max = self.add_weight(name = '{}_max'.format(self.name), shape = (), initializer = tf.keras.initializers.Constant(MaxVal), trainable = False)
        
    def call(self, inputs):
        return self.min+tf.nn.relu(inputs-self.min)-tf.nn.relu(inputs-self.max)
    
class DRCGNet:
    def __init__(self, K, J, A, img_size, method, cov_params, cov_style, scale_cov = (False, 1.0), u0_style = 'Tik', normalize_init = True, b = 10.0, D = 8, filters = 32, kernel_size = 3, eta = 1.0, normalize_grad = True, B = 1.0, eps = 1e-2, bias = True, project = False, denoise = False):
        self.K = K
        self.J = J
        self.A = A
        self.m, n = tf.shape(A)
        self.Initial = InitialLayer(normalize = normalize_init, name = 'x0')
        self.project = project
        if project:
            self.P = mReLU(0, b, name = 'P0')  # Radon measurements
        self.Tikhonov = TikhonovLayer(self.m, n, cov_params, cov_style, eps, scale_cov, name = 'Tik')
        self.u0_style = u0_style
        self.Z_updates = list()

        if method == 'ISTAConv':
#            Params = (img_size, num_layers, channels, filters, kernel_size, UseBias, eta, normalize, B)
            Params = (img_size, D, 1, filters, kernel_size, bias, eta, normalize_grad, B)
            from blocks import ISTAConv
            for k in range(K):
                model = ISTAConv.ISTAConv(J, self.m, n, Params, project = project, name = 'ISTAConv_{}'.format(k))
                self.Z_updates.append(model.call(A, k))
        elif method == 'GradConv':
#            Params = (img_size, num_layers, channels, filters, kernel_size, UseBias, eta, normalize, B)
            Params = (img_size, D, 1, filters, kernel_size, bias, eta, normalize_grad, B)
            from blocks import GradConv
            for k in range(K):
                model = GradConv.GradConv(J, self.m, n, Params, project = project, name = 'GradConv_{}'.format(k))
                self.Z_updates.append(model.call(A, k))                
                
        self.method = method
        self.denoise = denoise
        if denoise:
            if method == 'ISTAConv':
                model = ISTAConv.ISTAConv(1, self.m, n, Params, project = False, name = 'ISTAConv_final')
            if method == 'GradConv':
                model = GradConv.GradConv(1, self.m, n, Params, project = False, name = 'GradConv_final')
            self.D = model.call(A, 'final')
                
    def call(self):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        inputs = tf.keras.Input(shape = (self.m, ))
        z = self.Initial(self.A, inputs)
        if self.project:
            z = self.P(z)
        if self.u0_style == 'Tik':
            u = self.Tikhonov(self.A, z, inputs)
        elif self.u0_style == 'Ones':
            u = tf.ones_like(z)
        for k in range(self.K):
            z = self.Z_updates[k]((inputs, z, u))
            u = self.Tikhonov(self.A, z, inputs)
        outputs = tf.multiply(z, u)
        if self.denoise:
            outputs = self.D((inputs, outputs, tf.ones_like(outputs)))
        model = tf.keras.Model(inputs, outputs, name = 'DRCGNet')
        return model

class Grad:
    def __init__(self):
        return

    def call(self, model, loss, inputs, targets):
        with tf.GradientTape() as tape:
            start = time.time()
            estimates = model(inputs, training=True)
            loss_value = loss.call(targets, estimates)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)               
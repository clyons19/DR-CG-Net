import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.activations as Act
import numpy as np

seed = 19
np.random.seed(seed)
tf.random.set_seed(seed)

class rLayer(tf.keras.layers.Layer):
    def __init__(self, eta = 1e-3, normalize = False, B = 1.0, name = None):
        super(rLayer, self).__init__(name = name)
        self.eta = self.add_weight(name = '{}_eta'.format(self.name), shape = (), initializer = tf.keras.initializers.Constant(eta), trainable = True)
        self.norm = normalize
        self.B = B
        
    def call(self, A, z, y):
        v = tf.transpose(A, perm = [0, 2, 1])@(tf.expand_dims(y, -1)-A@tf.expand_dims(z, -1))
        v_norm = tf.norm(v, ord='euclidean', axis=1, keepdims=True)
        v = v/(tf.maximum(1.0, v_norm/self.B))**self.norm
        return z - self.eta*tf.reshape(v, tf.shape(z))

class CNN(tf.keras.Model):  # Basic CNN model with ReLU activations
    def __init__(self, img_size, num_layers, channels, filters, kernel_size, UseBias):
        super(CNN, self).__init__()
        num_layers = int(np.floor(np.maximum(1,num_layers)))  
        self.Mat = L.Reshape((img_size, img_size, channels))
        self.Vec = L.Reshape((channels*img_size**2, ))
        self.options = []
        for layer in range(num_layers-1):
            self.options.append(L.Conv2D(filters, kernel_size, padding='same', activation = 'relu', use_bias = UseBias))
        self.options.append(L.Conv2D(channels, kernel_size, padding='same', use_bias = UseBias))

    def call(self, inputs):
        x = self.Mat(inputs)
        for l in self.options:
            x = l(x)    
        x = self.Vec(x)
        return x   

class ISTAConv(tf.keras.Model):
    def __init__(self, J, m, n, Params, project = False, name = None):
        super(ISTAConv, self).__init__(name = name)
        self.J = J
        self.m = m
        self.n = n
        self.project = project
        self.prox = list()
        self.r = list()
        for j in range(J):
            self.prox.append(CNN(*Params[:6]))
            self.r.append(rLayer(*Params[6:], name = '{}_{}_r'.format(name, j)))
                            
    def call(self, A, k):
        y = tf.keras.Input(shape = (self.m, ))
        z_input = tf.keras.Input(shape = (self.n, ))
        u = tf.keras.Input(shape = (self.n, ))
        z = z_input
        for j in range(self.J):
            r = self.r[j](A@tf.linalg.diag(u), z, y)
            z = r+self.prox[j](r)
            if self.project:
                z = tf.nn.relu(z)
        model = tf.keras.Model(inputs = [y, z_input, u], outputs = z, name = 'ISTAConv_{}'.format(k))
        return model
import tensorflow as tf

from utils import Initialize, LossFunctions, LoadMatrices
import numpy as np
import pandas as pd
import time
import os

from config import conf

class MultiRun:
    def __init__(self, method, grad_fnc):
        self.method = method
        self.grad = grad_fnc
        
        self.epoch_check = conf.train.epoch_check
        self.loss = conf.train.loss
        self.use_Phi = conf.train.use_sparsity
        self.phi_type = conf.train.sparsity_type
        self.rearrange = conf.train.rearrange_inputs  
        
    def val_loop(self, data, batch_size, m, Phi, Phi_Change):
        data_size = len(data)
        num_batches = int(np.ceil(data_size/batch_size))
        loss_avg = tf.keras.metrics.Mean()
        
        from utils import Validate
        V = Validate.AverageMetrics(Phi, self.img_size, self.scale, power = 1, normalize = False)

        for batch in range(num_batches):  # Loop over data. Reconstruct and calculate quality metrics
            y = self.scale*np.array(data.iloc[batch*batch_size:min([(batch+1)*batch_size,data_size])])
            c_act, y = y[:,m:], y[:,:m]
            if self.rearrange:
                size = y.shape[0]
                sinogram = np.reshape(y, (size,m//self.msmnt,self.msmnt))
                y = np.reshape(np.transpose(sinogram, [0,2,1]), (size, m))
            c_act = c_act if Phi_Change is None else c_act@Phi_Change
            x_est = self.model(y, training = False)
            if type(x_est) == tuple:
                x_est = x_est[0]
            loss_value = self.loss.call(c_act,x_est)
            loss_avg.update_state(loss_value)
            V.call(c_act, x_est)
        return V, loss_avg
    
    def training_loop(self, train_data, test_data, val_data, m, Phi, Phi_Change):
        # Keep results for plotting
        train_loss_results = []
        val_loss_results = []
        training_time = []
        outputs = []
        val_results = {'SSIM':[], 'PSNR':[], 'MSE':[], 'MAE':[], 'LOSS':[]}
        
        ValBatchSize, batch_size = self.test_batch_size, self.batch_size
        train_data_size, test_data_size, val_data_size = len(train_data), len(test_data), len(val_data)
        num_batches, num_test_batches, num_val_batches = int(np.ceil(train_data_size/batch_size)), int(np.ceil(test_data_size/ValBatchSize)), int(np.ceil(val_data_size/ValBatchSize))
        print('Number of updates per epoch = ', num_batches)

        start = time.time()
        for epoch in range(self.num_epochs):  # Training Loop
            if conf.optimizer.use_lr_scheduler:
                self.optimizer.lr.assign(self.lr_schedule(epoch).numpy())
            epoch_loss_avg = tf.keras.metrics.Mean()
            shuffled_data = train_data.sample(frac = 1)
          # Training loop - using batches of size batch_size
            start2 = time.time()
            for batch in range(num_batches):
                y = self.scale*np.array(shuffled_data.iloc[batch*batch_size:min([(batch+1)*batch_size,train_data_size])])
                c_act, y = y[:,m:], y[:,:m]
                if self.rearrange:
                    size = y.shape[0]
                    sinogram = np.reshape(y, (size,m//self.msmnt,self.msmnt))
                    y = np.reshape(np.transpose(sinogram, [0,2,1]), (size, m))
                c_act = c_act if Phi_Change is None else c_act@Phi_Change
            # Optimize the model
                loss_value, grads = self.grad.call(self.model, self.loss, y, c_act)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                            
            training_time.append(time.time()-start2) 
            train_loss_results.append(epoch_loss_avg.result())

            if (epoch+1)%self.epoch_check==0:
                save_route = os.path.join(self.path_to_save, '{} epochs'.format(epoch+1))
                if not os.path.exists(save_route):
                    os.makedirs(save_route)
                self.model.save_weights(os.path.join(save_route, '{}.h5'.format(self.save_name)))
                
                #### ################################### ####
                #### Average Quality Metrics on Val Data ####
                #### ################################### ####
                V, loss_avg = self.val_loop(val_data, ValBatchSize, m, Phi, Phi_Change)
                
                for entry in V.data:
                    val_results[entry].append(np.mean(V.data[entry]))
                val_results['LOSS'].append(loss_avg.result().numpy())
                    
                if not conf.general.quiet:
                    print("Epoch {:02d}: Model Loss: {:.3e} | Test Loss: {:.3e} | Time: {:.3e}".format(epoch+1, epoch_loss_avg.result(), val_results['LOSS'][-1], time.time() - start))
                    print("Training Time per Epoch: {:.3e} | Total Training Time: {:.3e}".format(sum(training_time)/(epoch+1), sum(training_time)))

                outputs.append([epoch+1, epoch_loss_avg.result().numpy(), val_results['LOSS'][-1], sum(training_time)])
                df = pd.DataFrame(np.array(outputs))
                df.to_csv(os.path.join(save_route, '{}_results.csv'.format(self.save_name)))
                
                data = []
                for entry in V.data:
                    data.append([entry, np.mean(V.data[entry]), np.var(V.data[entry]), 2.576*np.sqrt(np.var(V.data[entry])/val_data_size)])
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(save_route, '{}_ave_val_results.csv'.format(self.save_name)))                     
                
                data = []
                for entry in val_results:
                    data.append([entry]+val_results[entry])
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(save_route, '{}_val_results.csv'.format(self.save_name)))
                
                #### ################################### ####
                #### ################################### ####
                #### ################################### ####
                
                #### Average Quality Metrics on Test Data ####

                V, loss_avg = self.val_loop(test_data, ValBatchSize, m, Phi, Phi_Change)
                
                data = []
                for entry in V.data:
                    data.append([entry, np.mean(V.data[entry]), np.var(V.data[entry]), 2.576*np.sqrt(np.var(V.data[entry])/test_data_size)])
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(save_route, '{}_ave_small_test_results.csv'.format(self.save_name)))
                
                ### Overfitting Check for Terminating Training ###
                
                if val_results['LOSS'][-1] > self.best_loss:
                    self.cnt += 1
                else:
                    self.cnt = 0
                    self.best_loss = val_results['LOSS'][-1]
                    self.best_epoch_check = epoch+1
                if self.cnt >= conf.general.max_increases:
                    break
                    
                ### ########################################## ###
                
        if self.num_epochs%self.epoch_check != 0:
            save_route = os.path.join(self.path_to_save, '{} epochs'.format(self.num_epochs))
            if not os.path.exists(save_route):
                os.makedirs(save_route)
            self.model.save_weights(os.path.join(save_route, '{}.h5'.format(self.save_name)))
            
            V, loss_avg = self.val_loop(val_data, ValBatchSize, m, Phi, Phi_Change)
                
            for entry in V.data:
                val_results[entry].append(np.mean(V.data[entry]))
            val_results['LOSS'].append(loss_avg.result().numpy())

            if not conf.general.quiet:
                print("Epoch {:02d}: Model Loss: {:.3e} | Test Loss: {:.3e} | Time: {:.3e}".format(epoch+1, epoch_loss_avg.result(), val_results['LOSS'][-1], time.time() - start))
                print("Training Time per Epoch: {:.3e} | Total Training Time: {:.3e}".format(sum(training_time)/(epoch+1), sum(training_time)))
            
            outputs.append([epoch+1, epoch_loss_avg.result().numpy(), val_results['LOSS'][-1], sum(training_time)])
            
            data = []
            for entry in V.data:
                data.append([entry, np.mean(V.data[entry]), np.var(V.data[entry]), 2.576*np.sqrt(np.var(V.data[entry])/val_data_size)])
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(save_route, '{}_ave_results_val_dataset.csv'.format(self.save_name)))

            data = []
            for entry in val_results:
                data.append([entry]+val_results[entry])
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(save_route, '{}_val_results.csv'.format(self.save_name)))                

            df = pd.DataFrame(np.array(outputs))
            df.to_csv(os.path.join(save_route, '{}_training_losses.csv'.format(self.save_name)))
    
    def call(self, img_size, msmnt_list, noise_list, training_list, epoch_list, msmnt_type):
        self.img_size = img_size
        self.batch_size = conf.train.batch_size
        self.test_batch_size = conf.train.test_batch_size
        self.scale = conf.data.scale
        Load = LoadMatrices.Load()
        
        ### ########################################## ###        
        ### Read in Sparsity Transform (as applicable) ###
        ### ########################################## ###
        Phi, Phi_Wavelet = Load.load_Phi(img_size, conf.train.use_sparsity, conf.train.sparsity_type, conf.data.true_data_type)
        
        ## ########################## ##
        ## Set Change-of-Basis Matrix ##
        ## ########################## ##
        if Phi_Wavelet is None:
            if Phi is None:
                Phi_Change = None
            else: 
                Phi_Change = Phi
        else:
            if Phi is None:
                Phi_Change = tf.transpose(tf.constant(Phi_Wavelet, dtype = 'float32'))
            else: 
                Phi_Change = tf.constant(np.transpose(Phi_Wavelet)@Phi, dtype = 'float32')    
        ## ########################## ##
        ## ########################## ##
        ## ########################## ##
        
        ### ########################################## ###
        ### ########################################## ###
        ### ########################################## ###
        
        for i in range(len(msmnt_list)):  # Loop over all measurements
            msmnt = msmnt_list[i]
            self.msmnt = msmnt
            
            ### ########################## ###  
            ### Read in Measurement Matrix ###
            ### ########################## ###  
            
            Psi = Load.load_Psi(img_size, msmnt, msmnt_type)
            
            A = tf.constant(Psi, dtype = 'float32') if Phi is None else tf.constant(Psi@Phi, dtype = 'float32')  # Set A = Psi@Phi for whether the DNN method is recovering the images or sparisty coefficients.
            m, n = tf.shape(A)
            del Psi
            if 'GDu' in conf.model.method:
                A_norm = tf.linalg.norm(A, ord=2)
                A = tf.sparse.from_dense(A)
            ### ########################## ###                  
            ### ########################## ###
            ### ########################## ###
            
            for j in range(len(noise_list[i])):  # Loop over all noise options for the given measurement
                noise = noise_list[i][j]
                for k in range(len(training_list[i][j])):  # Loop over all training dataset sizes for the choice of measurement and noise
                    num_training_samples = training_list[i][j][k]
                    print('Beginning Trial {} {} {} dB {} training size'.format(msmnt, msmnt_type, noise, num_training_samples))
                    self.num_epochs = epoch_list[i][j][k]
                    
                    self.best_loss = np.inf
                    self.cnt = 0
                    
                    seed = conf.train.random_seed
                    np.random.seed(seed)
                    tf.random.set_seed(seed)
                    
                    ### ################################### ###
                    ### Set Up Directory to Save Results In ###
                    ### ################################### ###
                    
                    train_details = r'{} {} {} dB {} training size'.format(msmnt, msmnt_type, noise, num_training_samples)
                    self.path_to_save = os.path.join(conf.general.SaveBasePath, 'Results {} imgsize'.format(self.img_size), conf.general.SaveFolder, train_details)
                    if not os.path.exists(self.path_to_save):
                        os.makedirs(self.path_to_save)
                    else:
                        original_save_path = os.path.join(conf.general.SaveBasePath, 'Results {} imgsize'.format(self.img_size), conf.general.SaveFolder, train_details)
                        trials_cnt = 1
                        while os.path.exists(self.path_to_save):
                            folder_label = '_{}'.format(trials_cnt)
                            self.path_to_save = original_save_path + folder_label
                            trials_cnt += 1
                        os.makedirs(self.path_to_save)
                        
                    self.save_name = '{}_{}{}_{}dB'.format(conf.general.SaveName, msmnt, msmnt_type, noise)
                    
                    ### ################################### ###  
                    ### ################################### ###
                    ### ################################### ###

                    ### ##################### ###
                    ### Load in Training Data ###
                    ### ##################### ###
                    
                    if self.img_size == 32:
                        dataset_name = 'cifar10_{}{}_{}dB_train_data.pkl'.format(msmnt, msmnt_type, noise)
                        path_to_train_data = os.path.join(conf.general.TrainDataBasePath, dataset_name)
                        train_data = pd.read_pickle(path_to_train_data)
                    elif self.img_size == 64:
                        dataset_name = 'caltech101_{}{}_{}dB_train_data.csv'.format(msmnt, msmnt_type, noise)
                        path_to_train_data = os.path.join(conf.general.TrainDataBasePath, dataset_name)
                        train_data = pd.read_csv(path_to_train_data)
                        train_data.drop(train_data.columns[0], axis=1, inplace=True)
                    elif self.img_size == 128:
                        if conf.data.true_data_type == 'coefs':
                            dataset_name = 'lopabct_{}{}_{}dB_train_data.pkl'.format(msmnt, msmnt_type, noise)
                        elif conf.data.true_data_type == 'imgs':
                            dataset_name = 'lopabct_{}{}_{}dB_train_data_imgs.pkl'.format(msmnt, msmnt_type, noise)                        
                        path_to_train_data = os.path.join(conf.general.TrainDataBasePath, dataset_name)
                        train_data = pd.read_pickle(path_to_train_data)                  
                    
                    ## Create Validation Dataset from Remaining Unused Train Data ##
                    
                    val_data = train_data.iloc[num_training_samples:min(len(train_data), num_training_samples+1000)]  # Select up to 1000 additional data points
                    shuffled_data = val_data.sample(frac = 1)  # Randomly Shuffle
                    val_data = shuffled_data[:min(len(val_data),conf.data.num_validation_points)]  # Choose validation points
                    del shuffled_data
                    ## ########################################################## ##             
                    train_data = train_data.iloc[0:num_training_samples]
                    
                    ### ##################### ###
                    ### ##################### ###
                    ### ##################### ###
                    
                    ### ################# ###
                    ### Load in Test Data ###
                    ### ################# ###
                    
                    if self.img_size == 32 or self.img_size == 64:
                        dataset_name = '{}x{}_{}{}_{}dB_test_data.csv'.format(self.img_size, self.img_size, msmnt, msmnt_type, noise)
                        path_to_test_data = os.path.join(conf.general.TestDataBasePath, dataset_name)
                        test_data = pd.read_csv(path_to_test_data)
                        test_data.drop(test_data.columns[0], axis=1, inplace=True)
                    elif self.img_size == 128:
                        if conf.data.true_data_type == 'coefs':
                            dataset_name = 'lopabct_{}{}_{}dB_test_data.pkl'.format(msmnt, msmnt_type, noise)
                        elif conf.data.true_data_type == 'imgs':
                            dataset_name = 'lopabct_{}{}_{}dB_test_data_imgs.pkl'.format(msmnt, msmnt_type, noise)                        
                        path_to_test_data = os.path.join(conf.general.TestDataBasePath, dataset_name)
                        test_data = pd.read_pickle(path_to_test_data)
                    
                    test_data = test_data.iloc[0:min(len(test_data), conf.data.num_test_points)]

                    ### ################# ###  
                    ### ################# ###
                    ### ################# ###
                    
############### ################ ###############  
############### Create DR-CG-Net ###############
############### ################ ###############  

                    import tensorflow.keras.initializers as Init
                    cov_params = {'const': [Init.Constant(conf.model.lamb_init[noise])],
                                  'diag': [Init.Constant(conf.model.lamb_init[noise])],
                                  'tridiag': [Init.Constant(conf.model.lamb_init[noise]), Init.Constant(0.0), 'SRDD']
                                 }
                    if conf.model.cov_style == 'full':
                        n = self.img_size**2
                        Q = np.zeros((n*(n+1)//2,))
                        indx = np.array([i*(n+1)+1 for i in range(n-n//2)]+[i*(n+1) for i in range(n-n//2-n%2, 0, -1)])-1
                        Q[indx] = conf.model.lamb_init[noise]*np.ones(n)
                        cov_params[conf.model.cov_style] = [Init.Constant(Q), 'DEFAULT'] 
                        del Q 
                                   
                    self.optimizer = conf.optimizer.optimizer(conf.optimizer.starting_lr)
                    if conf.optimizer.use_lr_scheduler: 
                        self.lr_schedule = conf.optimizer.lr_schedule(conf.optimizer.starting_lr, decay_steps = self.num_epochs, end_learning_rate = conf.optimizer.starting_lr)

                    if 'GDu' not in conf.model.method:                        
                        self.model_pre_call = self.method(
                                              conf.model.K, 
                                              conf.model.J, 
                                              A, 
                                              self.img_size, 
                                              conf.model.CNN_option, 
                                              cov_params[conf.model.cov_style], 
                                              conf.model.cov_style,
                                              scale_cov = conf.model.scale_cov,
                                              u0_style = conf.model.u0_style, 
                                              normalize_init = conf.model.normalize_A_in_z0,
                                              b = conf.model.b,
                                              D = conf.model.D,
                                              filters = conf.model.num_filters,
                                              kernel_size = conf.model.kernel_size,
                                              eta = conf.model.step_size,
                                              normalize_grad = conf.model.normalize_grad,
                                              B = conf.model.B,
                                              eps = conf.model.epsilon,                      
                                              bias = conf.model.use_bias,
                                              project = conf.model.project_z_est,
                                              denoise = conf.model.use_refinement)
                        
                    elif 'GDu' in conf.model.method:
                        self.model_pre_call = self.method(
                                              conf.model.K, 
                                              conf.model.J,
                                              conf.model.GDsteps,
                                              A, 
                                              A_norm,
                                              self.img_size, 
                                              conf.model.CNN_option, 
                                              cov_params[conf.model.cov_style], 
                                              conf.model.cov_style,
                                              scale_cov = conf.model.scale_cov,
                                              u0_style = conf.model.u0_style, 
                                              normalize_init = conf.model.normalize_A_in_z0,
                                              b = conf.model.b,
                                              D = conf.model.D,
                                              filters = conf.model.num_filters,
                                              kernel_size = conf.model.kernel_size,
                                              eta = conf.model.step_size,
                                              normalize_grad = conf.model.normalize_grad,
                                              B = conf.model.B,
                                              eps = conf.model.epsilon,                      
                                              bias = conf.model.use_bias,
                                              project = conf.model.project_z_est,
                                              denoise = conf.model.use_refinement,
                                              shared_u_weights = conf.model.shared_u_weights)
                    
############### ################ ###############
############### ################ ###############  
############### ################ ###############  

                    self.model = self.model_pre_call.call()

                    self.training_loop(train_data, test_data, val_data, m, Phi, Phi_Change)
                    
                    ### ################################################# ### 
                    ### Load in Best Weights Over Training and Test Model ###
                    ### ################################################# ### 
                    if conf.general.best_model_test:
                        save_route = os.path.join(self.path_to_save, 'best')
                        if not os.path.exists(save_route):
                            os.makedirs(save_route)

                        best_weight_file = os.path.join(self.path_to_save, '{} epochs'.format(self.best_epoch_check), '{}.h5'.format(self.save_name))
                        self.model.load_weights(best_weight_file)                   
                        test_data_size = len(test_data)
                        V, loss_avg = self.val_loop(test_data, self.test_batch_size, m, Phi, Phi_Change)
                        data = []
                        for entry in V.data:
                            data.append([entry, np.mean(V.data[entry]), np.var(V.data[entry]), 2.576*np.sqrt(np.var(V.data[entry])/test_data_size)])
                        df = pd.DataFrame(data)
                        df.to_csv(os.path.join(save_route, '{}_small_test_dataset_results.csv'.format(self.save_name)))

                    ### ################################################# ### 
                    ### ################################################# ### 
                    ### ################################################# ### 
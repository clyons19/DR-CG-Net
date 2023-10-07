import os
import pandas as pd
import numpy as np

class Load:
    def __init__(self):
        if not os.path.exists(os.path.join('Matrices')):
            os.makedirs(os.path.join('Matrices'))
        return
    
    def load_Psi(self, img_size, msmnt, msmnt_type):
        Psi = None
        
        ### ########################## ###
        ### Load in measurement matrix ###
        ### ########################## ###
        try:
            Psi = pd.read_csv(os.path.join('Matrices', 'Psi_{}x{}_{}{}.csv'.format(img_size, img_size, msmnt, msmnt_type)))
        except:
            if msmnt_type == 'angles':
                print("Radon Transform Psi has not been created. Creating it now.") 
                from utils import Initialize
                B = Initialize.create_measurements('barbara.png', [[154, 154+img_size],[64,64+img_size]], import_phi = (True, None, None), import_psi = (False, [], msmnt))
                df = pd.DataFrame(B.Psi)
                df.to_csv(os.path.join('Matrices', 'Psi_{}x{}_{}{}.csv'.format(img_size, img_size, msmnt, msmnt_type)))
                Psi = np.array(B.Psi)
            elif msmnt_type == 'ratio':
                print('Gaussian Matrix Psi has not been created. Double check it located in the correct path:')
                print(os.path.join('Matrices'))
            else:
                print('Unknown msmnt_type. Must be one of: [angles, ratio].')
        else:
            Psi.drop(Psi.columns[0], axis = 1, inplace = True) # Drop uneeded label column from save_csv
            Psi = np.array(Psi)
        ### ########################## ###
        ### ########################## ###
        ### ########################## ###
        
        return Psi
        
    def load_Phi(self, img_size, use_sparsity, sparsity_type, true_data_type):
        Phi, Phi_Wavelet = None, None

        ### ############################## ###    
        ### Load in change-of-basis matrix ###
        ### ############################## ### 
        if true_data_type == 'coefs':  # If data contains image wavelet coefficents load in discrete wavelet transform matrix
            try:
                Phi_Wavelet = pd.read_csv(os.path.join('Matrices', 'Phi_biowavelet_{}x{}.csv'.format(img_size, img_size)))
            except:
                print('Wavelet Transform Phi has not been created. Creating it now.')
                from utils import Initialize
                B = Initialize.create_measurements('barbara.png', [[154, 154+img_size],[64,64+img_size]], import_phi = (False, [], 'bior1.1'), import_psi = (True, None, None))
                df = pd.DataFrame(B.Phi)
                df.to_csv(os.path.join('Matrices', 'Phi_biowavelet_{}x{}.csv'.format(img_size, img_size)))
                Phi_Wavelet = np.array(B.Phi)
            else:
                Phi_Wavelet.drop(Phi_Wavelet.columns[0], axis = 1, inplace = True) # Drop uneeded label column from save_csv
                Phi_Wavelet = np.array(Phi_Wavelet)
        elif true_data_type == 'imgs':  # If data contains images wavelet transform matrix is not needed so set as None
            Phi_Wavelet = None

        if use_sparsity:  # If DR-CG-Net is recovering image sparsity coefficents load in corresponding sparsity transformation matrix
            Phi = Phi_Wavelet if sparsity_type == 'biowavelet' and Phi_Wavelet is not None else None
            if Phi is None:
                try:
                    Phi = pd.read_csv(os.path.join('Matrices', 'Phi_{}_{}x{}.csv'.format(sparsity_type, img_size, img_size)))
                except:
                    print('{} Phi has not been created. Creating it now.'.format(sparsity_type))
                    import Initialize
                    sp = sparsity_type if sparsity_type == 'dct' else 'bior1.1'
                    B = Initialize.create_measurements('barbara.png', [[154, 154+img_size],[64,64+img_size]], import_phi = (False, [], sp), import_psi = (True, None, None))
                    df = pd.DataFrame(B.Phi)
                    df.to_csv(os.path.join('Matrices', 'Phi_{}_{}x{}.csv'.format(sparsity_type, img_size, img_size)))
                    Phi = np.array(B.Phi)
                else:
                    Phi.drop(Phi.columns[0], axis = 1, inplace = True) # Drop uneeded label column from save_csv
                    Phi = np.array(Phi)
        else:  # Otherwise DR-CG-Net is recovering the images and no sparsity transformation matrix is needed
            Phi = None 
        ### ############################## ###   
        ### ############################## ### 
        ### ############################## ### 
        
        return Phi, Phi_Wavelet
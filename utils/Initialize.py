# Initialize.py 
# * Created by: Carter Lyons
# * Last Edited: **7/15/2022**

# Contains:
# * Function "create_PHI" to create a wavelet dictionary matrix $\Phi$
# * Function "create_PSI" to create a Radon transform matrix $\Psi$
# * Class which takes an image, clips it and creates a wavelet dictionary, Radon transform matrix, and Radon transform measurements based upon the clipped image

# Required Packages (in addition to Anaconda):
# numpy, scikit-image

import copy
import numpy as np
import pywt
from skimage.transform import radon, rescale, iradon
from PIL import Image, ImageOps


#Create matrix PHI of size (rows*cols, rows*cols) using wavelets wname
def create_PHI_wavelet(rows, cols, wname):
    PHI_temp = np.zeros((rows*cols, rows*cols))
    Ci = pywt.wavedec2(np.zeros((rows, cols)), wname)
    
    ## LL Rec ##
    Ci_copy = copy.deepcopy(Ci)
    Ci_copy[0][0][0] = 1
    timg = pywt.waverec2(Ci_copy, wname)
    PHI_temp[:, 0] = np.reshape(timg, rows*cols)
    ############
    
    cnt = 1
    for level in range(1, len(Ci)):
        for arr_indx in range(len(Ci[level])):
            for row_indx in range(np.shape(Ci[level][arr_indx])[0]):
                for col_indx in range(np.shape(Ci[level][arr_indx])[1]):
                    Ci_copy = copy.deepcopy(Ci)
                    Ci_copy[level][arr_indx][row_indx][col_indx] = 1
                    timg = pywt.waverec2(Ci_copy, wname)
                    PHI_temp[:,cnt] = np.reshape(timg, rows*cols)
                    cnt += 1
    return PHI_temp

#Create matrix dct PHI matrix 
def create_PHI_dct(size):
    L = size**2
    C = np.zeros((L,L))
    for k in range(L):
        for n in range(L):
            if k == 0:
                C[k,n] = np.sqrt(1/L)
            else:
                C[k,n] = np.sqrt(2/L)*np.cos((np.pi*k*(1/2+n))/L)
    return C

#Create Radon transform matrix PSI using num_angles number of angles for the Radon transform and wavelet matrix PHI
def create_PSI(rows, cols, num_angles):
    theta_imrot = np.pi
    del_rot = theta_imrot/num_angles
    theta_vec = np.linspace(del_rot, theta_imrot, num_angles)*(180/np.pi)
    y_orig = radon(np.ones((rows, cols)), theta = theta_vec, circle = False)
    radscf = np.shape(y_orig)[0]
    PSI = np.zeros((radscf*num_angles, rows*cols))
    for cc in range(rows*cols):
        vec = np.zeros(( rows*cols,1))
        vec[cc] = 1
        rvec = radon(np.reshape(vec, (rows, cols)), theta = theta_vec, circle = False)
        rvec = np.reshape(rvec, (radscf*num_angles,1))   
        PSI[:, cc] = np.reshape(rvec, (np.shape(rvec)[0],))
    return PSI

#Class to convert non-grayscale images to grayscale and extract a designated patch
class create_measurements:
    
    # Initialization function for the class. When class is called the inputs to __init__ can be passed to set up the variables below.
    def __init__(self, image, cell_selection, import_phi = (False, [], 'bior1.1'), import_psi = (False, [], 15)):
        I = np.array(ImageOps.grayscale(Image.open(image)))      # Import image and convert to grayscle
        self.I = I[cell_selection[0][0]:cell_selection[0][1], cell_selection[1][0]:cell_selection[1][1]]   # Patch selection
        self.I = self.I/np.max(self.I)
        self.create_phi(import_phi)
        self.create_psi(import_psi)
#         self.y = self.Psi@np.reshape(self.I, (self.I.size,1))    
    
    # Take Radon transform at num_angles uniformly spaced angles
    def Radon(self, num_angles):
        theta = np.linspace(0., 180., num_angles, endpoint=False)+(180/num_angles)
        self.sinogram = radon(self.I, theta=theta, circle = False)
    
    # Create or import wavelet dictionary. importing_phi is a tuple containing:
      # 1.) True or False Boolean specifying if Phi is being imported. If True then matrix Phi must be provided. If False then Phi is created from scratch
      # 2.) Numpy array containing matrix Phi if 1.) is True. Else an empty array [] can be passed in
      # 3.) Wavename, as a string, for the type of wavelets to be used in creation of Phi if 1.) is False. Else this entry can be ignored
    def create_phi(self, importing_phi):
        if not importing_phi[0]:
            if importing_phi[2] == 'dct':
                self.Phi = create_PHI_dct(np.shape(self.I)[0])
            else:
                self.Phi = create_PHI_wavelet(*np.shape(self.I), importing_phi[2])
        else:
            self.Phi = importing_phi[1]
    
    # Create or import the discrete matrix form of radon transform. importing_psi is a tuple containing:
      # 1.) True or False Boolean specifying if Psi is being imported. If True then matrix Psi must be provided. If False then Psi is created from scratch
      # 2.) Numpy array containing matrix Psi if 1.) is True. Else an empty array [] can be passed in
      # 3.) Number of angles for the Radon transform used in creation of Psi if 1.) is False. Else this entry can be ignored
    def create_psi(self, importing_psi):
        if not importing_psi[0]:
            self.Psi = create_PSI(*np.shape(self.I), importing_psi[2])
        else:
            self.Psi = importing_psi[1]  
            
    #Add noise to signal y at set SNR_dB
    def add_noise(self, SNR_dB, y):
        snr = 10.0**(SNR_dB/10.0)
        sig_pwr = np.linalg.norm(y)**2/y.size
        noise_pwr = sig_pwr/snr
        noise = np.sqrt(noise_pwr)*np.random.normal(0, 1, np.shape(y))
        return y + noise                  

### Example Using the above class ###

#img_size = 32
#B = create_measurements('barbara.png', [[154, 154+img_size],[64,64+img_size]], import_phi = (False, [], 'bior1.1'), import_psi = (False, [], 15))
# F = create_measurements('flower.jfif', [[262, 262+64], [1128, 1128+64]], import_phi = (True, B.Phi), import_psi = (True, B.Psi))





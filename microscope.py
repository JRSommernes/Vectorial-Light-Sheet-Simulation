from scipy.optimize import curve_fit
from other_functions import *
import numpy as np
import tifffile
import json
import os

################################################################################
#Below is a test simulation not requiring the GUI. To alter the test system,
#alter the code in the functions "make_system" and "run_simulation".
################################################################################

def make_system(path,ex,em,alpha,O3='Glass'):
    """Creates the system and adds the optical path.

    Parameters
    ----------
    path : String
        Determines the save location of the simulation files
    ex : Float
        Excitation wavelength
    em : Float
        Emission wavelength
    alpha : Float
        Rotation of the sample plane
    O3 : String
        Immersion medium ['Glass', 'Water', or 'Air']

    Returns
    -------
    Class
        microscope class
    """
    #Creates microscope class
    system = Microscope(ex,em,path) 

    #Lens 1
    NA_1 = 1.35
    RI_1 = 1.4
    system.add_lens(NA_1,RI_1)

    #Lens 2
    NA_2 = 0.025
    system.add_lens(NA_2)

    #Lens 4
    NA_4 = 0.95
    rot_4 = alpha
    # rot_4 = 0
    throughput_4 = np.array([[2.21057481e-24,1.73931774e-11,-8.22755585e-21,-1.88712388e-06,3.83174080e-18,9.79322936e-01],[1.45298466e-24,-1.73285965e-10,-6.69094419e-21,-5.33563605e-05,6.53859519e-18,9.91885292e-01]])
    system.add_lens(NA_4,rot=rot_4,thr=throughput_4)

    #Lens 3
    NA_3 = NA_2*(RI_1*NA_4)/(NA_1)
    system.add_lens(NA_3,pos=2)

    #Lens 5
    if O3 == 'Glass':
        NA_5 = 1
        RI_5 = 1.7
        throughput_5 = np.array([5.42858373e-29,-1.52269005e-12,-1.34761517e-25,1.60371526e-09,-1.35290597e-21,-3.77445778e-06,2.83736542e-18,9.88848755e-01])
    elif O3 == 'Water':
        NA_5 = 1
        RI_5 = 1.33
        throughput_5 = None
    elif O3 == 'Air':
        NA_5 = 0.5
        RI_5 = 1
        throughput_5 = None
    system.add_lens(NA_5,RI_5,thr=throughput_5)

    #Lens 6
    NA_6 = NA_5/40
    system.add_lens(NA_6)
    
    return system

def run_simulation():
    """Determines the parameters of the system and runs the simulation.
    """
    
    path = 'lens_throughput'
    excitation = 488e-9 #Excitation wavelength
    emission = 507e-9 #Emission wavelength
    alpha = 30*np.pi/180 #Rotation of the sample plane
    O3 = 'Glass' #Immersion medium ['Glass', 'Water', or 'Air']

    #Adds lenses to the system
    system = make_system(path,excitation,emission,alpha,O3) 

    res = 128 #Camera resolution
    vox = 2e-6 #Camera voxel size
    bias_offset = 100 #Camera bias offset
    RMS = 1.4 #Camera readout noise

    #Defines camera configuration
    system.add_camera(res,vox,bias_offset,RMS) 

    system.ensamble = 10 #Number of dipoles in ensamble
    system.OTF_res = 256 #Size of OTF in pixels
    system.ls_pol = 'p' #Ls polarization ['p', 's', or 'u']
    system.anisotropy = 0.4 #Anisotropy [0, or 0.4]
    system.ls_opening = 15*np.pi/180 #Ls opening half-angle in degrees
    system.SNR = 100 #Signal to noise ratio for poisson noise

    #Runs the simulation
    system.calculate_system_specs()
    system.calculate_PSF()
    #Calculates the MTF and analyzes is
    system.make_MTF()
    system.save_stacks()
    system.save_data()

################################################################################
#The simulation code starts below. Alter at your own risk.
################################################################################

class Lens:
    """ Class containing lens parameters. Also contains
        functions for focusing and collimating a beam.
    """
    def __init__(self,NA,RI,rot,thr):
        """Initialization of the lens class

        Parameters
        ----------
        NA : Float
            Numerical aperture of the lens
        RI : Float
            Refractive index of the medium in image space
        rot : Float
            Rotation between the current and new optical axis
        """
        self.NA = NA    #Numerical aperture
        self.RI = RI    #Refractive index
        self.rot = rot  #Rotation of lens
        self.thr = thr  #Light throughput of lens

    def collimating(self,theta,phi,dir):
        """Function for transform matrix
           corresponding to a collimating lens.

        Parameters
        ----------
        theta : float64 numpy array
            Polar angles of rays before lens refraction
        phi : float64 numpy array
            Azimuthal angles of rays before lens refraction
        dir : int
            Direction of of ray rotation in lens refraction.
            Direction is given around the azimuthal axis.
        """
        transform = []
        #Lens refraction
        self.theta = theta
        self.phi = phi
        transform.append(L_refraction(dir*theta))

        #If the lens has a predetermined light throughout, use that
        if self.thr is not None:
            transform.append(R_y(dir*theta))
            transform.append(lens_transmission(theta,self.thr))
            transform.append(R_y(-1*dir*theta))

        #Rotation of the optical axis
        if self.rot != 0:
            Rx = R_x(-self.rot)
            ki = np.array((np.sin(theta)*np.cos(phi),
                            np.sin(theta)*np.sin(phi),
                            np.cos(theta)))
            kf = np.einsum('ij,jkl->ikl',Rx,ki)
            phi = (np.arctan2(kf[1], kf[0]))
            theta = np.arctan2(np.sqrt(kf[0]**2+kf[1]**2),kf[2])
            theta[kf[2]<0] = np.NaN
            transform.append(R_z(self.phi))
            transform.append(np.broadcast_to(R_x(self.rot),(len(phi),len(phi),3,3)))
            transform.append(np.linalg.inv(R_z(phi)))

        #Downstream ray angles
        self.theta_prev = theta
        self.phi_prev = phi
        #Effective lens matrix and apodization
        self.transform = multidot(np.array(transform))
        self.apodization = 1/(np.sqrt(np.cos(self.theta)/self.RI))
    
    def focusing(self,theta,phi,RI_next,thr_next,dir):
        """Function for transform matrix
           corresponding to a focusing lens.

        Parameters
        ----------
        theta : float64 numpy array
            Polar angles of rays before lens refraction
        phi : float64 numpy array
            Azimuthal angles of rays before lens refraction
        RI_next : Float
            Refractive index of the next lens in the system.
            If there is no next lens, input the same refractive 
            index as the current lens.
        dir : int
            Direction of of ray rotation in lens refraction.
            Direction is given around the azimuthal axis.
        """
        transform = []
        self.theta_next = theta
        self.phi_next = phi

        #If the lens has a predetermined light throughout, use that
        if self.thr is not None:
            transform.append(R_y(-1*dir*theta))
            transform.append(lens_transmission(theta,self.thr))
            transform.append(R_y(dir*theta))
        
        #Transmission through a refractive index change
        if RI_next != self.RI:
            theta = np.arcsin(RI_next*np.sin(theta)/self.RI)
            phi = phi
            transform.append(R_y(-1*dir*self.theta_next))
            if thr_next is None:
                transform.append(Fresnel(theta,self.theta_next,self.RI,RI_next))
            transform.append(R_y(dir*theta))
      
        #Rotation of the optical axis
        if self.rot != 0:
            Rx = R_x(-self.rot)
            ki = np.array((np.sin(theta)*np.cos(phi),
                            np.sin(theta)*np.sin(phi),
                            np.cos(theta)))
            kf = np.einsum('ij,jkl->ikl',Rx,ki)
            theta = np.arctan2(np.sqrt(kf[0]**2+kf[1]**2),kf[2])
            theta[kf[2]<0] = np.NaN
            transform.append(R_z(phi))
            phi = (np.arctan2(kf[1], kf[0]))
            transform.append(np.broadcast_to(R_x(self.rot),(len(phi),len(phi),3,3)))
            transform.append(np.linalg.inv(R_z(phi)))

        #Downstream ray angles
        self.theta = theta
        self.theta[theta>np.arcsin(self.NA/self.RI)] = np.NaN
        self.phi = phi
        #Lens refraction
        transform.append(L_refraction(dir*self.theta))

        #Effective lens matrix and apodization
        self.transform = multidot(np.array(transform))
        self.apodization = np.sqrt(np.cos(self.theta)/self.RI)

class Camera:
    """Class containin camera parameters

    Parameters
    ----------
    res : Integer
        Number of pixels on the camera chip in each axis
    vox : Float
        Distance between sampling points on camera chip

    Attributes
    ----------
    res
    vox

    """

    def __init__(self,res,vox,offset,RMS):
        self.res = res #Field of view [pixels]
        self.vox = vox #Voxel size [m]
        self.offset = offset
        self.RMS = RMS


class Microscope:
    """Class containing all simulation functions

    Parameters
    ----------
    lam_ex : Float
        Excitation wavelength
    lam_em : type
        Emission wavelength

    Attributes
    ----------
    lenses : List
        List of all lenses contained in the system
    k0 : Float
        Wavenumber of emission light
    lam_ex
    lam_em

    """
    def __init__(self,lam_ex,lam_em,path):
        self.lenses = []              #List of all lenses in system
        self.lam_ex = lam_ex          #Excitation wavelength [m]
        self.lam_em = lam_em          #Emission wavelength [m]
        self.k0 = 2*np.pi/self.lam_em #Emission wavenumber [rad/m]
        self.path = path              #Save folder location

    def add_lens(self,NA,RI=1,rot=0,thr=None,pos=False):
        """Adds lens to the system

        Parameters
        ----------
        NA : Float
            Numerical aperture of the lens
        RI : Float
            Refractive index in the image space
        rot : Float
            Rotation of the lens around the x-axis
        pos : Int
            Position of the lens in the setup

        """
        lens = Lens(NA,RI,rot,thr)
        if pos == False:
            self.lenses.append(lens)
        elif not type(pos) is int:
            raise TypeError('Lens position can only be integer')
        elif len(self.lenses) < pos:
            raise ValueError('Lens position out of range')
        else:
            self.lenses.insert(pos,lens)

    def add_camera(self,res,vox,offset,RMS):
        """Updates the camera of the system

        Parameters
        ----------
        res : Integer
            Number of pixels on the camera chip in each axis
        vox : Float
            Distance between sampling points on camera chip

        """
        if not type(res) is int:
            raise TypeError('Pixel count can only be integer')
        else:
            self.camera = Camera(res,vox,offset,RMS)

    def calculate_system_specs(self):
        """Calculates the system specifications

        Specs
        -------
        mag
            Transverse magnification
        alpha
            Rotation of the sample plane
        axial_mag
            Axial magnification
        z_voxel_size
            z-sampling in image space to get uniform
            sampling of the object space
        scaling
            Scaling constant used for the electric field evaluation

        """
        #Calculates system magnification and light-sheet rotation
        self.mag = 1
        self.alpha = np.pi/2-np.sum([lens.rot for lens in self.lenses])
        for i,lens in enumerate(self.lenses):
            if i % 2 == 0:
                self.mag *= lens.NA
            else:
                self.mag /= lens.NA
        self.axial_mag = self.mag**2*self.lenses[-1].RI/self.lenses[0].RI
        self.z_voxel_size = self.camera.vox/self.mag*self.axial_mag
        self.scaling = self.lam_em/(2*self.camera.vox*self.lenses[-1].NA)

        const = self.lam_em*self.mag/(4*self.lenses[0].NA)
        if self.camera.vox >= const:
            print('Warning: Voxel size too large for accurate simulations')

        #Clculates system FoV in sample space
        if hasattr(self, 'camera'):
            self.FoV = self.camera.vox*self.camera.res/self.mag

    def light_sheet(self):
        """Calculates the light-sheet of the system. The light-sheet is assumed
           to be generated by uniform polarized input light entering the back
           focal plane of the illumination objective. The light is passed
           through a mask and focused by the objective.

        """

        #Defines the resolution of the electric field
        res = self.camera.res
        M = res//2
        x = y = np.linspace(-M,M,res)
        xx,yy = np.meshgrid(x,y)
        RR = np.sqrt(xx**2+yy**2)

        #Calculates the wavenumber proportiones by the NA of the light-sheet
        NA = self.lenses[0].RI*np.sin(self.ls_opening)
        del_K = self.k0*NA/M
        k_xy = del_K*RR
        k_z = np.sqrt((self.k0*self.lenses[0].RI)**2 - k_xy**2)

        #Defines spherical coordinates of the lens
        theta = np.arcsin((del_K/(self.k0*self.lenses[0].RI))*RR)
        phi = np.arctan2(yy,xx)

        #Defines the initial electric field in the fourier space of the lens
        if self.ls_pol == 'p':
            self.Ei_base = np.array((1,0,0))
        elif self.ls_pol == 's':
            self.Ei_base = np.array((0,1,0))
        elif self.ls_pol == 'u':
            self.Ei_base = np.array((np.sqrt(2)/2,np.sqrt(2)/2,0))
        Ei = np.ones((res,res,3))*self.Ei_base

        #Defines the mask shaping the light in the back focal plane and
        #multiplies it with the initial electric field
        mask = np.ones_like(theta)
        mask[theta>self.ls_opening] = np.NaN
        mask[:,:M] = np.NaN
        mask[:,M+1:] = np.NaN
        Ei *= mask.reshape(res,res,1)

        #Calculates the coordinate transform the lens induces
        #and transform the electric field
        jones_mat = [np.linalg.inv(R_z(phi))]
        jones_mat.append(L_refraction(-theta))
        jones_mat.append(R_z(phi))
        jones_mat = np.array(jones_mat)
        transform = multidot(jones_mat)
        apodization = np.sqrt(np.cos(theta)/self.lenses[0].RI).reshape(res,res,1)
        Ef = apodization*np.nan_to_num(np.einsum('abji,abi->abj', transform, Ei))

        #Defines the back aperture obliqueness of the electric field
        bao = np.nan_to_num(1 / np.cos(theta))

        #Calculate the z-sampling to get uniform sampling of the object space
        z_max = res//2 * self.camera.vox/self.mag
        z_val = np.linspace(-z_max,z_max,res)

        #Performs the Debye integral to evaluate the
        #electric field (usong a Fourier transform)
        scaling = self.lam_ex/(2*self.camera.vox/self.mag*NA)
        k_z = np.nan_to_num(k_z)
        self.ls_PSF = dft2_volume(Ef,k_z,z_val,bao,res,scaling).transpose(1,2,0)

    def field_trace(self):
        #Defines the resolution of the electric field trace
        M = self.camera.res//2
        x = y = np.linspace(-M,M,self.camera.res)
        xx,yy = np.meshgrid(x,y)
        RR = np.sqrt(xx**2+yy**2)

        rotation_direction = np.flip((-1)**(np.arange(len(self.lenses))//2%2+1))
        
        del_K = self.k0*self.lenses[-1].NA/M
        theta_tmp = np.arcsin((del_K/(self.k0*self.lenses[-1].RI))*RR)
        phi_tmp = np.arctan2(yy,xx)
        RI_next = self.lenses[-1].RI
        thr_next = None

        #Iterates through the lenses and append the coordinate transform
        #of the lens to a list
        trans = [np.linalg.inv(R_z(phi_tmp))]
        self.apodization = np.ones_like(phi_tmp)
        for i,lens in enumerate(list(reversed(self.lenses))):
            #Determines if the lens is focusing or collimating
            dir = rotation_direction[i]
            if i%2 == 0:
                lens.focusing(theta_tmp,phi_tmp,RI_next,thr_next,dir)
                pupil = lens.RI*np.sin(lens.theta)/lens.NA
                phi_tmp = lens.phi
            if i%2 == 1:
                theta_tmp = np.arcsin(lens.NA*pupil/lens.RI)
                lens.collimating(theta_tmp,phi_tmp,dir)
                theta_tmp = lens.theta_prev
                phi_tmp = lens.phi_prev
                RI_next = lens.RI
                thr_next = lens.thr

            trans.append(lens.transform)
            self.apodization *= lens.apodization

        trans.append(R_z(self.lenses[0].phi_prev))
        self.transform = multidot(np.array(trans))

    def make_MTF(self):
        """Calculates the MTF of the system using a Fourier transform.

        """
        #Zero pads the effective PSF to the desired OTF resolution to get the
        #desired base frequency
        padding = (self.OTF_res-self.camera.res)//2
        product = np.pad(self.eff_PSF,padding)

        #Adds noise to the PSF
        product, poisson = add_noise(product,self.SNR**2,self.camera.offset,self.camera.RMS)

        #Snips out the effective PSF with noise to save
        self.PSF_poisson = poisson[padding:padding+self.camera.res,
                                   padding:padding+self.camera.res,
                                   padding:padding+self.camera.res]
        self.PSF_readout = product[padding:padding+self.camera.res,
                                   padding:padding+self.camera.res,
                                   padding:padding+self.camera.res]

        #Fourier transform the effective PSF to get the MTF
        OTF_noiseless = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(product)))
        OTF_poisson = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(poisson)))
        OTF_readout = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(product)))
        self.MTF_noiseless = np.abs(OTF_noiseless)
        self.MTF_poisson = np.abs(OTF_poisson)
        self.MTF_readout = np.abs(OTF_readout)

        #Calculates the base frequency of the MTF
        self.base_freq = 1/(len(self.MTF_readout)*(self.camera.vox/(self.mag)))

    def save_stacks(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        with tifffile.TiffWriter(self.path+'/PSF.tiff') as stack:
            stack.write(img16(self.PSF).transpose(2,1,0),contiguous=True)
        with tifffile.TiffWriter(self.path+'/PSF_effective.tiff') as stack:
            stack.write(img16(self.eff_PSF).transpose(2,1,0),contiguous=True)
        with tifffile.TiffWriter(self.path+'/PSF_poisson.tiff') as stack:
            stack.write(self.PSF_poisson.transpose(2,1,0),contiguous=True)
        with tifffile.TiffWriter(self.path+'/PSF_readout.tiff') as stack:
            stack.write(self.PSF_readout.transpose(2,1,0),contiguous=True)
        with tifffile.TiffWriter(self.path+'/MTF_noiseless.tiff') as stack:
            stack.write(self.MTF_noiseless.transpose(2,1,0).astype(np.float32),contiguous=True)
        with tifffile.TiffWriter(self.path+'/MTF_poisson.tiff') as stack:
            stack.write(self.MTF_poisson.transpose(2,1,0).astype(np.float32),contiguous=True)
        with tifffile.TiffWriter(self.path+'/MTF_readout.tiff') as stack:
            stack.write(self.MTF_readout.transpose(2,1,0).astype(np.float32),contiguous=True)

    def analyze(self):
            """Automated analysis script used to extract resolution from the MTF.

            """
            #Define the half length of the MTF
            N = len(self.MTF_poisson)//2
            #Subtracts the square root of the DC term from the poisson MTF
            #to find the noise floor of the Fourier transform
            poisson = self.MTF_poisson-np.sqrt(self.MTF_poisson.max())
            poisson[poisson<0] = 0

            #Evil statistical magic to find the predicted noise floor of the readout MTF
            readout_noise = np.sqrt(self.MTF_readout.max()-self.camera.offset*(2*N)**3+self.camera.RMS**2*(2*N)**3)
            readout = self.MTF_readout-readout_noise
            readout[readout<0] = 0

            xp = poisson[N,N:,N]
            XP_cut = np.where(xp<=0)[0][0]
            XP_res = 1/(self.base_freq*XP_cut/1e9)

            yp = poisson[N:,N,N]
            YP_cut = np.where(yp<=0)[0][0]
            YP_res = 1/(self.base_freq*YP_cut/1e9)

            zp = poisson[N,N,N:]
            ZP_cut = np.where(zp<=0)[0][0]
            ZP_res = 1/(self.base_freq*ZP_cut/1e9)

            xr = readout[N,N:,N]
            XR_cut = np.where(xr<=0)[0][0]
            XR_res = 1/(self.base_freq*XR_cut/1e9)

            yr = readout[N:,N,N]
            YR_cut = np.where(yr<=0)[0][0]
            YR_res = 1/(self.base_freq*YR_cut/1e9)

            zr = readout[N,N,N:]
            ZR_cut = np.where(zr<=0)[0][0]
            ZR_res = 1/(self.base_freq*ZR_cut/1e9)

            self.XYZ_res_poisson = np.array([XP_res,YP_res,ZP_res])
            self.XYZ_res_readout = np.array([XR_res,YR_res,ZR_res])

    def FWHM_measurement(self):
        """Script to find the FWHM of the PSF

        """
        PSF = self.PSF_readout - self.PSF_readout.min()

        #Define the axis lines of the PSF
        xx = PSF[self.camera.res//2,:,self.camera.res//2]
        yy = PSF[:,self.camera.res//2,self.camera.res//2]
        zz = PSF[self.camera.res//2,self.camera.res//2,:]

        guess = np.array((xx.max(), 0, self.lam_ex/(2*self.lenses[0].NA)))

        x = np.linspace(-self.FoV/2,self.FoV/2,self.camera.res)
        x_fit, _ = curve_fit(gaussian, x, xx, p0=guess)
        y_fit, _ = curve_fit(gaussian, x, yy, p0=guess)
        z_fit, _ = curve_fit(gaussian, x, zz, p0=guess)

        _, _, x_sigma = x_fit
        _, _, y_sigma = y_fit
        _, _, z_sigma = z_fit

        self.FWHM = np.array((x_sigma,y_sigma,z_sigma))*2*np.sqrt(2*np.log(2))

    def save_data(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        readout_SNR = self.SNR**2/np.sqrt(self.SNR**2+self.camera.RMS**2)

        data = {'Anisotrpy' : self.anisotropy,
                'Dipoles in ensamble' : self.ensamble,
                'Emission wavelength [nm]' : np.round(self.lam_em*1e9,2),
                'Excitation wavelength [nm]' : np.round(self.lam_ex*1e9,2),
                'Full FoV [pixels]' : self.camera.res,
                'Full FoV in object space [microns]' : self.FoV*1e6,
                'Lens NA\'s' : [lens.NA for lens in self.lenses],
                'Lens immersion RI\'s' : [lens.RI for lens in self.lenses],
                'Light sheet angle [degrees]' : 90-np.round(self.alpha*180/np.pi,1),
                'Light sheet opening [degrees]' : np.round(self.ls_opening*180/np.pi),
                'Light sheet polarization' : self.ls_pol,
                'Magnification transverse' : self.mag,
                'Magnification axial' : self.axial_mag,
                'MTF base frequency' : self.base_freq,
                'MTF size [pixels]' : self.OTF_res,
                'Optical efficiency' : self.tti,
                'SNR poisson'   :   self.SNR,
                'SNR readout'   :   readout_SNR,
                'Voxel size [microns]' : self.camera.vox*1e6}
    
        try:
            self.analyze()
            res = {'X_res_poisson [nm]' : self.XYZ_res_poisson[0],
                   'Y_res_poisson [nm]' : self.XYZ_res_poisson[1],
                   'Z_res_poisson [nm]' : self.XYZ_res_poisson[2],
                   'X_res_readout [nm]' : self.XYZ_res_readout[0],
                   'Y_res_readout [nm]' : self.XYZ_res_readout[1],
                   'Z_res_readout [nm]' : self.XYZ_res_readout[2]}
            data = data|res
        except:
            print('Auto analyze failed')

        try:
            self.FWHM_measurement()
            FWHM = {'X_FWHM [nm]' : self.FWHM[0]*1e9,
                    'Y_FWHM [nm]' : self.FWHM[1]*1e9,
                    'Z_FWHM [nm]' : self.FWHM[2]*1e9}
            data = data|FWHM
        except:
            print('FWHM could not be found')

        with open(self.path+'/data.json', 'w') as output:
            json.dump(data, output, indent=4)

        self.data = data

    def calculate_PSF(self,pbar=None):
        """Main function that creates the system PSF and MTF

        Parameters
        ----------
        GUI : Class
            PyQt5 GUI class

        """
        #Simplifies the res to shorten the lines
        res = self.camera.res

        #Traces the system and light-sheet
        self.field_trace()
        self.light_sheet()

        #Defines the z-sampling in image space
        z_max = self.z_voxel_size*res/2
        z_val = np.linspace(-z_max,z_max,res)

        #See the tracing funtion for explanation of the same code block
        M = res//2
        x = y = np.linspace(-M,M,res)
        xx,yy = np.meshgrid(x,y)
        RR = np.sqrt(xx**2+yy**2)
        delta_k = self.k0*self.lenses[-1].NA/(res//2)
        k_xy = delta_k*RR
        k_z = np.sqrt(self.k0**2 - k_xy**2)
        k_z = np.nan_to_num(k_z)

        #Back aperture obliqueness of the last lens in the system
        bao = np.nan_to_num(1 / np.cos(self.lenses[-1].theta))

        #Generates a dipole ensamble using a Fibonacci lattice
        phi,theta = make_pol(self.ensamble)

        #Calculates average light-sheet polarization in image space
        l_p = np.array(((np.cos(self.alpha), 0, -np.sin(self.alpha)),
                        (0, 1, 0),
                        (np.sin(self.alpha), 0, np.cos(self.alpha))))@self.Ei_base

        #Empty arrays for storing total transmitted intensity, dipole
        #excitation coeficcient for all dipoles in ensamble, and the PSF
        tti = []
        ani = []
        self.PSF = np.zeros((res,res,res))
        #Iterates through the ensamble
        for i in range(len(phi)):
            #If the user uses the GUI, the load bar is directed to the GUI. If
            #the user simulates from the terminal, the loadbar is directed to
            #the terminal
            try:
                pbar.emit(100*(i+1)//len(phi))
            except:
                loadbar(i,len(phi))

            #Current dipole polarization
            dip_th = theta[i]
            dip_ph = phi[i]
            pol = np.array((np.sin(dip_th)*np.cos(dip_ph),
                            np.sin(dip_th)*np.sin(dip_ph),
                            np.cos(dip_th)))

            #Excitation coeficcient is calculated
            if self.anisotropy == 0:
                Ae = 1
            elif self.anisotropy == 0.4:
                Ae = np.abs(pol@l_p)
            ani.append(collected_field(pol,np.nanmax(self.lenses[0].theta))*Ae)

            #Calculates the initial and final field
            Ei = E_0(pol, self.lenses[0].phi, self.lenses[0].theta, Ae)
            Ef = np.nan_to_num(self.apodization.reshape(res,res,1)*np.einsum('abji,abi->abj', self.transform, Ei))
            

            #Finds the electric field in the pupil of O2 and O3
            P2 = multidot(np.array([self.lenses[2].transform,self.lenses[1].transform,
                                                   self.lenses[0].transform,R_z(self.lenses[0].phi)]))
            P3 = multidot(np.array([np.linalg.inv(self.lenses[-1].transform),
                                                   R_z(self.lenses[-1].phi),
                                                   self.transform]))
            E_z_1 = np.nan_to_num(np.einsum('abji,abi->abj', P2, Ei))
            E_z_2 = np.nan_to_num(np.einsum('abji,abi->abj', P3, Ei))

            #Calculates the total transmitted field intensity
            initial_intensity = np.sum(np.abs(E_z_1)**2)
            final_intensity = np.sum(np.abs(E_z_2)**2)
            tti.append(final_intensity/initial_intensity)

            #Calculates the PSF based on the final field
            self.PSF += dft2_volume(Ef,k_z,z_val,bao,res,self.scaling)

        #Calculates the effective PSF
        self.eff_PSF = (self.PSF/self.PSF.max())*(self.ls_PSF/self.ls_PSF.max())

        #Calculates the optical eficciency of the system configuration
        throughput = np.array(ani)*np.array(tti)
        self.tti = np.mean(throughput)

##############################################################
#This function is executed if the current file is run directly
if __name__ == '__main__':
    #Runs the test code given in the beginning of the file
    run_simulation() 

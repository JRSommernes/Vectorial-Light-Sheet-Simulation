from numba import njit
import numpy as np
import sys

@njit
def dft2(X, k, a, N):
    """Fourier transform implementation
       Source: https://github.com/jdmanton/debye_diffraction_code

    """
    f1 = np.linspace(-1 / (2 * a[0]), 1 / (2 * a[0]), N[0]) - k[0] / X.shape[0]
    f1 = f1.reshape((-1, 1))
    f2 = np.linspace(-1 / (2 * a[1]), 1 / (2 * a[1]), N[1]) - k[1] / X.shape[1]
    x1 = np.arange(0, X.shape[0])
    x2 = np.arange(0, X.shape[1])
    x2 = x2.reshape((-1, 1))
    F1 = np.exp(-1j * 2 * np.pi * f1 * x1)
    F2 = np.exp(-1j * 2 * np.pi * x2 * f2)
    Xhat = np.dot(F1,np.dot(X,F2))
    return Xhat

@njit(parallel=True)
def dft2_volume(Ef,k_z,z_val,bao,res,scaling):
    """Funtion to fourier transform a volume in the transverse direction

    Parameters
    ----------
    Ef : Floating point array
        Electric field to be evaluated
    k_z : Floating point array
        z-contribution of the wavenumber
    z_val : Floating point vector
        sampling points in the optical axis
    bao : Floating point array
        The obliquness of the rays
    res : Integer
        Number of sampling points in each axis of the PSF
    scaling : Float
        Scaling coeficcient used in the dft2 function

    """
    PSF = np.zeros((res,res,res))
    for i in range(res):
        #Adding the diffraction pattens to the electric field
        field_x = np.exp(1j * k_z * z_val[i]) * bao * Ef[:,:,0]
        field_y = np.exp(1j * k_z * z_val[i]) * bao * Ef[:,:,1]
        field_z = np.exp(1j * k_z * z_val[i]) * bao * Ef[:,:,2]

        #Fourier transforming every field component separately
        field_x = dft2(field_x, np.array([0, 0]), np.array((scaling,scaling)), np.array((res,res)))
        field_y = dft2(field_y, np.array([0, 0]), np.array((scaling,scaling)), np.array((res,res)))
        field_z = dft2(field_z, np.array([0, 0]), np.array((scaling,scaling)), np.array((res,res)))

        #Calculating the PSF of the slice
        PSF[:,:,i] = np.abs(field_x)**2 + np.abs(field_y)**2 + np.abs(field_z)**2
    return PSF

def multidot(arr):
    """Dot product of multiple arrays.

    Parameters
    ----------
    arr : Array
        Numpy array of shape (M,N,N,3,3) where M is the number of matrices to
        be dotted, and N is the number of sampling points in each axis.

    Returns
    -------
    Array
        Transform matrix of shape (N,N,3,3)

    """
    num,res,a,b,c = arr.shape
    out = np.zeros_like(arr[0])
    for i in range(res):
        for j in range(res):
            tmp = arr[0,i,j]@arr[1,i,j]
            for k in range(2,num):
                tmp = tmp@arr[k,i,j]
            out[i,j] = tmp

    return out

def make_pol(timepoints):
    """Using a Fibonacci lattice to evenly distribute dipole orientations
       in the ensamble

    Parameters
    ----------
    timepoints : Integer
        Number of dipoles in the ensamble

    Returns
    -------
    Array
        Orientation of the dipole in phi and theta direction

    """
    N_sensors = timepoints
    sensors = np.zeros((N_sensors,2))
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(N_sensors):
        y = (1 - (i / float(N_sensors - 1)) * 2)  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        ph = np.arctan2(y, x)
        th = np.arctan2(np.sqrt(x**2+y**2),z)

        sensors[i] = (ph,th)

    phi = sensors[:,0]
    theta = sensors[:,1]

    return phi,theta

def collected_field(pol,theta_max):
    """Calculates the relative dipole emission collected by a aperture

    Parameters
    ----------
    pol : Array
        Polarization of the dipole emitter
    theta_max : Float
        Half angle of the acceptance cone of the aperture

    Returns
    -------
    Float
        The ammount of the emission collected by the aperture

    """
    N_sensors = 1000
    sensors = np.zeros((N_sensors,2))
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(N_sensors):
        y = (1 - (i / float(N_sensors - 1)) * 2)  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        ph = np.arctan2(y, x)
        th = np.arctan2(np.sqrt(x**2+y**2),z)

        sensors[i] = (ph,th)

    phi,theta = np.meshgrid(sensors[:,0],sensors[:,1])

    field = E_0(pol,phi,theta,1)
    used_field = np.copy(field)
    used_field[theta>theta_max] = 0

    total_field = np.sum(np.abs(field)**2)
    total_used_field = np.sum(np.abs(used_field)**2)

    return total_used_field/total_field

def loadbar(counter,len):
    #Its just a loadbar
    counter +=1
    done = (counter*100)//len
    sys.stdout.write('\r')
    sys.stdout.write("[%-100s] %d%%" % ('='*done, done))
    sys.stdout.flush()
    if counter == len:
        print('\n')

def img16(image):
    """Function to convert a floating point array to a 16bit array

    Parameters
    ----------
    image : Array
        Floating point array

    Returns
    -------
    Array
        16bit array

    """
    image = image.astype(np.float64)
    image /= np.amax(image)
    img_16 = ((2**16-1)*image).astype(np.uint16)
    return img_16

def poisson_noise(image,SNR):
    """Function to add poisson noise to a N-D stack

    Parameters
    ----------
    image : Array
        Noiseless data
    SNR : Float
        Signal to noise ratio

    Returns
    -------
    Array
        Noisy array

    """
    image *= SNR**2
    noisy = np.random.poisson(image)

    return noisy

def add_noise(image,photon_count,cam_offset=100,cam_sigma=2):
    """Function to add noise to a N-D stack

    Parameters
    ----------
    image : Array
        Noiseless data
    SNR : Float
        Signal to noise ratio

    Returns
    -------
    Array
        Noisy array

    """
    poisson = poisson_noise(image,np.sqrt(photon_count))
    gaussian_background = np.random.normal(cam_offset, cam_sigma, image.shape)
    image = (poisson+gaussian_background).astype(np.uint16)

    return image, poisson, gaussian_background

def calculate_histogram(stack):
    """Calculates and returns the histogram of a N-D array

    Parameters
    ----------
    stack : Array
        Data to be turned into histogram

    Returns
    -------
    Array
        Vectors containing the histogram and intensity values of the histogram

    """
    M = len(stack)
    stack[stack==0] = 1
    hist,values = np.histogram(np.log(stack), bins=200)
    values = values[:-1]

    return hist, values

def gaussian(x, a, x0, sigma):
    """Gaussian function

    Parameters
    ----------
    x : Float or array of floats
        Position to be calculated
    a : Float
        This is the scaling factor of the gaussian
    x0 : Float
        Mean value of the gaussian
    sigma : Float
        Standard deviation

    Returns
    -------
    Float
        Value of the Gaussian at point x

    """
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def smooth_histogram(stack):
    """Function to smooth out the histogram made above. Used by convolving the
       histogram with a gaussian distribution

    Parameters
    ----------
    stack : Array
        Array to be turned into histogram

    Returns
    -------
    Array
        Smooth histogram and intensity values

    """
    count,val = calculate_histogram(stack)

    N = 31
    sig = 4
    ax = np.linspace(-(N-1)/2,(N-1)/2,N)
    gauss = np.exp(-0.5 * ax**2 / sig**2)
    gauss /= np.sum(gauss)

    count_smooth = np.convolve(count,gauss,mode='same')

    return val,count_smooth

def R_x(alpha):
    """Making the coordinate rotation matrix for clockwise x-rotation

    Parameters
    ----------
    alpha : floating point array
        Rotation angle in radians

    Returns
    -------
    floating point array
        Complete rotation matrix
    """
    zero = np.zeros_like(alpha)
    one = np.ones_like(alpha)
    return np.array(((one, zero, zero),
                    (zero, np.cos(alpha), -np.sin(alpha)),
                    (zero, np.sin(alpha), np.cos(alpha))))

def R_y(alpha):
    """Making the coordinate rotation matrix for clockwise y-rotation

    Parameters
    ----------
    alpha : floating point array
        Rotation angle in radians

    Returns
    -------
    floating point array
        Complete rotation matrix
    """
    zero = np.zeros_like(alpha)
    one = np.ones_like(alpha)
    return np.array(((np.cos(alpha), zero, -np.sin(alpha)),
                     (zero, one, zero),
                     (np.sin(alpha), zero, np.cos(alpha)))).transpose(2,3,0,1)

def R_z(alpha):
    """Making the coordinate rotation matrix for clockwise z-rotation

    Parameters
    ----------
    alpha : floating point array
        Rotation angle in radians

    Returns
    -------
    floating point array
        Complete rotation matrix
    """
    zero = np.zeros_like(alpha)
    one = np.ones_like(alpha)
    return np.array(((np.cos(alpha), np.sin(alpha), zero),
                     (-np.sin(alpha), np.cos(alpha), zero),
                     (zero, zero, one))).transpose(2,3,0,1)

def Fresnel(th_i,th_f,RI_i,RI_f):
    """Making the Fresnel transmission matrix

    Parameters
    ----------
    tp : floating point array
        Parallel transmission coefficient
    ts : floating point array
        Sagittal transmission coefficient

    Returns
    -------
    floating point array
        Complete Fresnel matrix
    """
    tp = 2*RI_i*np.cos(th_i)/(RI_f*np.cos(th_i)+RI_i*np.cos(th_f))
    ts = 2*RI_i*np.cos(th_i)/(RI_i*np.cos(th_i)+RI_f*np.cos(th_f))
    Tp = np.abs(tp)**2*(RI_f*np.cos(th_f))/(RI_i*np.cos(th_i))
    Ts = np.abs(ts)**2*(RI_f*np.cos(th_f))/(RI_i*np.cos(th_i))
    zero = np.zeros_like(tp)
    one = np.ones_like(tp)
    return np.array(((Tp, zero, zero),
                     (zero, Ts, zero),
                     (zero, zero, one))).transpose(2,3,0,1)

def L_refraction(theta):
    """Making the ray refraction matrix for the meridional plane

    Parameters
    ----------
    theta : floating point array
        Refraction angle in radians

    Returns
    -------
    floating point array
        Complete refraction matrix
    """
    zero = np.zeros_like(theta)
    one = np.ones_like(theta)
    return np.array(((np.cos(theta),zero,np.sin(theta)),
                     (zero,one,zero),
                     (-np.sin(theta),zero,np.cos(theta)))).transpose(2,3,0,1)

def k_0(phi, theta):
    """Generating x-, y-, and z-component of k0 based on lens position

    Parameters
    ----------
    phi : floating point array
        Azimuthal angle on lens
    theta : floating point array
        Polar angle on lens

    Returns
    -------
    floating point array
        k0 in Cartesian coordinates
    """
    return np.array((np.sin(theta)*np.cos(phi),
                     np.sin(theta)*np.sin(phi),
                     np.cos(theta)))

def E_0(p, phi, theta, Ae):
    """Generating the initial electric field based
       on dipole orientation and anisotropy

    Parameters
    ----------
    p : floating point array
        Polarization of dipole in Cartesian coordinates
    phi : floating point array
        Lens azimuth angle
    theta : floating point array
        Lens polar angle
    Ae : float
        Anisotropy coefficient

    Returns
    -------
    floating point array
        Initial electric field
    """
    k0 = np.transpose(k_0(phi,theta),(1,2,0))
    return Ae*np.cross(np.cross(k0,p),k0)

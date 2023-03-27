# VectorialLightSheetSimulation

This repository is a supplementary software dataset for an article "S-polarized light-sheets improve resolution and light-efficiency in oblique plane microscopy" (link below).

The software is written to simulate and analyze light-sheet imaging systems while preserving polarization information for an ensemble of dipoles. To trace the emission of a dipole while preserving the polarization information, the system is modelled using vectorial raytracing with non-paraxial 3x3 Jones matrices. To find the PSF, the field is then evaluated using vectorial Debye diffraction integrals, here implemented using a Fourier transform. Evaluating the field at different offsets from the focus, the 3D PSF is calculated. This can be done for an ensemble of dipoles of arbitrary size. Using the traced field, the optical power loss due to refractive index changes can also be found. The OTF is then automatically evaluated to find the x-, y-, and z-resolution of the system. This is done by finding a cutoff power in the OTF, then finding the extent of the OTF in each axis. For each lens in the system, the user can define the: NA, immersion medium refractive index, and lens rotation around the x-axis. The user can also define the: camera grid size, voxel size, OTF sampling, light-sheet NA, light-sheet polarization, excitation and emission wavelength, number of dipoles in the ensemble, and signal to noise ratio.

# Links

[Corresponding article](https://arxiv.org/abs/2303.14018)

[Dataverse repository](https://dataverse.no/dataset.xhtml?persistentId=doi:10.18710/YAYNNL&version=DRAFT)

# Prerequisites

* Python 3.9.6 or above
* Python packages
	- Numpy 1.12.3
	- Numba 0.53.1
	- Matplotlib 3.5.2
	- PyQt5 5.15.4
	- Tifffile 2021.8.8
	- Scipy 1.7.1

# How to use

1. Run main.py
2. Create system
	* Create lenses
		- Specify numerical aperture
		- Specify refractive index of immersion medium
		- Specify lens rotation around the x-axis
		- Specify the position of the lens
	* Create camera
		- Specify number of camera pixels
		- Specify voxel size of the pixels
		- Specify camera readout noise
		- Specify camera base pixel count
		- Specify OTF sampling points
	* Make light-sheet
		- Specify opening angle of light-sheet
		- Specify excitation wavelength
		- Specify excitation polarization
	* Define sample and trace
		- Specify the number of dipole orientations in ensemble
		- Specify emission wavelength
		- Specify if sample anisotropy
		- Define path for file saving
		- Begin trace
3. Wait for trace to finish

	When hitting the trace button, the program might seemingly freeze. Give it a moment to do its thing, and a loadbar will show up. When the loadbar is at 100%, give it some more time, the button text will swap from "Working" to "Done!". Depending on the number of camera pixels and the size of the OTF, the trace time and the time before and after the loadbar might vary.

4. See the results

	Go to the save directory, and behold the results. In here should be a series of files:
	* PSF.tiff
	
		This is the PSF of the dipole ensamble after passing through the system.
		
	* PSF_effective.tiff
	
		This is the effective PSF of the system when taking the light-sheet into consideration.
		
	* PSF_poisson.tiff
	
		This is the PSF after factoring in Poisson noise (shot noise).
		
	* PSF_readout.tiff
	
		This is the PSF after factoring in Poisson and readout noise
		
	* MTF_noiseless.tiff
	
		This is the Fourier transform of the effective PSF
		
	* MTF_poisson.tiff
	
		This is the Fourier transform of the PSF with Poisson noise
		
	* MTF_readout.tiff
	
		This is the Fourier transform of the PSF with readout noise
		
	* data.json
	
		The data file includes system parameters as well as the auto-generated PSF evaluation
		
# Help

For questions about the software or its use, please contact the lead author Jon-Richard Sommernes (jon-richard.sommernes@uit.no), or corresponding author Florian Ströhl (florian.strohl@uit.no).

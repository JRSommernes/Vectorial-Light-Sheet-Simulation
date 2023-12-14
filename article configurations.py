import numpy as np
from microscope import *

def find_configs(SNR,ls_incl,pol,O3):
    configs = np.array(np.meshgrid(SNR,ls_incl,pol,O3)).T.reshape(-1,4)
    return configs

def simulate(path,config,constants,noise_samples):
    alpha = float(config[1])*np.pi/180
    O3 = config[3]

    ex = constants[0]
    em = constants[1]
    res = constants[2]
    vox = constants[3]
    bias_offset = constants[4]
    RMS = constants[5]

    system = make_system(path,ex,em,alpha,O3)

    system.SNR = int(config[0]) #Signal to noise ratio
    system.ls_pol = config[2] #Ls polarization ['p', 's', or 'u']

    system.anisotropy = float(constants[8]) #Anisotropy [0, or 0.4]
    system.ensamble = constants[7] #Number of dipoles in ensamble
    system.OTF_res = constants[6] #Size of OTF in pixels
    system.ls_opening = np.arcsin(system.lenses[0].NA/system.lenses[0].RI)-(np.pi/2-alpha)

    system.add_camera(res,vox,bias_offset,RMS)
    system.calculate_system_specs()
    system.calculate_PSF()

    poisson_res = []
    readout_res = []
    FWHM = []
    for i in range(noise_samples):
        system.make_MTF()
        system.save_data()
        poisson_res.append(system.XYZ_res_poisson)
        readout_res.append(system.XYZ_res_readout)
        FWHM.append(system.FWHM)
    system.save_stacks()

    poisson_res = np.array(poisson_res)
    readout_res = np.array(readout_res)
    FWHM = np.array(FWHM)

    data = system.data
    data['X_res_poisson [nm]'] = np.mean(poisson_res,axis=0)[0]
    data['Y_res_poisson [nm]'] = np.mean(poisson_res,axis=0)[1]
    data['Z_res_poisson [nm]'] = np.mean(poisson_res,axis=0)[2]
    data['X_res_readout [nm]'] = np.mean(readout_res,axis=0)[0]
    data['Y_res_readout [nm]'] = np.mean(readout_res,axis=0)[1]
    data['Z_res_readout [nm]'] = np.mean(readout_res,axis=0)[2]
    data['X_FWHM [nm]'] = np.mean(FWHM,axis=0)[0]*1e9
    data['Y_FWHM [nm]'] = np.mean(FWHM,axis=0)[1]*1e9
    data['Z_FWHM [nm]'] = np.mean(FWHM,axis=0)[2]*1e9
    
    with open(system.path+'/data.json', 'w') as output:
        json.dump(data, output, indent=4)

    res_all = {'X_res_poisson [nm]': poisson_res[:,0].tolist(),
               'Y_res_poisson [nm]': poisson_res[:,1].tolist(),
               'Z_res_poisson [nm]': poisson_res[:,2].tolist(),
               'X_res_readout [nm]': readout_res[:,0].tolist(),
               'Y_res_readout [nm]': readout_res[:,1].tolist(),
               'Z_res_readout [nm]': readout_res[:,2].tolist(),
               'X_FWHM [nm]': (FWHM[:,0]*1e9).tolist(),
               'Y_FWHM [nm]': (FWHM[:,1]*1e9).tolist(),
               'Z_FWHM [nm]': (FWHM[:,2]*1e9).tolist()}

    with open(system.path+'/res.json', 'w') as output:
        json.dump(res_all, output, indent=4)

def main():
    SNR = np.array([10,20,50,100])
    ls_inclination = np.array([20,25,30,35,40])
    polarizations = ['p','s']
    O3 = ['Glass','Water']

    excitation = 488e-9
    emission = 507e-9
    anisotropy = 0.4

    res = 256
    vox = 1.5e-6
    bias_offset = 100
    RMS = 1.4

    OTF_res = 256
    ensamble = 100
    noise_samples = 10

    constants = [excitation,emission,res,vox,bias_offset,RMS,OTF_res,ensamble,anisotropy]
    
    configs = find_configs(SNR,ls_inclination,polarizations,O3)

    for i,config in enumerate(configs):
        print('Simulation '+str(i+1)+'/'+str(len(configs)))
        path = 'tilt_'+config[1]+'_degrees__polarization_'+config[2]+'__'+config[3]+'_O3__Photon_count_'+str(int(config[0])**2)
        if path in os.listdir():
            continue
        else:
            simulate(path,config,constants,noise_samples)


if __name__ == '__main__':
    main()
import matplotlib.pyplot as plt
import numpy as np
import os, json

class DataAnalysis:
    def __init__(self,path) -> None:
        self.path = path
        self.num_files = len(os.listdir(path))

    def get_data(self):
        res_p = []
        res_r = []
        angle = []
        polarization = []
        O3 = []
        photons = []
        for fol in os.listdir(self.path):
            with open(self.path+fol+'/data_new.json') as f:
                data1 = json.load(f)
                res_p.append([data1['X_res_poisson [nm]'],data1['Y_res_poisson [nm]'],data1['Z_sec_poisson [nm]']])
                res_r.append([data1['X_res_readout [nm]'],data1['Y_res_readout [nm]'],data1['Z_sec_readout [nm]']])
                angle.append(data1['Light sheet angle [degrees]'])
                polarization.append(data1['Light sheet polarization'])
                O3.append(data1['Lens immersion RI\'s'][4])
                photons.append(int(data1['SNR poisson'])**2)
        self.res_p = np.array(res_p,dtype=float)
        self.res_r = np.array(res_r,dtype=float)
        self.angle = np.array(angle,dtype=float)
        self.polarization = np.array(polarization,dtype=str)
        self.O3 = np.array(O3,dtype=float)
        self.photons = np.array(photons,dtype=float)

    def plot_resolution(self,vars=['angle','photons','pol'],const=['O3',1.7],save=False):
        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        result1 = (4*np.pi/3)*self.res_r[:,0]*self.res_r[:,1]*self.res_r[:,2]/1e3**3
        z_lab1 = 'PSF Volume [$\mu m^3$]'

        result2 = np.pi*self.res_r[:,0]*self.res_r[:,1]/1e3**2
        z_lab2 = 'PSF Area [$\mu m^2$]'

        var_map = {'photons': (self.photons, 'Photons'),
                   'O3': (self.O3, 'O3'),
                   'angle': (self.angle, 'Angle'),
                   'pol': (self.polarization, 'Polarization')}

        var_list, label_list = [], []
        for var in vars:
            if var in var_map:
                var_list.append(var_map[var][0])
                label_list.append(var_map[var][1])
            else:
                raise ValueError('Invalid variable')
            
        if const[0] in var_map:
            const_val = var_map[const[0]][0]
            const_lab = var_map[const[0]][1]
        else:
            raise ValueError('Invalid constant')
        
        x_vals = np.unique(var_list[0])
        y_vals = np.flip(np.unique(var_list[1]))
        z_vals = np.unique(var_list[2])
        c_val = np.where(const_val==const[1])[0]
        data1 = np.zeros((len(x_vals),len(y_vals),len(z_vals)))
        data2 = np.zeros((len(x_vals),len(y_vals),len(z_vals)))

        for i in range(len(x_vals)):
            for j in range(len(y_vals)):
                for k in range(len(z_vals)):
                    l = np.where(var_list[0]==x_vals[i])[0]
                    m = np.where(var_list[1]==y_vals[j])[0]
                    n = np.where(var_list[2]==z_vals[k])[0]
                    #Find the common indices to get the correct psf volume
                    ind = np.intersect1d(np.intersect1d(l,m),np.intersect1d(n,c_val))
                    data1[i,j,k] = result1[ind]
                    data2[i,j,k] = result2[ind]   


        print(np.sort((data2[:,:,1]/data2[:,:,0]).flatten()))
        print(np.sort((data1[:,:,1]/data1[:,:,0]).flatten()))
        exit()     

        #Loop over the data points and plot the bars
        for i in range(len(x_vals)):
            for j in range(len(y_vals)):
                if data1[i,j,0] > data1[i,j,1]:
                    ax1.bar3d(i, j, 0, 1, 1, data1[i,j,1], color='r', alpha=0.7, shade=True)
                    ax1.bar3d(i, j, data1[i,j,1], 1, 1, data1[i,j,0]-data1[i,j,1], color='b', alpha=0.7, shade=True)
                else:
                    ax1.bar3d(i, j, 0, 1, 1, data1[i,j,0], color='b', alpha=0.7, shade=True)
                    ax1.bar3d(i, j, data1[i,j,0], 1, 1, data1[i,j,1]-data1[i,j,0], color='r', alpha=0.7, shade=True)

                if data2[i,j,0] > data2[i,j,1]:
                    ax2.bar3d(i, j, 0, 1, 1, data2[i,j,1], color='r', alpha=0.7, shade=True)
                    ax2.bar3d(i, j, data2[i,j,1], 1, 1, data2[i,j,0]-data2[i,j,1], color='b', alpha=0.7, shade=True)
                else:
                    ax2.bar3d(i, j, 0, 1, 1, data2[i,j,0], color='b', alpha=0.7, shade=True)
                    ax2.bar3d(i, j, data2[i,j,0], 1, 1, data2[i,j,1]-data2[i,j,0], color='r', alpha=0.7, shade=True)

        x_p = np.arange(len(x_vals))
        y_p = np.arange(len(y_vals))

        z_ticks1 = np.linspace(0,round(np.max(data1),1),5)
        z_ticks2 = np.linspace(0,round(np.max(data2),1),5)
        z_vals1 = ['{:.2f}'.format(i) for i in z_ticks1]
        z_vals2 = ['{:.2f}'.format(i) for i in z_ticks2]


        ax1.set_xlabel(label_list[0],fontsize=16,labelpad=10)
        ax1.set_ylabel(label_list[1],fontsize=16,labelpad=10)
        ax1.set_zlabel(z_lab1,fontsize=16,labelpad=10)
        ax1.set_xticks(x_p+0.5,x_vals,fontsize=16)
        ax1.set_yticks(y_p+0.5,y_vals,fontsize=16)
        ax1.set_zticks(z_ticks1,z_vals1,fontsize=16)
        ax1.view_init(30,-40)
        #Add text in upper left corner of the plot
        ax1.text2D(0.25, 0.9, 'a)', transform=ax1.transAxes, fontsize=18)

        ax2.set_xlabel(label_list[0],fontsize=16,labelpad=10)
        ax2.set_ylabel(label_list[1],fontsize=16,labelpad=10)
        ax2.set_zlabel(z_lab2,fontsize=16,labelpad=10)
        ax2.set_xticks(x_p+0.5,x_vals,fontsize=16)
        ax2.set_yticks(y_p+0.5,y_vals,fontsize=16)
        ax2.set_zticks(z_ticks2,z_vals2,fontsize=16)
        ax2.view_init(30,-40)
        #Add text in upper left corner of the plot
        ax2.text2D(0.25, 0.9, 'b)', transform=ax2.transAxes, fontsize=18)

        # Make labels for the bar colors
        blue_patch = plt.matplotlib.patches.Patch(color='blue', label=z_vals[0]+'-'+label_list[2])
        red_patch = plt.matplotlib.patches.Patch(color='red', label=z_vals[1]+'-'+label_list[2])
        #Make a common legend for both subplots
        fig.legend(handles=[blue_patch,red_patch],loc=(0.42,0.73),fontsize=16)
        plt.show()
        
        
def main():
    path = 'D:/Simulations_October_23_high_res/'
    data1 = DataAnalysis(path)
    data1.get_data()
    data1.plot_resolution()

if __name__ == "__main__":
    main()
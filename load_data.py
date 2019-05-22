#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class Waterfalls(Dataset):
    
    def __init__(self, filepath, transform=None):
        # Open the HDF file in read mode (keep it opened during the data loading to speed up the process)
        hdf_file = h5py.File(filepath, 'r')
            
        # Print datasets and shapes
        print('#######################')
        print('HDF File structure:')
        for key in hdf_file.keys():
            if type(hdf_file[key]) is h5py.Dataset:
                print('%s - %s' % (key, hdf_file[key].shape))
        print('#######################')
        
        self.transform = transform
        self.hdf_file = hdf_file
        
    def __len__(self):
        return self.hdf_file['Waterfalls'].shape[2]
        
    def __getitem__(self, idx):
        
        # Initialize sample
        sample = {}
        
        # Get Parameters
        sample['Parameters'] = {}
        parameters = self.hdf_file['Parameters'].value[:, idx]
        for param_idx, param_value in enumerate(parameters):
            param_name_ref = self.hdf_file['ParametersNames'][param_idx, 0]
            param_name = ''.join(map(chr, self.hdf_file[param_name_ref][:, 0]))
            sample['Parameters'][param_name] = param_value
        
        # Get Path
        path_ref = self.hdf_file['Paths'].value[0, idx]
        sample['Paths'] = self.hdf_file[path_ref][:]
        
        # Get Waterfalls and Waterfalls Signal
        sample['SignalWaterfalls'] = self.hdf_file['SignalWaterfalls'][:, :, idx]
        sample['Waterfalls'] = self.hdf_file['Waterfalls'][:, :, idx]
        
        # Sample transformation
        if self.transform:
            sample = self.transform(sample)
        
        return sample
        
    def close(self):
        # Close the HDF file
        self.hdf_file.close()


# this line helps the code to understand if this file has been executed as script or has been imported as a module
#%%

if __name__ == '__main__':
        
    # Initialize dataset
    filepath = '../Waterfalls/Waterfalls_fish.mat'
    dataset = Waterfalls(filepath)
    
    # Load sample
    idx = 1
    sample = dataset[idx]


    #%% Plot
    
    ### Create figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    params_str = ' - '.join(['%s: %s' % (p_name, p_value) for p_name, p_value in sample['Parameters'].items()])
    title = 'SAMPLE INDEX %d (%s)' % (idx, params_str)
    
    ### RAW data
    axs[0].set_title(title + '\nRAW data (Waterfalls)')
    axs[0].imshow(sample['Waterfalls'], aspect='auto', interpolation='bilinear', origin='lower')
    
    ### True paths (binary representation)
    axs[1].set_title('True paths - binary representation (SignalWaterfalls)')
    axs[1].imshow(sample['SignalWaterfalls'], aspect='auto', interpolation='none', origin='lower')
    
    ### True paths (each target)
    axs[2].set_title('True paths - each target (Paths)')
    cmap = plt.get_cmap('rainbow')
    num_targets = int(sample['Parameters']['num_Targets'])
    # Set background
    axs[2].imshow(sample['Waterfalls']*0, aspect='auto', interpolation='none', origin='lower')
    # Plot paths
    for path_idx in range(num_targets):
        color = cmap((path_idx + 1) / num_targets)
        path = sample['Paths'][:, path_idx]
        axs[2].plot(path, color=color)
        
    ###
    [ax.set_ylabel('Distance') for ax in axs]
    axs[-1].set_xlabel('Time')
    fig.tight_layout()
    
    fig.show()
    #%% Close HDF file
    dataset.close()
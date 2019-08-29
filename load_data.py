#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import os


class Waterfalls(Dataset):
    default_filepath = '../DATASETS/Waterfalls/Waterfalls_fish.mat'

    def __init__(self, fpath=default_filepath, verbose=False, transform=None):
        # Open the HDF file in read mode (keep it opened during the data loading to speed up the process)
        hdf_file = h5py.File(fpath, 'r')

        if verbose:
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

        if isinstance(idx, slice):
            return self.hdf_file['Parameters'].value[:, idx]
        else:
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


def get_indices():
    # The used Training Dataset is the 80% of the entire dataset. The first 48000 samples are used to train the models
    # and tune the hyper-parameters whereas the remaining 20% is used to test them. Among the training samples,
    # 80% are used to train the model and the 20% is used as validation to get an unbiased estimation of the performance

    train_part = np.linspace(0, 38400, 38400, endpoint=False, dtype=int)
    validation_part = np.linspace(38400, 48000, 9600, endpoint=False, dtype=int)
    test_part = np.linspace(48000, 60000, 12000, endpoint=False, dtype=int)

    # [0 |--------------------- Training Dataset ---------------------| 48000
    # [0 |------------ TRAINING -----------| 38400 |--- VALIDATION ---| 48000 |----- TEST -----| 60000]
    # TRAINING      38400 samples
    # VALIDATION     9600 samples
    # TEST          12000 samples

    return train_part.tolist(), validation_part.tolist(), test_part.tolist()


if __name__ == '__main__':

    # Initialize dataset
    filepath = '../DATASETS/Waterfalls/Waterfalls_fish.mat'
    dataset = Waterfalls(filepath, verbose=True)

    # Load sample
    idx = np.random.randint(0, 60000)
    sample = dataset[idx]

    # %% Plot

    # Load style file
    plt.style.use('./styles.mplstyle')

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
    axs[2].imshow(sample['Waterfalls'] * 0, aspect='auto', interpolation='none', origin='lower')
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

    fig.savefig('figs/sample.pdf', dpi=300)

    # %% Close HDF file
    dataset.close()

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import utils
from load_data import Waterfalls, get_indices
import numpy as np
from models.denoisingnet import DenoisingNet

WATERFALLS_SIZE = (1200, 20)
train_part, validation_part, test_part = get_indices()
tested_min_velocity = np.array([15, 18])
tested_min_width = np.linspace(3, 10, 8)
tested_min_snr = np.linspace(1, 3, 10)

results_matrix = np.zeros((np.size(tested_min_velocity), np.size(tested_min_width), np.size(tested_min_snr), 1))
results = {
    'Accuracy': results_matrix,
    'BalancedAccuracy': results_matrix.copy(),
    'F-Score': results_matrix.copy(),
    'Precision': results_matrix.copy(),
    'Recall': results_matrix.copy(),
    'Samples': results_matrix.copy()
}

# Select the GPU if available
if torch.cuda.is_available():
    current_device = torch.device("cuda")
else:
    current_device = torch.device("cpu")

# Load the trained model
load_configuration = 'DenoisingNet-5e'
configuration = torch.load('models/data/' + load_configuration + '_net_parameters.torch', map_location=current_device)

# Initialize the network a set the  the loaded parameters
net = DenoisingNet(verbose=False)
net.load_state_dict(configuration['DenoisingNet'])
net.to(current_device)

# Load the dataset
dataset = Waterfalls(fpath='../DATASETS/Waterfalls/Waterfalls_fish.mat',
                     transform=utils.NormalizeSignal((1200, 20)))
parameters = dataset[:]


for idx_v, min_v in enumerate(tested_min_velocity):
    for idx_w, min_w in enumerate(tested_min_width):
        for idx_s, min_s in enumerate(tested_min_snr):

            s = tested_min_snr[idx_s]
            w = tested_min_width[idx_w]
            v = tested_min_velocity[idx_v]

            indices = utils.filter_indices(test_part, parameters, min_snr=s, min_width=w, min_velocity=v)

            print('v: '+v.__str__()+' w: '+w.__str__()+' s: '+s.__str__()+' ('+len(indices).__str__()+')')

            # Load the samples
            test_samples = DataLoader(dataset, batch_size=32,
                                      shuffle=False,
                                      sampler=SubsetRandomSampler(indices),
                                      collate_fn=utils.waterfalls_collate)

            # Compute the accuracy metrics
            metrics = net.accuracy(test_samples, device=current_device, print_summary=False, verbose=True)
            results['Accuracy'][idx_v, idx_w, idx_s] = metrics['Accuracy']
            results['BalancedAccuracy'][idx_v, idx_w, idx_s] = metrics['BalancedAccuracy']
            results['F-Score'][idx_v, idx_w, idx_s] = metrics['F-Score']
            results['Precision'][idx_v, idx_w, idx_s] = metrics['Precision']
            results['Recall'][idx_v, idx_w, idx_s] = metrics['Recall']
            results['Samples'][idx_v, idx_w, idx_s] = len(indices)

print('Saving the results..')
np.save('data/denoising_results', results)

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import utils
from load_data import Waterfalls, get_indices
import numpy as np
from models.denoisingnet import DenoisingNet
from models.fishclassifier import FishClassifier

WATERFALLS_SIZE = (1200, 20)
train_part, validation_part, test_part = get_indices()
load_configuration = 'Classifier-FT-15e'
tested_min_velocity = np.array([15, 18])
tested_min_width = np.linspace(3, 10, 8)
tested_min_snr = np.linspace(1, 3, 10)
# test_part = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
results_matrix = np.zeros((np.size(tested_min_velocity), np.size(tested_min_width), np.size(tested_min_snr), 13, 13))

# Select the GPU if available
if torch.cuda.is_available():
    current_device = torch.device("cuda")
else:
    current_device = torch.device("cpu")

# Load the trained models
# Encoder
encoder = DenoisingNet(verbose=True)
state_dict = torch.load('models/data/' + load_configuration + '_net_parameters.torch', map_location=current_device)
# Set the loaded parameters to the network
encoder.load_state_dict(state_dict['encoder_tuned'])
encoder.to(current_device)

# Classifier
classes, config = utils.get_classes()
classifier = FishClassifier(config, estimate_parameters=False, conv_part=True)
classifier.load_state_dict(state_dict['classifier_post_tuning'])
classifier.to(current_device)

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
            metrics = classifier.accuracy(encoder=encoder, dataloader_eval=test_samples,
                                          classes=classes, device=current_device)
            results_matrix[idx_v, idx_w, idx_s, :, :] = metrics['matrix']

print('Saving the results..')
np.save('data/classifier_results', results_matrix)

import torch
import utils
from torch.utils.data import DataLoader
from load_data import Waterfalls
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import time
from models.denoisingnet import DenoisingNet
from models.fishclassifier import FishClassifier, train_network
from torch.backends import cudnn

WATERFALLS_SIZE = (1200, 20)

##########################
# TODO Set all the following parameters by args. Improve this section
# Parameters
verbose_flag = True
plot_flag = False
epochs = 4
min_SNR = 2.5
lr = 1e-3  # Learning rate
load_configuration = 'DAE9B-9B-fd'
##########################
try:
    print('MKL support: ' + torch.backends.mkl.is_available().__str__())
    print('Intel MKL-DNN support: ' + torch.backends.mkldnn.is_available().__str__())
except:
    print('Intel MKL-DNN library not available')

print('\n\n[Loading data...]')
dataset = Waterfalls(transform=utils.NormalizeSignal(WATERFALLS_SIZE))

# Load the test-sample indices from the saved configuration
# TODO remove max_samples
train_indices = np.load('models/data/' + load_configuration + '_data.npy')[1]
test_indices = np.load('models/data/' + load_configuration + '_data.npy')[2]

# TODO change
parameters = dataset[:]  # Get all the parameters
snr_cond = parameters[0] >= min_SNR  # Check where data satisfies the SNR condition
train_indices = train_indices[snr_cond[train_indices]]
test_indices = test_indices[snr_cond[test_indices]]

# Write a log file containing useful information with in 'logs/$reference_time$.log'
reference_time = time.strftime("%H_%M_%S")
log_name = 'logs/' + reference_time + '.log'
with open(log_name, 'w') as logfile:  # Append mode
    logfile.write('LOG FILE - ' + reference_time
                  # + '\n\tMin SNR ' + min_SNR.__str__()
                  # + '\n\tMin Width ' + min_width.__str__()
                  # + '\n\tSamples\t' + len(dataset_indices).__str__() + '/60000'
                  + '\n\tCUDA Support\t' + torch.cuda.is_available().__str__())

# Load train data efficiently
train_dataloader = DataLoader(dataset, batch_size=32,
                              shuffle=False,  # The shuffle has been already performed by splitting the dataset
                              sampler=SubsetRandomSampler(train_indices),
                              collate_fn=utils.waterfalls_collate)
# Load test data efficiently
test_dataloader = DataLoader(dataset, batch_size=32,
                             shuffle=False,
                             sampler=SubsetRandomSampler(test_indices),
                             collate_fn=utils.waterfalls_collate)

encoder = DenoisingNet(verbose=verbose_flag)

classes = {
    # TODO adjust in case of contraints for the parameters
    't': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'v': [15, 16, 17, 18, 19, 20],
    'w': [3, 4, 5, 6, 7, 8, 9, 10]
}
config = {
    't_dim': len(classes['t']),
    'v_dim': len(classes['v']),
    'w_dim': len(classes['w']),
    'IntermediateDimension': None,
    'HiddenFeaturesDimension': None
}

classifier = FishClassifier(classification_parameters=config, conv_part=True, estimate_all=False)

# If cuda is available set the device to GPU otherwise use the CPU
if torch.cuda.is_available():
    current_device = torch.device("cuda")
    print('Available GPUs: ' + torch.cuda.device_count().__str__())
    if torch.cuda.device_count() > 1:
        # if available, use multiple GPUs
        encoder = torch.nn.DataParallel(encoder)
        classifier = torch.nn.DataParallel(classifier)
    try:
        # Enable the NVIDIA CUDA Deep Neural Network library (cuDNN)
        cudnn.enabled = True
        print('cuDNN library enabled')
    except:
        print('cuDNN library is not available')
else:
    current_device = torch.device("cpu")

# Load the trained AutoEncoder Model
net_state_dict = torch.load('models/data/' + load_configuration + '_net_parameters.torch', map_location=current_device)
# Set the loaded parameters
encoder.load_state_dict(net_state_dict)

# Move all the network parameters to the selected device
encoder.to(current_device)

# Move all the network parameters to the selected device
classifier.to(current_device)

# PHASE 1 - Train the classifier
print('PHASE 1')
optim1 = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-5)
train_network(classifier=classifier, encoder=encoder, dataloader_train=train_dataloader,
              dataloader_eval=test_dataloader, num_epochs=epochs, optimizer=optim1,
              device=current_device, finetuning=False)

# # PHASE 2 - Fine-tuning of the encoder
# print('\nPHASE 2')
# optim2 = torch.optim.Adam([{'params': encoder.parameters(), 'lr': lr, 'weight_decay': 1e-5},
#                            {'params': classifier.parameters(), 'lr': lr, 'weight_decay': 1e-5}])
# train_network(classifier=classifier, encoder=encoder, dataloader_train=train_dataloader,
#               dataloader_eval=test_dataloader, num_epochs=epochs, optimizer=optim2,
#               device=current_device, finetuning=True)

# Save the network state
statistics = []
net_state_dict = classifier.state_dict()  # The state dictionary includes all the parameters of the network
torch.save(net_state_dict, 'data/' + reference_time + '_net_parameters.torch')  # Save the state dict to a file
# Save train statistics and train/test indices to data files
np.save('data/' + reference_time + '_data', [statistics, train_indices, test_indices])

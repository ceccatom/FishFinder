import torch
import utils
from torch.utils.data import DataLoader
from load_data import Waterfalls
from torch.utils.data.sampler import SubsetRandomSampler
from torch.backends import cudnn
import numpy as np
import time
from models.denoisingnet import DenoisingNet, train_network, freeze_block
from sklearn.model_selection import train_test_split

WATERFALLS_SIZE = (1200, 20)
min_SNR = 0
min_width = 0

##########################
# TODO Set all the following parameters by args. Improve this section
# Parameters
verbose_flag = True
plot_flag = False
lr = 1e-3  # Learning rate
train_percentage = 0.98
max_sample = 60000
train_blocks = []
epochs = 3
open_config = 'DAE9B-9B-fd'
final_train = True
##########################

print('[Loading data...]')
dataset = Waterfalls(transform=utils.NormalizeSignal(WATERFALLS_SIZE))
parameters = dataset[0:max_sample]  # Get all the parameters
snr_cond = parameters[0] >= min_SNR  # Check where data satisfies the SNR condition
width_cond = parameters[2] >= min_width  # Check where data satisfies the width condition
dataset_indices = np.nonzero(np.logical_and(snr_cond, width_cond))[0]  # Get indices of the valid part of the dataset

# Write a log file containing useful information with in 'logs/$reference_time$.log'
reference_time = time.strftime("%H_%M_%S")
log_name = 'logs/' + reference_time + '.log'
with open(log_name, 'w') as logfile:  # Append mode
    logfile.write('LOG FILE - ' + reference_time
                  + '\n\tMin SNR ' + min_SNR.__str__()
                  + '\n\tMin Width ' + min_width.__str__()
                  + '\n\tSamples\t' + len(dataset_indices).__str__() + '/60000'
                  + '\n\tCUDA Support\t' + torch.cuda.is_available().__str__())

if open_config is not None:
    # Load the previously computed indices of the test/train samples
    train_indices = np.load('models/data/' + open_config + '_data.npy')[1]
    test_indices = np.load('models/data/' + open_config + '_data.npy')[2]
else:
    # Split the considered dataset into train and test
    train_indices, test_indices = train_test_split(dataset_indices, train_size=train_percentage, shuffle=True)

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

# Initialize the network
net = DenoisingNet(verbose=verbose_flag)
# Load the pre-trained network
if open_config is not None:
    net_state_dict = torch.load('models/data/' + open_config + '_net_parameters.torch', map_location='cpu')
    # Set the loaded parameters to the network
    net.load_state_dict(net_state_dict)

# If cuda is available set the device to GPU otherwise use the CPU
if torch.cuda.is_available():
    current_device = torch.device("cuda")
    print('Available GPUs: ' + torch.cuda.device_count().__str__())
    if torch.cuda.device_count() > 1:
        # if available, use multiple GPUs
        net = torch.nn.DataParallel(net)
    try:
        # Enable the NVIDIA CUDA Deep Neural Network library (cuDNN)
        cudnn.enabled = True
        print('cuDNN library enabled')
    except:
        print('cuDNN library is not available')
else:
    current_device = torch.device("cpu")

# Move all the network parameters to the selected device
net.to(current_device)

statistics = []

# Training
start_time = time.time()
for index, block in enumerate(train_blocks):
    print('Training of block ' + block.__str__())

    # Freeze Blocks not involved in the training
    for i, b in enumerate(net.blocks):
        freeze_block(b, i != block - 1, verbose=True)

    # Define the optimizer on the unfrozen parameters
    optim = torch.optim.Adam(net.blocks[block-1].parameters(), lr=lr, weight_decay=1e-5)

    # Perform Greedy block-wise training
    phase_train, phase_eval = train_network(net, train_dataloader, test_dataloader,
                                            device=current_device,
                                            optimizer=optim,
                                            num_epochs=epochs,
                                            depth=block)

    statistics.append((phase_train, phase_eval))


if final_train:
    # Perform the final training the involves all the blocks
    for b in net.blocks:
        freeze_block(b, False)

    # Define the optimizer for the entire network
    optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

    # Train the network
    phase_train, phase_eval = train_network(net, train_dataloader, test_dataloader,
                                            device=current_device,
                                            optimizer=optim,
                                            num_epochs=epochs,
                                            depth=9)

elapsed_time = time.time() - start_time

# Save the network state
net_state_dict = net.state_dict()  # The state dictionary includes all the parameters of the network
torch.save(net_state_dict, 'data/' + reference_time + '_net_parameters.torch')  # Save the state dict to a file

# Write train statistics to the log file
with open(log_name, 'a') as logfile:  # Append mode
    logfile.write('\nLoss\t' + statistics.__str__()
                  + '\n\nTraining time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                  + '\n' + time.strftime("%H:%M:%S"))

# Save train statistics and train/test indices to data files
np.save('data/' + reference_time + '_data', [statistics, train_indices, test_indices])

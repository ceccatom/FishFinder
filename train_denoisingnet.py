import torch
import utils
from torch.utils.data import DataLoader
from load_data import Waterfalls, get_indices
from torch.utils.data.sampler import SubsetRandomSampler
from torch.backends import cudnn
import numpy as np
import time
from models.denoisingnet import DenoisingNet, train_network, freeze_block

WATERFALLS_SIZE = (1200, 20)
train_part, validation_part, test_part = get_indices()
##########################
# TODO Set all the following parameters by args. Improve this section
# Parameters
verbose_flag = True
plot_flag = False
lr = 1e-3  # Learning rate
batchsize = 32
# max_sample = 60000
train_blocks = [1, 2, 3, 4, 5, 6]
block_epochs = 3
final_train = True
open_config = None
##########################

print('[Loading data...]')
dataset = Waterfalls(transform=utils.NormalizeSignal(WATERFALLS_SIZE))
# parameters = dataset[0:max_sample]  # Get all the parameters
# snr_cond = parameters[0] >= min_SNR  # Check where data satisfies the SNR condition
# dataset_indices = np.nonzero(snr_cond)[0]  # Get indices of the valid part of the dataset

# Write a log file containing useful information with in 'logs/$reference_time$.log'
reference_time = time.strftime("%H_%M_%S")
log_name = 'logs/' + reference_time + '.log'
with open(log_name, 'w') as logfile:  # Append mode
    logfile.write('LOG FILE - ' + reference_time
                  + '\n\tCUDA Support\t' + torch.cuda.is_available().__str__())

# Load train data efficiently
train_dataloader = DataLoader(dataset, batch_size=batchsize,
                              shuffle=False,  # The shuffle is performed by the RandomSampler
                              sampler=SubsetRandomSampler(train_part),
                              collate_fn=utils.waterfalls_collate)
# Load test data efficiently
eval_dataloader = DataLoader(dataset, batch_size=batchsize,
                             shuffle=False,  # The shuffle is performed by the RandomSampler
                             sampler=SubsetRandomSampler(validation_part),
                             collate_fn=utils.waterfalls_collate)

# Initialize the network
net = DenoisingNet(verbose=verbose_flag)

print('[ Hardware configuration ]')
# If cuda is available set the device to GPU otherwise use the CPU
if torch.cuda.is_available():
    current_device = torch.device("cuda")
    print('\nAvailable GPUs: ' + torch.cuda.device_count().__str__())
    if torch.cuda.device_count() > 1:
        # if available, use multiple GPUs
        net = torch.nn.DataParallel(net)
    try:
        # Enable the NVIDIA CUDA Deep Neural Network library (cuDNN)
        cudnn.enabled = True
        print('\tNVIDIA cuDNN library enabled')
    except:
        print('\tNVIDIA cuDNN library is not available')
else:
    current_device = torch.device("cpu")
try:
    print('\tIntel MKL-DNN library: ' + torch.backends.mkldnn.is_available().__str__())
except:
    print('\tIntel MKL-DNN library not available')

# Load the pre-trained network
if open_config is not None:
    configuration = torch.load('models/data/' + open_config + '_net_parameters.torch', map_location=current_device)
    # Set the loaded parameters to the network
    net.load_state_dict(configuration['DenoisingNet'])
    print('[Configuration "' + open_config + '" loaded]')

# Move all the network parameters to the selected device
net.to(current_device)

statistics = {
    'loss': {
        'train': [[], [], [], [], [], [], []],
        'eval': [[], [], [], [], [], [], []]
    },
    'accuracy': {
        'train': [[], [], [], [], [], [], []],
        'eval': [[], [], [], [], [], [], []]
    }
}

# Training
for index, block in enumerate(train_blocks):
    print('[ Training of block ' + block.__str__() + ' ]')

    # Freeze Blocks not involved in the training
    for i, b in enumerate(net.blocks):
        freeze_block(b, i != block - 1, verbose=True)

    # Define the optimizer on the unfrozen parameters
    optim = torch.optim.Adam(net.blocks[block - 1].parameters(), lr=lr, weight_decay=1e-5)

    # Perform Greedy block-wise training
    loss_train, loss_eval, acc_train, acc_eval = train_network(net, train_dataloader, eval_dataloader,
                                                               device=current_device,
                                                               optimizer=optim,
                                                               num_epochs=block_epochs,
                                                               depth=block)

    # Save the training results
    statistics['loss']['train'][index].append(loss_train)
    statistics['loss']['eval'][index].append(loss_eval)
    statistics['accuracy']['train'][index].append(acc_train)
    statistics['accuracy']['train'][index].append(acc_eval)

if final_train:
    # Perform the final training the involves all the blocks
    for b in net.blocks:
        freeze_block(b, False)

    # Define the optimizer for the entire network
    optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

    # Train the network
    loss_train, loss_eval, acc_train, acc_eval = train_network(net, train_dataloader, eval_dataloader,
                                                               device=current_device,
                                                               optimizer=optim,
                                                               num_epochs=block_epochs,
                                                               depth=6)

    # Save the training results
    statistics['loss']['train'][10].append(loss_train)
    statistics['loss']['eval'][10].append(loss_eval)
    statistics['accuracy']['train'][10].append(acc_train)
    statistics['accuracy']['train'][10].append(acc_eval)

# Save the network state
net_state_dict = net.state_dict()  # The state dictionary includes all the parameters of the network
torch.save({
    'DenoisingNet': net_state_dict,
    'statistics': statistics,
    'hyperparam': {
        'lr': lr,
        'batchsize': batchsize,
        'block_epochs': block_epochs
    }
}, 'data/' + reference_time + '_net_parameters.torch')  # Save the state dict to a file

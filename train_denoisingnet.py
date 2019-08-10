import torch
import utils
from torch.utils.data import DataLoader
from load_data import Waterfalls
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import time
from models.denoisingnet import DenoisingNet, train_network
from sklearn.model_selection import train_test_split

WATERFALLS_SIZE = (1200, 20)

##########################
# TODO Set all the following parameters by args. Improve this section
# Parameters
verbose_flag = True
plot_flag = False
lr = 5*1e-3  # Learning rate
train_percentage = 0.98
min_SNR = 0
min_width = 0
max_sample = 59999

# TODO improve frozen configuration management
# Epochs, Depth, Freeze L1, Freeze L2, Freeze L3
training_config = [[3, 1, False, False, False],
                   [3, 2, True,  False, False],
                   [3, 3, True,  True,  False],
                   [3, 4, True,  True,  True]]
##########################

print('[Loading data...]')
dataset = Waterfalls(transform=utils.NormalizeSignal(WATERFALLS_SIZE))
parameters = dataset[0:max_sample]                                     # Get all the parameters
snr_cond = parameters[0] >= min_SNR                                    # Check where data satisfies the SNR condition
width_cond = parameters[2] >= min_width                                # Check where data satisfies the width condition
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

# If cuda is available set the device to GPU otherwise use the CPU
if torch.cuda.is_available():
    current_device = torch.device("cuda")
else:
    current_device = torch.device("cpu")

# Move all the network parameters to the selected device
net.to(current_device)

# Define the optimizer
optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

statistics = []

# Training
start_time = time.time()
phases = len(training_config)
for phase in range(phases):
    if verbose_flag:
        print('[ PHASE ' + (phase + 1).__str__() + '/' + phases.__str__() + ' | Config:' + training_config[
            phase].__str__() + ']')

    phase_train, phase_eval = train_network(net, train_dataloader, test_dataloader,
                                            device=current_device,
                                            optimizer=optim,
                                            num_epochs=training_config[phase][0],
                                            encoding_depth=training_config[phase][1],
                                            freeze_l1=training_config[phase][2],
                                            freeze_l2=training_config[phase][3],
                                            freeze_l3=training_config[phase][4])

    statistics.append((phase_train, phase_eval))
    if plot_flag:
        utils.plot_loss(phase_train, phase_eval, phase=phase)

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

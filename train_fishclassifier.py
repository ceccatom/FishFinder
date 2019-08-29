import torch
import utils
from torch.utils.data import DataLoader
from load_data import Waterfalls, get_indices
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import time
from models.denoisingnet import DenoisingNet
from models.fishclassifier import FishClassifier, train_network
from torch.backends import cudnn

WATERFALLS_SIZE = (1200, 20)
train_part, validation_part, test_part = get_indices(list=False)
##########################
# TODO Set all the following parameters by args. Improve this section
# Parameters
verbose_flag = True
plot_flag = False
epochs = 20
min_SNR = 0
lr = 1e-4  # Learning rate
load_configuration = 'DenoisingNet-5e'
##########################
print('\n\n[Loading data...]')
dataset = Waterfalls(transform=utils.NormalizeSignal(WATERFALLS_SIZE))

# TODO change
parameters = dataset[:]  # Get all the parameters
snr_cond = parameters[0] >= min_SNR  # Check where data satisfies the SNR condition
train_part = train_part[snr_cond[train_part]].tolist()
validation_part = validation_part[snr_cond[validation_part]].tolist()

# Write a log file containing useful information with in 'logs/$reference_time$.log'
reference_time = time.strftime("%H_%M_%S")
log_name = 'logs/' + reference_time + '.log'
with open(log_name, 'w') as logfile:  # Append mode
    logfile.write('LOG FILE - ' + reference_time
                  + '\n\tCUDA Support\t' + torch.cuda.is_available().__str__())

# Load train data efficiently
train_dataloader = DataLoader(dataset, batch_size=32,
                              shuffle=False,
                              sampler=SubsetRandomSampler(train_part),
                              collate_fn=utils.waterfalls_collate)
# Load test data efficiently
test_dataloader = DataLoader(dataset, batch_size=32,
                             shuffle=False,
                             sampler=SubsetRandomSampler(validation_part),
                             collate_fn=utils.waterfalls_collate)

encoder = DenoisingNet(verbose=verbose_flag)

classes, config = utils.get_classes()
classifier = FishClassifier(classification_parameters=config)

# If cuda is available set the device to GPU otherwise use the CPU
print('[ Hardware configuration ]')
if torch.cuda.is_available():
    current_device = torch.device("cuda")
    print('\tAvailable GPUs: ' + torch.cuda.device_count().__str__())
    if torch.cuda.device_count() > 1:
        # if available, use multiple GPUs
        encoder = torch.nn.DataParallel(encoder)
        classifier = torch.nn.DataParallel(classifier)
    try:
        # Enable the NVIDIA CUDA Deep Neural Network library (cuDNN)
        cudnn.enabled = True
        print('\tcuDNN library enabled')
    except:
        print('\tcuDNN library is not available')
else:
    current_device = torch.device("cpu")
try:
    print('\tMKL support: ' + torch.backends.mkl.is_available().__str__())
    print('\tIntel MKL-DNN support: ' + torch.backends.mkldnn.is_available().__str__())
except:
    print('\tIntel MKL-DNN library not available')

# Load the trained AutoEncoder Model
net_state_dict = torch.load('models/data/' + load_configuration + '_net_parameters.torch', map_location=current_device)
# Set the loaded parameters
encoder.load_state_dict(net_state_dict['DenoisingNet'])

# Move the networks to the selected device
encoder.to(current_device)
classifier.to(current_device)

# PHASE 1 - Train the classifier
print('[ PHASE 1 ]')
optim1 = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-5)
train_loss, eval_loss, eval_accuracy = train_network(classifier=classifier, encoder=encoder,
                                                     dataloader_train=train_dataloader,
                                                     dataloader_eval=test_dataloader, num_epochs=epochs,
                                                     optimizer=optim1,
                                                     device=current_device, finetuning=False)

# save the network's weights
dict_classifier = classifier.state_dict()
statistics_pre_tuning = {
    'loss': {
        'train': train_loss,
        'eval': eval_loss
    },
    'accuracy': eval_accuracy
}

# PHASE 2 - Fine-tuning of the encoder
print('\n[ PHASE 2 ]')
optim2 = torch.optim.Adam([{'params': encoder.parameters(), 'lr': lr, 'weight_decay': 1e-5},
                           {'params': classifier.parameters(), 'lr': lr, 'weight_decay': 1e-5}])
train_loss, eval_loss, eval_accuracy = train_network(classifier=classifier, encoder=encoder,
                                                     dataloader_train=train_dataloader,
                                                     dataloader_eval=test_dataloader, num_epochs=epochs,
                                                     optimizer=optim2,
                                                     device=current_device, finetuning=True)
statistics_post_tuning = {
    'loss': {
        'train': train_loss,
        'eval': eval_loss
    },
    'accuracy': eval_accuracy
}

# Save the parameters of the two networks and configuration
torch.save({
    'encoder_tuned': encoder.state_dict(),
    'classifier_pre_tuning': dict_classifier,
    'classifier_post_tuning': classifier.state_dict(),
    'epochs': epochs,
    'lr': lr,
    'statistics_pre': statistics_pre_tuning,
    'statistics_post': statistics_post_tuning,
}, 'data/' + reference_time + '_net_parameters.torch')

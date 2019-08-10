import torch
from load_data import Waterfalls
import numpy as np
from models.denoisingnet import DenoisingNet
import matplotlib.pyplot as plt


def plot_kernels(data, h_num, v_num, title):
    fig, axs = plt.subplots(h_num, v_num, figsize=(8, 8))
    shape = data.shape
    data = data.reshape(shape[0] * shape[1], shape[2], shape[3])
    for idx, ax in enumerate(axs.flatten()):
        ax.set_xticks([])
        ax.set_yticks([])
        if idx < len(data):
            ax.imshow(data[idx, :, :], cmap='gray')
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0, w_pad=0)
    plt.show()


# Load the trained model
load_configuration = 'DAE4B_fd'
net_state_dict = torch.load('models/data/' + load_configuration + '_net_parameters.torch', map_location='cpu')

# Initialize the network
net = DenoisingNet(verbose=True)
# Set the loaded parameters to the network
net.load_state_dict(net_state_dict)

# %% Analyze network weights

# EXTRACT WEIGHTS
weights_l1 = net.encoder_l1[0].weight.data.numpy()  # Block 1 weights
weights_l2 = net.encoder_l2[0].weight.data.numpy()  # Block 2 weights
weights_l3 = net.encoder_l3[0].weight.data.numpy()  # Block 3 weights
weights_l4 = net.encoder_l4[0].weight.data.numpy()  # Block 4 weights

plt.close('all')
plot_kernels(weights_l1, 4, 2, 'Block 1 convolutional kernels')
plot_kernels(weights_l2, 4, 4, 'Block 2 convolutional kernels')
plot_kernels(weights_l2, 6, 5, 'Block 3 convolutional kernels')
plot_kernels(weights_l2, 8, 8, 'Block 4 convolutional kernels')

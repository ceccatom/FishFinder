import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import utils
import matplotlib.pyplot as plt
from torch import nn
from models.denoisingnet import DenoisingNet
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from load_data import Waterfalls, get_indices
from torchvision import transforms
import numpy as np
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
from skimage import data

import matplotlib.pyplot as plt


class FishTracker(nn.Module):

    def __init__(self, verbose=True):
        super().__init__()

        # Verbose Mode
        self.verbose = verbose

        self.n = 1  # 12

        self.loss_fn = torch.nn.BCELoss()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(29, 3), stride=1, padding=(14, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=(10, 3), dilation=2, stride=1, padding=(9, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )

        self.recurrent = nn.Sequential(
            nn.LSTM(input_size=1200, hidden_size=1200, num_layers=1, bidirectional=True),
        )

        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.network(x)
        x, hid = self.recurrent(x.squeeze(1).permute(2, 0, 1))
        return self.final_activation(x)



def train(tracker, autoencoder, data_train, data_eval, epochs, optimizer, device):
    # Empty lists to store training statistics
    train_loss = []
    eval_loss = []

    # Use autoencoder as a features extractor
    autoencoder.eval()
    # The the tracker in training mode

    for epoch in range(epochs):
        epoch_progress = 'Epoch ' + (epoch + 1).__str__() + '/' + epochs.__str__()
        data_train = tqdm(data_train, bar_format='{l_bar}|{bar}| ['
                                                 + epoch_progress
                                                 + ' {postfix}, {elapsed}<{remaining}]')

        batch_loss_train = []
        tracker.train()
        for batch in data_train:
            waterfalls = batch['SignalWaterfalls'].to(device)
            paths = batch['Paths2D'].squeeze(1).permute(2, 0, 1).to(device)

            # Reset the parameters' gradient to zero
            optimizer.zero_grad()


            # Initialise hidden state
            # TODO LSTM stateful?
            # model.hidden = model.init_hidden()

            # # Forward pass
            # with torch.no_grad():  # No need to track the gradients
            #     clean_waterfalls = autoencoder.forward(waterfalls, depth=6)
            output = tracker.forward(waterfalls)
            output_forward = output.view(20, 32, 2, 1200)
            loss = tracker.loss_fn(output_forward[:, :, 0, :], nn.Softmax(dim=-1).forward(paths))

            # Backward pass
            loss.backward()
            optimizer.step()

            # Get loss value
            batch_loss_train.append(loss.data.cpu().numpy())
            data_train.set_postfix_str('Partial Train Loss: ' + np.array2string(batch_loss_train[-1]))

        train_loss.append(np.mean(batch_loss_train))

        batch_loss_eval = []

        # Validation Phase
        tracker.eval()
        with torch.no_grad():  # No need to track the gradients

            for batch in data_eval:
                waterfalls = batch['SignalWaterfalls'].to(device)
                paths = batch['SignalWaterfalls'].to(device).squeeze(1).permute(2, 0, 1).to(device) # batch['Paths2D'].squeeze(1).permute(2, 0, 1).to(device)

                # Reset the parameters' gradient to zero
                optimizer.zero_grad()

                # Forward pass
                with torch.no_grad():  # No need to track the gradients
                    # clean_waterfalls = autoencoder.forward(waterfalls, depth=6)
                    output = tracker.forward(waterfalls)

                output_forward = output.view(20, 32, 2, 1200)
                loss = tracker.loss_fn(output_forward[:, :, 0, :], nn.Softmax(dim=-1).forward(paths))
                batch_loss_eval.append(loss.data.cpu().numpy())

            eval_loss.append(np.mean(batch_loss_eval))
            print('Validation Loss: ' + eval_loss[-1].__str__())

    return train_loss, eval_loss


# Show a summary of FishTracker
if __name__ == '__main__':
    train_part, validation_part, test_part = get_indices()


    # Load the trained model
    # load_configuration_ae = 'DAE9B-9B-fd'
    load_configuration_tracker = 'test'
    #
    # # Initialize the Autoencoder
    # ae = DenoisingNet(verbose=True)
    #
    # pretrained_dict = torch.load('data/' + load_configuration_ae + '_net_parameters.torch', map_location='cpu')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in ae.state_dict()}
    # # 2. overwrite entries in the existing state dict
    # ae.state_dict().update(pretrained_dict)
    # # 3. load the new state dict
    # ae.load_state_dict(pretrained_dict)
    #
    # Inizialize the tracker
    tracer = FishTracker()
    pretrained_dict = torch.load('data/' + load_configuration_tracker + '_net_parameters.torch', map_location='cpu')
    tracer.load_state_dict(pretrained_dict['tracer'])

    dataset = Waterfalls(fpath='../../DATASETS/Waterfalls/Waterfalls_fish.mat',
                         transform=transforms.Compose(
                             [utils.NormalizeSignal((1200, 20)),
                              utils.Paths2D((1200, 20))]
                         ))


    # %% Evaluate the accuracy metrics
    test_samples = DataLoader(dataset, batch_size=32,
                              shuffle=False,
                              sampler=SubsetRandomSampler(test_part),
                              collate_fn=utils.waterfalls_collate)


 # %% View the reconstruction performance with random test sample

    # Select randomly one sample
    idx = test_part[np.random.randint(len(test_part))]

    # Extracts data from the sample and transform them into tensors
    waterfalls = utils.get_tensor(dataset[idx]['SignalWaterfalls'], float_cast=True, unsqueeze=2).cpu()
    paths = utils.get_tensor(dataset[idx]['Paths2D'], float_cast=True, unsqueeze=2).cpu()

    tracer.eval()
    with torch.no_grad():
        output = tracer.forward(waterfalls)
        output_forward = output.view(20, 1, 2, 1200)
    # Extract parameters from the sample
    parameters = dataset[idx]['Parameters']


    # Plot the waterfalls, the output of the network and the waterfalls without the noise
    utils.plot_reconstruction(waterfalls, output_forward[:, 0, 0, :].permute([1, 0]), paths, parameters,
                              hard_threshold=False, show_error=False)


    ### Create figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    params_str = ' - '.join(['%s: %s' % (p_name, p_value) for p_name, p_value in dataset[idx]['Parameters'].items()])
    title = 'SAMPLE INDEX %d (%s)' % (idx, params_str)

    ### RAW data
    axs[0].set_title(title + '\nRAW data (Waterfalls)')
    axs[0].imshow(dataset[idx]['Waterfalls'], aspect='auto', interpolation='bilinear', origin='lower')

    ### True paths (binary representation)
    axs[1].set_title('True paths - binary representation (SignalWaterfalls)')
    axs[1].imshow(dataset[idx]['SignalWaterfalls'], aspect='auto', interpolation='none', origin='lower')

    ### True paths (each target)
    axs[2].set_title('True paths - each target (Paths)')
    cmap = plt.get_cmap('rainbow')
    num_targets = int(dataset[idx]['Parameters']['num_Targets'])
    # Set background
    axs[2].imshow(dataset[idx]['Waterfalls'] * 0, aspect='auto', interpolation='none', origin='lower')
    # Plot paths
    for path_idx in range(num_targets):
        color = cmap((path_idx + 1) / num_targets)
        path = dataset[idx]['Paths'][:, path_idx]
        axs[2].plot(path, color=color)

    ###
    [ax.set_ylabel('Distance') for ax in axs]
    axs[-1].set_xlabel('Time')
    fig.tight_layout()

    fig.show()

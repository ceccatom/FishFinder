import torch
import utils
from torch import nn
from tqdm import tqdm
from load_data import Waterfalls
import numpy as np


class DenoisingNet(nn.Module):

    def __init__(self, verbose=True):
        super().__init__()

        # Loss function - Binary Cross Entropy
        self.loss_fn = torch.nn.BCELoss()

        # Verbose Mode
        self.verbose = verbose

        # ENCODER BLOCKS
        # Encoder Block L1
        self.encoder_l1 = nn.Sequential(
            nn.Conv2d(1, 8, (6, 3), stride=(2, 1), padding=(2, 1)),  # output dimensions: 8 * 600 * 20
            nn.ReLU(True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=0.1)
        )

        # Encoder Block L2
        self.encoder_l2 = nn.Sequential(
            nn.Conv2d(8, 16, (4, 3), stride=(2, 1), padding=(1, 2),
                      dilation=(1, 2)),  # output dimensions: 16 * 300 * 20
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.1)
        )

        # Encoder Block L3
        self.encoder_l3 = nn.Sequential(
            nn.Conv2d(16, 32, 5, stride=(3, 1), padding=(1, 0)),  # output dimensions: 32 * 100 * 16
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1)
        )

        # Encoder Block L4
        self.encoder_l4 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # output dimensions: 64 * 50 * 8
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1)
        )

        # DECODER BLOCKS
        # Decoder Block L4
        self.decoder_l4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # output dimensions: 32 * 100 * 16
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1)
        )

        # Decoder Block L3
        self.decoder_l3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 5, stride=(3, 1), padding=(1, 0)),  # output dimensions: 16 * 300 * 20
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.1)
        )
        # Decoder Block L2
        self.decoder_l2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, (4, 3), stride=(2, 1), padding=(1, 2),
                               dilation=(1, 2)),  # output dimensions: 8 * 600 * 20
            nn.ReLU(True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.1),
        )

        # Decoder Block L1
        self.decoder_l1 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, (6, 3), stride=(2, 1), padding=(2, 1)),  # output dimensions: 1 * 1200 * 20
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # Blocks pointer
        self.block_l1 = [self.encoder_l1, self.decoder_l1]
        self.block_l2 = [self.encoder_l2, self.decoder_l2]
        self.block_l3 = [self.encoder_l3, self.decoder_l3]
        self.block_l4 = [self.encoder_l4, self.decoder_l4]

    def forward(self, x, depth=3):
        x = self.encode(x, encoder_depth=depth)
        x = self.decode(x, decoder_depth=depth)
        return x

    def encode(self, x, encoder_depth=3):
        # Apply L1 Encoder
        x = self.encoder_l1(x)

        if encoder_depth > 1:
            # Apply L2 Encoder
            x = self.encoder_l2(x)

        if encoder_depth > 2:
            # Apply L3 Encoder
            x = self.encoder_l3(x)

        if encoder_depth > 3:
            # Apply L4 Encoder
            x = self.encoder_l4(x)

        return x

    def decode(self, x, decoder_depth=3):

        if decoder_depth > 3:
            # Apply L4 Decoder
            x = self.decoder_l4(x)

        if decoder_depth > 2:
            # Apply L3 Decoder
            x = self.decoder_l3(x)

        if decoder_depth > 1:
            # Apply L2 Decoder
            x = self.decoder_l2(x)

        # Apply L1 Decoder
        x = self.decoder_l1(x)

        return x


def freeze_block(block, freeze):
    for components in block:
        for param in components:
            param.requires_grad = not freeze


def train_network(net, dataloader_train, dataloader_eval, num_epochs, optimizer, device,
                  encoding_depth=3, freeze_l1=False, freeze_l2=False, freeze_l3=False):

    # Perform Greedy block-wise Training according with train configuration parameters
    freeze_block(net.block_l1, freeze_l1)  # Freeze Block L1 if needed
    freeze_block(net.block_l2, freeze_l2)  # Freeze Block L2 if needed
    freeze_block(net.block_l3, freeze_l3)  # Freeze Block L3 if needed

    # Empty lists to store training statistics
    train_loss = []
    eval_loss = []

    # Training Phase
    net.train()
    for epoch in range(num_epochs):

        # Show the progress bar
        if net.verbose:
            epoch_progress = 'Epoch ' + (epoch + 1).__str__() + '/' + num_epochs.__str__()
            dataloader_train = tqdm(dataloader_train, bar_format='{l_bar}|{bar}| ['
                                                                 + epoch_progress
                                                                 + ' {postfix}, {elapsed}<{remaining}]')
        batch_loss_train = []

        for batch in dataloader_train:

            # Extract useful data and move tensors to the selected device
            waterfalls = batch['Waterfalls'].to(device)
            signals = batch['SignalWaterfalls'].to(device)

            # Reset the parameters' gradient to zero
            optimizer.zero_grad()

            # Forward pass
            output = net.forward(waterfalls, depth=encoding_depth)
            loss = net.loss_fn(output, signals)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Get loss value
            batch_loss_train.append(loss.data.cpu().numpy())
            if net.verbose:
                dataloader_train.set_postfix_str('Partial Train Loss: ' + np.array2string(batch_loss_train[-1]))

        train_loss.append(np.mean(batch_loss_train))

        batch_loss_eval = []

        # Validation Phase
        net.eval()
        with torch.no_grad():  # No need to track the gradients

            for batch in dataloader_eval:
                # Extract useful data and move tensors to the selected device
                waterfalls = batch['Waterfalls'].to(device)
                signals = batch['SignalWaterfalls'].to(device)

                # Forward pass
                output = net.forward(waterfalls, depth=encoding_depth)

                # Get loss value
                loss = net.loss_fn(output, signals)
                batch_loss_eval.append(loss.data.cpu().numpy())

        eval_loss.append(np.mean(batch_loss_eval))
        if net.verbose:
            print('Validation Loss: ' + eval_loss[-1].__str__())

    return train_loss, eval_loss


if __name__ == '__main__':

    # Load the trained model
    load_configuration = 'DAE4B_fd'
    net_state_dict = torch.load('data/' + load_configuration + '_net_parameters.torch', map_location='cpu')

    # Initialize the network
    net = DenoisingNet(verbose=True)
    # Set the loaded parameters to the network
    net.load_state_dict(net_state_dict)

    # Load the dataset
    dataset = Waterfalls(fpath='../../DATASETS/Waterfalls/Waterfalls_fish.mat',
                         transform=utils.NormalizeSignal((1200, 20)))

    # %% Test the network with random test sample

    # Load the test-sample indices from the saved configuration
    test_idx = np.load('data/' + load_configuration + '_data.npy')[2]

    # Select randomly one sample
    idx = test_idx[np.random.randint(len(test_idx))]

    # Extracts data from the sample and transform them into tensors
    waterfalls = utils.get_tensor(dataset[idx]['Waterfalls'], float_cast=True, unsqueeze=2).to('cpu')
    signals = utils.get_tensor(dataset[idx]['SignalWaterfalls'], float_cast=True, unsqueeze=2).to('cpu')

    # Extract parameters from the sample
    parameters = dataset[idx]['Parameters']

    # Use the waterfalls sample to evaluate the model
    net.eval()
    with torch.no_grad():
        output = net(waterfalls)

    # Plot the waterfalls, the output of the network and the waterfalls without the noise
    utils.plot_reconstruction(waterfalls, output, signals, parameters, hard_threshold=False, show_error=False)

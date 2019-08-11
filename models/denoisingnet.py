import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import utils
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
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
            nn.Conv2d(1, 32, (6, 3), stride=(2, 1), padding=(2, 1)),  # output dimensions: 32 * 600 * 20
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.1)
        )

        # Encoder Block L2
        self.encoder_l2 = nn.Sequential(
            nn.Conv2d(32, 64, (4, 3), stride=(2, 1), padding=(1, 2),
                      dilation=(1, 2)),  # output dimensions: 64 * 300 * 20
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1)
        )

        # Encoder Block L3
        self.encoder_l3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride=(3, 1), padding=(1, 0)),  # output dimensions: 128 * 100 * 16
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1)
        )

        # Encoder Block L4
        self.encoder_l4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # output dimensions: 256 * 50 * 8
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.1)
        )

        # DECODER BLOCKS
        # Decoder Block L4
        self.decoder_l4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # output dimensions: 128 * 100 * 16
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1)
        )

        # Decoder Block L3
        self.decoder_l3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, stride=(3, 1), padding=(1, 0)),  # output dimensions: 64 * 300 * 20
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1)
        )
        # Decoder Block L2
        self.decoder_l2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (4, 3), stride=(2, 1), padding=(1, 2),
                               dilation=(1, 2)),  # output dimensions: 32 * 600 * 20
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1),
        )

        # Decoder Block L1
        self.decoder_l1 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, (6, 3), stride=(2, 1), padding=(2, 1)),  # output dimensions: 1 * 1200 * 20
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # Blocks pointer
        self.block_l1 = [self.encoder_l1, self.decoder_l1]
        self.block_l2 = [self.encoder_l2, self.decoder_l2]
        self.block_l3 = [self.encoder_l3, self.decoder_l3]
        self.block_l4 = [self.encoder_l4, self.decoder_l4]

    def forward(self, x, depth=4):
        x = self.encode(x, encoder_depth=depth)
        x = self.decode(x, decoder_depth=depth)
        return x

    def encode(self, x, encoder_depth=4):
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

    def decode(self, x, decoder_depth=4):

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

    def accuracy(self, dataloader_eval, device=torch.device("cpu"), encoding_depth=4, print_summary=False):

        # Empty tensors to store predictions and labels
        predictions_soft = torch.Tensor().float().cpu().view((0, 0))
        labels = torch.Tensor().float().cpu().view((0, 0))

        print('[Evaluation of the test samples...]')
        self.eval()  # Validation Mode
        with torch.no_grad():  # No need to track the gradients

            for batch in tqdm(dataloader_eval):
                # Extract noisy waterfalls and move tensors to the selected device
                noisy_waterfalls = batch['Waterfalls'].to(device)

                # Forward pass - Reconstructed Waterfalls
                waterfalls_rec = self.forward(noisy_waterfalls, depth=encoding_depth)

                # Flatten the reconstructed waterfalls and append it to the predictions
                predictions_soft = torch.cat((predictions_soft, waterfalls_rec.view([waterfalls_rec.size(0), -1])),
                                             dim=0)

                # Flatten the signals and append it to the labels
                labels = torch.cat((labels, batch['SignalWaterfalls'].cpu().view([waterfalls_rec.size(0), -1])), dim=0)

        # Compute hard prediction and convert data form tensors to numpy vectors
        predictions = utils.binary_threshold(predictions_soft, threshold=0.5).view([-1]).numpy()
        labels = labels.view([-1]).numpy()

        print('[Computation of the accuracy metrics...]')
        # Collects several evaluation metrics
        result_metrics = {
            'Accuracy': accuracy_score(labels, predictions),  # Compute the Accuracy metric
            'BalancedAccuracy': balanced_accuracy_score(labels, predictions),  # Compute the Balanced Accuracy metric
            'F-Score': f1_score(labels, predictions),  # Compute the F-Score metric
            'Precision': precision_score(labels, predictions),  # Compute the Precision metric
            'Recall': recall_score(labels, predictions)  # Compute the Recall metric
        }

        if print_summary:
            num_ones = np.count_nonzero(labels)
            num_zeros = len(labels) - num_ones
            print('Performance evaluation (' + predictions_soft.shape[0].__str__() + ' Samples)'
                  + '\nPixels: ' + num_zeros.__str__() + ' Zeros, ' + num_ones.__str__() + ' Ones'
                  + '\n\tAccuracy: ' + result_metrics['Accuracy'].__str__()
                  + '\n\tBalanced Accuracy: ' + result_metrics['BalancedAccuracy'].__str__()
                  + '\n\tF-Score: ' + result_metrics['F-Score'].__str__()
                  + '\n\tPrecision: ' + result_metrics['Precision'].__str__()
                  + '\n\tRecall: ' + result_metrics['Recall'].__str__())

        return result_metrics


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


# Show a summary of DenoisingNet
if __name__ == '__main__':
    # Load the trained model
    load_configuration = 'DAE4B_fd_c'
    net_state_dict = torch.load('data/' + load_configuration + '_net_parameters.torch', map_location='cpu')

    # Initialize the network
    net = DenoisingNet(verbose=True)
    # Set the loaded parameters to the network
    net.load_state_dict(net_state_dict)

    # Load the dataset
    dataset = Waterfalls(fpath='../../DATASETS/Waterfalls/Waterfalls_fish.mat',
                         transform=utils.NormalizeSignal((1200, 20)))

    # %% Evaluate the accuracy metrics

    # Load the test-sample indices from the saved configuration
    test_indices = np.load('data/' + load_configuration + '_data.npy')[2]

    # Load test data efficiently
    test_samples = DataLoader(dataset, batch_size=32,
                              shuffle=False,
                              sampler=SubsetRandomSampler(test_indices),
                              collate_fn=utils.waterfalls_collate)

    # Compute the accuracy metrics
    metrics = net.accuracy(test_samples, print_summary=True)

    # %% Analyze network weights

    # Extract weights from the trained network
    weights_l1 = net.encoder_l1[0].weight.data.numpy()  # Block 1 weights
    weights_l2 = net.encoder_l2[0].weight.data.numpy()  # Block 2 weights
    weights_l3 = net.encoder_l3[0].weight.data.numpy()  # Block 3 weights
    weights_l4 = net.encoder_l4[0].weight.data.numpy()  # Block 4 weights

    # Show learnt Conv2D kernels
    utils.plot_kernels(weights_l1, 4, 2, 'Encoder Block 1 Kernels')
    utils.plot_kernels(weights_l2, 4, 4, 'Encoder Block 2 Kernels')
    utils.plot_kernels(weights_l2, 6, 5, 'Encoder Block 3 Kernels')
    utils.plot_kernels(weights_l2, 8, 8, 'Encoder Block 4 Kernels')

    # %% View the reconstruction performance with random test sample

    # Select randomly one sample
    idx = test_indices[np.random.randint(len(test_indices))]

    # Extracts data from the sample and transform them into tensors
    waterfalls = utils.get_tensor(dataset[idx]['Waterfalls'], float_cast=True, unsqueeze=2).cpu()
    signals = utils.get_tensor(dataset[idx]['SignalWaterfalls'], float_cast=True, unsqueeze=2).cpu()

    # Extract parameters from the sample
    parameters = dataset[idx]['Parameters']

    # Use the waterfalls sample to evaluate the model
    net.eval()
    with torch.no_grad():
        output = net(waterfalls, depth=4)

    # Plot the waterfalls, the output of the network and the waterfalls without the noise
    utils.plot_reconstruction(waterfalls, output, signals, parameters, hard_threshold=False, show_error=False)

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import utils
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from load_data import Waterfalls, get_indices
import numpy as np
from torchvision import transforms
from models.support_modules import EncoderBlock


class DenoisingNet(nn.Module):

    def __init__(self, verbose=True):
        super().__init__()

        # Loss function - Binary Cross Entropy
        self.loss_fn = torch.nn.BCELoss()

        # Verbose Mode
        self.verbose = verbose

        # Layers configuration
        configuration = {
            'N': 6,
            'in_channels': [1, 16, 32, 64, 128, 128],
            'out_channels': [16, 32, 64, 128, 128, 128],
            'kernel': [(6, 3), (4, 3), 5, 4, 4, (3, 4)],
            'stride': [(2, 1), (2, 1), (3, 1), 2, 2, 1],
            'padding': [(2, 1), (1, 1), (1, 0), 1, 1, (1, 0)],
            'dilation': [1, 1, 1, 1, 1, 1],
            'dropout': 0.1
        }

        # ENCODER BLOCKS
        self.blocks = nn.ModuleList([EncoderBlock(configuration, i) for i in range(configuration['N'])])

    def forward(self, x, depth):
        x = self.encode(x, encoder_depth=depth)
        x = self.decode(x, decoder_depth=depth)
        return x

    def train_block(self, idx):
        for n, block in enumerate(self.blocks):
            if n == idx:
                block.train()
            else:
                block.eval()

    def encode(self, x, encoder_depth):
        for i in range(encoder_depth):
            x = self.blocks[i].encoder(x)

        return x

    def decode(self, x, decoder_depth):
        for i in reversed(range(decoder_depth)):
            x = self.blocks[i].decoder(x)

        return x

    def accuracy(self, dataloader, device=torch.device("cpu"), depth=6, print_summary=False,
                 verbose=False, threshold=0.5):

        # Empty tensors to store predictions and labels
        predictions_soft = torch.zeros([0, 24000]).float().to(device)
        labels = torch.zeros([0, 24000]).float().to(device)

        print('[Evaluation of the test samples...]')
        self.eval()  # Validation Mode
        with torch.no_grad():  # No need to track the gradients

            if verbose:
                dataloader = tqdm(dataloader)

            for batch in dataloader:
                # Extract noisy waterfalls and move tensors to the selected device
                noisy_waterfalls = batch['Waterfalls'].to(device)
                signals = batch['SignalWaterfalls'].to(device)

                # Forward pass - Reconstructed Waterfalls
                waterfalls_rec = self.forward(noisy_waterfalls, depth=depth)

                # Flatten the reconstructed waterfalls and append it to the predictions
                predictions_soft = torch.cat((predictions_soft,
                                              waterfalls_rec.view([waterfalls_rec.size(0), -1])),
                                             dim=0)

                # Flatten the signals and append it to the labels
                labels = torch.cat((labels, signals.view([waterfalls_rec.size(0), -1])), dim=0)

        # Compute hard prediction and convert data form tensors to numpy vectors
        predictions = utils.binary_threshold(predictions_soft, threshold=threshold,
                                             device=device).view([-1]).cpu().numpy()
        labels = labels.view([-1]).cpu().numpy()

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


def freeze_block(block, freeze, verbose=False):
    for param in block.parameters():
        param.requires_grad = not freeze
    if verbose:
        print('\tBlock ' + block.index.__str__() + ' Frozen: ' + freeze.__str__())


def train_network(net, dataloader_train, dataloader_eval, num_epochs, optimizer, device,
                  depth=1, full_train=False, mode='autoecoder'):
    # Empty lists to store training statistics
    train_loss = []
    eval_loss = []
    predictions = []
    train_accuracy = []
    eval_accuracy = []

    for epoch in range(num_epochs):

        # Training Phase
        if full_train:
            # Set the entire network in train mode
            net.train()
        else:
            # Set only the desired block in train mode
            net.train_block(depth - 1)

        # Show the progress bar
        if net.verbose:
            epoch_progress = 'Epoch ' + (epoch + 1).__str__() + '/' + num_epochs.__str__()
            dataloader_train = tqdm(dataloader_train, bar_format='{l_bar}|{bar}| ['
                                                                 + epoch_progress
                                                                 + ' {postfix}, {elapsed}<{remaining}]')
        batch_loss_train = []
        batch_predictions = []

        for batch in dataloader_train:

            # Extract useful data and move tensors to the selected device
            waterfalls = batch['Waterfalls'].to(device)
            if mode == 'tracker':
                labels = batch['Paths2D'].to(device)
            else:
                labels = batch['SignalWaterfalls'].to(device)

            # Reset the parameters' gradient to zero
            optimizer.zero_grad()

            # Forward pass
            output = net.forward(waterfalls, depth=depth)
            loss = net.loss_fn(output, labels)

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
                if mode == 'tracker':
                    labels = batch['Paths2D'].to(device)
                else:
                    labels = batch['SignalWaterfalls'].to(device)

                # Forward pass
                output = net.forward(waterfalls, depth=depth)

                # Get loss value
                loss = net.loss_fn(output, labels)
                batch_loss_eval.append(loss.data.cpu().numpy())

        eval_loss.append(np.mean(batch_loss_eval))
        if net.verbose:
            print('Validation Loss: ' + eval_loss[-1].__str__())

    return train_loss, eval_loss, train_accuracy, eval_accuracy


# Show a summary of DenoisingNet
if __name__ == '__main__':
    WATERFALLS_SIZE = (1200, 20)
    train_part, validation_part, test_part = get_indices()

    # Load the trained model
    mode = 'a'
    load_configuration = 'DenoisingNet-5e'
    configuration = torch.load('data/' + load_configuration + '_net_parameters.torch', map_location='cpu')

    # Initialize the network
    net = DenoisingNet(verbose=True)


    # Set the loaded parameters to the network
    net.load_state_dict(configuration['DenoisingNet'])

    # Load the dataset
    if mode == 'tracking':
        transform = transforms.Compose(
                             [utils.NormalizeSignal(WATERFALLS_SIZE),
                              utils.Paths2D(WATERFALLS_SIZE)]
                         )
    else:
        transform = utils.NormalizeSignal((1200, 20))

    dataset = Waterfalls(fpath='../../DATASETS/Waterfalls/Waterfalls_fish.mat',
                         transform=transform)

    # %% Evaluate the accuracy metrics
    test_samples = DataLoader(dataset, batch_size=32,
                              shuffle=False,
                              sampler=SubsetRandomSampler(test_part),
                              collate_fn=utils.waterfalls_collate)

    # Compute the accuracy metrics
    # metrics = net.accuracy(test_samples, print_summary=True)

    # %% Analyze network weights

    # Extract weights from the trained network
    weights_l1 = net.blocks[0].encoder[0].weight.data.numpy()  # Block 1 weights
    weights_l2 = net.blocks[1].encoder[0].weight.data.numpy()  # Block 2 weights
    weights_l3 = net.blocks[2].encoder[0].weight.data.numpy()  # Block 3 weights
    weights_l4 = net.blocks[3].encoder[0].weight.data.numpy()  # Block 4 weights

    # Show learnt Conv2D kernels
    # utils.plot_kernels(weights_l1, 4, 2, 'Encoder Block 1 Kernels')
    # utils.plot_kernels(weights_l2, 4, 4, 'Encoder Block 2 Kernels')
    # utils.plot_kernels(weights_l2, 6, 5, 'Encoder Block 3 Kernels')
    # utils.plot_kernels(weights_l2, 8, 8, 'Encoder Block 4 Kernels')

    # %% View the reconstruction performance with random test sample

    # Select randomly one sample
    idx = test_part[np.random.randint(len(test_part))]

    # Extracts data from the sample and transform them into tensors
    waterfalls = utils.get_tensor(dataset[idx]['Waterfalls'], float_cast=True, unsqueeze=2).cpu()
    signals = utils.get_tensor(dataset[idx]['SignalWaterfalls'], float_cast=True, unsqueeze=2).cpu()

    # Extract parameters from the sample
    parameters = dataset[idx]['Parameters']

    # Use the waterfalls sample to evaluate the model
    net.eval()
    with torch.no_grad():
        output = net.forward(waterfalls, depth=6)

    # Plot the waterfalls, the output of the network and the waterfalls without the noise
    utils.plot_reconstruction(waterfalls, output, signals, parameters, hard_threshold=False,
                              show_error=False, save=True)



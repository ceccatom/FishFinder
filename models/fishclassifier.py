import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import utils
from torch import nn
from models.denoisingnet import DenoisingNet
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from load_data import Waterfalls
import numpy as np


class FeatureClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, dropout_p=0.1):
        super().__init__()

        self.num_classes = num_classes

        # Loss function (same for all the branches) - Cross Entropy Loss
        # TODO Consider the weight parameter for imbalanced classes
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # The set of layer that compose the classifier
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(dropout_p),
            nn.Linear(128, 96),
            nn.ReLU(True),
            nn.Dropout(dropout_p),
            nn.Linear(96, 64),
            nn.ReLU(True),
            nn.Dropout(dropout_p),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Dropout(dropout_p),
            nn.Linear(32, num_classes)
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)


class FishClassifier(nn.Module):

    def __init__(self, classification_parameters, estimate_all=False, conv_part=False, verbose=True):
        super().__init__()

        # Verbose Mode
        self.verbose = verbose

        if classification_parameters['IntermediateDimension'] is None:
            self.intermediate_dim = 384
        else:
            self.intermediate_dim = classification_parameters['IntermediateDimension']

        if classification_parameters['HiddenFeaturesDimension'] is None:
            self.features_hidden_dim = 64
        else:
            self.features_hidden_dim = classification_parameters['HiddenFeaturesDimension']

        self.branches = nn.ModuleDict({
            'NumTarget': FeatureClassifier(self.intermediate_dim,
                                           self.features_hidden_dim,
                                           classification_parameters['t_dim'],
                                           dropout_p=0.1),
        })

        self.estimate_all = estimate_all

        if estimate_all:
            self.branches.update({
                'Velocity': FeatureClassifier(self.intermediate_dim,
                                              self.features_hidden_dim,
                                              classification_parameters['v_dim'],
                                              dropout_p=0.1),
                'Width': FeatureClassifier(self.intermediate_dim,
                                           self.features_hidden_dim,
                                           classification_parameters['w_dim'],
                                           dropout_p=0.1),
            })

        # # Convolutional part of the common classifier
        self.conv_part = conv_part
        if conv_part:
            self.conv_part = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=4, stride=1, padding=1, dilation=1),  # output dimensions 128 * 24
                nn.ReLU(True),
                nn.MaxPool1d(kernel_size=2),  # output dimensions 128 * 12
                nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),  # output dimensions 64 * 12
                nn.ReLU(True),
                nn.Conv1d(64, 64, kernel_size=3, stride=1, dilation=2),  # output dimensions 64 * 8
                nn.ReLU(True)
            )
        # self.conv_part = nn.Sequential(
        #     nn.Conv2d(256, 512, 3, stride=1, dilation=2, padding=1),  # output dimensions: 512 * 48 * 6
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(512),
        #     nn.MaxPool2d(2, padding=0),  # output dimensions: 512 * 24 * 3
        #     nn.Conv2d(512, 1024, 3, stride=(2, 1), padding=1),  # output dimensions: 1024 * 12 * 3
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(1024),
        #     nn.MaxPool2d(3, padding=0),  # output dimensions: 1024 * 4 * 1
        # )

        # Linear part of the classifier
        self.lin_part = nn.Sequential(
            nn.Linear(512, self.intermediate_dim),
            nn.ReLU(True),
            nn.Dropout(0.1),
        )

    # Notice that the input x must be encoded by an external encoder
    def forward(self, x):
        if self.conv_part:
            # Apply the convolutional part of the classifier
            x = x.squeeze()
            x = self.conv_part(x)
            # Flatten the output
            x = x.view([x.shape[0], -1])

        # Apply the dense part of the classifier
        x = self.lin_part(x)

        if self.estimate_all:
            # The output takes 3 different branches, one for each desired feature
            velocity = self.branches['Velocity'](x)
            width = self.branches['Width'](x)
            targets = self.branches['NumTarget'].forward(x)
            return velocity, width, targets
        else:
            return self.branches['NumTarget'].forward(x)

    def accuracy(self, encoder, dataloader_eval, classes, device=torch.device("cpu"), print_summary=False):

        if self.conv_part:
            encoder_depth = 6
        else:
            encoder_depth = 9

        # Empty tensors to store predictions and labels
        predictions_soft_v = torch.Tensor().float().cpu()
        predictions_soft_w = torch.Tensor().float().cpu()
        predictions_soft_t = torch.Tensor().float().cpu()
        labels_v = np.array([])
        labels_w = np.array([])
        labels_t = np.array([])

        print('[Evaluation of the test samples...]')
        self.eval()  # Validation Mode
        with torch.no_grad():  # No need to track the gradients

            for batch in tqdm(dataloader_eval):
                # Extract noisy waterfalls and move tensors to the selected device
                noisy_waterfalls = batch['Waterfalls'].to(device)

                encoded_waterfalls = encoder.encode(noisy_waterfalls, encoder_depth=encoder_depth)
                # Forward pass - Reconstructed Waterfalls

                if classifier.estimate_all:
                    v, w, t = self.forward(encoded_waterfalls)
                    # Flatten the reconstructed waterfalls and append it to the predictions
                    predictions_soft_v = torch.cat((predictions_soft_v, v), dim=0)
                    predictions_soft_w = torch.cat((predictions_soft_w, w), dim=0)
                    predictions_soft_t = torch.cat((predictions_soft_t, t), dim=0)
                else:
                    t = self.forward(encoded_waterfalls)
                    predictions_soft_t = torch.cat((predictions_soft_t, t), dim=0)



                # Flatten the signals and append it to the labels
                labels_v = np.append(labels_v, batch['Parameters']['velocity'])
                labels_w = np.append(labels_w, batch['Parameters']['width'])
                labels_t = np.append(labels_t, batch['Parameters']['num_Targets'])

        # Compute hard prediction and convert data form tensors to numpy vectors
        _, preds_v = torch.max(predictions_soft_v.data, 1)
        _, preds_w = torch.max(predictions_soft_w.data, 1)
        _, preds_t = torch.max(predictions_soft_t.data, 1)

        preds_v = 15 + preds_v.numpy()
        preds_w = 3 + preds_w.numpy()
        preds_t = preds_t.numpy()

        print('[Computation of the accuracy metrics...]')
        # Collects several evaluation metrics
        result_metrics = {
            'v': confusion_matrix(labels_v, preds_v, labels=classes['v']),
            'w': confusion_matrix(labels_w, preds_w, labels=classes['w']),
            't': confusion_matrix(labels_t, preds_t, labels=classes['t']),

        }

        return result_metrics


def train_network(classifier, encoder, dataloader_train, dataloader_eval, num_epochs, optimizer, device,
                  finetuning=False):
    if classifier.conv_part:
        encoder_depth = 6
    else:
        encoder_depth = 9

    # Empty lists to store training statistics
    train_loss = []
    eval_loss = []

    # Training Phase
    classifier.train()
    if finetuning:
        # The encoder is trained, its parameters are improved for the classification task [Slower]
        encoder.train()
    else:
        # The encoder is used as fixed feature extractor and its parameter are not modified [Faster]
        encoder.eval()
    for epoch in range(num_epochs):

        # Show the progress bar
        if classifier.verbose:
            epoch_progress = 'Epoch ' + (epoch + 1).__str__() + '/' + num_epochs.__str__()
            dataloader_train = tqdm(dataloader_train, bar_format='{l_bar}|{bar}| ['
                                                                 + epoch_progress
                                                                 + ' {postfix}, {elapsed}<{remaining}]')
        batch_loss_train = []

        for batch in dataloader_train:

            # Move waterfalls and labels tensor to the selected device
            waterfalls, velocity_labels, width_labels, targets_labels = utils.get_labels(batch, device)

            # Forward part 1: Encode the waterfalls by using the external encoder.
            if finetuning:
                encoded_waterfalls = encoder.encode(waterfalls, encoder_depth=encoder_depth)
            else:
                with torch.no_grad():
                    encoded_waterfalls = encoder.encode(waterfalls, encoder_depth=encoder_depth)

            # Forward part 2
            # Reset the parameters' gradient to zero
            optimizer.zero_grad()



            if classifier.estimate_all:
                velocity_out, width_out, targets_out = classifier.forward(encoded_waterfalls)
                # Calculate the loss for each branch
                loss_velocity = classifier.branches['Velocity'].loss_fn(velocity_out, velocity_labels)
                loss_width = classifier.branches['Width'].loss_fn(width_out, width_labels)
                loss_targets = classifier.branches['NumTarget'].loss_fn(targets_out, targets_labels)
                # Sum up all the losses that have to be back-propagated into the network
                loss = loss_velocity + loss_width + loss_targets
            else:
                targets_out = classifier.forward(encoded_waterfalls)
                loss = classifier.branches['NumTarget'].loss_fn(targets_out, targets_labels)



            # Backward pass
            loss.backward()
            optimizer.step()

            # Get loss value
            # TODO differentiate the loss for the three branches
            batch_loss_train.append(loss.data.cpu().numpy())
            if classifier.verbose:
                dataloader_train.set_postfix_str('Partial Train Loss: ' + np.array2string(batch_loss_train[-1]))

        train_loss.append(np.mean(batch_loss_train))

        batch_loss_eval = []

        # Validation Phase
        encoder.eval()
        classifier.eval()
        with torch.no_grad():  # No need to track the gradients

            for batch in dataloader_eval:
                # Move waterfalls and labels tensor to the selected device
                waterfalls, velocity_labels, width_labels, targets_labels = utils.get_labels(batch, device)

                # Forward pass
                encoded_waterfalls = encoder.encode(waterfalls, encoder_depth=encoder_depth)
                # decoded_waterfalls = encoder.decode(encoded_waterfalls, decoder_depth=encoder_depth)
                # # Plot the waterfalls, the output of the network and the waterfalls without the noise
                # par = {
                #     'SNR': batch['Parameters']['SNR'][0],
                #     'velocity': batch['Parameters']['velocity'][0],
                #     'width': batch['Parameters']['width'][0],
                #     'num_Targets': batch['Parameters']['num_Targets'][0]
                # }
                # utils.plot_reconstruction(waterfalls[0], decoded_waterfalls[0], batch['SignalWaterfalls'][0], par, hard_threshold=False,
                #                           show_error=False)

                if classifier.estimate_all:
                    velocity_out, width_out, targets_out = classifier.forward(encoded_waterfalls)
                    # Calculate the loss for each branch
                    loss_velocity = classifier.branches['Velocity'].loss_fn(velocity_out, velocity_labels)
                    loss_width = classifier.branches['Width'].loss_fn(width_out, width_labels)
                    loss_targets = classifier.branches['NumTarget'].loss_fn(targets_out, targets_labels)
                    # Sum up all the losses that have to be back-propagated into the network
                    loss = loss_velocity + loss_width + loss_targets
                else:
                    targets_out = classifier.forward(encoded_waterfalls)
                    loss = classifier.branches['NumTarget'].loss_fn(targets_out, targets_labels)

                batch_loss_eval.append(loss.data.cpu().numpy())

        eval_loss.append(np.mean(batch_loss_eval))
        if classifier.verbose:
            print('Validation Loss: ' + eval_loss[-1].__str__())

    return train_loss, eval_loss


if __name__ == '__main__':
    # Load the trained model
    load_configuration_encoder = 'DAE9B-9B-fd'
    load_configuration_classifier = 'FishC-2'

    net_state_dict = torch.load('data/' + load_configuration_encoder + '_net_parameters.torch', map_location='cpu')

    # Initialize the network
    encoder = DenoisingNet(verbose=True)
    # Set the loaded parameters to the network
    encoder.load_state_dict(net_state_dict)

    # Load the dataset
    dataset = Waterfalls(fpath='../../DATASETS/Waterfalls/Waterfalls_fish.mat',
                         transform=utils.NormalizeSignal((1200, 20)))

    # Load the test-sample indices from the saved configuration
    test_indices = np.load('data/' + load_configuration_encoder + '_data.npy')[2]

    # Load test data efficiently
    test_samples = DataLoader(dataset, batch_size=32,
                              shuffle=False,
                              sampler=SubsetRandomSampler(test_indices),
                              collate_fn=utils.waterfalls_collate)

    # targets = np.unique(dataset[:][3])
    # velocities = np.unique(dataset[:][1])
    # widths = np.unique(dataset[:][2])

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

    classifier = FishClassifier(config, estimate_all=False, conv_part=True)
    class_state_dict = torch.load('data/' + load_configuration_classifier + '_net_parameters.torch', map_location='cpu')
    classifier.load_state_dict(class_state_dict)

    metrics = classifier.accuracy(encoder=encoder, dataloader_eval=test_samples, classes=classes)

    utils.plot_confusion_matrix(metrics['t'], classes['t'])
    utils.plot_confusion_matrix(metrics['v'], classes['v'])
    utils.plot_confusion_matrix(metrics['w'], classes['w'])

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import utils
from torch import nn
from models.denoisingnet import DenoisingNet
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from load_data import Waterfalls, get_indices
import numpy as np


class FeatureClassifier(nn.Module):

    def __init__(self, input_dim, num_classes, dropout_p=0.1):
        super().__init__()

        self.num_classes = num_classes

        # Loss function (same for all the branches) - Cross Entropy Loss
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
        )

    def forward(self, x):
        return self.network(x)


class FishClassifier(nn.Module):

    def __init__(self, classification_parameters, estimate_parameters=False, conv_part=True, verbose=True):
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
                                           classification_parameters['t_dim']),
        })

        self.estimate_parameters = estimate_parameters

        if estimate_parameters:
            self.branches.update({
                'Velocity': FeatureClassifier(self.intermediate_dim,
                                              classification_parameters['v_dim']),
                'Width': FeatureClassifier(self.intermediate_dim,
                                           classification_parameters['w_dim']),
            })

        # Convolutional part of the classifier
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
            x = x.squeeze(-1)
            x = self.conv_part(x)
            # Flatten the output
            x = x.view([x.shape[0], -1])

        # Apply the dense part of the classifier
        x = self.lin_part(x)

        if self.estimate_parameters:
            # The output takes 3 different branches, one for each desired feature
            velocity = self.branches['Velocity'](x)
            width = self.branches['Width'](x)
            targets = self.branches['NumTarget'].forward(x)
            return velocity, width, targets
        else:
            # Just return only the branch that classifier the number of targets
            return self.branches['NumTarget'].forward(x)

    def accuracy(self, encoder, dataloader_eval, classes, device=torch.device("cpu"), print_summary=False):

        if self.conv_part:
            encoder_depth = 6
        else:
            encoder_depth = 9

        # Empty tensors to store predictions and labels
        predictions_soft = torch.Tensor().float().to(device)
        labels = np.array([])

        print('[Evaluation of the samples...]')
        self.eval()  # Validation Mode
        encoder.eval()
        with torch.no_grad():  # No need to track the gradients

            for batch in tqdm(dataloader_eval):
                # Extract noisy waterfalls and move tensors to the selected device
                noisy_waterfalls, _, _, targets_labels = utils.get_labels(batch, device)

                encoded_waterfalls = encoder.encode(noisy_waterfalls.to(device), encoder_depth=encoder_depth)

                targets = self.forward(encoded_waterfalls)
                predictions_soft = torch.cat((predictions_soft, targets), dim=0)
                # loss = self.branches['NumTarget'].loss_fn(targets,  targets_labels)

                # Flatten the signals and append it to the labels
                labels = np.append(labels, batch['Parameters']['num_Targets'])

        # Compute hard prediction and convert data form tensors to numpy vectors
        try:
            _, preds = torch.max(predictions_soft.data, dim=1)
        except:
            preds = torch.zeros(0)

        preds = preds.cpu().numpy()

        print('[Computation of the accuracy metrics...]')
        # Collects several evaluation metrics
        conf_marix = np.zeros((13, 13))
        try:
            conf_marix = confusion_matrix(labels, preds, labels=classes['t'])
        except:
            pass
        result_metrics = {
            'matrix': conf_marix,
            'accuracy': accuracy_score(labels, preds),
            'balanced_accuracy': balanced_accuracy_score(labels, preds)
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
    eval_accuracy = []
    labels = np.array([])
    # Empty tensor to store predictions and labels
    predictions_soft = torch.Tensor().float().to(device)


    if finetuning:
        # The encoder is trained, its parameters are improved for the classification task [Slower]
        encoder.train()
    else:
        # The encoder is used as fixed feature extractor and its parameter are not modified [Faster]
        encoder.eval()
    for epoch in range(num_epochs):

        # Training Phase
        classifier.train()

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

            targets_out = classifier.forward(encoded_waterfalls)
            loss = classifier.branches['NumTarget'].loss_fn(targets_out, targets_labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Get loss value
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
                waterfalls.to(device)
                targets_labels.to(device)

                # Forward pass
                encoded_waterfalls = encoder.encode(waterfalls, encoder_depth=encoder_depth)

                targets_out = classifier.forward(encoded_waterfalls)
                loss = classifier.branches['NumTarget'].loss_fn(targets_out, targets_labels)

                batch_loss_eval.append(loss.data.cpu().numpy())
                predictions_soft = torch.cat((predictions_soft, targets_out), dim=0)
                labels = np.append(labels, batch['Parameters']['num_Targets'])

        eval_loss.append(np.mean(batch_loss_eval))
        _, preds = torch.max(predictions_soft.data, 1)
        preds = preds.cpu().numpy()
        accuracy = accuracy_score(labels, preds)
        eval_accuracy.append(accuracy)
        if classifier.verbose:
            print('Validation Loss: ' + eval_loss[-1].__str__() + ' Validation accuracy:' + accuracy.__str__())

    return train_loss, eval_loss, eval_accuracy


if __name__ == '__main__':
    _, _, test_part = get_indices()

    # Load the trained model
    load_configuration_classifier = 'Classifier-FT-15e'

    # Initialize the network
    encoder = DenoisingNet(verbose=True)
    encoder_state_dict = torch.load('data/' + load_configuration_classifier + '_net_parameters.torch', map_location='cpu')
    # Set the loaded parameters to the network
    encoder.load_state_dict(encoder_state_dict['encoder_tuned'])

    # Load the dataset
    dataset = Waterfalls(fpath='../../DATASETS/Waterfalls/Waterfalls_fish.mat',
                         transform=utils.NormalizeSignal((1200, 20)))

    test_part = utils.filter_indices(test_part, parameters=dataset[:], min_snr=2)

    # Load test data efficiently
    test_samples = DataLoader(dataset, batch_size=32,
                              shuffle=False,
                              sampler=SubsetRandomSampler(test_part),
                              collate_fn=utils.waterfalls_collate)

    classes, config = utils.get_classes()
    classifier = FishClassifier(config, estimate_parameters=False, conv_part=True)
    class_state_dict = torch.load('data/' + load_configuration_classifier + '_net_parameters.torch', map_location='cpu')
    classifier.load_state_dict(class_state_dict['classifier_post_tuning'])

    metrics = classifier.accuracy(encoder=encoder, dataloader_eval=test_samples, classes=classes)

    utils.plot_confusion_matrix(metrics['matrix'], classes['t'], normalize=True, save=True)

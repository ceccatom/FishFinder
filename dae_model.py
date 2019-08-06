import os
import torch
import matplotlib.pyplot as plt
import random
import utils
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from load_data import Waterfalls
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split


# %% Define the network architecture

class DenoisingAutoencoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()

        ### Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 4, (12, 3), stride=(10, 1), padding=(1, 2)),
            nn.ReLU(True),
            nn.Conv2d(4, 8, (4, 2), stride=(2, 2), padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, (10, 2), stride=(4, 2), padding=(1, 0)),
            nn.ReLU(True)
        )
        self.encoder_lin = nn.Sequential(
            nn.Linear(1344, 256),
            nn.ReLU(True),
            nn.Linear(256, encoded_space_dim)
        )

        ### Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 1344),
            nn.ReLU(True)
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, (10, 2), stride=(4, 2), padding=(1, 0)),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 4, (4, 2), stride=(2, 2), padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 1, (12, 3), stride=(10, 1), padding=(1, 2)),
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = x.view([x.size(0), -1])
        # Apply linear layers
        x = self.encoder_lin(x)
        return x

    def decode(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Reshape
        x = x.view([-1, 16, 14, 6])
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


def swap_in(lst, fro, to):
    lst[fro], lst[to] = lst[to], lst[fro]


### Initialize the network
encoded_space_dim = 64
net = DenoisingAutoencoder(encoded_space_dim=encoded_space_dim)

## Load dataset
dataset = Waterfalls()
idx = 0
sample = dataset[idx]['SignalWaterfalls']
img = utils.get_tensor(sample, float_cast=True, unsqueeze=2)


print('Original image shape:', img.shape)
# Encode the image
img_enc = net.encode(img)
print('Encoded image shape:', img_enc.shape)
# Decode the image
dec_img = net.decode(img_enc)
print('Decoded image shape:', dec_img.shape)

print('[Loading data...]')

# TODO remove this hardcoded parameters
m = 100  # len(dataset)
train_percentage = 0.9
dataset_indices = np.linspace(0, m - 1, m, dtype=np.int32)

# Split the considered dataset into train and test
train_indices, test_indices = train_test_split(dataset_indices, train_size=train_percentage)

# Load test&train data efficiently
train_dataloader = DataLoader(dataset, batch_size=32,
                              shuffle=False,  # The shuffle has been already performed by splitting the dataset
                              sampler=SubsetRandomSampler(train_indices),
                              collate_fn=utils.waterfalls_collate)

test_dataloader = DataLoader(dataset, batch_size=32,
                             shuffle=False,
                             sampler=SubsetRandomSampler(test_indices),
                             collate_fn=utils.waterfalls_collate)

### Define a loss function
loss_fn = torch.nn.MSELoss()

### Define an optimizer
lr = 1e-3  # Learning rate
optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

### If cuda is available set the device to GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# Move all the network parameters to the selected device (if they are already on that device nothing happens)
net.to(device)


# %% Network training

# Training function
def train_epoch(net, dataloader, loss_fn, optimizer, epoch):
    # Training
    net.train()

    progress = tqdm(dataloader, bar_format='{percentage:3.0f}%|{bar}| ['+epoch+' {postfix}, {elapsed}<{remaining}]')

    for sample_batch in progress:

        # Extract data and move tensors to the selected device
        waterfalls_batch = sample_batch['Waterfalls'].to(device)
        signal_batch = sample_batch['SignalWaterfalls'].to(device)
        # Forward pass
        output = net(waterfalls_batch)
        loss = loss_fn(output, signal_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print loss
        progress.set_postfix_str('Train Loss: '+np.array2string(loss.data.numpy()))
        # print('\t partial train loss: %f' % )


# Testing function
def test_epoch(net, dataloader, loss_fn, optimizer):
    # Validation
    net.eval()  # Evaluation mode (e.g. disable dropout)
    with torch.no_grad():  # No need to track the gradients
        conc_out = torch.Tensor().float()
        conc_label = torch.Tensor().float()

        for sample_batch in dataloader:

            # Extract data and move tensors to the selected device
            waterfalls_batch = sample_batch['Waterfalls'].to(device)
            signal_batch = sample_batch['SignalWaterfalls'].to(device)

            # Forward pass
            output = net(waterfalls_batch)

            # Concatenate with previous outputs
            conc_out = torch.cat([conc_out, output.cpu()])
            conc_label = torch.cat([conc_label, signal_batch.cpu()])

        # Evaluate global loss
        validation_loss = loss_fn(conc_out, conc_label)

    return validation_loss.data


# Training cycle
num_epochs = 5
for epoch in range(num_epochs):
    epoch_str = 'Epoch '+(epoch + 1).__str__()+'/'+num_epochs.__str__()
    ### Training
    train_epoch(net, train_dataloader, loss_fn=loss_fn, optimizer=optim, epoch=epoch_str)
    ### Validation
    val_loss = test_epoch(net, test_dataloader, loss_fn=loss_fn, optimizer=optim)
    # Print Validationloss
    print('\t Validation - Epoch %d/%d - loss: %f' % (epoch + 1, num_epochs, val_loss))

# %% Plot example

img = utils.get_tensor(dataset[test_indices[0]]['Waterfalls'], unsqueeze=2).to(device)
signal = utils.get_tensor(dataset[test_indices[0]]['SignalWaterfalls'], float_cast=True, unsqueeze=2).to(device)

net.eval()
with torch.no_grad():
    rec_img = net(img)
    error_loss = loss_fn(rec_img, signal)
    print('Sample Loss: ' + error_loss.numpy().__str__())

fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

### RAW data
axs[0].set_title('Denoising Autoencoder\n\nRAW data (Waterfalls)')
axs[0].imshow(img.cpu().squeeze().numpy(), aspect='auto', interpolation='bilinear', origin='lower')

### DAE Output
axs[1].set_title('DAE Output (Reconstructed Signal Waterfalls)')
axs[1].imshow(rec_img.cpu().squeeze().numpy(), aspect='auto', interpolation='none', origin='lower')

### True paths
axs[2].set_title('True paths (Signal Waterfalls)')
axs[2].imshow(signal.cpu().squeeze().numpy(), aspect='auto', interpolation='none', origin='lower')

fig.tight_layout()
fig.show()

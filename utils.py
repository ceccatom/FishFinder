import torch
from scipy.stats import halfnorm
import matplotlib.pyplot as plt
import numpy as np


def waterfalls_collate(batch):
    parameters = {
        'SNR': [],
        'velocity': [],
        'width': [],
        'num_Targets': []
    }
    paths = []
    waterfalls = []
    signal_waterfalls = []

    # TODO Consider to create two different versions of this function optimized for the two networks.
    #  In fact DenoisingNet just need waterfalls and signals whereas FishClassifier does not need signals but waterfalls
    #  and Parameters
    for item in batch:
        # Append items in the corresponding parameters list
        parameters['SNR'].append(item['Parameters']['SNR'])
        parameters['velocity'].append(item['Parameters']['velocity'])
        parameters['width'].append(item['Parameters']['width'])
        parameters['num_Targets'].append(item['Parameters']['num_Targets'])
        paths.append(item['Paths'])

        # Append waterfalls and convert them into tensors
        waterfalls.append(get_tensor(item['Waterfalls'], float_cast=True))

        # Append signals and convert them into tensors
        signal_waterfalls.append(get_tensor(item['SignalWaterfalls'], float_cast=True))

    batch_dictionary = {
        'Parameters': parameters,
        'Paths': paths,
        'Waterfalls': torch.stack(waterfalls).unsqueeze(1),
        'SignalWaterfalls': torch.stack(signal_waterfalls).unsqueeze(1)
    }

    return batch_dictionary


def fast_collate(batch):
    parameters = {
        'num_Targets': []
    }
    signal_waterfalls = []

    # TODO Consider to create two different versions of this function optimized for the two networks.
    #  In fact DenoisingNet just need waterfalls and signals whereas FishClassifier does not need signals but waterfalls
    #  and Parameters
    for item in batch:
        # Append items in the corresponding parameters list
        parameters['num_Targets'].append(item['Parameters']['num_Targets'])

        # Append signals and convert them into tensors
        signal_waterfalls.append(get_tensor(item['SignalWaterfalls'], float_cast=True))

    batch_dictionary = {
        'Parameters': parameters,
        'SignalWaterfalls': torch.stack(signal_waterfalls).unsqueeze(1)
    }

    return batch_dictionary


def get_tensor(element, float_cast=False, unsqueeze=0):
    out_tensor = torch.LongTensor(element)

    if float_cast:
        out_tensor = out_tensor.float()

    for _ in range(unsqueeze):
        out_tensor = out_tensor.unsqueeze(0)

    return out_tensor


def binary_threshold(batch, threshold=0.5):
    assert isinstance(threshold, float)
    t = torch.Tensor([threshold])  # threshold
    return (batch > t).float()


def plot_reconstruction(raw, rec, original, parameters, hard_threshold=True, show_error=False):
    num_subplots = 4

    data = np.zeros((num_subplots, 1200, 20))
    data[0][:][:] = raw.cpu().squeeze().numpy()
    if hard_threshold:
        data[1][:][:] = binary_threshold(rec.cpu().squeeze().numpy(), 0.5)
    else:
        data[1][:][:] = rec.cpu().squeeze().numpy()
    data[2][:][:] = original.cpu().squeeze().numpy()

    titles = ['SNR ' + parameters['SNR'].__str__()
              + ' | Velocity ' + parameters['velocity'].__str__()
              + ' | Width ' + parameters['width'].__str__()
              + ' | #Targets ' + parameters['num_Targets'].__str__()
              + '\nRAW data (Waterfalls)',
              'DAE Output (Reconstructed Signal Waterfalls)',
              'True paths (Signal Waterfalls)']

    if show_error:
        data[3][:][:] = (original - rec).cpu().squeeze().numpy()
        titles.append('Reconstruction Error')
    else:
        num_subplots = 3

    v_min = data.min()
    v_max = data.max()

    fig, axs = plt.subplots(num_subplots, 1, figsize=(12, 10), sharex=True)

    for idx in range(num_subplots):
        axs[idx].set_title(titles[idx])
        axs[idx].imshow(data[idx], aspect='auto', interpolation='none', origin='lower', vmin=v_min, vmax=v_max)

    fig.tight_layout()
    fig.show()


class AddNoise(object):
    """Add random noise to the image in a sample. The noise has an Half-normal distribution

    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        out_size = sample['SignalWaterfalls'].shape
        noise = halfnorm.rvs(size=out_size)
        sample['Waterfalls'] = sample['SignalWaterfalls'] + noise
        return sample


class NormalizeSignal(object):
    """Normalize 'Waterfalls' between 0 and 1

    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        val_min = sample['Waterfalls'].min()
        val_max = sample['Waterfalls'].max()

        sample['Waterfalls'] = (sample['Waterfalls'] - val_min) / (val_max - val_min)

        return sample

class OneCol(object):
    """Normalize 'Waterfalls' between 0 and 1

    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        sample['SignalWaterfalls'] = np.array([sample['Parameters']['num_Targets'], sample['Parameters']['num_Targets']]) # np.sum(np.transpose(sample['SignalWaterfalls'])[:][0:1])

        return sample


def size_calculator(dim, kernel_size, stride, padding, dilatation=1, inverse=False):
    if not inverse:
        return 1 + (dim + 2 * padding - dilatation * (kernel_size - 1) - 1) / stride
    else:
        return (dim - 1) * stride - 2 * padding + dilatation * (kernel_size - 1) + 1


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


def update_onehot_labels(batch, labels_container, velocity_onehot, width_onehot, targets_onehot):

    # Move labels to its tensor container
    labels_container[:][0] = torch.LongTensor(batch['Parameters']['velocity'])
    labels_container[:][1] = torch.LongTensor(batch['Parameters']['width'])
    labels_container[:][2] = torch.LongTensor(batch['Parameters']['num_Targets'])

    # Reset labels_onehot for each mini-batch
    velocity_onehot.zero_()
    width_onehot.zero_()
    targets_onehot.zero_()

    # Rearrange labels by using the One-Hot-Encoding scheme. Fast operations by using tensors allocated in GPU
    velocity_onehot.scatter_(1, (labels_container[:][0]-15).view([-1, 1]), 1)
    width_onehot.scatter_(1, (labels_container[:][1]-3).view([-1, 1]), 1)
    targets_onehot.scatter_(1, labels_container[:][2].view([-1, 1]), 1)
    pass

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.show()
    pass





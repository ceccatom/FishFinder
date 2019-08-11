import torch
from scipy.stats import halfnorm
import matplotlib.pyplot as plt
import numpy as np


def waterfalls_collate(batch):
    parameters = [[], [], [], []]
    paths = []
    waterfalls = []
    signal_waterfalls = []

    # TODO Consider to swap also the inner elements of 'Parameters'
    for item in batch:
        parameters.append(item['Parameters'])
        paths.append(item['Paths'])
        waterfalls.append(get_tensor(item['Waterfalls'], float_cast=True))
        signal_waterfalls.append(get_tensor(item['SignalWaterfalls'], float_cast=True))

    batch_dictionary = {
        'Parameters': parameters,
        'Paths': paths,
        'Waterfalls': torch.stack(waterfalls).unsqueeze(1),
        'SignalWaterfalls': torch.stack(signal_waterfalls).unsqueeze(1)
    }

    return batch_dictionary


def get_tensor(element, float_cast=False, unsqueeze=0):
    out_tensor = torch.tensor(element)

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
        sample['Waterfalls'] = sample['SignalWaterfalls']  + noise
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


def size_calculator(dim, kernel_size, stride, padding, dilatation=1, inverse=False):
    if not inverse:
        return 1 + (dim + 2 * padding - dilatation * (kernel_size - 1) - 1) / stride
    else:
        return (dim-1) * stride - 2 * padding + dilatation * (kernel_size - 1) + 1


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


import torch
from scipy.stats import halfnorm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


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
    paths2d = []

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

        if 'Paths2D' in item.keys():
            paths2d.append(get_tensor(item['Paths2D'], float_cast=True))

    batch_dictionary = {
        'Parameters': parameters,
        'Paths': paths,
        'Waterfalls': torch.stack(waterfalls).unsqueeze(1),
        'SignalWaterfalls': torch.stack(signal_waterfalls).unsqueeze(1),
    }

    if 'Paths2D' in item.keys():
        batch_dictionary.update({
            'Paths2D': torch.stack(paths2d).unsqueeze(1)
        })

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


def get_tensor(element, float_cast=False, unsqueeze=0, long=False):
    if long:
        out_tensor = torch.LongTensor(element)
    else:
        out_tensor = torch.tensor(element)

    if float_cast:
        out_tensor = out_tensor.float()

    for _ in range(unsqueeze):
        out_tensor = out_tensor.unsqueeze(0)

    return out_tensor


def binary_threshold(batch, threshold=0.5, device=torch.device("cpu")):
    assert isinstance(threshold, float)
    t = torch.Tensor([threshold]).to(device)  # threshold
    return (batch > t).float()


def plot_reconstruction(raw, rec, original, parameters, hard_threshold=True, show_error=False, save=False):
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
              'DenoisingNet Output (Reconstructed Signal Waterfalls)',
              'True paths (Signal Waterfalls)']

    if show_error:
        data[3][:][:] = (original - rec).cpu().squeeze().numpy()
        titles.append('Reconstruction Error')
    else:
        num_subplots = 3

    v_min = data.min()
    v_max = data.max()

    # Load style file
    plt.style.use('../styles.mplstyle')

    fig, axs = plt.subplots(num_subplots, 1, figsize=(12, 10), sharex=True)

    for idx in range(num_subplots):
        axs[idx].set_title(titles[idx])
        axs[idx].imshow(data[idx], aspect='auto', interpolation='none', origin='lower', vmin=v_min, vmax=v_max)

    fig.tight_layout()

    if save:
        esito = fig.savefig('./sample_reconstruction.pdf', dpi=300)

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


class Paths2D(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        sample['Paths2D'] = np.zeros((1200, 20), dtype=bool)
        support = np.zeros(shape=(1200 * 20), dtype=bool)
        if sample['Parameters']['num_Targets'] > 0:
            for i in range(sample['Paths'].shape[1]):
                linear_idxs = (sample['Paths'][:, i].astype(int), np.linspace(0, 19, 20, dtype=int))
                support[np.ravel_multi_index(linear_idxs, (1200, 20))] = True
                sample['Paths2D'] = np.bitwise_or(sample['Paths2D'], np.reshape(support, (1200, 20)))
                support[:] = False
        sample['Paths2D'] = sample['Paths2D'].astype(int)
        return sample


class OrderPaths(object):
    """Sort the paths in ascending order by using their mean

    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # means = np.mean(sample['Paths'], axis=0)
        # sample['Paths'] = np.array(sample['Paths'][:, np.argsort(means)], dtype=int)
        tmp = np.zeros((12, 1200, 20))
        support = np.zeros((1200 * 20))
        branches = np.random.permutation([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        if sample['Parameters']['num_Targets'] > 0:
            for i in range(sample['Paths'].shape[1]):
                sub_idxs = (sample['Paths'][:, i], np.linspace(0, 19, 20, dtype=int))
                support[np.ravel_multi_index(sub_idxs, (1200, 20))] = 1
                tmp[branches[i], :, :] = np.reshape(support, (1200, 20))
                support[:] = 0
        sample['Paths2D'] = np.sum(tmp, axis=0)
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


def update_onehot_labels(batch, labels_container, velocity_labels, width_labels, targets_labels):
    # Move labels to its tensor container
    labels_container[:][0] = torch.LongTensor(batch['Parameters']['velocity'])
    labels_container[:][1] = torch.LongTensor(batch['Parameters']['width'])
    labels_container[:][2] = torch.LongTensor(batch['Parameters']['num_Targets'])

    # Reset labels_onehot for each mini-batch
    velocity_labels.zero_()
    width_labels.zero_()
    targets_labels.zero_()

    # Rearrange labels by using the One-Hot-Encoding scheme. Fast operations by using tensors allocated in GPU
    velocity_labels.scatter_(1, (labels_container[:][0] - 15).view([-1, 1]), 1)
    width_labels.scatter_(1, (labels_container[:][1] - 3).view([-1, 1]), 1)
    targets_labels.scatter_(1, labels_container[:][2].view([-1, 1]), 1)
    pass


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          save=False):
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

    if save:
        # Load style file
        # plt.style.use('./styles.mplstyle')
        fig.savefig('./conf_matrix.pdf', dpi=300)
    pass


def get_labels(batch, device):
    waterfalls = batch['Waterfalls'].to(device)
    velocity_labels = get_tensor(batch['Parameters']['velocity'], long=True).to(device) - 15
    width_labels = get_tensor(batch['Parameters']['width'], long=True).to(device) - 3
    targets_labels = get_tensor(batch['Parameters']['num_Targets'], long=True).to(device)

    return waterfalls, velocity_labels, width_labels, targets_labels


def get_classes():
    classes = {
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

    return classes, config


def filter_indices(indices, parameters, min_snr=0, min_width=3, min_velocity=15):
    # Convert list to a numpy array
    indices = np.array(indices)

    # Check where data satisfies the required condition
    snr_cond = parameters[0, indices] >= min_snr
    width_cond = parameters[2, indices] >= min_width
    velocity_cond = parameters[1, indices] >= min_velocity

    # Bit-wise AND to satisfy all the conditions at the same time
    mask = np.bitwise_and(snr_cond, width_cond)
    mask = np.bitwise_and(mask, velocity_cond)

    # Return the filtered indices as python list
    return indices[mask].tolist()


def classification_performance(fpath, save=False):
    results = np.load(fpath)

    classes, config = get_classes()

    sns.set(style="ticks")

    min_snr = np.linspace(1, 3, 10)
    min_w = np.linspace(3, 10, 8)
    min_v = np.array([15, 18])
    dict1 = {
        'Minimum width': [],
        'Minimum SNR': [],
        'Minimum velocity': [],
        'Accuracy': [],
        'align': []
    }

    for k in range(2):
        for i in range(8):
            for j in range(10):
                conf_matrix = results[k, i, j]
                tot = np.sum(conf_matrix, axis=None)
                tp = np.trace(conf_matrix)
                wrong = tot - tp
                if tot > 500:
                    dict1['Accuracy'].append(tp / tot)
                    dict1['Minimum SNR'].append(min_snr[j])
                    dict1['Minimum velocity'].append(min_v[k])
                    dict1['Minimum width'].append(min_w[i])
                    dict1['align'].append('dots')

    data = pd.DataFrame(data=dict1)  # sns.load_dataset("dots")

    # Define a palette to ensure that colors will be
    # shared across the facets
    palette = dict(zip(data['Minimum width'].unique(),
                       sns.cubehelix_palette(8, start=.5, rot=-.75)))

    # Load style file
    plt.style.use('./styles.mplstyle')

    # Plot the lines on two facets
    p = sns.relplot(x="Minimum SNR", y="Accuracy",
                    hue="Minimum width", col="Minimum velocity",  # size="Minimum velocity", #
                    size_order=[15, 18], palette=palette,
                    height=5, aspect=.75, facet_kws=dict(sharex=False),
                    kind="line", legend="full", data=data)

    p.savefig('figs/classification_accuracy.pdf', dpi=300)

def denoising_performance(fpath, save=False):
    results = np.load(fpath)

    sns.set(style="ticks")

    metric = 'F-Score'

    min_snr = np.linspace(1, 3, 10)
    min_w = np.linspace(3, 10, 8)
    min_v = np.array([15, 18])
    dict1 = {
        'Minimum width': [],
        'Minimum SNR': [],
        'Minimum velocity': [],
        metric: []
    }


    for k in range(2):
        for i in range(8):
            for j in range(10):
                dict1[metric].append(results.item()[metric][k, i, j][0])
                dict1['Minimum SNR'].append(min_snr[j])
                dict1['Minimum velocity'].append(min_v[k])
                dict1['Minimum width'].append(min_w[i])

    data = pd.DataFrame(data=dict1)

    # Define a palette to ensure that colors will be
    # shared across the facets
    palette = dict(zip(data['Minimum width'].unique(),
                       sns.cubehelix_palette(8, start=.5, rot=-.75)))

    # Load style file
    plt.style.use('./styles.mplstyle')

    # Plot the lines on two facets
    p = sns.relplot(x="Minimum SNR", y=metric,
                    hue="Minimum width", col="Minimum velocity",  # size="Minimum velocity", #
                    size_order=[15, 18], palette=palette,
                    height=5, aspect=.75, facet_kws=dict(sharex=False),
                    kind="line", legend="full", data=data)

    p.savefig('figs/denoising_' + metric + '.pdf', dpi=300)
    plt.show()


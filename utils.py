import torch


def waterfalls_collate(batch):
    parameters = [[], [], [], []]
    paths = []
    waterfalls = []
    signal_waterfalls = []

    # TODO Consider to swap also the inner elements of 'Parameters'
    for item in batch:
        parameters.append(item['Parameters'])
        paths.append(item['Paths'])
        waterfalls.append(get_tensor(item['Waterfalls']))
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

import numpy as np
import torch


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def hard_update_ddpg(target, source):
    target.hard_update(source)


def average_update(target, sources):
    sources_named_parameters = [dict(source.named_parameters()) for source in sources]
    frac = 1 / len(sources)

    for name, param in target.named_parameters():
        source_params = [frac * source_named_parameters[name].data for source_named_parameters
                         in sources_named_parameters]

        param.data.copy_(sum(source_params))


def to_torch_tensor(numpy_array, cpu=False):
    tensor = torch.from_numpy(numpy_array.astype(np.float32))

    # device = torch.device(device_name)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if cpu:
        device = torch.device('cpu')
    tensor = tensor.to(device)
    return tensor
